from __future__ import annotations

import json
import re
import tempfile
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import pandas as pd
from langgraph.graph import END, START, StateGraph

try:
    from .ckan_support import DEFAULT_CKAN_ENDPOINT, normalize_ckan_base_url, package_search, package_show
    from .openui_support import OpenUIValidationError, _json_arg, parse_openui_lang
except ImportError:
    from ckan_support import DEFAULT_CKAN_ENDPOINT, normalize_ckan_base_url, package_search, package_show
    from openui_support import OpenUIValidationError, _json_arg, parse_openui_lang


TaskType = Literal["dataset_retrieval", "analysis", "openui_generation", "general_agent"]
VisualizationType = Literal["summary", "schema", "bar", "histogram", "quality"]


class WorkflowStep(TypedDict):
    node: str
    title: str
    detail: str


@dataclass
class UserIntent:
    task_type: TaskType
    query_terms: list[str]
    desired_visualization: VisualizationType
    language: str = "en"
    ambiguous: bool = False


@dataclass
class RetrievalResource:
    package_id: str
    package_title: str
    resource_id: str
    name: str
    format: str
    url: str


@dataclass
class RetrievalResult:
    endpoint: str
    query: str
    packages: list[dict[str, Any]] = field(default_factory=list)
    resources: list[RetrievalResource] = field(default_factory=list)
    selected: RetrievalResource | None = None
    confidence: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    rows: int = 0
    columns: int = 0
    schema: list[dict[str, Any]] = field(default_factory=list)
    missingness: list[dict[str, Any]] = field(default_factory=list)
    sample_rows: list[dict[str, Any]] = field(default_factory=list)
    chart_rows: list[dict[str, Any]] = field(default_factory=list)
    chart_x: str = ""
    chart_y: str = ""
    numeric_column: str = ""
    observations: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class UiPlan:
    layout_type: VisualizationType
    title: str
    subtitle: str
    metrics: list[dict[str, str]] = field(default_factory=list)
    table_rows: list[dict[str, Any]] = field(default_factory=list)
    chart_rows: list[dict[str, Any]] = field(default_factory=list)
    chart_x: str = ""
    chart_y: str = ""
    histogram_values: list[float] = field(default_factory=list)
    histogram_column: str = ""
    followups: list[str] = field(default_factory=list)
    callout: str = ""


class AgentWorkflowState(TypedDict, total=False):
    prompt: str
    ckan_endpoint: str
    dataset_path: str
    steps: list[WorkflowStep]
    intent: dict[str, Any]
    retrieval_result: dict[str, Any]
    analysis_result: dict[str, Any]
    ui_plan: dict[str, Any]
    openui_lang: str
    next_action: str


def run_agent_workflow(prompt: str, ckan_endpoint: str | None = None, dataset_path: str | None = None) -> AgentWorkflowState:
    initial_state: AgentWorkflowState = {
        "prompt": prompt.strip() or "Summarize this dataset",
        "ckan_endpoint": _normalize_endpoint_or_default(ckan_endpoint),
        "steps": [],
    }
    if dataset_path:
        initial_state["dataset_path"] = dataset_path
    return cast(AgentWorkflowState, build_agent_workflow().invoke(initial_state))


def build_agent_workflow():
    graph = StateGraph(AgentWorkflowState)
    graph.add_node("route_intent", route_intent)
    graph.add_node("retrieve_ckan", retrieve_ckan)
    graph.add_node("analyze_data", analyze_data)
    graph.add_node("plan_openui", plan_openui)
    graph.add_node("translate_openui", translate_openui)
    graph.add_edge(START, "route_intent")
    graph.add_conditional_edges(
        "route_intent",
        route_next_action,
        {
            "retrieve_ckan": "retrieve_ckan",
            "analyze_data": "analyze_data",
            "plan_openui": "plan_openui",
        },
    )
    graph.add_edge("retrieve_ckan", "analyze_data")
    graph.add_edge("analyze_data", "plan_openui")
    graph.add_edge("plan_openui", "translate_openui")
    graph.add_edge("translate_openui", END)
    return graph.compile()


def route_next_action(state: AgentWorkflowState) -> str:
    return state.get("next_action", "plan_openui")


def route_intent(state: AgentWorkflowState) -> AgentWorkflowState:
    prompt = state.get("prompt", "")
    intent = classify_intent(prompt, has_dataset=bool(state.get("dataset_path")))
    if intent.task_type == "dataset_retrieval":
        next_action = "retrieve_ckan"
        detail = f"Deterministic router selected CKAN retrieval for query '{' '.join(intent.query_terms) or 'open data'}'."
    elif intent.task_type == "analysis":
        next_action = "analyze_data"
        detail = f"Deterministic router selected dataset analysis with {intent.desired_visualization} intent."
    else:
        next_action = "plan_openui"
        detail = "Deterministic router selected a general no-data response."
    return {
        "intent": asdict(intent),
        "next_action": next_action,
        "steps": [*state.get("steps", []), {"node": "route_intent", "title": intent.task_type, "detail": detail}],
    }


def retrieve_ckan(state: AgentWorkflowState) -> AgentWorkflowState:
    intent = _intent_from_state(state)
    endpoint = state.get("ckan_endpoint", DEFAULT_CKAN_ENDPOINT)
    query = " ".join(intent.query_terms) or _fallback_query(state.get("prompt", ""))
    result = RetrievalResult(endpoint=endpoint, query=query)
    try:
        search_result = package_search(endpoint, query, rows=5)
        packages = search_result.get("results", [])
        result.packages = [package for package in packages if isinstance(package, dict)]
        result.resources = _resources_from_packages(result.packages)
        result.selected = _select_resource(result.resources)
        result.confidence = 0.8 if result.selected else 0.35 if result.packages else 0.0
        if result.selected:
            try:
                package_detail = package_show(endpoint, result.selected.package_id)
                result.resources = _resources_from_packages([package_detail]) or result.resources
                result.selected = _select_resource(result.resources) or result.selected
            except Exception as exc:
                result.errors.append(f"package_show failed: {_short_error(exc)}")
    except Exception as exc:
        result.errors.append(_short_error(exc))
    detail = (
        f"Searched CKAN for '{query}' and selected {result.selected.name}."
        if result.selected
        else f"Searched CKAN for '{query}' but did not find a CSV-like resource."
    )
    return {
        "retrieval_result": _retrieval_to_dict(result),
        "steps": [*state.get("steps", []), {"node": "retrieve_ckan", "title": "ckan_retrieval", "detail": detail}],
    }


def analyze_data(state: AgentWorkflowState) -> AgentWorkflowState:
    intent = _intent_from_state(state)
    retrieval = _retrieval_from_state(state)
    source = state.get("dataset_path") or (retrieval.selected.url if retrieval and retrieval.selected else "")
    analysis = AnalysisResult()
    if not source:
        analysis.errors.append("No CSV-like dataset resource is available for analysis.")
    else:
        try:
            df = _load_dataframe(source)
            analysis = _analyze_dataframe(df, intent)
        except Exception as exc:
            analysis.errors.append(_short_error(exc))
    detail = (
        f"Computed deterministic stats for {analysis.rows:,} rows and {analysis.columns:,} columns."
        if not analysis.errors
        else f"Analysis could not load data: {analysis.errors[0]}"
    )
    return {
        "analysis_result": asdict(analysis),
        "steps": [*state.get("steps", []), {"node": "analyze_data", "title": "data_analysis", "detail": detail}],
    }


def plan_openui(state: AgentWorkflowState) -> AgentWorkflowState:
    intent = _intent_from_state(state)
    retrieval = _retrieval_from_state(state)
    analysis = _analysis_from_state(state)
    plan = build_ui_plan(state.get("prompt", ""), intent, retrieval, analysis)
    return {
        "ui_plan": asdict(plan),
        "steps": [*state.get("steps", []), {"node": "plan_openui", "title": "ui_plan", "detail": f"Prepared {plan.layout_type} OpenUI template plan."}],
    }


def translate_openui(state: AgentWorkflowState) -> AgentWorkflowState:
    plan = _ui_plan_from_state(state)
    openui_lang = build_openui_lang(plan)
    try:
        parse_openui_lang(openui_lang)
    except OpenUIValidationError:
        openui_lang = build_openui_lang(_fallback_ui_plan("OpenUI validation fallback", "The deterministic template recovered from invalid UI output."))
    return {
        "openui_lang": openui_lang,
        "steps": [*state.get("steps", []), {"node": "translate_openui", "title": "openui_generation", "detail": "Rendered template-backed OpenUI-Lang and validated it."}],
    }


def classify_intent(prompt: str, *, has_dataset: bool = False) -> UserIntent:
    lower = prompt.casefold()
    query_terms = _query_terms(prompt)
    desired: VisualizationType = "summary"
    if any(term in lower for term in ["schema", "columns", "fields"]):
        desired = "schema"
    elif any(term in lower for term in ["histogram", "distribution", "spread"]):
        desired = "histogram"
    elif any(term in lower for term in ["chart", "bar", "plot", "visualize", "compare"]):
        desired = "bar"
    elif any(term in lower for term in ["quality", "missing", "duplicates"]):
        desired = "quality"

    retrieval_terms = ["ckan", "dataset", "resource", "catalog", "search", "find", "retrieve", "open data"]
    analysis_terms = ["analy", "schema", "columns", "quality", "missing", "trend", "statistics", "chart", "histogram", "summary", "summarize"]
    openui_terms = ["openui", "render", "ui", "card"]
    if any(term in lower for term in retrieval_terms) and not has_dataset:
        task: TaskType = "dataset_retrieval"
    elif has_dataset or any(term in lower for term in analysis_terms):
        task = "analysis"
    elif any(term in lower for term in openui_terms):
        task = "openui_generation"
    else:
        task = "general_agent"
    return UserIntent(task, query_terms, desired, "de" if _looks_german(lower) else "en", len(query_terms) < 2)


def build_ui_plan(prompt: str, intent: UserIntent, retrieval: RetrievalResult | None, analysis: AnalysisResult) -> UiPlan:
    selected_title = retrieval.selected.package_title if retrieval and retrieval.selected else "Dataset"
    if analysis.errors:
        return UiPlan(
            "summary",
            "Dataset unavailable",
            selected_title,
            table_rows=_retrieval_rows(retrieval),
            callout=analysis.errors[0],
            followups=["Search CKAN for another dataset", "Try a broader query"],
        )
    if analysis.rows == 0 and retrieval and retrieval.packages:
        return UiPlan(
            "summary",
            "CKAN search results",
            f"{len(retrieval.packages)} packages found",
            table_rows=_retrieval_rows(retrieval),
            callout="Select a CSV-like resource before running analysis.",
            followups=["Analyze the selected resource", "Search with a narrower query"],
        )
    metrics = [
        {"label": "Rows", "value": f"{analysis.rows:,}", "caption": "records"},
        {"label": "Columns", "value": f"{analysis.columns:,}", "caption": "fields"},
        {"label": "Missing", "value": f"{sum(row['missing'] for row in analysis.missingness):,}", "caption": "cells"},
    ]
    title = {
        "schema": "Dataset schema",
        "bar": "Bar chart",
        "histogram": "Distribution",
        "quality": "Data quality",
    }.get(intent.desired_visualization, "Dataset summary")
    plan = UiPlan(
        intent.desired_visualization,
        title,
        selected_title,
        metrics=metrics,
        table_rows=analysis.schema if intent.desired_visualization in {"schema", "quality"} else analysis.sample_rows,
        chart_rows=analysis.chart_rows,
        chart_x=analysis.chart_x,
        chart_y=analysis.chart_y,
        histogram_values=[float(row.get(analysis.numeric_column, 0)) for row in analysis.chart_rows if analysis.numeric_column and _is_number(row.get(analysis.numeric_column))],
        histogram_column=analysis.numeric_column,
        callout=" ".join(analysis.observations[:2]),
        followups=["List the columns", "Show a bar chart", "Check missing values"],
    )
    if intent.task_type == "general_agent":
        plan.title = "smolnalysis"
        plan.callout = f"I can search CKAN datasets or analyze a CSV-like resource. Request: {prompt}"
    return plan


def build_openui_lang(plan: UiPlan) -> str:
    lines = [
        _root_for_plan(plan),
        f"header = CardHeader({_json_arg(plan.title)}, {_json_arg(plan.subtitle)})",
    ]
    if plan.metrics:
        metric_ids = []
        for index, metric in enumerate(plan.metrics):
            ident = f"metric{index + 1}"
            metric_ids.append(ident)
            lines.append(f'{ident} = ListItem({_json_arg(metric["label"])}, {_json_arg(f"{metric['value']} {metric.get('caption', '')}".strip())})')
        lines.append(f'metrics = ListBlock([{", ".join(metric_ids)}], "number")')
    if plan.table_rows:
        columns = list(plan.table_rows[0].keys())[:5]
        for index, column in enumerate(columns):
            values = [row.get(column) for row in plan.table_rows[:10]]
            lines.append(f'col{index + 1} = Col({_json_arg(column)}, {_json_arg(values)}, "string")')
        lines.append(f'table = Table([{", ".join(f"col{index + 1}" for index in range(len(columns)))}])')
    if plan.layout_type == "bar" and plan.chart_rows and plan.chart_x and plan.chart_y:
        labels = [str(row.get(plan.chart_x, "")) for row in plan.chart_rows[:12]]
        values = [float(row.get(plan.chart_y, 0) or 0) for row in plan.chart_rows[:12]]
        lines.append(f"series = Series({_json_arg(plan.chart_y)}, {_json_arg(values)})")
        lines.append(f"chart = BarChart({_json_arg(labels)}, [series], \"grouped\", {_json_arg(plan.chart_x)}, {_json_arg(plan.chart_y)})")
    if plan.layout_type == "histogram" and plan.histogram_values:
        counts, labels = _histogram_counts(plan.histogram_values)
        lines.append(f'series = Series("Count", {_json_arg(counts)})')
        lines.append(f"chart = BarChart({_json_arg(labels)}, [series], \"grouped\", {_json_arg(plan.histogram_column or 'Value')}, \"Rows\")")
    if plan.callout:
        lines.append(f'callout = Callout("info", "Result", {_json_arg(plan.callout)})')
    if plan.followups:
        followup_ids = []
        for index, followup in enumerate(plan.followups[:3]):
            ident = f"followup{index + 1}"
            followup_ids.append(ident)
            lines.append(f"{ident} = FollowUpItem({_json_arg(followup)})")
        lines.append(f"followups = FollowUpBlock([{', '.join(followup_ids)}])")
    return "\n".join(lines)


def _root_for_plan(plan: UiPlan) -> str:
    children = ["header"]
    if plan.metrics:
        children.append("metrics")
    if plan.layout_type in {"bar", "histogram"} and (plan.chart_rows or plan.histogram_values):
        children.append("chart")
    if plan.table_rows:
        children.append("table")
    if plan.callout:
        children.append("callout")
    if plan.followups:
        children.append("followups")
    return f"root = Card([{', '.join(children)}])"


def _analyze_dataframe(df: pd.DataFrame, intent: UserIntent) -> AnalysisResult:
    df = df.head(500)
    numeric_columns = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    text_columns = [column for column in df.columns if not pd.api.types.is_numeric_dtype(df[column])]
    selected_numeric = _find_column(" ".join(intent.query_terms), numeric_columns) or (numeric_columns[0] if numeric_columns else "")
    selected_label = _find_column(" ".join(intent.query_terms), text_columns) or (text_columns[0] if text_columns else "")
    sample = df.head(8).where(pd.notna(df.head(8)), None)
    chart_columns = [column for column in [selected_label, selected_numeric] if column]
    chart_rows = df[chart_columns].dropna().head(12).to_dict(orient="records") if chart_columns else []
    if chart_rows and not selected_label:
        chart_rows = [{"row": index + 1, selected_numeric: row[selected_numeric]} for index, row in enumerate(chart_rows)]
        selected_label = "row"
    missingness = [{"column": str(column), "missing": int(df[column].isna().sum())} for column in df.columns]
    return AnalysisResult(
        rows=len(df),
        columns=len(df.columns),
        schema=[
            {"column": str(column), "dtype": str(df[column].dtype), "missing": int(df[column].isna().sum()), "unique": int(df[column].nunique(dropna=True))}
            for column in df.columns
        ],
        missingness=missingness,
        sample_rows=[{str(key): value for key, value in row.items()} for row in sample.to_dict(orient="records")],
        chart_rows=chart_rows,
        chart_x=selected_label,
        chart_y=selected_numeric,
        numeric_column=selected_numeric,
        observations=[
            f"Loaded {len(df):,} rows across {len(df.columns):,} columns.",
            f"Detected {len(numeric_columns)} numeric columns and {len(text_columns)} text-like columns.",
            f"Missing cells total {sum(row['missing'] for row in missingness):,}.",
        ],
    )


def _load_dataframe(source: str) -> pd.DataFrame:
    parsed = urllib.parse.urlsplit(source)
    if parsed.scheme in {"http", "https"}:
        suffix = Path(parsed.path).suffix.lower()
        if suffix not in {".csv", ".txt"}:
            raise ValueError("Selected resource is not a CSV-like URL.")
        request = urllib.request.Request(source, headers={"User-Agent": "smolnalysis/0.1"})
        with urllib.request.urlopen(request, timeout=10) as response:
            content = response.read(2_000_000)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=True) as handle:
            handle.write(content)
            handle.flush()
            return pd.read_csv(handle.name, nrows=500)
    return pd.read_csv(source, nrows=500)


def _query_terms(prompt: str) -> list[str]:
    stop = {"find", "show", "search", "dataset", "datasets", "data", "ckan", "resource", "resources", "please", "the", "and", "for", "with", "about", "open"}
    words = [word for word in re.findall(r"[\wäöüÄÖÜß-]+", prompt.casefold()) if len(word) > 2 and word not in stop]
    return words[:5]


def _fallback_query(prompt: str) -> str:
    return " ".join(_query_terms(prompt)) or "open data"


def _resources_from_packages(packages: list[dict[str, Any]]) -> list[RetrievalResource]:
    resources: list[RetrievalResource] = []
    for package in packages:
        package_id = str(package.get("id") or package.get("name") or "")
        package_title = str(package.get("title") or package.get("name") or package_id or "CKAN package")
        for resource in package.get("resources") or []:
            if not isinstance(resource, dict):
                continue
            resources.append(
                RetrievalResource(
                    package_id=package_id,
                    package_title=package_title,
                    resource_id=str(resource.get("id") or ""),
                    name=str(resource.get("name") or resource.get("description") or resource.get("url") or "resource"),
                    format=str(resource.get("format") or resource.get("mimetype") or "").lower(),
                    url=str(resource.get("url") or ""),
                )
            )
    return resources


def _select_resource(resources: list[RetrievalResource]) -> RetrievalResource | None:
    for resource in resources:
        url_path = urllib.parse.urlsplit(resource.url).path.casefold()
        if "csv" in resource.format or url_path.endswith(".csv"):
            return resource
    return resources[0] if resources else None


def _retrieval_to_dict(result: RetrievalResult) -> dict[str, Any]:
    return {
        "endpoint": result.endpoint,
        "query": result.query,
        "packages": result.packages,
        "resources": [asdict(resource) for resource in result.resources],
        "selected": asdict(result.selected) if result.selected else None,
        "confidence": result.confidence,
        "errors": result.errors,
    }


def _intent_from_state(state: AgentWorkflowState) -> UserIntent:
    value = state.get("intent") or asdict(classify_intent(state.get("prompt", ""), has_dataset=bool(state.get("dataset_path"))))
    return UserIntent(**value)


def _retrieval_from_state(state: AgentWorkflowState) -> RetrievalResult | None:
    value = state.get("retrieval_result")
    if not value:
        return None
    selected = value.get("selected")
    return RetrievalResult(
        endpoint=value.get("endpoint", DEFAULT_CKAN_ENDPOINT),
        query=value.get("query", ""),
        packages=value.get("packages", []),
        resources=[RetrievalResource(**item) for item in value.get("resources", [])],
        selected=RetrievalResource(**selected) if isinstance(selected, dict) else None,
        confidence=float(value.get("confidence", 0.0)),
        errors=value.get("errors", []),
    )


def _analysis_from_state(state: AgentWorkflowState) -> AnalysisResult:
    value = state.get("analysis_result") or {}
    return AnalysisResult(**value)


def _ui_plan_from_state(state: AgentWorkflowState) -> UiPlan:
    value = state.get("ui_plan") or asdict(_fallback_ui_plan("smolnalysis", "Ask for a dataset search or analysis."))
    return UiPlan(**value)


def _fallback_ui_plan(title: str, message: str) -> UiPlan:
    return UiPlan("summary", title, "Fallback", callout=message, followups=["Search CKAN datasets", "Summarize a dataset"])


def _retrieval_rows(retrieval: RetrievalResult | None) -> list[dict[str, Any]]:
    if not retrieval:
        return []
    return [
        {
            "title": str(package.get("title") or package.get("name") or ""),
            "name": str(package.get("name") or package.get("id") or ""),
            "resources": len(package.get("resources") or []),
        }
        for package in retrieval.packages[:8]
    ]


def _find_column(prompt: str, columns: list[str]) -> str | None:
    normalized = prompt.casefold()
    return next((column for column in columns if column.casefold() in normalized), None)


def _histogram_counts(values: list[float], bucket_count: int = 8) -> tuple[list[int], list[str]]:
    low = min(values)
    high = max(values)
    span = high - low or 1
    counts = [0] * bucket_count
    for value in values:
        index = min(bucket_count - 1, int(((value - low) / span) * bucket_count))
        counts[index] += 1
    labels = [f"{low + (span / bucket_count) * index:.1f}" for index in range(bucket_count)]
    return counts, labels


def _looks_german(prompt: str) -> bool:
    return any(term in prompt for term in ["daten", "zeige", "spalten", "qualität", "verteilung", "diagramm"])


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _short_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc).strip() or type(exc).__name__}"


def _normalize_endpoint_or_default(endpoint: str | None) -> str:
    if not endpoint:
        return DEFAULT_CKAN_ENDPOINT
    try:
        return normalize_ckan_base_url(endpoint)
    except ValueError:
        return DEFAULT_CKAN_ENDPOINT


def dumps_artifact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
