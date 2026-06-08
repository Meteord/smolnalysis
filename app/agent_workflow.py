from __future__ import annotations

import os
import random
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ckan_support import DEFAULT_CKAN_ENDPOINT, normalize_ckan_base_url
from openui_support import _json_arg


MIN_NODE_DELAY_SECONDS = 0.35
MAX_NODE_DELAY_SECONDS = 0.9
DISABLE_DELAYS_ENV = "SMOLNALYSIS_WORKFLOW_DISABLE_DELAYS"


class WorkflowStep(TypedDict):
    node: str
    title: str
    detail: str


class AgentWorkflowState(TypedDict, total=False):
    prompt: str
    ckan_endpoint: str
    steps: list[WorkflowStep]
    ckan_result: dict[str, Any]
    analysis_result: dict[str, Any]
    openui_lang: str
    next_action: str
    retrieval_attempts: int
    analysis_attempts: int


def run_agent_workflow(prompt: str, ckan_endpoint: str | None = None) -> AgentWorkflowState:
    initial_state: AgentWorkflowState = {
        "prompt": prompt.strip() or "Summarize this dataset",
        "ckan_endpoint": _normalize_endpoint_or_default(ckan_endpoint),
        "steps": [],
        "retrieval_attempts": 0,
        "analysis_attempts": 0,
    }
    return build_agent_workflow().invoke(initial_state)


def build_agent_workflow():
    graph = StateGraph(AgentWorkflowState)
    graph.add_node("react_agent", react_agent)
    graph.add_node("retrieve_ckan", retrieve_ckan)
    graph.add_node("analyze_data", analyze_data)
    graph.add_node("translate_openui", translate_openui)
    graph.add_edge(START, "react_agent")
    graph.add_conditional_edges(
        "react_agent",
        route_next_action,
        {
            "retrieve_ckan": "retrieve_ckan",
            "analyze_data": "analyze_data",
            "translate_openui": "translate_openui",
        },
    )
    graph.add_edge("retrieve_ckan", "react_agent")
    graph.add_edge("analyze_data", "react_agent")
    graph.add_edge("translate_openui", END)
    return graph.compile()


def route_next_action(state: AgentWorkflowState) -> str:
    return state.get("next_action", "translate_openui")


def react_agent(state: AgentWorkflowState) -> AgentWorkflowState:
    _simulate_node_delay()
    next_action, thought = _decide_next_action(state)
    return {
        "next_action": next_action,
        "steps": [
            *state.get("steps", []),
            {
                "node": "react_agent",
                "title": "general_agent",
                "detail": thought,
            },
        ],
    }


def retrieve_ckan(state: AgentWorkflowState) -> AgentWorkflowState:
    _simulate_node_delay()
    endpoint = state["ckan_endpoint"]
    prompt = state["prompt"]
    attempt = state.get("retrieval_attempts", 0) + 1
    candidates = _mock_ckan_candidates(prompt)
    selected = random.choice(candidates)
    return {
        "retrieval_attempts": attempt,
        "ckan_result": {
            "endpoint": endpoint,
            "query": _mock_search_query(prompt),
            "attempt": attempt,
            "datasets": candidates,
            "selected": selected,
        },
        "steps": [
            *state.get("steps", []),
            {
                "node": "retrieve_ckan",
                "title": "ckan_tool",
                "detail": f"Attempt {attempt} searched {endpoint} and selected {selected['title']}.",
            },
        ],
    }


def analyze_data(state: AgentWorkflowState) -> AgentWorkflowState:
    _simulate_node_delay()
    ckan_result = state.get("ckan_result", {})
    selected = ckan_result.get("selected", {}) if isinstance(ckan_result, dict) else {}
    attempt = state.get("analysis_attempts", 0) + 1
    rows = random.randint(450, 12500)
    columns = random.randint(5, 18)
    missing_pct = round(random.uniform(0.0, 7.5), 1)
    chart_labels = random.sample(["2019", "2020", "2021", "2022", "2023", "2024", "Q1", "Q2", "Q3", "Q4"], 5)
    chart_values = [random.randint(12, 96) for _ in chart_labels]
    return {
        "analysis_attempts": attempt,
        "analysis_result": {
            "attempt": attempt,
            "summary": f"Stub analysis inspected {selected.get('resource', 'a CSV resource')} and found a usable table.",
            "rows": rows,
            "columns": columns,
            "missing_pct": missing_pct,
            "chart_labels": chart_labels,
            "chart_values": chart_values,
            "observations": [
                f"Estimated {rows:,} rows across {columns} columns.",
                f"Missing values are roughly {missing_pct}% in this mocked pass.",
                random.choice(
                    [
                        "A time-series chart looks promising for the selected resource.",
                        "A ranked bar chart would be a useful first visualization.",
                        "The resource looks suitable for a summary table and follow-up filtering.",
                    ]
                ),
            ],
        },
        "steps": [
            *state.get("steps", []),
            {
                "node": "analyze_data",
                "title": "data_analysis",
                "detail": f"Attempt {attempt} analyzed {rows:,} rows and prepared summary metrics.",
            },
        ],
    }


def translate_openui(state: AgentWorkflowState) -> AgentWorkflowState:
    _simulate_node_delay()
    final_steps: list[WorkflowStep] = [
        *state.get("steps", []),
        {
            "node": "translate_openui",
            "title": "openui_translator",
            "detail": "Stub translated the current ReAct state into OpenUI-Lang.",
        },
    ]
    openui_lang = _build_openui_response({**state, "steps": final_steps})
    return {
        "openui_lang": openui_lang,
        "steps": final_steps,
    }


def _decide_next_action(state: AgentWorkflowState) -> tuple[str, str]:
    prompt = state["prompt"].casefold()
    retrieval_attempts = state.get("retrieval_attempts", 0)
    analysis_attempts = state.get("analysis_attempts", 0)
    wants_deeper_search = any(term in prompt for term in ["again", "broader", "compare", "more", "rerun"])
    wants_deeper_analysis = any(term in prompt for term in ["chart", "compare", "distribution", "quality", "trend"])

    if retrieval_attempts == 0:
        return "retrieve_ckan", "Thought: I need CKAN candidates before I can inspect data."
    if wants_deeper_search and retrieval_attempts < 2:
        return "retrieve_ckan", "Thought: The request hints at comparison or breadth, so I will rerun CKAN retrieval once."
    if analysis_attempts == 0:
        return "analyze_data", "Thought: I have a CKAN candidate, so the next useful action is data analysis."
    if wants_deeper_analysis and analysis_attempts < 2:
        return "analyze_data", "Thought: The request asks for a chart or quality lens, so I will rerun analysis with a different pass."
    return "translate_openui", "Thought: I have enough retrieval and analysis context to produce the final OpenUI-Lang response."


def _build_openui_response(state: AgentWorkflowState) -> str:
    prompt = state["prompt"]
    endpoint = state["ckan_endpoint"]
    steps = state.get("steps", [])
    ckan_result = state.get("ckan_result", {})
    analysis_result = state.get("analysis_result", {})
    datasets = ckan_result.get("datasets", []) if isinstance(ckan_result, dict) else []
    selected = ckan_result.get("selected", {}) if isinstance(ckan_result, dict) else {}
    observations = analysis_result.get("observations", []) if isinstance(analysis_result, dict) else []
    variant = random.choice(["overview", "chart", "quality"])

    lines = [
        _root_line(variant),
        f'header = CardHeader("ReAct-style LangGraph agent", {_json_arg(f"Tool loop - {variant} result")})',
        f'request = TextContent({_json_arg(f"User request: {prompt}")}, "default")',
        f'endpoint = TextContent({_json_arg(f"CKAN endpoint used: {endpoint}")}, "small")',
        f'workflow = ListBlock([{", ".join(f"step{index + 1}" for index in range(len(steps)))}], "number")',
    ]

    for index, step in enumerate(steps):
        step_detail = f"{step['title']}: {step['detail']}"
        lines.append(f'step{index + 1} = ListItem({_json_arg(step["node"])}, {_json_arg(step_detail)})')

    lines.extend(
        [
            f'ckan = ListBlock([{", ".join(f"dataset{index + 1}" for index in range(len(datasets)))}], "number")',
            f'analysis = ListBlock([{", ".join(f"observation{index + 1}" for index in range(len(observations)))}], "number")',
        ]
    )
    for index, dataset in enumerate(datasets):
        lines.append(f'dataset{index + 1} = ListItem({_json_arg(dataset["title"])}, {_json_arg(dataset["resource"])})')
    for index, observation in enumerate(observations):
        lines.append(f'observation{index + 1} = ListItem({_json_arg(f"Observation {index + 1}")}, {_json_arg(observation)})')
    lines.extend(_variant_lines(variant, analysis_result, selected))
    lines.extend(
        [
            "followups = FollowUpBlock([f1, f2, f3])",
            'f1 = FollowUpItem("Search CKAN for population datasets")',
            'f2 = FollowUpItem("Rerun analysis with a quality lens")',
            'f3 = FollowUpItem("Translate the result to OpenUI-Lang")',
        ]
    )
    return "\n".join(lines)


def _simulate_node_delay() -> None:
    if os.environ.get(DISABLE_DELAYS_ENV, "").casefold() in {"1", "true", "yes", "on"}:
        return
    time.sleep(random.uniform(MIN_NODE_DELAY_SECONDS, MAX_NODE_DELAY_SECONDS))


def _mock_search_query(prompt: str) -> str:
    words = [word.strip(".,!?;:").casefold() for word in prompt.split() if len(word.strip(".,!?;:")) > 3]
    return " ".join(words[:4]) or "open data"


def _mock_ckan_candidates(prompt: str) -> list[dict[str, str]]:
    query = _mock_search_query(prompt)
    themes = [
        ("Population indicators", "population-indicators.csv"),
        ("Mobility counts", "mobility-counts.csv"),
        ("Public services by district", "public-services-districts.csv"),
        ("Environmental measurements", "environmental-measurements.csv"),
        ("Budget spending overview", "budget-spending.csv"),
    ]
    random.shuffle(themes)
    return [
        {
            "title": f"{title} for {query}",
            "resource": resource,
        }
        for title, resource in themes[:3]
    ]


def _root_line(variant: str) -> str:
    if variant == "chart":
        return "root = Card([header, request, endpoint, workflow, ckan, analysis, chart, callout, followups])"
    if variant == "quality":
        return "root = Card([header, callout, request, endpoint, workflow, ckan, analysis, table, followups])"
    return "root = Card([header, request, endpoint, workflow, ckan, analysis, table, callout, followups])"


def _variant_lines(variant: str, analysis_result: dict[str, Any], selected: dict[str, Any]) -> list[str]:
    labels = analysis_result.get("chart_labels", ["A", "B", "C"])
    values = analysis_result.get("chart_values", [24, 48, 72])
    rows = analysis_result.get("rows", 0)
    columns = analysis_result.get("columns", 0)
    missing_pct = analysis_result.get("missing_pct", 0)
    selected_title = selected.get("title", "selected CKAN resource")
    selected_resource = selected.get("resource", "resource.csv")

    if variant == "chart":
        return [
            f'series = Series("Mock value", {_json_arg(values)})',
            f'chart = BarChart({_json_arg(labels)}, [series], "grouped", "Period", "Value")',
            f'callout = Callout("success", "Chart-ready resource", {_json_arg(f"{selected_title} can be turned into a first exploratory chart.")})',
        ]
    if variant == "quality":
        return [
            f'col1 = Col("Metric", {_json_arg(["Rows", "Columns", "Missing values", "Resource"])}, "string")',
            f'col2 = Col("Value", {_json_arg([f"{rows:,}", str(columns), f"{missing_pct}%", selected_resource])}, "string")',
            "table = Table([col1, col2])",
            f'callout = Callout("info", "Data quality stub", {_json_arg("This result is randomized placeholder analysis until real CKAN resource loading is added.")})',
        ]
    return [
        f'col1 = Col("Candidate", {_json_arg([selected_title, "Rows", "Columns", "Missing values"])}, "string")',
        f'col2 = Col("Stub result", {_json_arg([selected_resource, f"{rows:,}", str(columns), f"{missing_pct}%"])}, "string")',
        "table = Table([col1, col2])",
        f'callout = Callout("neutral", "Randomized OpenUI-Lang", {_json_arg("The translator stub chose this layout randomly for this run.")})',
    ]


def _normalize_endpoint_or_default(endpoint: str | None) -> str:
    if not endpoint:
        return DEFAULT_CKAN_ENDPOINT
    try:
        return normalize_ckan_base_url(endpoint)
    except ValueError:
        return DEFAULT_CKAN_ENDPOINT
