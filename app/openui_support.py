from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from typing import Any

import gradio as gr
import pandas as pd


ASSISTANT_FALLBACK = "I could not render the OpenUI response, so I am showing a fallback."
EMPTY_RENDER_VALUE = {
    "encoded": "",
}


@dataclass
class AgentStep:
    role: str
    content: str


@dataclass
class ChatTurn:
    user_message: str
    agent_steps: list[AgentStep]
    openui_lang: str
    fallback_text: str


@dataclass
class OpenUIComponent:
    identifier: str
    component_type: str
    args: list[Any] = field(default_factory=list)


@dataclass
class ParsedOpenUI:
    root_children: list[str]
    components: dict[str, OpenUIComponent]


class OpenUIValidationError(ValueError):
    pass


def _json_arg(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _records(df: pd.DataFrame, limit: int = 8) -> list[dict[str, Any]]:
    sample = df.head(limit).where(pd.notna(df.head(limit)), None)
    return [{str(key): value for key, value in row.items()} for row in sample.to_dict(orient="records")]


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]


def _text_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if not pd.api.types.is_numeric_dtype(df[column])]


def _find_column(prompt: str, columns: list[str]) -> str | None:
    normalized = prompt.casefold()
    for column in columns:
        if column.casefold() in normalized:
            return column
    return None


def _build_column_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "column": column,
            "dtype": str(df[column].dtype),
            "missing": int(df[column].isna().sum()),
            "unique": int(df[column].nunique(dropna=True)),
        }
        for column in df.columns
    ]


def _workflow_trace_lines(prompt: str) -> list[str]:
    request = prompt or "User request"
    return [
        "workflow = ListBlock([wf1, wf2, wf3, wf4, wf5], \"number\")",
        f'wf1 = ListItem("User request received", {_json_arg(request)})',
        'wf2 = ListItem("general_agent", "Would plan CKAN search, data analysis, and OpenUI translation.")',
        'wf3 = ListItem("ckan_tool", "Would search the configured CKAN endpoint with tool calls.")',
        'wf4 = ListItem("data_analysis", "Would analyze the selected dataset/resource data.")',
        'wf5 = ListItem("openui_translator", "Would convert the analysis result into OpenUI-Lang for rendering.")',
    ]


def generate_openui_response(df: pd.DataFrame | None, prompt: str) -> ChatTurn:
    prompt = prompt.strip()
    steps = [AgentStep("planner", "Classified the request and selected a mock response path.")]

    if df is None or df.empty:
        openui_lang = "\n".join(
            [
                "root = Root([notice])",
                'notice = Notice("Upload a CSV dataset first, then ask me to summarize, inspect columns, or chart it.", "info")',
            ]
        )
        return ChatTurn(prompt, steps, openui_lang, "Upload a CSV dataset first.")

    rows = len(df)
    columns = len(df.columns)
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    numeric_columns = _numeric_columns(df)
    text_columns = _text_columns(df)
    selected_numeric = _find_column(prompt, numeric_columns) or (numeric_columns[0] if numeric_columns else None)
    selected_label = _find_column(prompt, text_columns) or (text_columns[0] if text_columns else None)
    lower_prompt = prompt.casefold()

    steps.append(AgentStep("tool", f"Loaded CSV with {rows:,} rows and {columns:,} columns."))

    if "invalid openui" in lower_prompt:
        steps.append(AgentStep("validator", "Returning intentionally invalid OpenUI-Lang for fallback testing."))
        return ChatTurn(
            prompt,
            steps,
            "root = Nope([missing])",
            "The intentionally invalid OpenUI response triggered the fallback path.",
        )

    if any(term in lower_prompt for term in ["columns", "schema", "fields"]):
        openui_lang = "\n".join(
            [
                "root = Root([summary, table])",
                f'summary = InsightCard("Dataset schema", "{columns:,} columns detected. Missing values are counted per column.")',
                f'table = DataTable("Columns", {_json_arg(_build_column_rows(df))})',
            ]
        )
        return ChatTurn(prompt, steps, openui_lang, "Rendered the dataset schema.")

    if any(term in lower_prompt for term in ["histogram", "distribution", "spread"]):
        if not selected_numeric:
            openui_lang = "\n".join(
                [
                    "root = Root([notice, table])",
                    'notice = Notice("This dataset has no numeric columns for a histogram.", "warning")',
                    f'table = DataTable("Sample rows", {_json_arg(_records(df))})',
                ]
            )
            return ChatTurn(prompt, steps, openui_lang, "No numeric histogram is available.")

        values = [float(value) for value in df[selected_numeric].dropna().head(500).tolist()]
        openui_lang = "\n".join(
            [
                "root = Root([summary, histogram, table])",
                f'summary = InsightCard("Distribution", "Histogram for {selected_numeric} based on the uploaded CSV.")',
                f'histogram = Histogram("Distribution of {selected_numeric}", "{selected_numeric}", {_json_arg(values)})',
                f'table = DataTable("Sample rows", {_json_arg(_records(df))})',
            ]
        )
        return ChatTurn(prompt, steps, openui_lang, f"Rendered a histogram for {selected_numeric}.")

    if any(term in lower_prompt for term in ["plot", "chart", "bar", "compare", "visualize", "show"]):
        if not selected_numeric:
            openui_lang = "\n".join(
                [
                    "root = Root([notice, table])",
                    'notice = Notice("This dataset has no numeric columns for charting.", "warning")',
                    f'table = DataTable("Sample rows", {_json_arg(_records(df))})',
                ]
            )
            return ChatTurn(prompt, steps, openui_lang, "No numeric chart is available.")

        chart_columns = [selected_numeric]
        if selected_label:
            chart_columns.insert(0, selected_label)
        chart_rows = df[chart_columns].dropna().head(18).to_dict(orient="records")
        x_column = selected_label or "__row__"
        if not selected_label:
            chart_rows = [{"__row__": index + 1, selected_numeric: row[selected_numeric]} for index, row in enumerate(chart_rows)]

        openui_lang = "\n".join(
            [
                "root = Root([summary, chart])",
                f'summary = InsightCard("Chart", "Bar chart for {selected_numeric}.")',
                f'chart = BarChart("{selected_numeric} overview", "{x_column}", "{selected_numeric}", {_json_arg(chart_rows)})',
            ]
        )
        return ChatTurn(prompt, steps, openui_lang, f"Rendered a bar chart for {selected_numeric}.")

    openui_lang = "\n".join(
        [
            "root = Root([summary, metrics, table])",
            f'summary = InsightCard("Dataset summary", "Loaded {rows:,} rows and {columns:,} columns from the uploaded CSV.")',
            f'm1 = Metric("Rows", "{rows:,}", "CSV records")',
            f'm2 = Metric("Columns", "{columns:,}", "Dataset fields")',
            f'm3 = Metric("Missing", "{missing:,}", "Empty cells")',
            f'm4 = Metric("Duplicates", "{duplicates:,}", "Repeated rows")',
            "metrics = MetricGrid([m1, m2, m3, m4])",
            f'table = DataTable("Sample rows", {_json_arg(_records(df))})',
        ]
    )
    return ChatTurn(prompt, steps, openui_lang, "Rendered a dataset summary.")


def generate_openui_chat_response(df: pd.DataFrame | None, prompt: str) -> str:
    prompt = prompt.strip()
    lower_prompt = prompt.casefold()

    if df is None or df.empty:
        return "\n".join(
            [
                "root = Card([header, workflow, callout, followups])",
                'header = CardHeader("smolnalysis", "OpenUI fullscreen chat")',
                *_workflow_trace_lines(prompt),
                'callout = Callout("info", "Ready for data questions", "Ask for a summary, schema, bar chart, histogram, or mocked fallback. This server-mode frontend is rendered by OpenUI, while Python serves the responses.")',
                "followups = FollowUpBlock([f1, f2, f3])",
                'f1 = FollowUpItem("Summarize this dataset")',
                'f2 = FollowUpItem("Show a bar chart of population by city")',
                'f3 = FollowUpItem("List the columns and missing values")',
            ]
        )

    rows = len(df)
    columns = len(df.columns)
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    numeric_columns = _numeric_columns(df)
    text_columns = _text_columns(df)
    selected_numeric = _find_column(prompt, numeric_columns) or (numeric_columns[0] if numeric_columns else None)
    selected_label = _find_column(prompt, text_columns) or (text_columns[0] if text_columns else None)

    if "invalid openui" in lower_prompt or "fallback" in lower_prompt:
        return "\n".join(
            [
                "root = Card([header, workflow, callout, code])",
                'header = CardHeader("Fallback path", "Mocked invalid OpenUI request")',
                *_workflow_trace_lines(prompt),
                'callout = Callout("warning", "Renderer guard", "The old prototype used a custom fallback renderer. The fullscreen chat keeps this as a mocked warning response for now.")',
                f'code = CodeBlock("openui-lang", {_json_arg("root = Nope([missing])")})',
            ]
        )

    if any(term in lower_prompt for term in ["columns", "schema", "fields"]):
        rows_by_column = _build_column_rows(df)
        return "\n".join(
            [
                "root = Card([header, workflow, table, followups])",
                f'header = CardHeader("Dataset schema", "{columns:,} columns detected")',
                *_workflow_trace_lines(prompt),
                f'c1 = Col("Column", {_json_arg([row["column"] for row in rows_by_column])}, "string")',
                f'c2 = Col("Type", {_json_arg([row["dtype"] for row in rows_by_column])}, "string")',
                f'c3 = Col("Missing", {_json_arg([row["missing"] for row in rows_by_column])}, "number")',
                f'c4 = Col("Unique", {_json_arg([row["unique"] for row in rows_by_column])}, "number")',
                "table = Table([c1, c2, c3, c4])",
                "followups = FollowUpBlock([f1, f2])",
                'f1 = FollowUpItem("Summarize this dataset")',
                'f2 = FollowUpItem("Show a chart")',
            ]
        )

    if any(term in lower_prompt for term in ["histogram", "distribution", "spread"]):
        if not selected_numeric:
            return "\n".join(
                [
                    "root = Card([header, workflow, callout])",
                    'header = CardHeader("Distribution", "No numeric column found")',
                    *_workflow_trace_lines(prompt),
                    'callout = Callout("warning", "No histogram available", "This dataset does not include numeric columns that can be bucketed.")',
                ]
            )

        values = [float(value) for value in df[selected_numeric].dropna().head(120).tolist()]
        if not values:
            values = [0]
        low = min(values)
        high = max(values)
        span = high - low or 1
        bucket_count = 8
        counts = [0] * bucket_count
        for value in values:
            bucket = min(bucket_count - 1, int(((value - low) / span) * bucket_count))
            counts[bucket] += 1
        labels = [f"{low + (span / bucket_count) * i:.1f}" for i in range(bucket_count)]
        return "\n".join(
            [
                "root = Card([header, workflow, chart, note, followups])",
                f'header = CardHeader("Distribution", "Histogram for {selected_numeric}")',
                *_workflow_trace_lines(prompt),
                f'series = Series("Count", {_json_arg(counts)})',
                f'chart = BarChart({_json_arg(labels)}, [series], "grouped", "{selected_numeric}", "Rows")',
                f'note = TextContent("Bucketed {len(values):,} numeric values from the uploaded/demo dataset.", "small")',
                "followups = FollowUpBlock([f1, f2])",
                'f1 = FollowUpItem("List the columns")',
                'f2 = FollowUpItem("Summarize this dataset")',
            ]
        )

    if any(term in lower_prompt for term in ["plot", "chart", "bar", "compare", "visualize", "show"]):
        if not selected_numeric:
            return "\n".join(
                [
                    "root = Card([header, workflow, callout])",
                    'header = CardHeader("Chart", "No numeric column found")',
                    *_workflow_trace_lines(prompt),
                    'callout = Callout("warning", "No chart available", "I need at least one numeric column for a chart.")',
                ]
            )

        chart_rows = df[[column for column in [selected_label, selected_numeric] if column]].dropna().head(12)
        labels = [str(value) for value in (chart_rows[selected_label].tolist() if selected_label else range(1, len(chart_rows) + 1))]
        values = [float(value) for value in chart_rows[selected_numeric].tolist()]
        return "\n".join(
            [
                "root = Card([header, workflow, chart, followups])",
                f'header = CardHeader("Bar chart", "{selected_numeric} by {selected_label or "row"}")',
                *_workflow_trace_lines(prompt),
                f'series = Series("{selected_numeric}", {_json_arg(values)})',
                f'chart = BarChart({_json_arg(labels)}, [series], "grouped", "{selected_label or "Row"}", "{selected_numeric}")',
                "followups = FollowUpBlock([f1, f2])",
                'f1 = FollowUpItem("Show a histogram")',
                'f2 = FollowUpItem("List the columns")',
            ]
        )

    sample = _records(df, limit=6)
    sample_columns = list(sample[0].keys())[:5] if sample else []
    table_lines = [
        f'c{index + 1} = Col({_json_arg(column)}, {_json_arg([row.get(column) for row in sample])}, "string")'
        for index, column in enumerate(sample_columns)
    ]
    return "\n".join(
        [
            "root = Card([header, workflow, metrics, table, followups])",
            f'header = CardHeader("Dataset summary", "{rows:,} rows x {columns:,} columns")',
            *_workflow_trace_lines(prompt),
            f'metrics = ListBlock([m1, m2, m3, m4], "number")',
            f'm1 = ListItem("Rows", "{rows:,} records")',
            f'm2 = ListItem("Columns", "{columns:,} fields")',
            f'm3 = ListItem("Missing cells", "{missing:,}")',
            f'm4 = ListItem("Duplicate rows", "{duplicates:,}")',
            *table_lines,
            f'table = Table([{", ".join(f"c{index + 1}" for index in range(len(sample_columns))) }])',
            "followups = FollowUpBlock([f1, f2, f3])",
            'f1 = FollowUpItem("Show a bar chart")',
            'f2 = FollowUpItem("Show a histogram")',
            'f3 = FollowUpItem("List the columns")',
        ]
    )


def _split_args(args_text: str) -> list[str]:
    args: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False

    for index, char in enumerate(args_text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote:
            escaped = True
            continue
        if char in {'"', "'"}:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
            continue
        if quote:
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            args.append(args_text[start:index].strip())
            start = index + 1

    tail = args_text[start:].strip()
    if tail:
        args.append(tail)
    return args


def _parse_value(value: str) -> Any:
    value = value.strip()
    if value == "null":
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(part) for part in _split_args(inner)]
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        return {"$ref": value}
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise OpenUIValidationError(f"Invalid argument value: {value}") from exc


def parse_openui_lang(openui_lang: str) -> ParsedOpenUI:
    allowed_components = {
        "Root",
        "InsightCard",
        "Notice",
        "Metric",
        "MetricGrid",
        "DataTable",
        "BarChart",
        "Histogram",
    }
    components: dict[str, OpenUIComponent] = {}
    root_children: list[str] | None = None

    for raw_line in openui_lang.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue

        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)", line)
        if not match:
            raise OpenUIValidationError(f"Invalid OpenUI statement: {line}")

        identifier, component_type, args_text = match.groups()
        if component_type not in allowed_components:
            raise OpenUIValidationError(f"Unsupported component: {component_type}")

        component = OpenUIComponent(
            identifier=identifier,
            component_type=component_type,
            args=[_parse_value(part) for part in _split_args(args_text)],
        )
        components[identifier] = component

        if identifier == "root":
            if component_type != "Root":
                raise OpenUIValidationError("`root` must be a Root(...) component.")
            if not component.args or not isinstance(component.args[0], list):
                raise OpenUIValidationError("Root must receive a child reference list.")
            root_children = [
                item["$ref"]
                for item in component.args[0]
                if isinstance(item, dict) and "$ref" in item
            ]

    if not root_children:
        raise OpenUIValidationError("OpenUI-Lang must include `root = Root([...])`.")

    missing = [child for child in root_children if child not in components]
    if missing:
        raise OpenUIValidationError(f"Missing component definitions: {', '.join(missing)}")

    return ParsedOpenUI(root_children=root_children, components=components)


def _encode_openui(openui_lang: str) -> str:
    return base64.b64encode(openui_lang.encode("utf-8")).decode("ascii")


def render_openui_value(parsed: ParsedOpenUI, openui_lang: str) -> dict[str, Any]:
    return {"encoded": _encode_openui(openui_lang), "openui_lang": openui_lang}


def render_openui_error(openui_lang: str, error: str) -> dict[str, Any]:
    fallback_openui = "\n".join(
        [
            "root = Root([notice])",
            f"notice = Notice({_json_arg(f'{ASSISTANT_FALLBACK} {error}')}, \"warning\")",
        ]
    )
    return {"encoded": _encode_openui(fallback_openui), "openui_lang": openui_lang, "error": error}


OPENUI_HTML_TEMPLATE = """
<div class="openui-host">
  <div data-openui-mount data-openui-encoded="${value.encoded}"></div>
</div>
"""


OPENUI_CSS_TEMPLATE = """
.openui-host { width: 100%; }
.openui-host [data-openui-mount] {
  display: block;
  min-height: 48px;
}
.openui-host [data-openui-mount]:empty::before {
  content: "Upload a CSV and ask a question to render OpenUI-Lang.";
  display: block;
  border: 1px dashed #cbd5e1;
  border-radius: 8px;
  padding: 14px;
  color: #64748b;
  background: #f8fafc;
}
"""


OPENUI_JS_ON_LOAD = """
const loadOpenUIRenderer = () => new Promise((resolve, reject) => {
  if (window.SmolnalysisOpenUIRenderer) {
    resolve();
    return;
  }
  const existing = document.querySelector('script[data-smolnalysis-openui-renderer]');
  if (existing) {
    existing.addEventListener('load', resolve, { once: true });
    existing.addEventListener('error', reject, { once: true });
    return;
  }
  const script = document.createElement('script');
  script.src = '/gradio_api/file=app/static/openui-renderer.js';
  script.dataset.smolnalysisOpenuiRenderer = 'true';
  script.onload = resolve;
  script.onerror = reject;
  document.head.appendChild(script);
});

loadOpenUIRenderer().then(() => {
  window.SmolnalysisOpenUIRenderer.mount(element, props.value?.encoded || "");
});
"""


class OpenUIRenderer(gr.HTML):
    def __init__(self, value: dict[str, Any] | None = None, **kwargs: Any):
        super().__init__(
            value=value or EMPTY_RENDER_VALUE,
            html_template=OPENUI_HTML_TEMPLATE,
            css_template=OPENUI_CSS_TEMPLATE,
            js_on_load=OPENUI_JS_ON_LOAD,
            apply_default_css=False,
            **kwargs,
        )


def openui_component(value: dict[str, Any] | None = None, **kwargs: Any) -> gr.HTML:
    return gr.HTML(
        value=value or EMPTY_RENDER_VALUE,
        html_template=OPENUI_HTML_TEMPLATE,
        css_template=OPENUI_CSS_TEMPLATE,
        js_on_load=OPENUI_JS_ON_LOAD,
        apply_default_css=False,
        **kwargs,
    )


def app_styles() -> str:
    return """
<style>
.gradio-container {
  background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
}
.app-shell { width: min(980px, calc(100vw - 32px)); margin: 0 auto; }
.app-hero { padding: 8px 4px 4px; }
.app-kicker {
  margin: 0 0 6px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #0369a1;
}
.app-hero h1 {
  margin: 0;
  font-size: clamp(30px, 5vw, 44px);
  line-height: 1;
  color: #0f172a;
}
.app-subtitle { max-width: 720px; margin: 10px 0 0; color: #475569; font-size: 15px; }
.upload-shell,
.chat-shell,
.render-shell {
  background: rgba(255, 255, 255, 0.86);
  border: 1px solid rgba(148, 163, 184, 0.24);
  border-radius: 8px;
  box-shadow: 0 14px 36px rgba(15, 23, 42, 0.06);
}
.upload-shell { margin-top: 12px; margin-bottom: 14px; }
.chat-shell,
.render-shell { padding: 10px; }
.composer-row { align-items: end; gap: 10px; margin-top: 10px; }
.raw-openui textarea { font-family: Consolas, monospace; font-size: 12px; }
</style>
"""
