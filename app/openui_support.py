from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from typing import Any

import gradio as gr
import pandas as pd


ASSISTANT_FALLBACK = "I could not render that UI response, so here is the text fallback."


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


def build_message_schema_example() -> dict[str, Any]:
    return {
        "user_message": "Show a histogram of population",
        "agent_steps": [
            {"role": "planner", "content": "Classify request as chart analysis."},
            {"role": "tool", "content": "Read uploaded CSV and select numeric column."},
        ],
        "openui_lang": 'root = Root([chart])\nchart = Chart("histogram", "Population", "population", null, [])',
        "fallback_text": "Histogram of population.",
    }


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _json_arg(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _top_records(df: pd.DataFrame, limit: int = 8) -> list[dict[str, Any]]:
    records = df.head(limit).where(pd.notna(df.head(limit)), None).to_dict(orient="records")
    return [{str(key): value for key, value in row.items()} for row in records]


def _find_column(query: str, columns: list[str]) -> str | None:
    normalized = query.casefold()
    for column in columns:
        if column.casefold() in normalized:
            return column
    return None


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes("number").columns)


def _text_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(exclude="number").columns)


def generate_openui_response(df: pd.DataFrame | None, prompt: str) -> ChatTurn:
    prompt = prompt.strip()
    steps = [AgentStep("planner", "Classified the user request and selected an analysis path.")]

    if df is None or df.empty:
        openui_lang = "\n".join(
            [
                "root = Root([notice])",
                'notice = Notice("Upload a CSV file first.", "warning")',
            ]
        )
        return ChatTurn(prompt, steps, openui_lang, "Upload a CSV file first.")

    numeric_columns = _numeric_columns(df)
    text_columns = _text_columns(df)
    selected_numeric = _find_column(prompt, numeric_columns) or (numeric_columns[0] if numeric_columns else None)
    selected_label = _find_column(prompt, text_columns) or (text_columns[0] if text_columns else None)

    lower_prompt = prompt.casefold()
    steps.append(AgentStep("tool", f"Loaded dataset with {len(df):,} rows and {len(df.columns):,} columns."))

    if "invalid openui" in lower_prompt:
        steps.append(AgentStep("validator", "Generated intentionally invalid OpenUI-Lang for fallback testing."))
        return ChatTurn(
            prompt,
            steps,
            "root = UnknownComponent([missing])",
            "This intentionally invalid response should render as a fallback.",
        )

    if any(word in lower_prompt for word in ["histogram", "distribution", "spread"]):
        if not selected_numeric:
            openui_lang = "\n".join(
                [
                    "root = Root([notice])",
                    'notice = Notice("No numeric column is available for a histogram.", "warning")',
                ]
            )
            return ChatTurn(prompt, steps, openui_lang, "No numeric column is available for a histogram.")

        chart_data = [{selected_numeric: value} for value in df[selected_numeric].dropna().head(250).tolist()]
        openui_lang = "\n".join(
            [
                "root = Root([chart, table])",
                f'chart = Chart("histogram", "Distribution of {selected_numeric}", "{selected_numeric}", null, {_json_arg(chart_data)})',
                f'table = DataTable("Sample rows", {_json_arg(_top_records(df))})',
            ]
        )
        return ChatTurn(prompt, steps, openui_lang, f"Rendered a histogram for {selected_numeric}.")

    if any(word in lower_prompt for word in ["plot", "chart", "bar", "compare", "visualize", "show"]):
        if not selected_numeric:
            openui_lang = "\n".join(
                [
                    "root = Root([notice, table])",
                    'notice = Notice("No numeric column is available for charting.", "warning")',
                    f'table = DataTable("Sample rows", {_json_arg(_top_records(df))})',
                ]
            )
            return ChatTurn(prompt, steps, openui_lang, "No numeric column is available for charting.")

        chart_columns = [selected_numeric]
        if selected_label:
            chart_columns.insert(0, selected_label)
        chart_data = df[chart_columns].dropna().head(20).to_dict(orient="records")
        x_column = selected_label if selected_label else None
        openui_lang = "\n".join(
            [
                "root = Root([chart])",
                f'chart = Chart("bar", "{selected_numeric} overview", {_json_arg(x_column)}, "{selected_numeric}", {_json_arg(chart_data)})',
            ]
        )
        return ChatTurn(prompt, steps, openui_lang, f"Rendered a chart for {selected_numeric}.")

    if any(word in lower_prompt for word in ["columns", "schema", "fields"]):
        rows = [
            {"column": column, "dtype": str(dtype), "missing": int(df[column].isna().sum())}
            for column, dtype in df.dtypes.items()
        ]
        openui_lang = "\n".join(
            [
                "root = Root([summary, table])",
                f'summary = InsightCard("Dataset schema", "{len(df.columns)} columns detected.")',
                f'table = DataTable("Columns", {_json_arg(rows)})',
            ]
        )
        return ChatTurn(prompt, steps, openui_lang, "Rendered the dataset schema.")

    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    openui_lang = "\n".join(
        [
            "root = Root([summary, metrics, table])",
            f'summary = InsightCard("Dataset summary", "Loaded {len(df):,} rows and {len(df.columns):,} columns.")',
            f'm1 = Metric("Rows", "{len(df):,}")',
            f'm2 = Metric("Columns", "{len(df.columns):,}")',
            f'm3 = Metric("Missing cells", "{missing_cells:,}")',
            f'm4 = Metric("Duplicate rows", "{duplicate_rows:,}")',
            "metrics = Metrics([m1, m2, m3, m4])",
            f'table = DataTable("Sample rows", {_json_arg(_top_records(df))})',
        ]
    )
    return ChatTurn(prompt, steps, openui_lang, "Rendered a dataset summary.")


def _split_args(args_text: str) -> list[str]:
    args: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escape_next = False

    for index, char in enumerate(args_text):
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and quote:
            escape_next = True
            continue
        if char in {'"', "'"}:
            quote = None if quote == char else char if quote is None else quote
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
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            return {"$ref": value}
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                return []
            return [_parse_value(part) for part in _split_args(inner)]
        raise OpenUIValidationError(f"Invalid argument value: {value}")


def parse_openui_lang(openui_lang: str) -> ParsedOpenUI:
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
        if component_type not in {"Root", "InsightCard", "Metric", "Metrics", "DataTable", "Chart", "Notice", "TextBlock"}:
            raise OpenUIValidationError(f"Unsupported component: {component_type}")

        args = [_parse_value(part) for part in _split_args(args_text)]
        components[identifier] = OpenUIComponent(identifier, component_type, args)

        if identifier == "root":
            if component_type != "Root":
                raise OpenUIValidationError("The `root` statement must use Root(...).")
            if not args or not isinstance(args[0], list):
                raise OpenUIValidationError("Root must receive a list of child references.")
            root_children = [item["$ref"] for item in args[0] if isinstance(item, dict) and "$ref" in item]

    if not root_children:
        raise OpenUIValidationError("OpenUI output must include `root = Root([...])`.")

    missing_refs = [ref for ref in root_children if ref not in components]
    if missing_refs:
        raise OpenUIValidationError(f"Missing component definitions: {', '.join(missing_refs)}")

    return ParsedOpenUI(root_children=root_children, components=components)


def render_openui_html(parsed: ParsedOpenUI) -> str:
    chunks = ['<div class="openui-render">']
    for child_ref in parsed.root_children:
        component = parsed.components[child_ref]
        rendered = _render_component(component, parsed.components)
        if rendered:
            chunks.append(rendered)
    chunks.append("</div>")
    return "\n".join(chunks)


def _render_component(component: OpenUIComponent, components: dict[str, OpenUIComponent]) -> str:
    args = component.args
    if component.component_type == "InsightCard":
        title = args[0] if len(args) > 0 else "Insight"
        body = args[1] if len(args) > 1 else ""
        return f'<section class="openui-card"><h3>{_escape(title)}</h3><p>{_escape(body)}</p></section>'

    if component.component_type == "Notice":
        message = args[0] if args else "Something needs attention."
        tone = args[1] if len(args) > 1 else "info"
        return f'<section class="openui-notice openui-{_escape(tone)}">{_escape(message)}</section>'

    if component.component_type == "TextBlock":
        return f'<p class="openui-text">{_escape(args[0] if args else "")}</p>'

    if component.component_type == "Metrics":
        refs = args[0] if args else []
        cards = []
        for ref in refs:
            if not isinstance(ref, dict) or "$ref" not in ref:
                continue
            metric = components.get(ref["$ref"])
            if not metric:
                continue
            label = metric.args[0] if metric.args else ""
            value = metric.args[1] if len(metric.args) > 1 else ""
            cards.append(
                '<div class="openui-metric">'
                f'<span>{_escape(label)}</span>'
                f'<strong>{_escape(value)}</strong>'
                "</div>"
            )
        return f'<section class="openui-metrics">{"".join(cards)}</section>'

    if component.component_type == "DataTable":
        title = args[0] if args else "Data"
        rows = args[1] if len(args) > 1 and isinstance(args[1], list) else []
        return _render_table(str(title), rows)

    if component.component_type == "Chart":
        return _render_chart_html(component)

    return ""


def _render_table(title: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f'<section class="openui-card"><h3>{_escape(title)}</h3><p>No rows to display.</p></section>'

    columns = list(rows[0].keys())
    head = "".join(f"<th>{_escape(column)}</th>" for column in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{_escape(row.get(column, ''))}</td>" for column in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        '<section class="openui-card">'
        f"<h3>{_escape(title)}</h3>"
        '<div class="openui-table-wrap"><table>'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div></section>"
    )


def _render_chart_html(component: OpenUIComponent) -> str:
    chart_type = str(component.args[0]) if len(component.args) > 0 else "bar"
    title = str(component.args[1]) if len(component.args) > 1 else "Chart"
    x_column = component.args[2] if len(component.args) > 2 else None
    y_column = component.args[3] if len(component.args) > 3 else None
    rows = component.args[4] if len(component.args) > 4 and isinstance(component.args[4], list) else []

    if not rows:
        return f'<section class="openui-card"><h3>{_escape(title)}</h3><p>No chart data to display.</p></section>'

    bars: list[tuple[str, float]] = []
    if chart_type == "histogram" and isinstance(x_column, str):
        numeric_values = [float(row[x_column]) for row in rows if isinstance(row, dict) and row.get(x_column) is not None]
        if not numeric_values:
            return f'<section class="openui-card"><h3>{_escape(title)}</h3><p>No numeric values available for histogram rendering.</p></section>'
        bin_count = min(8, max(3, len(numeric_values) // 25 or 3))
        minimum = min(numeric_values)
        maximum = max(numeric_values)
        if minimum == maximum:
            bars = [(f"{minimum:.1f}", float(len(numeric_values)))]
        else:
            step = (maximum - minimum) / bin_count
            counts = [0 for _ in range(bin_count)]
            for value in numeric_values:
                index = min(int((value - minimum) / step), bin_count - 1)
                counts[index] += 1
            for index, count in enumerate(counts):
                start = minimum + step * index
                end = start + step
                bars.append((f"{start:.0f}-{end:.0f}", float(count)))
    elif chart_type == "bar" and isinstance(y_column, str):
        for row in rows[:12]:
            if not isinstance(row, dict):
                continue
            raw_value = row.get(y_column)
            if raw_value is None:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if isinstance(x_column, str) and row.get(x_column) is not None:
                label = str(row[x_column])
            else:
                label = str(len(bars) + 1)
            bars.append((label, value))

    if not bars:
        return f'<section class="openui-card"><h3>{_escape(title)}</h3><p>Chart type `{_escape(chart_type)}` could not be rendered inline.</p></section>'

    max_value = max(value for _, value in bars) or 1.0
    bar_markup = []
    for label, value in bars:
        width = max(4.0, (value / max_value) * 100.0)
        bar_markup.append(
            '<div class="openui-chart-row">'
            f'<div class="openui-chart-label">{_escape(label)}</div>'
            '<div class="openui-chart-track">'
            f'<div class="openui-chart-bar" style="width: {width:.2f}%"></div>'
            '</div>'
            f'<div class="openui-chart-value">{_escape(f"{value:.0f}" if float(value).is_integer() else f"{value:.2f}")}</div>'
            '</div>'
        )

    axis_label = y_column if isinstance(y_column, str) else x_column if isinstance(x_column, str) else chart_type
    return (
        '<section class="openui-card openui-chart-card">'
        f'<h3>{_escape(title)}</h3>'
        f'<p class="openui-chart-meta">Inline { _escape(chart_type) } chart for {_escape(axis_label)}</p>'
        '<div class="openui-chart-grid">'
        f"{''.join(bar_markup)}"
        '</div>'
        '</section>'
    )


OPENUI_HTML_TEMPLATE = """
<div class="openui-render" data-openui-root>
  <section class="openui-placeholder">Ask a question to render OpenUI output.</section>
</div>
"""


OPENUI_CSS_TEMPLATE = """
.openui-render { display: grid; gap: 12px; }
.openui-host { display: grid; gap: 12px; }
.openui-placeholder {
  border: 1px dashed #d1d5db;
  border-radius: 8px;
  padding: 14px;
  color: #6b7280;
  background: #f9fafb;
}
.openui-card {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 14px;
  background: #ffffff;
}
.openui-card h3 { margin: 0 0 8px; font-size: 16px; }
.openui-card p { margin: 0; color: #374151; }
.openui-notice {
  border-radius: 8px;
  padding: 12px 14px;
  border: 1px solid #f59e0b;
  background: #fffbeb;
  color: #78350f;
}
.openui-notice.openui-error {
    border-color: #ef4444;
    background: #fef2f2;
    color: #991b1b;
}
.openui-metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 10px;
}
.openui-metric {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 12px;
  background: #f9fafb;
}
.openui-metric span { display: block; color: #6b7280; font-size: 12px; }
.openui-metric strong { display: block; margin-top: 4px; font-size: 20px; color: #111827; }
.openui-table-wrap { max-height: 320px; overflow: auto; }
.openui-table-wrap table { border-collapse: collapse; width: 100%; font-size: 13px; }
.openui-table-wrap th, .openui-table-wrap td {
  border-bottom: 1px solid #e5e7eb;
  padding: 8px;
  text-align: left;
  vertical-align: top;
}
.openui-table-wrap th { background: #f9fafb; position: sticky; top: 0; }
"""


OPENUI_JS_ON_LOAD = """
const runtimeCacheKey = "__smolnalysis_openui_runtime";
const rootCacheKey = "__smolnalysis_openui_roots";

const rootCache = window[rootCacheKey] ||= new WeakMap();

const renderPlaceholder = (host, message, tone = "") => {
    host.innerHTML = `<section class="openui-placeholder${tone ? ` ${tone}` : ""}">${message}</section>`;
};

const loadRuntime = async () => {
    if (!window[runtimeCacheKey]) {
        window[runtimeCacheKey] = (async () => {
            const [ReactModule, ReactDOMModule, LangModule, ZodModule] = await Promise.all([
                import("https://esm.sh/react@18?bundle"),
                import("https://esm.sh/react-dom@18/client?bundle"),
                import("https://esm.sh/@openuidev/react-lang?bundle"),
                import("https://esm.sh/zod@4?bundle"),
            ]);

            const React = ReactModule.default || ReactModule;
            const { createRoot } = ReactDOMModule;
            const { Renderer, createLibrary, defineComponent } = LangModule;
            const { z } = ZodModule;
            const h = React.createElement;

            const Metric = defineComponent({
                name: "Metric",
                description: "Displays a metric label and value.",
                props: z.object({ label: z.string(), value: z.string() }),
                component: ({ props }) => h(
                    "div",
                    { className: "openui-metric" },
                    h("span", null, props.label),
                    h("strong", null, props.value),
                ),
            });

            const InsightCard = defineComponent({
                name: "InsightCard",
                description: "Displays a titled insight card.",
                props: z.object({ title: z.string(), body: z.string() }),
                component: ({ props }) => h(
                    "section",
                    { className: "openui-card" },
                    h("h3", null, props.title),
                    h("p", null, props.body),
                ),
            });

            const Notice = defineComponent({
                name: "Notice",
                description: "Displays an informational or warning notice.",
                props: z.object({ message: z.string(), tone: z.string().optional() }),
                component: ({ props }) => h(
                    "section",
                    { className: `openui-notice${props.tone ? ` openui-${props.tone}` : ""}` },
                    props.message,
                ),
            });

            const TextBlock = defineComponent({
                name: "TextBlock",
                description: "Displays a text paragraph.",
                props: z.object({ text: z.string() }),
                component: ({ props }) => h("p", { className: "openui-text" }, props.text),
            });

            const DataTable = defineComponent({
                name: "DataTable",
                description: "Displays rows of structured data.",
                props: z.object({
                    title: z.string(),
                    rows: z.array(z.record(z.string(), z.any())),
                }),
                component: ({ props }) => {
                    const rows = Array.isArray(props.rows) ? props.rows : [];
                    if (!rows.length) {
                        return h(
                            "section",
                            { className: "openui-card" },
                            h("h3", null, props.title),
                            h("p", null, "No rows to display."),
                        );
                    }

                    const columns = Object.keys(rows[0] || {});
                    return h(
                        "section",
                        { className: "openui-card" },
                        h("h3", null, props.title),
                        h(
                            "div",
                            { className: "openui-table-wrap" },
                            h(
                                "table",
                                null,
                                h(
                                    "thead",
                                    null,
                                    h(
                                        "tr",
                                        null,
                                        ...columns.map((column) => h("th", { key: column }, column)),
                                    ),
                                ),
                                h(
                                    "tbody",
                                    null,
                                    ...rows.map((row, index) => h(
                                        "tr",
                                        { key: `${index}` },
                                        ...columns.map((column) => h("td", { key: `${index}-${column}` }, String(row[column] ?? ""))),
                                    )),
                                ),
                            ),
                        ),
                    );
                },
            });

            const Chart = defineComponent({
                name: "Chart",
                description: "Displays metadata for a chart rendered elsewhere in the interface.",
                props: z.object({
                    chartType: z.string(),
                    title: z.string(),
                    xColumn: z.string().nullable().optional(),
                    yColumn: z.string().nullable().optional(),
                    rows: z.array(z.record(z.string(), z.any())),
                }),
                component: ({ props }) => h(
                    "section",
                    { className: "openui-card" },
                    h("h3", null, props.title),
                    h(
                        "p",
                        null,
                        `Chart type: ${props.chartType}. The plotted figure is shown in the chart panel below.`,
                    ),
                ),
            });

            const Metrics = defineComponent({
                name: "Metrics",
                description: "Displays multiple metric cards.",
                props: z.object({ items: z.array(Metric.ref) }),
                component: ({ props, renderNode }) => h(
                    "section",
                    { className: "openui-metrics" },
                    renderNode(props.items),
                ),
            });

            const Root = defineComponent({
                name: "Root",
                description: "Root container for the full OpenUI response.",
                props: z.object({
                    children: z.array(
                        z.union([
                            InsightCard.ref,
                            Metrics.ref,
                            DataTable.ref,
                            Chart.ref,
                            Notice.ref,
                            TextBlock.ref,
                        ])
                    ),
                }),
                component: ({ props, renderNode }) => h(
                    "div",
                    { className: "openui-host" },
                    renderNode(props.children),
                ),
            });

            const library = createLibrary({
                root: "Root",
                components: [Root, InsightCard, Metric, Metrics, DataTable, Chart, Notice, TextBlock],
            });

            return { React, Renderer, createRoot, library, h };
        })();
    }

    return window[runtimeCacheKey];
};

const normalizeValue = (value) => {
    if (!value || typeof value !== "object") {
        return { status: "empty", response: "", error: "" };
    }
    return {
        status: typeof value.status === "string" ? value.status : "empty",
        response: typeof value.response === "string" ? value.response : "",
        error: typeof value.error === "string" ? value.error : "",
    };
};

const renderOpenUI = async () => {
    const host = element.querySelector("[data-openui-root]");
    if (!host) return;

    const value = normalizeValue(props.value);
    if (!value.response) {
        renderPlaceholder(host, "Ask a question to render OpenUI output.");
        return;
    }

    const runtime = await loadRuntime();
    let root = rootCache.get(element);
    if (!root) {
        host.innerHTML = "";
        root = runtime.createRoot(host);
        rootCache.set(element, root);
    }

    root.render(
        runtime.h(
            runtime.React.Fragment,
            null,
            value.status === "error"
                ? runtime.h(
                        "section",
                        { className: "openui-notice openui-error" },
                        value.error || "The OpenUI response was invalid and was replaced with a fallback notice.",
                    )
                : null,
            runtime.h(runtime.Renderer, {
                library: runtime.library,
                response: value.response,
                isStreaming: false,
                onError: (errors) => {
                    if (Array.isArray(errors) && errors.length) {
                        console.warn("OpenUI renderer errors", errors);
                    }
                },
            }),
        ),
    );
};

renderOpenUI().catch((error) => {
    const host = element.querySelector("[data-openui-root]");
    if (!host) return;
    console.error(error);
    host.innerHTML = `<section class="openui-notice openui-error">Failed to load the OpenUI React renderer: ${String(error)}</section>`;
});
"""


class OpenUIRenderer(gr.HTML):
    def __init__(self, value: dict[str, Any] | None = None, **kwargs: Any):
        super().__init__(
            value=value or {"status": "empty", "response": "", "error": ""},
            html_template=OPENUI_HTML_TEMPLATE,
            css_template=OPENUI_CSS_TEMPLATE,
            js_on_load=OPENUI_JS_ON_LOAD,
            apply_default_css=False,
            **kwargs,
        )


def openui_styles() -> str:
    return """
<style>
.gradio-container {
    background:
        radial-gradient(circle at top left, rgba(14, 165, 233, 0.12), transparent 28%),
        linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
}
.app-shell {
    width: min(960px, calc(100vw - 32px));
    margin: 0 auto;
}
.app-hero {
    padding: 8px 4px 4px;
}
.app-kicker {
    margin: 0 0 6px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #0369a1;
}
.app-hero h1 {
    margin: 0;
    font-size: clamp(30px, 5vw, 44px);
    line-height: 0.95;
    letter-spacing: -0.04em;
    color: #0f172a;
}
.app-subtitle {
    max-width: 680px;
    margin: 10px 0 0;
    color: #475569;
    font-size: 15px;
}
.upload-shell,
.chat-shell {
    background: rgba(255, 255, 255, 0.82);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 18px;
    box-shadow: 0 16px 40px rgba(15, 23, 42, 0.06);
}
.upload-shell {
    margin-top: 12px;
    margin-bottom: 14px;
}
.chat-shell {
    padding: 10px;
}
.composer-row {
    align-items: end;
    gap: 10px;
    margin-top: 10px;
}
.composer-input label,
.composer-input .block-title {
    display: none !important;
}
.composer-input textarea {
    border-radius: 14px !important;
}
.openui-debug {
    margin-top: 10px;
    border-top: 1px solid #e5e7eb;
    padding-top: 10px;
}
.openui-debug summary {
    cursor: pointer;
    font-size: 12px;
    font-weight: 600;
    color: #475569;
}
.openui-debug ul {
    margin: 10px 0 8px;
    padding-left: 18px;
    color: #475569;
}
.openui-debug pre {
    margin: 0;
    padding: 12px;
    overflow: auto;
    border-radius: 10px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    font-size: 12px;
}
.openui-render { display: grid; gap: 12px; }
.openui-card {
  border: 1px solid #e5e7eb;
    border-radius: 12px;
  padding: 14px;
  background: #ffffff;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
}
.openui-card h3 { margin: 0 0 8px; font-size: 16px; }
.openui-card p { margin: 0; color: #374151; }
.openui-notice {
    border-radius: 12px;
  padding: 12px 14px;
  border: 1px solid #f59e0b;
  background: #fffbeb;
  color: #78350f;
}
.openui-metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 10px;
}
.openui-metric {
  border: 1px solid #e5e7eb;
    border-radius: 12px;
  padding: 12px;
  background: #f9fafb;
}
.openui-metric span { display: block; color: #6b7280; font-size: 12px; }
.openui-metric strong { display: block; margin-top: 4px; font-size: 20px; color: #111827; }
.openui-table-wrap { max-height: 320px; overflow: auto; }
.openui-table-wrap table { border-collapse: collapse; width: 100%; font-size: 13px; }
.openui-table-wrap th, .openui-table-wrap td {
  border-bottom: 1px solid #e5e7eb;
  padding: 8px;
  text-align: left;
  vertical-align: top;
}
.openui-table-wrap th { background: #f9fafb; position: sticky; top: 0; }
.openui-chart-card { gap: 12px; }
.openui-chart-meta { color: #6b7280; margin-bottom: 12px; }
.openui-chart-grid { display: grid; gap: 10px; }
.openui-chart-row {
    display: grid;
    grid-template-columns: minmax(80px, 120px) minmax(0, 1fr) minmax(48px, 64px);
    gap: 10px;
    align-items: center;
}
.openui-chart-label,
.openui-chart-value { font-size: 12px; color: #374151; }
.openui-chart-track {
    height: 14px;
    border-radius: 999px;
    background: #e5e7eb;
    overflow: hidden;
}
.openui-chart-bar {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #2563eb 0%, #0ea5e9 100%);
}
@media (max-width: 720px) {
    .app-shell {
        width: min(100vw - 20px, 960px);
    }
    .composer-row {
        gap: 8px;
    }
    .openui-chart-row {
        grid-template-columns: minmax(64px, 88px) minmax(0, 1fr) minmax(40px, 52px);
        gap: 8px;
    }
}
</style>
"""
