from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from typing import Any

import gradio as gr
import matplotlib.pyplot as plt
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


def render_openui_value(parsed: ParsedOpenUI) -> dict[str, Any]:
    return {
        "status": "ok",
        "root_children": parsed.root_children,
        "components": {
            name: {
                "identifier": component.identifier,
                "component_type": component.component_type,
                "args": component.args,
            }
            for name, component in parsed.components.items()
        },
    }


def render_openui_error(message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "error": message,
        "root_children": ["fallback"],
        "components": {
            "fallback": {
                "identifier": "fallback",
                "component_type": "Notice",
                "args": [message, "warning"],
            }
        },
    }


def render_openui_plot(parsed: ParsedOpenUI) -> Any:
    for child_ref in parsed.root_children:
        component = parsed.components[child_ref]
        if component.component_type == "Chart":
            return _render_chart(component)
    return None


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


def _render_chart(component: OpenUIComponent) -> Any:
    chart_type = component.args[0] if len(component.args) > 0 else "bar"
    title = component.args[1] if len(component.args) > 1 else "Chart"
    x_column = component.args[2] if len(component.args) > 2 else None
    y_column = component.args[3] if len(component.args) > 3 else None
    rows = component.args[4] if len(component.args) > 4 and isinstance(component.args[4], list) else []

    if not rows:
        return None

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    if chart_type == "histogram" and y_column is None and x_column in df:
        df[x_column].plot(kind="hist", bins=20, ax=ax)
        ax.set_xlabel(str(x_column))
    elif chart_type == "bar" and y_column in df:
        if x_column and x_column in df:
            df.plot(kind="bar", x=x_column, y=y_column, ax=ax, legend=False)
        else:
            df[y_column].plot(kind="bar", ax=ax)
        ax.set_ylabel(str(y_column))
    else:
        return None

    ax.set_title(str(title))
    fig.tight_layout()
    return fig


OPENUI_HTML_TEMPLATE = """
<div class="openui-render" data-openui-root>
  <section class="openui-placeholder">Ask a question to render OpenUI output.</section>
</div>
"""


OPENUI_CSS_TEMPLATE = """
.openui-render { display: grid; gap: 12px; }
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
const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");

const renderTable = (title, rows) => {
  if (!Array.isArray(rows) || rows.length === 0) {
    return `<section class="openui-card"><h3>${escapeHtml(title)}</h3><p>No rows to display.</p></section>`;
  }
  const columns = Object.keys(rows[0]);
  const head = columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("");
  const body = rows.map((row) => {
    const cells = columns.map((column) => `<td>${escapeHtml(row[column])}</td>`).join("");
    return `<tr>${cells}</tr>`;
  }).join("");
  return `<section class="openui-card"><h3>${escapeHtml(title)}</h3><div class="openui-table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div></section>`;
};

const renderComponent = (component, components) => {
  if (!component) return "";
  const args = component.args || [];
  if (component.component_type === "InsightCard") {
    return `<section class="openui-card"><h3>${escapeHtml(args[0] || "Insight")}</h3><p>${escapeHtml(args[1] || "")}</p></section>`;
  }
  if (component.component_type === "Notice") {
    return `<section class="openui-notice">${escapeHtml(args[0] || "Something needs attention.")}</section>`;
  }
  if (component.component_type === "TextBlock") {
    return `<p class="openui-text">${escapeHtml(args[0] || "")}</p>`;
  }
  if (component.component_type === "Metrics") {
    const cards = (args[0] || []).map((ref) => {
      const metric = components[ref?.$ref];
      if (!metric) return "";
      return `<div class="openui-metric"><span>${escapeHtml(metric.args?.[0] || "")}</span><strong>${escapeHtml(metric.args?.[1] || "")}</strong></div>`;
    }).join("");
    return `<section class="openui-metrics">${cards}</section>`;
  }
  if (component.component_type === "DataTable") {
    return renderTable(args[0] || "Data", args[1] || []);
  }
  if (component.component_type === "Chart") {
    return `<section class="openui-card"><h3>${escapeHtml(args[1] || "Chart")}</h3><p>Chart rendered below.</p></section>`;
  }
  return "";
};

const renderOpenUI = () => {
  const root = element.querySelector("[data-openui-root]");
  const value = props.value || {};
  const components = value.components || {};
  const children = value.root_children || [];
  if (!children.length) return;
  root.innerHTML = children.map((name) => renderComponent(components[name], components)).join("");
};

renderOpenUI();
"""

from gradio.events import Dependency

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
    from typing import Callable, Literal, Sequence, Any, TYPE_CHECKING
    from gradio.blocks import Block
    if TYPE_CHECKING:
        from gradio.components import Timer
        from gradio.components.base import Component


def openui_styles() -> str:
    return """
<style>
.openui-render { display: grid; gap: 12px; }
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
</style>
"""