from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Any

import pandas as pd


_PRESET_INDEX = 0

PRESET_NAMES = [
    "metrics-cards",
    "callout-highlights",
    "tabs-view",
    "chart-details",
    "schema-table",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class AgentStep:
    role: str
    content: str


@dataclass
class ChatTurn:
    user_message: str
    agent_steps: list[AgentStep]
    rendered_html: str
    preset_name: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _e(text: Any) -> str:
    return html.escape(str(text))


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]


def _text_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]


def _top_records(df: pd.DataFrame, n: int = 8) -> list[dict[str, Any]]:
    return df.head(n).to_dict(orient="records")


def _find_column(prompt: str, columns: list[str]) -> str | None:
    lower = prompt.casefold()
    for col in columns:
        if col.casefold() in lower:
            return col
    return None


def _next_preset() -> str:
    global _PRESET_INDEX
    name = PRESET_NAMES[_PRESET_INDEX % len(PRESET_NAMES)]
    _PRESET_INDEX += 1
    return name


# ---------------------------------------------------------------------------
# Preset 1 — metrics-cards
# ---------------------------------------------------------------------------


def _preset_metrics_cards(df: pd.DataFrame, prompt: str) -> str:
    rows = len(df)
    cols = len(df.columns)
    missing = int(df.isna().sum().sum())
    numeric = len(_numeric_columns(df))
    text = len(_text_columns(df))

    metrics_html = "".join(
        f'<div class="openui-metric"><span>{_e(label)}</span><strong>{_e(value)}</strong></div>'
        for label, value in [
            ("Rows", f"{rows:,}"),
            ("Columns", f"{cols:,}"),
            ("Numeric", f"{numeric}"),
            ("Text cols", f"{text}"),
            ("Missing", f"{missing:,}"),
        ]
    )

    records = _top_records(df)
    col_headers = "".join(f"<th>{_e(c)}</th>" for c in df.columns)
    table_rows = "".join(
        "<tr>" + "".join(f"<td>{_e(row.get(c, ''))}</td>" for c in df.columns) + "</tr>"
        for row in records
    )

    return (
        '<div class="openui-render">'
        '<section class="openui-card">'
        f'<h3>{_e(prompt)}</h3>'
        f'<section class="openui-metrics">{metrics_html}</section>'
        '</section>'
        '<section class="openui-card"><h3>Sample rows</h3>'
        '<div class="openui-table-wrap">'
        f'<table><thead><tr>{col_headers}</tr></thead><tbody>{table_rows}</tbody></table>'
        '</div></section>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Preset 2 — callout-highlights
# ---------------------------------------------------------------------------


def _preset_callout_highlights(df: pd.DataFrame, prompt: str) -> str:
    rows = len(df)
    cols = len(df.columns)
    missing = int(df.isna().sum().sum())
    numeric_cols = _numeric_columns(df)

    insights: list[tuple[str, str]] = [
        ("Dataset size", f"{rows:,} rows × {cols:,} columns"),
        ("Missing values", f"{missing:,} cells ({missing / max(rows * cols, 1) * 100:.1f}%)"),
    ]
    for col in numeric_cols[:4]:
        series = df[col].dropna()
        if not series.empty:
            insights.append((col, f"min {series.min():.2f}  /  max {series.max():.2f}  /  mean {series.mean():.2f}"))

    cards_html = "".join(
        f'<section class="openui-card"><h3>{_e(title)}</h3><p>{_e(body)}</p></section>'
        for title, body in insights
    )

    return (
        '<div class="openui-render">'
        '<section class="openui-notice openui-info">'
        f'<strong>{_e(prompt)}</strong>'
        f'<p>{rows:,} rows loaded — key column insights below.</p>'
        '</section>'
        f'<div class="openui-split">{cards_html}</div>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Preset 3 — tabs-view
# ---------------------------------------------------------------------------


def _preset_tabs_view(df: pd.DataFrame, prompt: str) -> str:
    rows = len(df)
    cols = len(df.columns)
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    summary_html = "".join(
        f'<div class="openui-metric"><span>{_e(k)}</span><strong>{_e(v)}</strong></div>'
        for k, v in [("Rows", f"{rows:,}"), ("Columns", f"{cols:,}"), ("Missing", f"{missing:,}"), ("Dupes", f"{duplicates:,}")]
    )

    col_rows = "".join(
        f'<tr><td>{_e(c)}</td><td>{_e(str(df[c].dtype))}</td><td>{_e(int(df[c].isna().sum()))}</td></tr>'
        for c in df.columns
    )

    records = _top_records(df)
    col_headers = "".join(f"<th>{_e(c)}</th>" for c in df.columns)
    sample_rows_html = "".join(
        "<tr>" + "".join(f"<td>{_e(row.get(c, ''))}</td>" for c in df.columns) + "</tr>"
        for row in records
    )

    return (
        '<div class="openui-render"><section class="openui-card">'
        f'<h3>{_e(prompt)}</h3>'
        '<div class="openui-tabs">'
        '<div class="openui-tab-nav">'
        '<span class="openui-tab-pill openui-tab-active">Summary</span>'
        '<span class="openui-tab-pill">Columns</span>'
        '<span class="openui-tab-pill">Sample</span>'
        '</div>'
        f'<section class="openui-tab-panel"><section class="openui-metrics">{summary_html}</section></section>'
        f'<section class="openui-tab-panel openui-tab-panel--hidden">'
        '<div class="openui-table-wrap"><table>'
        '<thead><tr><th>Column</th><th>Type</th><th>Missing</th></tr></thead>'
        f'<tbody>{col_rows}</tbody></table></div></section>'
        f'<section class="openui-tab-panel openui-tab-panel--hidden">'
        '<div class="openui-table-wrap"><table>'
        f'<thead><tr>{col_headers}</tr></thead>'
        f'<tbody>{sample_rows_html}</tbody></table></div></section>'
        '</div></section></div>'
    )


# ---------------------------------------------------------------------------
# Preset 4 — chart-details
# ---------------------------------------------------------------------------


def _preset_chart_details(df: pd.DataFrame, prompt: str) -> str:
    numeric_cols = _numeric_columns(df)
    text_cols = _text_columns(df)

    selected_num = _find_column(prompt, numeric_cols) or (numeric_cols[0] if numeric_cols else None)
    selected_label = _find_column(prompt, text_cols) or (text_cols[0] if text_cols else None)

    if selected_num:
        subset_cols = [c for c in ([selected_label, selected_num] if selected_label else [selected_num]) if c]
        chart_data = df[subset_cols].dropna().head(14).to_dict(orient="records")
        values = [float(row[selected_num]) for row in chart_data]
        labels = [str(row[selected_label]) if selected_label else str(i + 1) for i, row in enumerate(chart_data)]
        max_val = max(values) or 1.0

        bar_rows = "".join(
            f'<div class="openui-chart-row">'
            f'<div class="openui-chart-label">{_e(label)}</div>'
            f'<div class="openui-chart-track"><div class="openui-chart-bar" style="width:{max(4.0, v / max_val * 100):.1f}%"></div></div>'
            f'<div class="openui-chart-value">{_e(f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}")}</div>'
            f'</div>'
            for label, v in zip(labels, values)
        )
        chart_html = (
            '<section class="openui-card openui-chart-card">'
            f'<h3>{_e(selected_num)}</h3>'
            f'<p class="openui-chart-meta">Bar chart — {_e(selected_num)}</p>'
            f'<div class="openui-chart-grid">{bar_rows}</div>'
            '</section>'
        )
    else:
        chart_html = (
            '<section class="openui-notice openui-info">'
            '<strong>No numeric column found</strong>'
            '<p>Upload a dataset with numeric columns to see a chart.</p>'
            '</section>'
        )

    schema_rows = "".join(
        f'<tr><td>{_e(c)}</td><td>{_e(str(df[c].dtype))}</td><td>{_e(int(df[c].isna().sum()))}</td></tr>'
        for c in df.columns
    )

    return (
        '<div class="openui-render">'
        f'{chart_html}'
        '<section class="openui-card"><h3>Column schema</h3>'
        '<div class="openui-table-wrap"><table>'
        '<thead><tr><th>Column</th><th>Type</th><th>Missing</th></tr></thead>'
        f'<tbody>{schema_rows}</tbody></table></div></section>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Preset 5 — schema-table
# ---------------------------------------------------------------------------


def _preset_schema_table(df: pd.DataFrame, prompt: str) -> str:
    missing_total = int(df.isna().sum().sum())

    schema_rows = "".join(
        f'<tr>'
        f'<td><strong>{_e(c)}</strong></td>'
        f'<td>{_e(str(df[c].dtype))}</td>'
        f'<td>{_e(df[c].nunique())}</td>'
        f'<td>{_e(int(df[c].isna().sum()))}</td>'
        f'<td>{_e(str(df[c].iloc[0]) if not df[c].empty else "—")}</td>'
        f'</tr>'
        for c in df.columns
    )

    tone = "openui-warning" if missing_total > 0 else "openui-success"
    callout_msg = (
        f"{missing_total:,} missing values detected across {len(df.columns)} columns."
        if missing_total
        else f"No missing values — dataset looks clean ({len(df):,} rows)."
    )

    return (
        '<div class="openui-render">'
        f'<section class="openui-notice {_e(tone)}">'
        f'<strong>{_e(prompt)}</strong>'
        f'<p>{_e(callout_msg)}</p>'
        '</section>'
        '<section class="openui-card"><h3>Column details</h3>'
        '<div class="openui-table-wrap"><table>'
        '<thead><tr><th>Column</th><th>Type</th><th>Unique</th><th>Missing</th><th>First value</th></tr></thead>'
        f'<tbody>{schema_rows}</tbody></table></div></section>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# No-dataset fallback
# ---------------------------------------------------------------------------


def _preset_no_dataset(prompt: str, preset_name: str) -> str:
    return (
        '<div class="openui-render">'
        '<section class="openui-notice openui-info">'
        f'<strong>{_e(prompt)}</strong>'
        f'<p>Upload a CSV file to see an analysis. This would use the <em>{_e(preset_name)}</em> layout.</p>'
        '</section>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_openui_response(df: "pd.DataFrame | None", prompt: str) -> ChatTurn:
    steps: list[AgentStep] = [AgentStep("planner", f"Received: {prompt[:80]}")]
    preset_name = _next_preset()
    steps.append(AgentStep("planner", f"Rotating to preset: {preset_name}"))

    if df is None:
        steps.append(AgentStep("tool", "No dataset loaded."))
        return ChatTurn(prompt, steps, _preset_no_dataset(prompt, preset_name), preset_name)

    steps.append(AgentStep("tool", f"Dataset: {len(df):,} rows x {len(df.columns):,} columns"))

    dispatch = {
        "metrics-cards": _preset_metrics_cards,
        "callout-highlights": _preset_callout_highlights,
        "tabs-view": _preset_tabs_view,
        "chart-details": _preset_chart_details,
        "schema-table": _preset_schema_table,
    }
    html_out = dispatch.get(preset_name, _preset_schema_table)(df, prompt)

    steps.append(AgentStep("renderer", f"Rendered {preset_name} preset."))
    return ChatTurn(prompt, steps, html_out, preset_name)


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------


def openui_styles() -> str:
    return """
<style>
.gradio-container {
    background:
        radial-gradient(circle at top left, rgba(14, 165, 233, 0.12), transparent 28%),
        linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
}
.app-shell { width: min(960px, calc(100vw - 32px)); margin: 0 auto; }
.app-hero { padding: 8px 4px 4px; }
.app-kicker {
    margin: 0 0 6px; font-size: 11px; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase; color: #0369a1;
}
.app-hero h1 {
    margin: 0; font-size: clamp(30px, 5vw, 44px);
    line-height: 0.95; letter-spacing: -0.04em; color: #0f172a;
}
.app-subtitle { max-width: 680px; margin: 10px 0 0; color: #475569; font-size: 15px; }
.upload-shell,
.chat-shell {
    background: rgba(255,255,255,0.82); backdrop-filter: blur(10px);
    border: 1px solid rgba(148,163,184,0.22); border-radius: 18px;
    box-shadow: 0 16px 40px rgba(15,23,42,0.06);
}
.upload-shell { margin-top: 12px; margin-bottom: 14px; }
.chat-shell { padding: 10px; }
.composer-row { align-items: end; gap: 10px; margin-top: 10px; }
.openui-debug { margin-top: 10px; border-top: 1px solid #e5e7eb; padding-top: 10px; }
.openui-debug summary { cursor: pointer; font-size: 12px; font-weight: 600; color: #475569; }
.openui-debug ul { margin: 10px 0 8px; padding-left: 18px; color: #475569; }
.openui-debug pre { margin:0; padding:12px; overflow:auto; border-radius:10px; background:#f8fafc; border:1px solid #e2e8f0; font-size:12px; }
.openui-render { display: grid; gap: 12px; }
.openui-split { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
.openui-card { border:1px solid #e5e7eb; border-radius:12px; padding:14px; background:#fff; box-shadow:0 8px 24px rgba(15,23,42,0.04); }
.openui-card h3 { margin: 0 0 10px; font-size: 15px; font-weight: 600; color: #0f172a; }
.openui-card p  { margin: 0; color: #374151; font-size: 14px; }
.openui-notice { border-radius: 12px; padding: 12px 16px; font-size: 14px; }
.openui-notice strong { display: block; margin-bottom: 4px; font-weight: 600; }
.openui-notice p { margin: 0; }
.openui-info    { border: 1px solid #bae6fd; background: #f0f9ff; color: #0c4a6e; }
.openui-success { border: 1px solid #bbf7d0; background: #f0fdf4; color: #14532d; }
.openui-warning { border: 1px solid #fde68a; background: #fffbeb; color: #78350f; }
.openui-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 10px; margin-top: 10px; }
.openui-metric { border:1px solid #e5e7eb; border-radius:10px; padding:12px; background:#f9fafb; }
.openui-metric span   { display:block; color:#6b7280; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; }
.openui-metric strong { display:block; margin-top:4px; font-size:22px; color:#111827; font-weight:700; }
.openui-table-wrap { max-height: 320px; overflow: auto; margin-top: 6px; }
.openui-table-wrap table { border-collapse: collapse; width: 100%; font-size: 13px; }
.openui-table-wrap th,
.openui-table-wrap td { border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; vertical-align: top; }
.openui-table-wrap th { background:#f9fafb; position:sticky; top:0; font-weight:600; font-size:12px; color:#374151; }
.openui-tabs { margin-top: 6px; }
.openui-tab-nav { display: flex; gap: 6px; margin-bottom: 12px; flex-wrap: wrap; }
.openui-tab-pill { padding:4px 12px; border-radius:999px; background:#f1f5f9; font-size:12px; font-weight:600; color:#475569; cursor:pointer; border:1px solid #e2e8f0; }
.openui-tab-pill.openui-tab-active { background:#0ea5e9; color:#fff; border-color:#0ea5e9; }
.openui-tab-panel--hidden { display: none; }
.openui-chart-card { gap: 12px; }
.openui-chart-meta { color:#6b7280; margin:0 0 12px; font-size:12px; }
.openui-chart-grid { display: grid; gap: 8px; }
.openui-chart-row { display:grid; grid-template-columns:minmax(80px,120px) minmax(0,1fr) minmax(44px,64px); gap:10px; align-items:center; }
.openui-chart-label, .openui-chart-value { font-size:12px; color:#374151; }
.openui-chart-track { height:14px; border-radius:999px; background:#e5e7eb; overflow:hidden; }
.openui-chart-bar { height:100%; border-radius:999px; background:linear-gradient(90deg,#2563eb 0%,#0ea5e9 100%); }
@media (max-width: 720px) {
    .app-shell { width: min(100vw - 20px, 960px); }
    .openui-chart-row { grid-template-columns: minmax(64px,88px) minmax(0,1fr) minmax(40px,52px); gap:8px; }
}
</style>
"""