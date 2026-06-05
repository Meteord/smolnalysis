from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd


MAX_PREVIEW_ROWS = 25
MAX_UPLOAD_MB = 25


def _empty_result(message: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, Any]:
    return (
        pd.DataFrame(),
        pd.DataFrame({"message": [message]}),
        pd.DataFrame(),
        message,
        None,
    )


def _read_csv(file_path: str | None) -> pd.DataFrame:
    if not file_path:
        raise ValueError("Upload a CSV file to begin.")

    path = Path(file_path)
    if path.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"CSV file is larger than {MAX_UPLOAD_MB} MB.")

    return pd.read_csv(path)


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "Rows", "value": f"{len(df):,}"},
        {"metric": "Columns", "value": f"{len(df.columns):,}"},
        {"metric": "Numeric columns", "value": f"{len(df.select_dtypes('number').columns):,}"},
        {"metric": "Text-like columns", "value": f"{len(df.select_dtypes('object').columns):,}"},
        {"metric": "Duplicate rows", "value": f"{int(df.duplicated().sum()):,}"},
        {"metric": "Missing cells", "value": f"{int(df.isna().sum().sum()):,}"},
    ]
    return pd.DataFrame(rows)


def _build_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    profile = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "missing": df.isna().sum().values,
            "missing_pct": (df.isna().mean() * 100).round(2).values,
            "unique": df.nunique(dropna=True).values,
        }
    )
    return profile.sort_values(["missing", "column"], ascending=[False, True]).reset_index(drop=True)


def _new_figure() -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    return fig, ax


def _default_plot(df: pd.DataFrame) -> Any:
    numeric_columns = list(df.select_dtypes("number").columns)
    if not numeric_columns:
        return None

    first = numeric_columns[0]
    fig, ax = _new_figure()
    df[first].head(50).plot(kind="line", ax=ax)
    ax.set_title(f"First 50 values of {first}")
    ax.set_xlabel("Row")
    ax.set_ylabel(first)
    fig.tight_layout()
    return fig


def _format_insights(df: pd.DataFrame) -> str:
    insights: list[str] = []
    numeric_columns = list(df.select_dtypes("number").columns)
    missing = df.isna().sum()
    high_missing = missing[missing > 0].sort_values(ascending=False).head(5)

    insights.append(f"Loaded {len(df):,} rows and {len(df.columns):,} columns.")

    if numeric_columns:
        insights.append(f"Numeric columns detected: {', '.join(numeric_columns[:8])}.")
    else:
        insights.append("No numeric columns detected yet, so charting is limited to previews and profiles.")

    if not high_missing.empty:
        missing_bits = [f"{col} ({int(count):,})" for col, count in high_missing.items()]
        insights.append(f"Columns with missing values: {', '.join(missing_bits)}.")
    else:
        insights.append("No missing values were found.")

    if len(numeric_columns) >= 2:
        corr = df[numeric_columns].corr(numeric_only=True).abs()
        corr_values = corr.copy()
        for column in corr_values.columns:
            corr_values.loc[column, column] = pd.NA
        top_pair = corr_values.stack().sort_values(ascending=False).head(1)
        if not top_pair.empty:
            (left, right), value = top_pair.index[0], float(top_pair.iloc[0])
            insights.append(f"Strongest numeric relationship: {left} and {right} with |r|={value:.2f}.")

    return "\n".join(f"- {insight}" for insight in insights)


def analyze_csv(file_path: str | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, Any]:
    try:
        df = _read_csv(file_path)
    except Exception as exc:
        return _empty_result(str(exc))

    if df.empty:
        return _empty_result("The uploaded CSV is empty.")

    return (
        df.head(MAX_PREVIEW_ROWS),
        _build_summary(df),
        _build_column_profile(df),
        _format_insights(df),
        _default_plot(df),
    )


def _find_column(query: str, columns: list[str]) -> str | None:
    normalized = query.casefold()
    for column in columns:
        if column.casefold() in normalized:
            return column
    return None


def run_openui_command(file_path: str | None, command: str) -> tuple[str, Any]:
    command = command.strip()
    if not command:
        return "Enter a command such as `/plot histogram of population`.", None

    try:
        df = _read_csv(file_path)
    except Exception as exc:
        return str(exc), None

    numeric_columns = list(df.select_dtypes("number").columns)
    column = _find_column(command, list(df.columns))

    if command.startswith("/summary"):
        return _format_insights(df), None

    if command.startswith("/columns"):
        columns = ", ".join(df.columns)
        return f"Available columns: {columns}", None

    if command.startswith("/plot"):
        if not numeric_columns:
            return "This dataset has no numeric columns to plot.", None

        selected = column if column in numeric_columns else numeric_columns[0]
        title_column = selected

        if "hist" in command.lower():
            fig, ax = _new_figure()
            df[selected].dropna().plot(kind="hist", bins=20, ax=ax)
            ax.set_title(f"Histogram of {title_column}")
            ax.set_xlabel(title_column)
            fig.tight_layout()
            return (
                f"Rendered histogram for `{title_column}`.",
                fig,
            )

        match = re.search(r"against\s+([\w\s-]+)", command, re.IGNORECASE)
        second_column = _find_column(match.group(1), numeric_columns) if match else None
        if second_column and second_column != selected:
            fig, ax = _new_figure()
            df[[selected, second_column]].dropna().head(500).plot(
                kind="scatter",
                x=selected,
                y=second_column,
                ax=ax,
            )
            ax.set_title(f"{selected} vs {second_column}")
            fig.tight_layout()
            return (
                f"Rendered scatter plot for `{selected}` against `{second_column}`.",
                fig,
            )

        fig, ax = _new_figure()
        df[selected].dropna().head(100).plot(kind="line", ax=ax)
        ax.set_title(f"{title_column} over rows")
        ax.set_xlabel("Row")
        ax.set_ylabel(title_column)
        fig.tight_layout()
        return (
            f"Rendered line-style plot for `{title_column}`.",
            fig,
        )

    return (
        "Supported commands: `/summary`, `/columns`, `/plot <column>`, "
        "`/plot histogram of <column>`, `/plot <column> against <other column>`.",
        None,
    )


with gr.Blocks(title="smolnalysis", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # smolnalysis

        Upload a CSV file, inspect its structure, and try simple OpenUI-style commands.
        """
    )

    with gr.Row():
        csv_file = gr.File(label="CSV upload", file_types=[".csv"], type="filepath")
        analyze_button = gr.Button("Analyze CSV", variant="primary")

    with gr.Tabs():
        with gr.Tab("Overview"):
            preview = gr.Dataframe(label="Preview", interactive=False, wrap=True)
            summary = gr.Dataframe(label="Dataset summary", interactive=False)
            insights = gr.Markdown(label="Baseline insights")

        with gr.Tab("Columns"):
            profile = gr.Dataframe(label="Column profile", interactive=False)

        with gr.Tab("Chart"):
            default_chart = gr.Plot(label="Auto chart")

        with gr.Tab("OpenUI"):
            command = gr.Textbox(
                label="Command",
                placeholder="/plot histogram of population",
                value="/summary",
            )
            command_button = gr.Button("Run command")
            command_output = gr.Markdown()
            command_chart = gr.Plot(label="Command chart")

    analyze_button.click(
        analyze_csv,
        inputs=csv_file,
        outputs=[preview, summary, profile, insights, default_chart],
    )
    csv_file.change(
        analyze_csv,
        inputs=csv_file,
        outputs=[preview, summary, profile, insights, default_chart],
    )
    command_button.click(
        run_openui_command,
        inputs=[csv_file, command],
        outputs=[command_output, command_chart],
    )


if __name__ == "__main__":
    demo.launch()
