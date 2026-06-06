from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from openui_support import (
    app_styles,
    generate_openui_response,
    openui_component,
    parse_openui_lang,
    render_openui_error,
    render_openui_value,
)


MAX_UPLOAD_MB = 25
STATIC_DIR = Path(__file__).parent / "static"


gr.set_static_paths(paths=[STATIC_DIR])


def _read_csv(file_path: str | None) -> pd.DataFrame:
    if not file_path:
        raise ValueError("Upload a CSV file to begin.")

    path = Path(file_path)
    if path.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"CSV file is larger than {MAX_UPLOAD_MB} MB.")

    return pd.read_csv(path)


def _dataset_status(file_path: str | None) -> str:
    try:
        df = _read_csv(file_path)
    except Exception as exc:
        return f"Dataset not loaded: {exc}"
    return f"Dataset loaded: {len(df):,} rows x {len(df.columns):,} columns."


def chat_with_openui(
    file_path: str | None,
    prompt: str,
    history: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], str]:
    history = history or []
    prompt = prompt.strip()
    if not prompt:
        return history, ""

    try:
        df = _read_csv(file_path)
    except Exception:
        df = None

    turn = generate_openui_response(df, prompt)
    try:
        parsed = parse_openui_lang(turn.openui_lang)
        render_value = render_openui_value(parsed, turn.openui_lang)
    except Exception as exc:
        render_value = render_openui_error(turn.openui_lang, str(exc))

    rendered_response = openui_component(value=render_value, label="OpenUI response")

    history = [
        *history,
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": rendered_response},
    ]
    return history, ""


with gr.Blocks(title="smolnalysis") as demo:
    gr.HTML(app_styles())
    gr.HTML(
        """
        <section class="app-shell">
          <div class="app-hero">
            <p class="app-kicker">OpenUI Data Chat</p>
            <h1>smolnalysis</h1>
            <p class="app-subtitle">Upload a CSV, ask about the data, and render mocked OpenUI-Lang answers with a native Gradio HTML component.</p>
          </div>
        </section>
        """
    )

    with gr.Group(elem_classes=["app-shell", "upload-shell"]):
        csv_file = gr.File(label="Dataset", file_types=[".csv"], type="filepath")
        dataset_status = gr.Markdown("Dataset not loaded.")

    with gr.Group(elem_classes=["app-shell", "chat-shell"]):
        openui_chatbot = gr.Chatbot(
            label=None,
            height=620,
            container=False,
            layout="bubble",
        )
        with gr.Row(elem_classes=["composer-row"]):
            openui_prompt = gr.Textbox(
                label="Message",
                show_label=False,
                placeholder="Ask for a summary, schema, bar chart, or histogram",
                scale=8,
                max_lines=4,
                autofocus=True,
            )
            openui_send = gr.Button("Send", variant="primary", scale=1, min_width=120)

        gr.Examples(
            examples=[
                ["Summarize this dataset"],
                ["Show a bar chart of population by city"],
                ["Show a histogram of median_age"],
                ["List the columns and missing values"],
                ["Return invalid OpenUI for fallback testing"],
            ],
            inputs=openui_prompt,
            )

    csv_file.change(_dataset_status, inputs=csv_file, outputs=dataset_status)
    openui_send.click(
        chat_with_openui,
        inputs=[csv_file, openui_prompt, openui_chatbot],
        outputs=[openui_chatbot, openui_prompt],
    )
    openui_prompt.submit(
        chat_with_openui,
        inputs=[csv_file, openui_prompt, openui_chatbot],
        outputs=[openui_chatbot, openui_prompt],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
