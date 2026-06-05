from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from openui_support import (
    ASSISTANT_FALLBACK,
    generate_openui_response,
    openui_styles,
    parse_openui_lang,
    render_openui_html,
)


MAX_UPLOAD_MB = 25


def _read_csv(file_path: str | None) -> pd.DataFrame:
    if not file_path:
        raise ValueError("Upload a CSV file to begin.")

    path = Path(file_path)
    if path.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"CSV file is larger than {MAX_UPLOAD_MB} MB.")

    return pd.read_csv(path)


def _render_debug_block(debug_steps: str, raw_openui: str) -> str:
    steps_html = "".join(
        f"<li>{html.escape(line[2:] if line.startswith('- ') else line)}</li>"
        for line in debug_steps.splitlines()
        if line.strip()
    )
    return (
        "<details class='openui-debug'>"
        "<summary>OpenUI debug</summary>"
        f"<ul>{steps_html}</ul>"
        f"<pre><code class='language-openui'>{html.escape(raw_openui)}</code></pre>"
        "</details>"
    )


def _compose_assistant_html(rendered_value: str, debug_steps: str, raw_openui: str) -> str:
    return f"{rendered_value}{_render_debug_block(debug_steps, raw_openui)}"


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
    raw_openui = turn.openui_lang
    debug_steps = "\n".join(f"- {step.role}: {step.content}" for step in turn.agent_steps)

    try:
        parsed = parse_openui_lang(raw_openui)
        rendered_value = render_openui_html(parsed)
    except Exception as exc:
        rendered_value = (
            "<section class='openui-notice openui-warning'>"
            f"{ASSISTANT_FALLBACK} {exc}"
            "</section>"
        )

    assistant_message = _compose_assistant_html(rendered_value, debug_steps, raw_openui)

    history = [
        *history,
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant_message},
    ]
    return history, ""


with gr.Blocks(title="smolnalysis") as demo:
    gr.HTML(openui_styles())
    gr.HTML(
        """
        <section class="app-shell">
          <div class="app-hero">
            <p class="app-kicker">OpenUI Data Chat</p>
            <h1>smolnalysis</h1>
            <p class="app-subtitle">Upload a dataset, ask a question, and inspect the rendered OpenUI response directly in chat.</p>
          </div>
        </section>
        """
    )

    with gr.Group(elem_classes=["app-shell", "upload-shell"]):
        csv_file = gr.File(label="Dataset", file_types=[".csv"], type="filepath")

    with gr.Group(elem_classes=["app-shell", "chat-shell"]):
        openui_chatbot = gr.Chatbot(
            label="Conversation",
            height=420,
            container=False,
        )
        with gr.Row(elem_classes=["composer-row"]):
            openui_prompt = gr.Textbox(
                label="",
                show_label=False,
                placeholder="Ask something like: show a bar chart of population by city",
                scale=8,
                max_lines=4,
                autofocus=True,
                container=False,
                elem_classes=["composer-input"],
            )
            openui_send = gr.Button("Send", variant="primary", scale=1, min_width=120)

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
