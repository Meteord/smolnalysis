from __future__ import annotations

import html
import json
import logging
import os
import time
from functools import lru_cache
from typing import Any

import gradio as gr
from dotenv import load_dotenv

try:
    import spaces
except ImportError:
    class _SpacesFallback:
        @staticmethod
        def GPU(*args: Any, **kwargs: Any):
            def decorator(fn):
                return fn

            return decorator

    spaces = _SpacesFallback()

try:
    from .backend.smolnalysis_model_wrapper import SmolnalysisMoE
    from .openui_support import OpenUIValidationError, parse_openui_lang, render_openui_error, render_openui_value
except ImportError:
    from backend.smolnalysis_model_wrapper import SmolnalysisMoE
    from openui_support import OpenUIValidationError, parse_openui_lang, render_openui_error, render_openui_value


load_dotenv()
logging.basicConfig(
    level=os.getenv("SMOLNALYSIS_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MAX_NEW_TOKENS = int(os.getenv("SMOLNALYSIS_MINICPM_MAX_NEW_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.getenv("SMOLNALYSIS_MINICPM_TEMPERATURE", "0.7"))
DEFAULT_TOP_P = float(os.getenv("SMOLNALYSIS_MINICPM_TOP_P", "0.95"))
DEFAULT_TOP_K = int(os.getenv("SMOLNALYSIS_MINICPM_TOP_K", "64"))
LOAD_IN_4BIT = os.getenv("SMOLNALYSIS_MINICPM_LOAD_IN_4BIT", "true").casefold() not in {"0", "false", "no", "off"}


def _renderer_head() -> str:
    return """
<script>
(() => {
  const loadRenderer = () => {
    if (window.SmolnalysisOpenUIRenderer) {
      window.SmolnalysisOpenUIRenderer.mountAll();
      return;
    }
    if (document.querySelector("script[data-smolnalysis-openui-renderer]")) return;
    const script = document.createElement("script");
    script.src = "/gradio_api/file=app/static/openui-renderer.js";
    script.dataset.smolnalysisOpenuiRenderer = "true";
    document.head.appendChild(script);
  };
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", loadRenderer, { once: true });
  } else {
    loadRenderer();
  }
})();
</script>
"""


def _css() -> str:
    return """
<style>
.gradio-container {
  max-width: 1120px !important;
  margin: 0 auto;
}
.smol-shell {
  padding: 10px 0 4px;
}
.smol-title {
  margin: 0;
  font-size: 28px;
  line-height: 1.15;
  font-weight: 700;
}
.smol-subtitle {
  margin: 6px 0 0;
  color: #475569;
  font-size: 14px;
}
.smol-render-card {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #fff;
  padding: 12px;
  margin-top: 8px;
}
.smol-meta {
  margin-top: 8px;
  color: #64748b;
  font-size: 12px;
}
.openui-host [data-openui-mount]:empty::before {
  content: "Rendering OpenUI...";
  display: block;
  border: 1px dashed #cbd5e1;
  border-radius: 8px;
  padding: 12px;
  color: #64748b;
}
</style>
"""


@lru_cache(maxsize=1)
def _model() -> SmolnalysisMoE:
    logger.info("Loading SmolnalysisMoE wrapper")
    model = SmolnalysisMoE(load_in_4bit=LOAD_IN_4BIT)
    logger.info("SmolnalysisMoE wrapper loaded")
    return model


def _generation_kwargs() -> dict[str, Any]:
    return {
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "top_k": DEFAULT_TOP_K,
    }


def _run_model_chat(messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
    model = _model()
    started = time.perf_counter()
    result = model.generate_chat(messages, **_generation_kwargs())
    trace = {
        "stages": result["stages"],
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
    }
    if result.get("tool_result"):
        trace["tool_result"] = result["tool_result"]
    return str(result["content"]), trace


def _is_openui_lang(content: str) -> bool:
    return any(line.strip().startswith("root =") for line in content.splitlines())


def _openui_mount_html(openui_lang: str) -> str:
    try:
        parsed = parse_openui_lang(openui_lang)
        value = render_openui_value(parsed, openui_lang)
    except OpenUIValidationError as exc:
        value = render_openui_error(openui_lang, str(exc))
    encoded = html.escape(str(value.get("encoded") or ""))
    source = html.escape(openui_lang)
    return (
        '<div class="smol-render-card openui-host">'
        f'<div data-openui-mount data-openui-encoded="{encoded}"></div>'
        "</div>"
        '<details class="smol-openui-source">'
        "<summary>OpenUI Lang</summary>"
        f"<pre>{source}</pre>"
        "</details>"
    )


def _assistant_message(content: str, trace: dict[str, Any]) -> str:
    if _is_openui_lang(content):
        rendered = _openui_mount_html(content)
    else:
        rendered = html.escape(content).replace("\n", "<br>")
    meta = html.escape(json.dumps(trace, ensure_ascii=False, default=str))
    return f"{rendered}<div class=\"smol-meta\">{meta}</div>"


@spaces.GPU(duration=120)
def submit_message(
    user_message: str,
    chat_history: list[dict[str, Any]] | None,
    model_history: list[dict[str, str]] | None,
) -> tuple[str, list[dict[str, Any]], list[dict[str, str]]]:
    user_message = str(user_message or "").strip()
    if not user_message:
        return "", chat_history or [], model_history or []
    rendered_history = [*(chat_history or []), {"role": "user", "content": user_message}]
    messages = [*(model_history or []), {"role": "user", "content": user_message}]
    try:
        content, trace = _run_model_chat(messages)
        rendered_history.append({"role": "assistant", "content": _assistant_message(content, trace)})
        messages.append({"role": "assistant", "content": content})
    except Exception as exc:
        logger.exception("SmolnalysisMoE chat failed")
        detail = f"{type(exc).__name__}: {str(exc).strip() or type(exc).__name__}"
        rendered_history.append({"role": "assistant", "content": html.escape(detail)})
        messages.append({"role": "assistant", "content": detail})
    return "", rendered_history, messages


def clear_chat() -> tuple[list[dict[str, Any]], list[dict[str, str]], str]:
    return [], [], ""


with gr.Blocks(title="smolnalysis") as app:
    gr.HTML(
        _css()
        + _renderer_head()
        + """
        <div class="smol-shell">
          <h1 class="smol-title">smolnalysis</h1>
          <p class="smol-subtitle">MiniCPM SmolnalysisMoE chat with OpenUI rendering.</p>
        </div>
        """
    )
    chatbot = gr.Chatbot(
        height=620,
        show_label=False,
        sanitize_html=False,
        render_markdown=False,
        placeholder="Ask a general question or ask for open data that should render as OpenUI.",
    )
    model_messages = gr.State([])
    with gr.Row():
        prompt = gr.Textbox(
            placeholder="Ask smolnalysis...",
            show_label=False,
            scale=8,
            autofocus=True,
        )
        send = gr.Button("Send", variant="primary", scale=1)
        clear = gr.Button("Clear", scale=1)

    prompt.submit(submit_message, [prompt, chatbot, model_messages], [prompt, chatbot, model_messages])
    send.click(submit_message, [prompt, chatbot, model_messages], [prompt, chatbot, model_messages])
    clear.click(clear_chat, outputs=[chatbot, model_messages, prompt])


if __name__ == "__main__":
    app.launch()
