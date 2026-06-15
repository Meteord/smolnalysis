from __future__ import annotations

import html
import json
import logging
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

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
except ImportError:
    from backend.smolnalysis_model_wrapper import SmolnalysisMoE

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openui_adapter_demo import clean_component_output, render_component_preview  # type: ignore  # noqa: E402


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
EXAMPLE_PROMPTS = [
    ["Zeige mir Heizbedarf pro Monat für 2023 in Bürgerbüro Pasing."],
    ["Prüfe den Grenzwert für Stromverbrauch in Bogenhausen."],
    ["Say 'Hello World!' in Python"],
]

OPENUI_PREVIEW_CSS = """
.preview { border: 1px solid #d7dde8; border-radius: 8px; padding: 14px; background: #fff; color: #111827; }
.preview, .preview * { color: #111827; }
.preview h2 { margin: 0 0 10px; font-size: 18px; color: #0f172a; }
.preview p { color: #334155; }
.insight, .chart-preview, .table-preview { background: #fff; color: #111827; }
.insight { border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; margin-bottom: 10px; }
.insight p, .chart-preview p { color: #334155; margin: 0 0 10px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; }
.stat { border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; display: grid; gap: 3px; background: #f8fafc; }
.stat span, .stat small { color: #475569; font-size: 12px; }
.stat strong { color: #0f172a; font-size: 20px; }
.bar-row { display: grid; grid-template-columns: minmax(72px, 150px) 1fr minmax(72px, 120px); gap: 8px; align-items: center; margin: 8px 0; }
.bar-row span, .bar-row b { color: #1f2937; font-size: 12px; overflow-wrap: anywhere; }
.bar-row div, .progress { height: 14px; background: #e5e7eb; border-radius: 999px; overflow: hidden; }
.bar-row i, .progress i { display: block; height: 100%; background: #2563eb; border-radius: 999px; }
.alert.warning { border-color: #f59e0b; background: #fffbeb; }
.alert.danger { border-color: #ef4444; background: #fef2f2; }
.alert.success { border-color: #10b981; background: #ecfdf5; }
.preview table { width: 100%; border-collapse: collapse; }
.preview th { color: #111827; border-bottom: 1px solid #cbd5e1; padding: 7px 6px; font-size: 13px; text-align: left; }
.preview td { color: #1f2937; border-top: 1px solid #e5e7eb; padding: 7px 6px; font-size: 13px; }
.preview td:nth-child(2), .preview td:nth-child(3) { text-align: right; }
.histogram { display: flex; align-items: end; gap: 4px; height: 160px; padding-top: 8px; }
.histogram-bar { flex: 1; min-width: 8px; background: #2563eb; border-radius: 4px 4px 0 0; }
.error { border-color: #ef4444; background: #fef2f2; }
.error pre { white-space: pre-wrap; }
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
""" + OPENUI_PREVIEW_CSS + """
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
    content = str(result["content"])
    if any(stage.get("adapter") == "openui_translator" for stage in result.get("stages", [])):
        content = clean_component_output(content)
    trace = {
        "stages": result["stages"],
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
    }
    if result.get("tool_result"):
        trace["tool_result"] = result["tool_result"]
    return content, trace


def _is_openui_lang(content: str) -> bool:
    return any(line.strip().startswith("root =") for line in content.splitlines())


def _openui_mount_html(openui_lang: str) -> str:
    rendered = render_component_preview(openui_lang)
    source = html.escape(openui_lang)
    return (
        f'<div class="smol-render-card">{rendered}</div>'
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


class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = Field(default_factory=list)


def _server_page() -> str:
    examples = [
        {"label": "Monthly chart", "prompt": EXAMPLE_PROMPTS[0][0]},
        {"label": "Energy", "prompt": EXAMPLE_PROMPTS[1][0]},
        {"label": "District values", "prompt": EXAMPLE_PROMPTS[2][0]},
        #{"label": "Threshold", "prompt": EXAMPLE_PROMPTS[3][0]},
        #{"label": "Table", "prompt": EXAMPLE_PROMPTS[4][0]},
        #{"label": "General", "prompt": EXAMPLE_PROMPTS[5][0]},
    ]
    return (
        "<!doctype html>"
        '<html lang="en">'
        "<head>"
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>smolnalysis</title>"
        + _css()
        + """
<style>
:root {
  --ink: #172033;
  --muted: #647084;
  --line: #d7e0eb;
  --surface: #ffffff;
  --soft: #f6f8fb;
  --primary: #116d7b;
  --primary-strong: #0b5260;
  --primary-soft: #e7f5f4;
  --indigo: #4457c7;
  --green: #16866f;
  --rose: #c9435d;
  --shadow: 0 18px 44px rgba(23,32,51,0.11);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at 12% 0%, rgba(17,109,123,0.18), transparent 340px),
    radial-gradient(circle at 88% 8%, rgba(68,87,199,0.13), transparent 300px),
    linear-gradient(180deg, #f8fbfc 0%, #eef4f7 100%);
  color: var(--ink);
}
.gradio-container { max-width: 1180px; margin: 0 auto; padding: 20px; }
.topbar { display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; margin-bottom: 14px; }
.smol-shell { padding: 0; }
.smol-title { letter-spacing: 0; }
.status-card {
  display: flex;
  align-items: center;
  gap: 10px;
  border: 1px solid var(--line);
  background: rgba(255,255,255,0.82);
  border-radius: 8px;
  padding: 9px 12px;
  color: var(--muted);
  font-size: 13px;
  box-shadow: 0 8px 22px rgba(23,32,51,0.06);
}
.status-dot { width: 9px; height: 9px; border-radius: 999px; background: var(--green); box-shadow: 0 0 0 4px rgba(22,134,111,0.14); }
.workspace { display: grid; grid-template-columns: minmax(0, 1fr) 280px; gap: 14px; align-items: start; }
.chat-panel { border: 1px solid var(--line); background: rgba(255,255,255,0.9); border-radius: 8px; overflow: hidden; box-shadow: var(--shadow); }
.chat {
  min-height: 590px;
  max-height: calc(100vh - 230px);
  background: linear-gradient(180deg, #ffffff, #f7fafb);
  padding: 16px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.empty {
  margin: auto;
  max-width: 560px;
  text-align: center;
  color: var(--muted);
}
.empty strong { display: block; color: var(--ink); font-size: 22px; margin-bottom: 8px; }
.message-wrap { display: flex; flex-direction: column; gap: 4px; max-width: min(880px, 94%); }
.message-wrap.user { align-self: flex-end; align-items: flex-end; }
.message-wrap.assistant { align-self: flex-start; align-items: flex-start; }
.message-label { color: var(--muted); font-size: 11px; padding: 0 4px; }
.message {
  border-radius: 8px;
  padding: 11px 13px;
  line-height: 1.48;
  overflow-wrap: anywhere;
}
.message.user { background: linear-gradient(135deg, var(--primary), var(--indigo)); color: #fff; box-shadow: 0 12px 24px rgba(17,109,123,0.22); }
.message.assistant { background: #eef5f6; color: var(--ink); border: 1px solid #deeaee; }
.message.pending { min-width: 86px; color: var(--muted); }
.dots { display: inline-flex; gap: 4px; align-items: center; }
.dots i { width: 6px; height: 6px; border-radius: 999px; background: var(--muted); animation: pulse 1s infinite ease-in-out; }
.dots i:nth-child(2) { animation-delay: .15s; }
.dots i:nth-child(3) { animation-delay: .3s; }
@keyframes pulse { 0%, 80%, 100% { opacity: .25; transform: translateY(0); } 40% { opacity: 1; transform: translateY(-3px); } }
.composer {
  display: grid;
  grid-template-columns: 1fr auto auto;
  gap: 8px;
  padding: 10px;
  border-top: 1px solid var(--line);
  background: #fff;
}
.composer textarea {
  min-height: 48px;
  max-height: 150px;
  resize: vertical;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  padding: 12px;
  font: inherit;
  line-height: 1.35;
}
.composer textarea:focus { outline: 3px solid rgba(17,109,123,0.16); border-color: var(--primary); }
.composer button, .examples button {
  border: 1px solid #c9d5df;
  border-radius: 8px;
  background: linear-gradient(180deg, #ffffff, #f8fbfc);
  color: var(--ink);
  cursor: pointer;
  box-shadow: 0 1px 2px rgba(23,32,51,0.06);
  transition: transform .14s ease, border-color .14s ease, background .14s ease, box-shadow .14s ease;
}
.composer button { padding: 0 17px; font-weight: 700; }
.composer button:hover, .examples button:hover { transform: translateY(-1px); border-color: rgba(17,109,123,0.45); box-shadow: 0 8px 20px rgba(23,32,51,0.10); }
.composer button.primary {
  background: linear-gradient(135deg, var(--primary), var(--primary-strong));
  border-color: transparent;
  color: #fff;
  box-shadow: 0 10px 22px rgba(17,109,123,0.24);
}
.composer button.primary:hover { box-shadow: 0 12px 26px rgba(17,109,123,0.30); }
.composer button:disabled { cursor: wait; opacity: .6; transform: none; }
.side-panel { display: grid; gap: 12px; }
.panel { border: 1px solid var(--line); background: rgba(255,255,255,0.88); border-radius: 8px; padding: 12px; box-shadow: 0 12px 28px rgba(23,32,51,0.07); }
.panel h2 { margin: 0 0 8px; font-size: 14px; }
.panel p { margin: 0 0 10px; color: var(--muted); font-size: 13px; line-height: 1.4; }
.examples { display: grid; gap: 8px; }
.examples button { width: 100%; padding: 10px 11px; text-align: left; position: relative; overflow: hidden; }
.examples button::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 3px; background: linear-gradient(180deg, var(--primary), var(--indigo)); opacity: .9; }
.examples small { display: block; color: var(--primary); font-size: 11px; margin-bottom: 3px; font-weight: 750; }
.status { min-height: 20px; color: var(--muted); font-size: 12px; }
.smol-meta { overflow-wrap: anywhere; }
@media (max-width: 860px) {
  .gradio-container { padding: 12px; }
  .topbar { align-items: flex-start; flex-direction: column; }
  .workspace { grid-template-columns: 1fr; }
  .chat { min-height: 480px; max-height: none; }
  .composer { grid-template-columns: 1fr; }
  .composer button { min-height: 42px; }
}
</style>
        """
        "</head>"
        "<body>"
        '<main class="gradio-container">'
        '<header class="topbar">'
        '<div class="smol-shell">'
        '<h1 class="smol-title">smolnalysis</h1>'
        '<p class="smol-subtitle">Ask for open data, get a rendered interface back.</p>'
        "</div>"
        '<div class="status-card"><span class="status-dot"></span><span>Router + adapters online</span></div>'
        "</header>"
        '<div class="workspace">'
        '<section class="chat-panel">'
        '<section id="chat" class="chat" aria-live="polite">'
        '<div id="empty" class="empty"><strong>Start with an example prompt</strong><span>Results that produce OpenUI-Lang render directly in the chat.</span></div>'
        "</section>"
        '<form id="composer" class="composer">'
        '<textarea id="prompt" autocomplete="off" autofocus placeholder="Ask smolnalysis..."></textarea>'
        '<button id="send" class="primary" type="submit">Send</button>'
        '<button id="clear" type="button">Clear</button>'
        "</form>"
        "</section>"
        '<aside class="side-panel">'
        '<section class="panel">'
        '<h2>Examples</h2>'
        '<p>Try one of these examples and see what happens.</p>'
        '<div id="examples" class="examples"></div>'
        "</section>"
        '<section class="panel">'
        '<h2>Run State</h2>'
        '<p id="status" class="status">Idle</p>'
        "</section>"
        "</aside>"
        "</div>"
        "</main>"
        "<script>"
        f"const EXAMPLES = {json.dumps(examples, ensure_ascii=False)};"
        + r"""
const chat = document.getElementById("chat");
const prompt = document.getElementById("prompt");
const statusEl = document.getElementById("status");
const examplesEl = document.getElementById("examples");
const sendButton = document.getElementById("send");
let history = [];

function addMessage(role, content, htmlContent = false) {
  const emptyEl = document.getElementById("empty");
  if (emptyEl) emptyEl.remove();
  const wrap = document.createElement("div");
  wrap.className = `message-wrap ${role}`;
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = role === "user" ? "You" : "smolnalysis";
  const node = document.createElement("div");
  node.className = `message ${role}`;
  if (htmlContent) {
    node.innerHTML = content;
  } else {
    node.textContent = content;
  }
  wrap.appendChild(label);
  wrap.appendChild(node);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
  return wrap;
}

function addPendingMessage() {
  return addMessage("assistant", '<span class="dots"><i></i><i></i><i></i></span>', true);
}

function setBusy(isBusy) {
  prompt.disabled = isBusy;
  sendButton.disabled = isBusy;
}

async function sendMessage(message) {
  const text = (message || "").trim();
  if (!text) return;
  addMessage("user", text);
  prompt.value = "";
  statusEl.textContent = "Routing request and generating response...";
  setBusy(true);
  const pending = addPendingMessage();
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message: text, history})
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || payload.error || response.statusText);
    history = payload.history || history;
    pending.remove();
    addMessage("assistant", payload.html, true);
    const stages = (payload.trace && payload.trace.stages || []).map((stage) => stage.adapter || "base").join(" -> ");
    statusEl.textContent = stages ? `Last run: ${stages}` : "Idle";
  } catch (error) {
    pending.remove();
    addMessage("assistant", `${error.name}: ${error.message}`);
    statusEl.textContent = "Error";
  } finally {
    setBusy(false);
    prompt.focus();
  }
}

document.getElementById("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(prompt.value);
});
prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage(prompt.value);
  }
});
document.getElementById("clear").addEventListener("click", () => {
  history = [];
  chat.innerHTML = '<div id="empty" class="empty"><strong>Start with a training-style prompt</strong><span>Try a chart, table, threshold check, or a general question. Results that produce OpenUI-Lang render directly in the chat.</span></div>';
  statusEl.textContent = "Idle";
  prompt.value = "";
  prompt.focus();
});
for (const example of EXAMPLES) {
  const button = document.createElement("button");
  button.type = "button";
  button.innerHTML = `<small>${example.label}</small>${example.prompt}`;
  button.addEventListener("click", () => {
    prompt.value = example.prompt;
    prompt.focus();
  });
  examplesEl.appendChild(button);
}
        """
        "</script>"
        "</body></html>"
    )


app = gr.Server(title="smolnalysis")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(_server_page())


@app.post("/api/chat")
@spaces.GPU(duration=120)
def chat_api(request: ChatRequest) -> dict[str, Any]:
    user_message = str(request.message or "").strip()
    if not user_message:
        return {"html": "", "history": request.history, "trace": {}}
    messages = [*request.history, {"role": "user", "content": user_message}]
    try:
        content, trace = _run_model_chat(messages)
        messages.append({"role": "assistant", "content": content})
        return {
            "content": content,
            "html": _assistant_message(content, trace),
            "history": messages,
            "trace": trace,
        }
    except Exception as exc:
        logger.exception("SmolnalysisMoE chat failed")
        detail = f"{type(exc).__name__}: {str(exc).strip() or type(exc).__name__}"
        messages.append({"role": "assistant", "content": detail})
        return {
            "content": detail,
            "html": html.escape(detail),
            "history": messages,
            "trace": {"error": detail},
        }



if __name__ == "__main__":
    app.launch()
