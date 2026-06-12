from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from fastapi import Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

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

load_dotenv()
logging.basicConfig(
    level=os.getenv("SMOLNALYSIS_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

try:
    from .agent_workflow import run_agent_workflow
    from .ckan_support import DEFAULT_CKAN_ENDPOINT, default_ckan_status, validate_ckan_endpoint
    from .llm_support import llm_status, validate_llms
except ImportError:
    from agent_workflow import run_agent_workflow
    from ckan_support import DEFAULT_CKAN_ENDPOINT, default_ckan_status, validate_ckan_endpoint
    from llm_support import llm_status, validate_llms


APP_DIR = Path(__file__).parent
STATIC_DIR = APP_DIR / "static"
DEMO_CSV = APP_DIR / "examples" / "demo_cities.csv"
logger = logging.getLogger(__name__)


def _static_asset_version(filename: str) -> str:
    path = STATIC_DIR / filename
    if not path.exists():
        return "missing"
    return str(int(path.stat().st_mtime))


def _demo_dataframe() -> pd.DataFrame:
    if DEMO_CSV.exists():
        return pd.read_csv(DEMO_CSV)

    return pd.DataFrame(
        [
            {"city": "Berlin", "population": 3677000, "median_age": 42.6},
            {"city": "Hamburg", "population": 1906000, "median_age": 42.1},
            {"city": "Munich", "population": 1512000, "median_age": 41.5},
            {"city": "Cologne", "population": 1086000, "median_age": 42.3},
        ]
    )


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content)


def _last_user_prompt(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return _message_text(message).strip()
    return ""


def _chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    chat_messages = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        if role not in {"system", "user", "assistant"}:
            continue
        content = _message_text(message).strip()
        if content:
            chat_messages.append({"role": role, "content": content})
    return chat_messages


def build_openui_response(prompt: str) -> str:
    return build_workflow_response(prompt)


def build_workflow_response(prompt: str, ckan_endpoint: str | None = None) -> str:
    return run_agent_workflow(prompt or "Summarize this dataset", ckan_endpoint).get("openui_lang", "")


def build_model_openui_response(assistant_text: str, backend_label: str = "MiniCPM") -> str:
    return "\n".join(
        [
            "root = Card([header, response])",
            f"header = CardHeader({json.dumps(backend_label)}, \"Backend response\")",
            f"response = TextContent({json.dumps(assistant_text)}, \"default\")",
        ]
    )


build_gemma_openui_response = build_model_openui_response


def _looks_like_openui_lang(value: str) -> bool:
    return any(line.strip().startswith("root =") for line in value.splitlines())


def build_chat_openui_response(assistant_text: str) -> str:
    if _looks_like_openui_lang(assistant_text):
        return assistant_text
    return build_model_openui_response(assistant_text)


@spaces.GPU(duration=5)
def zerogpu_probe() -> dict[str, Any]:
    return {"ok": True, "runtime": "zerogpu-ready"}


def generate_chat_response(messages: list[dict[str, str]], *, adapter: str = "auto") -> str:
    try:
        try:
            from .backend.minicpm_llama_cpp import generate_chat_response as backend_generate_chat_response
        except ImportError:
            from backend.minicpm_llama_cpp import generate_chat_response as backend_generate_chat_response
    except Exception as exc:
        logger.warning("MiniCPM llama.cpp backend unavailable, using workflow fallback: %s", exc)
        prompt = next((message["content"] for message in reversed(messages) if message["role"] == "user"), "")
        return run_agent_workflow(prompt or "Summarize this dataset").get("openui_lang", "")

    try:
        return backend_generate_chat_response(messages, adapter=adapter)
    except Exception as exc:
        logger.warning("MiniCPM llama.cpp generation failed, using workflow fallback: %s", exc)
        prompt = next((message["content"] for message in reversed(messages) if message["role"] == "user"), "")
        return run_agent_workflow(prompt or "Summarize this dataset").get("openui_lang", "")


def _openai_sse_chunk(delta: dict[str, Any], finish_reason: str | None = None) -> str:
    payload = {"choices": [{"delta": delta, "finish_reason": finish_reason}]}
    return f"data: {json.dumps(payload)}\n\n"


async def _stream_openui_response(openui_lang: str):
    yield _openai_sse_chunk({"role": "assistant"})
    await asyncio.sleep(0)
    yield _openai_sse_chunk({"content": openui_lang})
    await asyncio.sleep(0)
    yield _openai_sse_chunk({}, "stop")
    yield "data: [DONE]\n\n"


app = gr.Server(title="smolnalysis")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.api(name="respond")
def respond(prompt: str) -> str:
    return build_openui_response(prompt)


@app.api(name="zerogpu_probe")
def api_zerogpu_probe() -> dict[str, Any]:
    return zerogpu_probe()


@app.post("/api/chat")
async def chat(request: Request) -> StreamingResponse:
    body = await request.json()
    messages = body.get("messages") or []
    adapter = str(body.get("adapter") or "auto").strip() or "auto"
    ckan = body.get("ckan") or {}
    chat_messages = _chat_messages(messages)
    logger.info(
        "chat request received: raw_messages=%d chat_messages=%d adapter=%s ckan_connected=%s ckan_base_url=%s",
        len(messages) if isinstance(messages, list) else 0,
        len(chat_messages),
        adapter,
        ckan.get("connected") if isinstance(ckan, dict) else None,
        ckan.get("base_url") if isinstance(ckan, dict) else None,
    )
    if chat_messages:
        logger.debug("chat last message: role=%s chars=%d", chat_messages[-1]["role"], len(chat_messages[-1]["content"]))
    assistant_text = await asyncio.to_thread(generate_chat_response, chat_messages, adapter=adapter)
    logger.info("chat response generated: adapter=%s response_chars=%d", adapter, len(assistant_text))
    openui_lang = build_chat_openui_response(assistant_text)
    return StreamingResponse(_stream_openui_response(openui_lang), media_type="text/event-stream")


@app.get("/api/ckan/default")
async def ckan_default() -> dict[str, Any]:
    status = default_ckan_status().to_dict()
    status["default_endpoint"] = DEFAULT_CKAN_ENDPOINT
    return status


@app.post("/api/ckan/connect")
async def ckan_connect(request: Request) -> dict[str, Any]:
    body = await request.json()
    return validate_ckan_endpoint(str(body.get("base_url", ""))).to_dict()


@app.get("/api/llms/status")
async def llms_status() -> dict[str, Any]:
    return llm_status()


@app.post("/api/llms/validate")
async def llms_validate() -> dict[str, Any]:
    return validate_llms()


@app.get("/api/minicpm/status")
async def minicpm_status() -> dict[str, Any]:
    try:
        try:
            from .backend.minicpm_llama_cpp import runtime_status
        except ImportError:
            from backend.minicpm_llama_cpp import runtime_status
        return runtime_status()
    except Exception as exc:
        return {"backend": "llama.cpp", "model_family": "MiniCPM", "error": str(exc)}


@app.get("/", response_class=HTMLResponse)
async def homepage() -> str:
    chat_css_version = _static_asset_version("openui-chat.css")
    chat_js_version = _static_asset_version("openui-chat.js")
    return f"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>smolnalysis</title>
    <link rel="stylesheet" href="/static/openui-chat.css?v={chat_css_version}" />
  </head>
  <body>
    <div id="root"></div>
    <script src="/static/openui-chat.js?v={chat_js_version}"></script>
  </body>
</html>
"""


if __name__ == "__main__":
    app.launch()
