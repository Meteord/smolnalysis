from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections import deque
from datetime import datetime, timezone
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
TRACE_LIMIT = int(os.getenv("SMOLNALYSIS_TRACE_LIMIT", "50"))
ENABLE_STUB_CHAT_FALLBACK = os.getenv("SMOLNALYSIS_ENABLE_STUB_CHAT_FALLBACK", "").casefold() in {"1", "true", "yes", "on"}
TRACE_STORE: deque[dict[str, Any]] = deque(maxlen=TRACE_LIMIT)
TRACE_LOCK = asyncio.Lock()


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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _remember_trace(trace: dict[str, Any]) -> dict[str, Any]:
    async with TRACE_LOCK:
        TRACE_STORE.appendleft(trace)
    return trace


async def _latest_traces(limit: int = 10) -> list[dict[str, Any]]:
    async with TRACE_LOCK:
        return list(TRACE_STORE)[: max(1, min(limit, TRACE_LIMIT))]


async def _trace_by_id(trace_id: str) -> dict[str, Any] | None:
    async with TRACE_LOCK:
        return next((trace for trace in TRACE_STORE if trace.get("request_id") == trace_id), None)


@spaces.GPU(duration=5)
def zerogpu_probe() -> dict[str, Any]:
    return {"ok": True, "runtime": "zerogpu-ready"}


def generate_chat_response(messages: list[dict[str, str]], *, adapter: str = "auto") -> str:
    response, _trace = generate_chat_response_with_trace(messages, adapter=adapter)
    return response


def generate_chat_response_with_trace(messages: list[dict[str, str]], *, adapter: str = "auto") -> tuple[str, dict[str, Any]]:
    started = time.perf_counter()
    try:
        try:
            from .backend.minicpm_llama_cpp import generate_chat_response_with_trace as backend_generate_chat_response_with_trace
        except ImportError:
            from backend.minicpm_llama_cpp import generate_chat_response_with_trace as backend_generate_chat_response_with_trace
    except Exception as exc:
        logger.exception("MiniCPM llama.cpp backend unavailable.")
        return _handle_backend_failure(messages, adapter, "backend_unavailable", exc, started)

    try:
        return backend_generate_chat_response_with_trace(messages, adapter=adapter)
    except Exception as exc:
        logger.exception("MiniCPM llama.cpp generation failed.")
        return _handle_backend_failure(messages, adapter, "generation_failed", exc, started)


def _handle_backend_failure(
    messages: list[dict[str, str]],
    adapter: str,
    reason: str,
    exc: Exception,
    started: float,
) -> tuple[str, dict[str, Any]]:
    if ENABLE_STUB_CHAT_FALLBACK:
        prompt = next((message["content"] for message in reversed(messages) if message["role"] == "user"), "")
        workflow = run_agent_workflow(prompt or "Summarize this dataset")
        return workflow.get("openui_lang", ""), _fallback_trace(
            messages,
            adapter,
            reason,
            _exception_detail(exc),
            workflow,
            started,
        )
    detail = _exception_detail(exc)
    return _backend_error_openui(reason, detail), _backend_error_trace(messages, adapter, reason, detail, started)


def _exception_detail(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def _backend_error_openui(reason: str, detail: str) -> str:
    return "\n".join(
        [
            "root = Card([header, callout, details])",
            'header = CardHeader("MiniCPM backend unavailable", "Generation did not complete")',
            f'callout = Callout("error", "Backend error", {json.dumps(reason)})',
            f"details = TextContent({json.dumps(detail)}, \"small\")",
        ]
    )


def _backend_error_trace(
    messages: list[dict[str, str]],
    adapter: str,
    reason: str,
    detail: str,
    started: float,
) -> dict[str, Any]:
    openui_lang = _backend_error_openui(reason, detail)
    return {
        "backend": "llama.cpp",
        "model_family": "MiniCPM",
        "requested_adapter": adapter,
        "role": "backend_error",
        "message_count": len(messages),
        "runtime": _minicpm_runtime_status_snapshot(),
        "events": [{"name": "backend_error", "detail": f"{reason}: {detail}"}],
        "fallback_reason": reason,
        "fallback_detail": detail,
        "stub_fallback_enabled": False,
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        "output_chars": len(openui_lang),
    }


def _minicpm_runtime_status_snapshot() -> dict[str, Any]:
    try:
        try:
            from .backend.minicpm_llama_cpp import runtime_status
        except ImportError:
            from backend.minicpm_llama_cpp import runtime_status
        return runtime_status()
    except Exception as exc:
        return {"error": _exception_detail(exc)}


def _fallback_trace(
    messages: list[dict[str, str]],
    adapter: str,
    reason: str,
    detail: str,
    workflow: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    steps = workflow.get("steps", [])
    events = [
        {"name": "fallback", "detail": f"{reason}: {detail}"},
        *[
            {"name": str(step.get("node", "workflow_step")), "detail": f"{step.get('title', '')}: {step.get('detail', '')}".strip()}
            for step in steps
            if isinstance(step, dict)
        ],
    ]
    return {
        "backend": "langgraph_fallback",
        "model_family": "stub",
        "requested_adapter": adapter,
        "role": "fallback",
        "message_count": len(messages),
        "events": events,
        "fallback_reason": reason,
        "fallback_detail": detail,
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        "output_chars": len(str(workflow.get("openui_lang", ""))),
    }


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
    request_id = str(uuid.uuid4())
    request_started = _now_iso()
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
    assistant_text, trace = await asyncio.to_thread(generate_chat_response_with_trace, chat_messages, adapter=adapter)
    trace = {
        "request_id": request_id,
        "thread_id": body.get("threadId"),
        "created_at": request_started,
        "completed_at": _now_iso(),
        "ckan": ckan if isinstance(ckan, dict) else {},
        **trace,
    }
    await _remember_trace(trace)
    logger.info("chat response generated: adapter=%s response_chars=%d", adapter, len(assistant_text))
    openui_lang = build_chat_openui_response(assistant_text)
    return StreamingResponse(
        _stream_openui_response(openui_lang),
        media_type="text/event-stream",
        headers={"x-smolnalysis-trace-id": request_id},
    )


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


@app.get("/api/traces/latest")
async def traces_latest(limit: int = 10) -> dict[str, Any]:
    traces = await _latest_traces(limit)
    return {"traces": traces}


@app.get("/api/traces/{trace_id}")
async def traces_get(trace_id: str) -> dict[str, Any]:
    trace = await _trace_by_id(trace_id)
    if trace is None:
        return {"error": "trace_not_found", "request_id": trace_id}
    return trace


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
