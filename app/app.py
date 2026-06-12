from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import traceback
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
    from .ckan_agent import AgentEvent, AgentResult, run_ckan_agent
    from .ckan_support import DEFAULT_CKAN_ENDPOINT, default_ckan_status, validate_ckan_endpoint
    from .backend import minicpm_transformers as _minicpm_startup
    from .llm_support import llm_status, validate_llms
    from .model_roles import call_role_model
except ImportError:
    from agent_workflow import run_agent_workflow
    from backend import minicpm_transformers as _minicpm_startup
    from ckan_agent import AgentEvent, AgentResult, run_ckan_agent
    from ckan_support import DEFAULT_CKAN_ENDPOINT, default_ckan_status, validate_ckan_endpoint
    from llm_support import llm_status, validate_llms
    from model_roles import call_role_model


APP_DIR = Path(__file__).parent
STATIC_DIR = APP_DIR / "static"
DEMO_CSV = APP_DIR / "examples" / "demo_cities.csv"
logger = logging.getLogger(__name__)
TRACE_LIMIT = int(os.getenv("SMOLNALYSIS_TRACE_LIMIT", "50"))
ENABLE_STUB_CHAT_FALLBACK = os.getenv("SMOLNALYSIS_ENABLE_STUB_CHAT_FALLBACK", "").casefold() in {"1", "true", "yes", "on"}
MINICPM_BACKEND = os.getenv("SMOLNALYSIS_MINICPM_BACKEND", "transformers").strip().casefold()
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


def _clean_user_prompt(text: str) -> str:
    value = str(text or "").strip()
    content_match = re.search(r"<content>(.*?)</content>", value, re.IGNORECASE | re.DOTALL)
    if content_match:
        value = content_match.group(1).strip()
    value = re.sub(r"<context>.*?</context>", " ", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _last_user_prompt(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return _clean_user_prompt(_message_text(message))
    return ""


def _chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    chat_messages = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        if role not in {"system", "user", "assistant"}:
            continue
        content = _message_text(message).strip()
        if role == "user":
            content = _clean_user_prompt(content)
        if content:
            chat_messages.append({"role": role, "content": content})
    return chat_messages


def build_openui_response(prompt: str) -> str:
    prompt = _clean_user_prompt(prompt)
    return build_workflow_response(prompt, dataset_path=_default_dataset_path_for_prompt(prompt))


def build_workflow_response(prompt: str, ckan_endpoint: str | None = None, dataset_path: str | None = None) -> str:
    return run_agent_workflow(prompt or "Summarize this dataset", ckan_endpoint, dataset_path).get("openui_lang", "")


def _default_dataset_path_for_prompt(prompt: str) -> str | None:
    prompt = _clean_user_prompt(prompt)
    lower = prompt.casefold()
    analysis_terms = ("this dataset", "summarize", "summary", "schema", "columns", "quality", "missing", "trend", "statistics", "chart", "histogram")
    if any(term in lower for term in analysis_terms):
        return str(DEMO_CSV) if DEMO_CSV.exists() else None
    retrieval_terms = ("ckan", "resource", "catalog", "search", "find", "retrieve", "open data", "bike", "bicycle", "bycycle", "bycycles", "fahrrad", "counter", "traffic")
    if any(term in lower for term in retrieval_terms):
        return None
    return str(DEMO_CSV) if DEMO_CSV.exists() else None


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
        backend_generate_chat_response_with_trace = _backend_generate_function()
    except Exception as exc:
        logger.exception("MiniCPM backend unavailable.")
        return _handle_backend_failure(messages, adapter, "backend_unavailable", exc, started)

    try:
        return backend_generate_chat_response_with_trace(messages, adapter=adapter)
    except Exception as exc:
        logger.exception("MiniCPM generation failed.")
        return _handle_backend_failure(messages, adapter, "generation_failed", exc, started)


def _backend_generate_function():
    if MINICPM_BACKEND in {"llama.cpp", "llamacpp", "llama_cpp", "gguf"}:
        try:
            from .backend.minicpm_llama_cpp import generate_chat_response_with_trace as backend_generate_chat_response_with_trace
        except ImportError:
            from backend.minicpm_llama_cpp import generate_chat_response_with_trace as backend_generate_chat_response_with_trace
        return backend_generate_chat_response_with_trace
    try:
        from .backend.minicpm_transformers import generate_chat_response_with_trace as backend_generate_chat_response_with_trace
    except ImportError:
        from backend.minicpm_transformers import generate_chat_response_with_trace as backend_generate_chat_response_with_trace
    return backend_generate_chat_response_with_trace


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
    return _backend_error_openui(reason, detail), _backend_error_trace(messages, adapter, reason, detail, exc, started)


def _exception_detail(exc: Exception) -> str:
    details = []
    current: BaseException | None = exc
    while current is not None:
        message = str(current).strip()
        details.append(f"{type(current).__name__}: {message}" if message else type(current).__name__)
        current = current.__cause__ or current.__context__
    return " <- ".join(details)


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
    exc: Exception,
    started: float,
) -> dict[str, Any]:
    openui_lang = _backend_error_openui(reason, detail)
    return {
        "backend": MINICPM_BACKEND,
        "model_family": "MiniCPM",
        "requested_adapter": adapter,
        "role": "backend_error",
        "message_count": len(messages),
        "runtime": _minicpm_runtime_status_snapshot(),
        "events": [{"name": "backend_error", "detail": f"{reason}: {detail}"}],
        "fallback_reason": reason,
        "fallback_detail": detail,
        "traceback": "".join(traceback.format_exception(exc)),
        "stub_fallback_enabled": False,
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        "output_chars": len(openui_lang),
    }


def _minicpm_runtime_status_snapshot() -> dict[str, Any]:
    try:
        return _selected_minicpm_runtime_status()
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
        "backend": "agent_fallback",
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


def _retrieval_stream_prefix(prompt: str, endpoint: str) -> str:
    progress_refs = ", ".join(f"progress{index}" for index in range(1, 13))
    return "\n".join(
        [
            "root = Card([header, suggestion, progress, candidates, callout, followups])",
            'header = CardHeader("Finding dataset", "Searching CKAN and inspecting candidates")',
            f'context = TextContent({_json_arg_safe(f"Request: {prompt} | Endpoint: {endpoint}")}, "small")',
            f'progress = ListBlock([{progress_refs}], "number")',
        ]
    )


def _agent_stream_suffix(result: AgentResult, progress_count: int) -> str:
    rows = [
        {
            "title": str(package.get("title") or package.get("name") or ""),
            "name": str(package.get("name") or package.get("id") or ""),
            "resources": len(package.get("resources") or []),
        }
        for package in result.packages[:8]
    ]
    if not rows:
        rows = [{"title": "No candidates", "name": result.prompt, "resources": 0}]
    lines = []
    for step in range(progress_count + 1, 13):
        lines.append(f'progress{step} = ListItem("waiting", "")')
    columns = list(rows[0].keys())
    for index, column in enumerate(columns):
        lines.append(f'candidate_col{index + 1} = Col({_json_arg_safe(column)}, {_json_arg_safe([row.get(column) for row in rows])}, "string")')
    lines.append(f'candidates = Table([{", ".join(f"candidate_col{index + 1}" for index in range(len(columns))) }])')
    if result.selected_resource:
        message = f"Selected {result.selected_resource.name} from {result.selected_resource.package_title}."
    elif result.clarification:
        message = result.clarification
    else:
        message = f"Stopped with status {result.status}."
    lines.append(f'callout = Callout("info", "Retrieval result", {_json_arg_safe(message)})')
    lines.extend(
        [
            'followup1 = FollowUpItem("Try a narrower CKAN search")',
            'followup2 = FollowUpItem("Search for CSV resources only")',
            'followup3 = FollowUpItem("Inspect the selected dataset")',
            "followups = FollowUpBlock([followup1, followup2, followup3])",
        ]
    )
    return "\n".join(lines)


def _json_arg_safe(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _is_retrieval_prompt(prompt: str, has_dataset: bool) -> bool:
    if has_dataset:
        return False
    lower = _clean_user_prompt(prompt).casefold()
    return any(term in lower for term in ("ckan", "dataset", "resource", "catalog", "search", "find", "retrieve", "open data", "bike", "bicycle", "bycycle", "bycycles", "fahrrad", "counter", "traffic"))


async def _stream_retrieval_workflow_response(
    *,
    prompt: str,
    endpoint: str,
    request_id: str,
    request_started: str,
    thread_id: Any,
    ckan: dict[str, Any],
    chat_messages: list[dict[str, str]],
    adapter: str,
):
    started = time.perf_counter()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_event(event: AgentEvent) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "event", "event": event})

    chunks: list[str] = []
    trace_events: list[dict[str, str]] = [{"name": "route_intent", "detail": f"dataset_retrieval: streaming CKAN agent loop for {endpoint}"}]
    yield _openai_sse_chunk({"role": "assistant"})
    prefix = _retrieval_stream_prefix(prompt, endpoint)
    chunks.append(prefix)
    yield _openai_sse_chunk({"content": f"{prefix}\n"})
    suggestion_line = 'suggestion = Callout("info", "CKAN specialist", "The CKAN retrieval model chooses each search or inspection action; Python validates and runs the tools.")'
    chunks.append(suggestion_line)
    yield _openai_sse_chunk({"content": f"{suggestion_line}\n"})

    def run_loop() -> None:
        try:
            result = run_ckan_agent(prompt, endpoint, history=chat_messages, on_event=on_event, model_caller=call_role_model)
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "done", "result": result})
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "error": _exception_detail(exc)})

    task = asyncio.create_task(asyncio.to_thread(run_loop))

    final_result: AgentResult | None = None
    progress_count = 0
    while True:
        item = await queue.get()
        if item["type"] == "event":
            event: AgentEvent = item["event"]
            progress_count += 1
            trace_events.append({"name": event.type, "detail": event.detail})
            if progress_count <= 12:
                line = f"progress{progress_count} = ListItem({_json_arg_safe(event.type)}, {_json_arg_safe(event.detail)})"
                chunks.append(line)
                yield _openai_sse_chunk({"content": f"{line}\n"})
            await asyncio.sleep(0)
            continue
        if item["type"] == "done":
            final_result = item["result"]
            suffix = _agent_stream_suffix(final_result, progress_count)
            chunks.append(suffix)
            yield _openai_sse_chunk({"content": suffix})
            break
        error_suffix = _agent_stream_error_suffix(str(item.get("error", "unknown error")), progress_count)
        chunks.append(error_suffix)
        yield _openai_sse_chunk({"content": error_suffix})
        break

    await task
    openui_lang = "\n".join(chunks)
    trace = {
        "request_id": request_id,
        "thread_id": thread_id,
        "created_at": request_started,
        "completed_at": _now_iso(),
        "ckan": ckan,
        "backend": "simple_ckan_agent",
        "model_family": "python",
        "requested_adapter": adapter,
        "role": "dataset_retrieval",
        "message_count": len(chat_messages),
        "events": trace_events,
        "runtime": {"ckan_endpoint": endpoint, "streaming": True},
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        "output_chars": len(openui_lang),
    }
    if final_result is not None:
        trace["retrieval"] = {
            "status": final_result.status,
            "selected": final_result.selected_resource.name if final_result.selected_resource else "",
            "confidence": final_result.confidence,
        }
    await _remember_trace(trace)
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
    prompt = _last_user_prompt(messages if isinstance(messages, list) else []) or "Summarize this dataset"
    ckan_base_url = ckan.get("base_url") if isinstance(ckan, dict) else None
    dataset_path = _default_dataset_path_for_prompt(prompt)
    if _is_retrieval_prompt(prompt, has_dataset=bool(dataset_path)):
        endpoint = ckan_base_url or DEFAULT_CKAN_ENDPOINT
        return StreamingResponse(
            _stream_retrieval_workflow_response(
                prompt=prompt,
                endpoint=endpoint,
                request_id=request_id,
                request_started=request_started,
                thread_id=body.get("threadId"),
                ckan=ckan if isinstance(ckan, dict) else {},
                chat_messages=chat_messages,
                adapter=adapter,
            ),
            media_type="text/event-stream",
            headers={"x-smolnalysis-trace-id": request_id},
        )
    workflow = await asyncio.to_thread(run_agent_workflow, prompt, ckan_base_url, dataset_path)
    assistant_text = str(workflow.get("openui_lang", ""))
    trace = _workflow_trace(chat_messages, adapter, workflow)
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


def _workflow_trace(messages: list[dict[str, str]], adapter: str, workflow: dict[str, Any]) -> dict[str, Any]:
    steps = workflow.get("steps", [])
    return {
        "backend": "simple_agent_workflow",
        "model_family": "python",
        "requested_adapter": adapter,
        "role": workflow.get("intent", {}).get("task_type", "workflow") if isinstance(workflow.get("intent"), dict) else "workflow",
        "message_count": len(messages),
        "events": [
            {"name": str(step.get("node", "workflow_step")), "detail": f"{step.get('title', '')}: {step.get('detail', '')}".strip()}
            for step in steps
            if isinstance(step, dict)
        ],
        "runtime": {
            "ckan_endpoint": workflow.get("ckan_endpoint", DEFAULT_CKAN_ENDPOINT),
            "dataset_path": workflow.get("dataset_path", ""),
        },
        "duration_ms": 0,
        "output_chars": len(str(workflow.get("openui_lang", ""))),
    }


def _agent_stream_error_suffix(error: str, progress_count: int) -> str:
    lines = []
    for step in range(progress_count + 1, 13):
        lines.append(f'progress{step} = ListItem("waiting", "")')
    lines.extend(
        [
            f'candidate_col1 = Col("title", {_json_arg_safe(["No candidates"])}, "string")',
            f'candidate_col2 = Col("name", {_json_arg_safe(["retrieval failed"])}, "string")',
            f'candidate_col3 = Col("resources", {_json_arg_safe([0])}, "number")',
            "candidates = Table([candidate_col1, candidate_col2, candidate_col3])",
            f'callout = Callout("error", "Retrieval failed", {_json_arg_safe(error)})',
            'followup1 = FollowUpItem("Try a narrower CKAN search")',
            'followup2 = FollowUpItem("Check the CKAN endpoint")',
            'followup3 = FollowUpItem("Search for CSV resources only")',
            "followups = FollowUpBlock([followup1, followup2, followup3])",
        ]
    )
    return "\n".join(lines)


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
        return _selected_minicpm_runtime_status()
    except Exception as exc:
        return {"backend": MINICPM_BACKEND, "model_family": "MiniCPM", "error": _exception_detail(exc)}


def _selected_minicpm_runtime_status() -> dict[str, Any]:
    if MINICPM_BACKEND in {"llama.cpp", "llamacpp", "llama_cpp", "gguf"}:
        try:
            from .backend.minicpm_llama_cpp import runtime_status
        except ImportError:
            from backend.minicpm_llama_cpp import runtime_status
        return runtime_status()
    try:
        from .backend.minicpm_transformers import runtime_status
    except ImportError:
        from backend.minicpm_transformers import runtime_status
    return runtime_status()


@app.get("/api/minicpm/probe")
async def minicpm_probe(role: str = "general_agent") -> dict[str, Any]:
    try:
        try:
            from .backend.minicpm_llama_cpp import probe_runtime
        except ImportError:
            from backend.minicpm_llama_cpp import probe_runtime
        return await asyncio.to_thread(probe_runtime, role)
    except Exception as exc:
        return {"backend": "llama.cpp", "model_family": "MiniCPM", "error": _exception_detail(exc)}


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
