from __future__ import annotations

from functools import lru_cache
import logging
import os
import threading
import time
from typing import Any

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

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = os.getenv("SMOLNALYSIS_MINICPM_TRANSFORMERS_MODEL_ID", os.getenv("MODEL_ID", "openbmb/MiniCPM5-1B"))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("SMOLNALYSIS_MINICPM_MAX_NEW_TOKENS", os.getenv("MAX_TOKENS", "384")))
DEFAULT_TEMPERATURE = float(os.getenv("SMOLNALYSIS_MINICPM_TEMPERATURE", os.getenv("TEMPERATURE", "0.7")))
DEFAULT_TOP_P = float(os.getenv("SMOLNALYSIS_MINICPM_TOP_P", os.getenv("TOP_P", "0.95")))
ZERO_GPU_DURATION_SECONDS = int(os.getenv("SMOLNALYSIS_MINICPM_ZEROGPU_DURATION_SECONDS", "120"))
EAGER_LOAD = os.getenv("SMOLNALYSIS_MINICPM_TRANSFORMERS_EAGER_LOAD", os.getenv("SPACE_ID", "")).casefold() not in {
    "",
    "0",
    "false",
    "no",
    "off",
}

MODEL_LOCK = threading.Lock()
EAGER_LOAD_STATUS: dict[str, Any] = {"enabled": EAGER_LOAD, "loaded": False, "error": "", "duration_ms": 0}

ROLE_SYSTEM_PROMPTS = {
    "general_agent": "You are smolnalysis, a concise assistant for exploring open data and planning analysis steps.",
    "ckan_retrieval": "You are the smolnalysis CKAN retrieval specialist. Help identify datasets, resources, filters, and catalog search steps.",
    "data_analysis": "You are the smolnalysis data analyst. Focus on columns, quality checks, aggregations, distributions, trends, and clear next analyses.",
    "openui_translator": "You are the smolnalysis OpenUI translator. When asked for UI, return valid OpenUI-Lang only.",
}

ROLE_ALIASES = {
    "auto": "auto",
    "router": "auto",
    "base": "general_agent",
    "none": "general_agent",
    "general": "general_agent",
    "general_agent": "general_agent",
    "ckan": "ckan_retrieval",
    "ckan_tool": "ckan_retrieval",
    "retrieval": "ckan_retrieval",
    "ckan_retrieval": "ckan_retrieval",
    "data": "data_analysis",
    "analysis": "data_analysis",
    "data_analysis": "data_analysis",
    "openui": "openui_translator",
    "openui_translator": "openui_translator",
}

ROLE_ENV_KEYS = {
    "general_agent": "GENERAL_AGENT",
    "ckan_retrieval": "CKAN_RETRIEVAL",
    "data_analysis": "DATA_ANALYSIS",
    "openui_translator": "OPENUI_TRANSLATOR",
}


def _hub_url(repo_id: str) -> str:
    value = repo_id.strip().strip("/")
    if not value or "/" not in value:
        return ""
    return f"https://huggingface.co/{value}"


def normalize_role(adapter: str | None) -> str:
    value = (adapter or "auto").strip().casefold()
    return ROLE_ALIASES.get(value, value)


def route_role(messages: list[dict[str, str]], adapter: str | None = "auto") -> str:
    requested = normalize_role(adapter)
    if requested != "auto":
        return requested

    last_user_text = next(
        (message["content"] for message in reversed(messages) if message.get("role") == "user"),
        "",
    ).casefold()
    if any(term in last_user_text for term in ("openui", "component", "render", "ui", "card", "chart")):
        return "openui_translator"
    if any(term in last_user_text for term in ("analy", "quality", "distribution", "trend", "statistics", "missing")):
        return "data_analysis"
    if any(term in last_user_text for term in ("ckan", "dataset", "resource", "search", "retrieve", "catalog")):
        return "ckan_retrieval"
    return "general_agent"


def _with_role_system_prompt(messages: list[dict[str, str]], role: str) -> list[dict[str, str]]:
    if any(message.get("role") == "system" for message in messages):
        return messages
    prompt = ROLE_SYSTEM_PROMPTS.get(role)
    if not prompt:
        return messages
    return [{"role": "system", "content": prompt}, *messages]


@lru_cache(maxsize=1)
def _load_runtime():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    logger.info("loading MiniCPM transformers runtime: model=%s", DEFAULT_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    logger.info("MiniCPM transformers runtime loaded in %.1f ms", (time.perf_counter() - started) * 1000)
    return tokenizer, model


def _runtime_device(model: Any):
    return next(model.parameters()).device


def _generate(
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, dict[str, Any]]:
    import torch

    cache_before = _load_runtime.cache_info()
    tokenizer, model = _load_runtime()
    cache_after = _load_runtime.cache_info()
    device = _runtime_device(model)
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    input_tokens = int(inputs["input_ids"].shape[-1])
    generate_kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "top_p": top_p,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generate_kwargs["temperature"] = temperature
    with torch.inference_mode():
        outputs = model.generate(**generate_kwargs)
    output_tokens = int(outputs[0].shape[-1] - input_tokens)
    text = tokenizer.decode(outputs[0][input_tokens:], skip_special_tokens=True).strip()
    return text, {
        "cache": {
            "hit": cache_after.hits > cache_before.hits,
            "loaded_models": cache_after.currsize,
            "hits": cache_after.hits,
            "misses": cache_after.misses,
        },
        "device": str(device),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


@spaces.GPU(duration=ZERO_GPU_DURATION_SECONDS)
def generate_chat_response_with_trace(
    messages: list[dict[str, str]],
    *,
    adapter: str | None = "auto",
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int | None = None,
) -> tuple[str, dict[str, Any]]:
    started = time.perf_counter()
    role = route_role(messages, adapter)
    routed_messages = _with_role_system_prompt(messages, role)
    with MODEL_LOCK:
        content, runtime = _generate(
            routed_messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    trace = {
        "backend": "transformers",
        "model_family": "MiniCPM",
        "model": DEFAULT_MODEL_ID,
        "requested_adapter": adapter or "auto",
        "role": role,
        "message_count": len(messages),
        "routed_message_count": len(routed_messages),
        "sampling": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        },
        "runtime": runtime,
        "cache": runtime["cache"],
        "events": [
            {"name": "route_role", "detail": f"{adapter or 'auto'} -> {role}"},
            {"name": "load_model", "detail": "cache hit" if runtime["cache"]["hit"] else "cache miss"},
            {"name": "generate", "detail": f"{runtime['output_tokens']} tokens in {elapsed_ms} ms on {runtime['device']}"},
        ],
        "duration_ms": elapsed_ms,
        "output_chars": len(content),
    }
    return content, trace


def runtime_status() -> dict[str, Any]:
    cache = _load_runtime.cache_info()
    return {
        "backend": "transformers",
        "model_family": "MiniCPM",
        "model": DEFAULT_MODEL_ID,
        "model_hub_url": _hub_url(DEFAULT_MODEL_ID),
        "configured": bool(DEFAULT_MODEL_ID),
        "roles": list(ROLE_ENV_KEYS),
        "cache": {
            "loaded_models": cache.currsize,
            "hits": cache.hits,
            "misses": cache.misses,
        },
        "eager_load": EAGER_LOAD_STATUS,
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
    }


def _eager_load_runtime() -> None:
    if not EAGER_LOAD:
        return
    started = time.perf_counter()
    try:
        _load_runtime()
        EAGER_LOAD_STATUS["loaded"] = True
    except Exception as exc:
        logger.exception("MiniCPM transformers eager load failed.")
        EAGER_LOAD_STATUS["error"] = f"{type(exc).__name__}: {str(exc).strip() or type(exc).__name__}"
    EAGER_LOAD_STATUS["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)


_eager_load_runtime()
