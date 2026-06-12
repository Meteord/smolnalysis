from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
import os
import threading
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

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

DEFAULT_MAX_NEW_TOKENS = int(os.getenv("SMOLNALYSIS_MINICPM_MAX_NEW_TOKENS", os.getenv("MAX_TOKENS", "850")))
DEFAULT_TEMPERATURE = float(os.getenv("SMOLNALYSIS_MINICPM_TEMPERATURE", os.getenv("TEMPERATURE", "0.7")))
DEFAULT_TOP_P = float(os.getenv("SMOLNALYSIS_MINICPM_TOP_P", os.getenv("TOP_P", "0.9")))
DEFAULT_N_CTX = int(os.getenv("SMOLNALYSIS_MINICPM_N_CTX", os.getenv("N_CTX", "4096")))
DEFAULT_N_BATCH = int(os.getenv("SMOLNALYSIS_MINICPM_N_BATCH", os.getenv("N_BATCH", "512")))
DEFAULT_N_GPU_LAYERS = int(os.getenv("SMOLNALYSIS_MINICPM_N_GPU_LAYERS", os.getenv("N_GPU_LAYERS", "0")))
ZERO_GPU_DURATION_SECONDS = int(os.getenv("SMOLNALYSIS_MINICPM_ZEROGPU_DURATION_SECONDS", "120"))

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


@dataclass(frozen=True)
class LlamaCppRoleConfig:
    role: str
    model_path: str
    model_repo_id: str
    model_filename: str
    lora_path: str


def _clean_env_value(name: str, default: str = "") -> str:
    raw = os.getenv(name, default)
    lines = []
    for line in str(raw).splitlines():
        value = line.strip().strip('"').strip("'")
        if value and not value.startswith("#"):
            lines.append(value)
    return lines[-1] if lines else default


def _role_env(role: str, suffix: str) -> str:
    return f"SMOLNALYSIS_MINICPM_{ROLE_ENV_KEYS[role]}_{suffix}"


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


def role_config(role: str) -> LlamaCppRoleConfig:
    if role not in ROLE_ENV_KEYS:
        available = ", ".join(ROLE_ENV_KEYS)
        raise KeyError(f"Unknown MiniCPM llama.cpp role '{role}'. Available roles: {available}")

    model_path = _clean_env_value(_role_env(role, "MODEL_PATH"), _clean_env_value("SMOLNALYSIS_MINICPM_MODEL_PATH", _clean_env_value("MODEL_PATH")))
    model_repo_id = _clean_env_value(
        _role_env(role, "MODEL_REPO_ID"),
        _clean_env_value("SMOLNALYSIS_MINICPM_MODEL_REPO_ID", _clean_env_value("MODEL_REPO_ID")),
    )
    model_filename = _clean_env_value(
        _role_env(role, "MODEL_FILENAME"),
        _clean_env_value("SMOLNALYSIS_MINICPM_MODEL_FILENAME", _clean_env_value("MODEL_FILENAME")),
    )
    lora_path = _clean_env_value(_role_env(role, "LORA_PATH"), "")
    return LlamaCppRoleConfig(role, model_path, model_repo_id, model_filename, lora_path)


def _resolve_model_path(config: LlamaCppRoleConfig) -> str:
    if config.model_path:
        path = Path(config.model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"MiniCPM GGUF model path does not exist: {path}")
        return str(path)
    if config.model_repo_id and config.model_filename:
        return hf_hub_download(repo_id=config.model_repo_id, filename=config.model_filename)
    raise RuntimeError(
        "MiniCPM llama.cpp model is not configured. Set MODEL_PATH or "
        "MODEL_REPO_ID and MODEL_FILENAME, or use the SMOLNALYSIS_MINICPM_* equivalents."
    )


@lru_cache(maxsize=4)
def _load_llama(role: str):
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError("llama-cpp-python is not installed in this runtime.") from exc

    config = role_config(role)
    model_path = _resolve_model_path(config)
    kwargs: dict[str, Any] = {
        "model_path": model_path,
        "n_ctx": int(_clean_env_value(_role_env(role, "N_CTX"), str(DEFAULT_N_CTX))),
        "n_batch": int(_clean_env_value(_role_env(role, "N_BATCH"), str(DEFAULT_N_BATCH))),
        "n_gpu_layers": int(_clean_env_value(_role_env(role, "N_GPU_LAYERS"), str(DEFAULT_N_GPU_LAYERS))),
        "verbose": _clean_env_value("SMOLNALYSIS_MINICPM_VERBOSE", "false").casefold() in {"1", "true", "yes", "on"},
    }
    n_threads = _clean_env_value(_role_env(role, "N_THREADS"), _clean_env_value("SMOLNALYSIS_MINICPM_N_THREADS", _clean_env_value("N_THREADS")))
    if n_threads:
        kwargs["n_threads"] = int(n_threads)
    if config.lora_path:
        lora_path = Path(config.lora_path).expanduser()
        if not lora_path.exists():
            raise FileNotFoundError(f"MiniCPM LoRA path does not exist for role {role}: {lora_path}")
        kwargs["lora_path"] = str(lora_path)

    logger.info("loading MiniCPM llama.cpp role=%s model=%s lora=%s", role, model_path, config.lora_path or "none")
    return Llama(**kwargs)


MODEL_LOCK = threading.Lock()


@spaces.GPU(duration=ZERO_GPU_DURATION_SECONDS)
def generate_chat_response(
    messages: list[dict[str, str]],
    *,
    adapter: str | None = "auto",
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int | None = None,
) -> str:
    role = route_role(messages, adapter)
    with MODEL_LOCK:
        llm = _load_llama(role)
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
            "stream": False,
        }
        if top_k is not None:
            payload["top_k"] = top_k
        response = llm.create_chat_completion(**payload)

    content = response["choices"][0]["message"]["content"]
    logger.info("MiniCPM llama.cpp response generated: role=%s chars=%d", role, len(content))
    return str(content).strip()


def runtime_status() -> dict[str, Any]:
    roles = {}
    for role in ROLE_ENV_KEYS:
        config = role_config(role)
        roles[role] = {
            "model_path": config.model_path,
            "model_repo_id": config.model_repo_id,
            "model_filename": config.model_filename,
            "lora_path": config.lora_path,
            "configured": bool(config.model_path or (config.model_repo_id and config.model_filename)),
            "loaded": _load_llama.cache_info().currsize > 0,
        }
    return {
        "backend": "llama.cpp",
        "model_family": "MiniCPM",
        "roles": roles,
        "n_ctx": DEFAULT_N_CTX,
        "n_gpu_layers": DEFAULT_N_GPU_LAYERS,
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
    }
