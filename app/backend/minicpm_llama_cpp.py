from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import lru_cache
import importlib.util
from importlib import metadata
import logging
import os
import site
import threading
import time
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
    lora_repo_id: str
    lora_filename: str


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
    lora_repo_id = _clean_env_value(_role_env(role, "LORA_REPO_ID"), "")
    lora_filename = _clean_env_value(_role_env(role, "LORA_FILENAME"), "")
    return LlamaCppRoleConfig(role, model_path, model_repo_id, model_filename, lora_path, lora_repo_id, lora_filename)


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


def _resolve_lora_path(config: LlamaCppRoleConfig) -> str:
    if config.lora_path:
        path = Path(config.lora_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"MiniCPM LoRA path does not exist for role {config.role}: {path}")
        return str(path)
    if config.lora_repo_id and config.lora_filename:
        return hf_hub_download(repo_id=config.lora_repo_id, filename=config.lora_filename)
    return ""


def _role_runtime_options(role: str) -> dict[str, Any]:
    options: dict[str, Any] = {
        "n_ctx": int(_clean_env_value(_role_env(role, "N_CTX"), str(DEFAULT_N_CTX))),
        "n_batch": int(_clean_env_value(_role_env(role, "N_BATCH"), str(DEFAULT_N_BATCH))),
        "n_gpu_layers": int(_clean_env_value(_role_env(role, "N_GPU_LAYERS"), str(DEFAULT_N_GPU_LAYERS))),
        "verbose": _clean_env_value("SMOLNALYSIS_MINICPM_VERBOSE", "false").casefold() in {"1", "true", "yes", "on"},
    }
    n_threads = _clean_env_value(_role_env(role, "N_THREADS"), _clean_env_value("SMOLNALYSIS_MINICPM_N_THREADS", _clean_env_value("N_THREADS")))
    if n_threads:
        options["n_threads"] = int(n_threads)
    return options


@lru_cache(maxsize=4)
def _load_llama_cached(
    model_path: str,
    lora_path: str,
    n_ctx: int,
    n_batch: int,
    n_gpu_layers: int,
    n_threads: int | None,
    verbose: bool,
):
    try:
        _preload_cuda_runtime()
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError("llama-cpp-python is not installed in this runtime.") from exc

    kwargs: dict[str, Any] = {
        "model_path": model_path,
        "n_ctx": n_ctx,
        "n_batch": n_batch,
        "n_gpu_layers": n_gpu_layers,
        "verbose": verbose,
    }
    if n_threads is not None:
        kwargs["n_threads"] = n_threads
    if lora_path:
        kwargs["lora_path"] = lora_path

    logger.info("loading MiniCPM llama.cpp model=%s lora=%s", model_path, lora_path or "none")
    return Llama(**kwargs)


def _preload_cuda_runtime() -> str:
    candidates = _cuda_runtime_library_candidates()
    errors = []
    for candidate in candidates:
        try:
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            return str(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    if errors:
        logger.debug("CUDA runtime preload attempts failed: %s", " | ".join(errors))
    return ""


def _cuda_runtime_library_candidates() -> list[Path]:
    candidates: list[Path] = []
    spec = importlib.util.find_spec("nvidia.cuda_runtime")
    if spec and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            candidates.append(Path(location) / "lib" / "libcudart.so.12")
    for root in [*site.getsitepackages(), site.getusersitepackages()]:
        candidates.append(Path(root) / "nvidia" / "cuda_runtime" / "lib" / "libcudart.so.12")
    return [candidate for candidate in candidates if candidate.exists()]


def _load_llama(role: str):
    config = role_config(role)
    model_path = _resolve_model_path(config)
    lora_path = _resolve_lora_path(config)
    options = _role_runtime_options(role)
    return _load_llama_cached(
        model_path,
        lora_path,
        options["n_ctx"],
        options["n_batch"],
        options["n_gpu_layers"],
        options.get("n_threads"),
        options["verbose"],
    )


def _load_llama_with_gpu_fallback(role: str):
    config = role_config(role)
    model_path = _resolve_model_path(config)
    lora_path = _resolve_lora_path(config)
    options = _role_runtime_options(role)
    try:
        llm = _load_llama_cached(
            model_path,
            lora_path,
            options["n_ctx"],
            options["n_batch"],
            options["n_gpu_layers"],
            options.get("n_threads"),
            options["verbose"],
        )
        return llm, options, ""
    except Exception as exc:
        if options["n_gpu_layers"] == 0:
            raise
        fallback_options = {**options, "n_gpu_layers": 0}
        logger.exception("MiniCPM llama.cpp GPU load failed for role=%s; retrying with CPU.", role)
        llm = _load_llama_cached(
            model_path,
            lora_path,
            fallback_options["n_ctx"],
            fallback_options["n_batch"],
            fallback_options["n_gpu_layers"],
            fallback_options.get("n_threads"),
            fallback_options["verbose"],
        )
        return llm, fallback_options, f"{type(exc).__name__}: {str(exc).strip() or type(exc).__name__}"


def role_runtime_status(role: str) -> dict[str, Any]:
    config = role_config(role)
    options = _role_runtime_options(role)
    model_path = ""
    lora_path = ""
    model_error = ""
    lora_error = ""
    try:
        model_path = _resolve_model_path(config)
    except Exception as exc:
        model_error = str(exc)
    try:
        lora_path = _resolve_lora_path(config)
    except Exception as exc:
        lora_error = str(exc)
    return {
        "role": role,
        "llama_cpp": llama_cpp_runtime_info(),
        "model_path": model_path or config.model_path,
        "model_repo_id": config.model_repo_id,
        "model_filename": config.model_filename,
        "model_error": model_error,
        "lora_path": lora_path or config.lora_path,
        "lora_repo_id": config.lora_repo_id,
        "lora_filename": config.lora_filename,
        "lora_error": lora_error,
        "options": options,
        "configured": bool(config.model_path or (config.model_repo_id and config.model_filename)),
        "loaded_models": _load_llama_cached.cache_info().currsize,
    }


def llama_cpp_runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "installed": False,
        "version": "",
        "supports_gpu_offload": None,
        "backend": "",
        "cuda_runtime_preload": "",
        "error": "",
    }
    try:
        info["version"] = metadata.version("llama-cpp-python")
    except metadata.PackageNotFoundError:
        info["error"] = "llama-cpp-python is not installed."
        return info
    except Exception as exc:
        info["error"] = str(exc)

    try:
        info["cuda_runtime_preload"] = _preload_cuda_runtime()
        import llama_cpp

        info["installed"] = True
        info["version"] = str(getattr(llama_cpp, "__version__", info["version"]))
        low_level = getattr(llama_cpp, "llama_cpp", None)
        supports_gpu_offload = getattr(low_level, "llama_supports_gpu_offload", None)
        if callable(supports_gpu_offload):
            info["supports_gpu_offload"] = bool(supports_gpu_offload())
        supports_mmap = getattr(low_level, "llama_supports_mmap", None)
        if callable(supports_mmap):
            info["supports_mmap"] = bool(supports_mmap())
        supports_mlock = getattr(low_level, "llama_supports_mlock", None)
        if callable(supports_mlock):
            info["supports_mlock"] = bool(supports_mlock())
    except Exception as exc:
        info["error"] = str(exc)
    return info

ROLE_SYSTEM_PROMPTS = {
    "general_agent": "You are smolnalysis, a concise assistant for exploring open data and planning analysis steps.",
    "ckan_retrieval": "You are the smolnalysis CKAN retrieval specialist. Help identify datasets, resources, filters, and catalog search steps.",
    "data_analysis": "You are the smolnalysis data analyst. Focus on columns, quality checks, aggregations, distributions, trends, and clear next analyses.",
    "openui_translator": "You are the smolnalysis OpenUI translator. When asked for UI, return valid OpenUI-Lang only.",
}


def _with_role_system_prompt(messages: list[dict[str, str]], role: str) -> list[dict[str, str]]:
    if any(message.get("role") == "system" for message in messages):
        return messages
    prompt = ROLE_SYSTEM_PROMPTS.get(role)
    if not prompt:
        return messages
    return [{"role": "system", "content": prompt}, *messages]


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
    response, _trace = generate_chat_response_with_trace(
        messages,
        adapter=adapter,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    return response


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
    runtime = role_runtime_status(role)
    routed_messages = _with_role_system_prompt(messages, role)
    cache_before = _load_llama_cached.cache_info()
    effective_options: dict[str, Any] = {}
    gpu_fallback_error = ""
    with MODEL_LOCK:
        try:
            llm, effective_options, gpu_fallback_error = _load_llama_with_gpu_fallback(role)
        except Exception as exc:
            raise RuntimeError(f"MiniCPM llama.cpp load failed for role '{role}'.") from exc
        cache_after_load = _load_llama_cached.cache_info()
        payload: dict[str, Any] = {
            "messages": routed_messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
            "stream": False,
        }
        if top_k is not None:
            payload["top_k"] = top_k
        try:
            response = llm.create_chat_completion(**payload)
        except Exception as exc:
            raise RuntimeError(f"MiniCPM llama.cpp generation failed for role '{role}'.") from exc

    content = response["choices"][0]["message"]["content"]
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    cache_hit = cache_after_load.hits > cache_before.hits
    trace = {
        "backend": "llama.cpp",
        "model_family": "MiniCPM",
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
        "effective_options": effective_options,
        "gpu_fallback_error": gpu_fallback_error,
        "cache": {
            "hit": cache_hit,
            "loaded_models": cache_after_load.currsize,
            "hits": cache_after_load.hits,
            "misses": cache_after_load.misses,
        },
        "events": [
            {"name": "route_role", "detail": f"{adapter or 'auto'} -> {role}"},
            {"name": "resolve_runtime", "detail": runtime.get("model_path") or runtime.get("model_repo_id") or "unconfigured"},
            {"name": "load_model", "detail": "cache hit" if cache_hit else "cache miss"},
            {"name": "generate", "detail": f"{len(str(content).strip())} chars in {elapsed_ms} ms"},
        ],
        "duration_ms": elapsed_ms,
        "output_chars": len(str(content).strip()),
    }
    logger.info("MiniCPM llama.cpp response generated: role=%s chars=%d", role, len(content))
    return str(content).strip(), trace


def runtime_status() -> dict[str, Any]:
    roles = {}
    for role in ROLE_ENV_KEYS:
        config = role_config(role)
        status = role_runtime_status(role)
        roles[role] = {
            "model_path": config.model_path,
            "model_repo_id": config.model_repo_id,
            "model_filename": config.model_filename,
            "lora_path": config.lora_path,
            "lora_repo_id": config.lora_repo_id,
            "lora_filename": config.lora_filename,
            "configured": bool(config.model_path or (config.model_repo_id and config.model_filename)),
            "loaded": _load_llama_cached.cache_info().currsize > 0,
            "resolved_model_path": status.get("model_path", ""),
            "resolved_lora_path": status.get("lora_path", ""),
            "model_error": status.get("model_error", ""),
            "lora_error": status.get("lora_error", ""),
            "options": status.get("options", {}),
            "llama_cpp": status.get("llama_cpp", {}),
        }
    return {
        "backend": "llama.cpp",
        "model_family": "MiniCPM",
        "llama_cpp": llama_cpp_runtime_info(),
        "roles": roles,
        "n_ctx": DEFAULT_N_CTX,
        "n_gpu_layers": DEFAULT_N_GPU_LAYERS,
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
    }


def probe_runtime(role: str = "general_agent") -> dict[str, Any]:
    started = time.perf_counter()
    normalized_role = normalize_role(role)
    if normalized_role == "auto":
        normalized_role = "general_agent"
    result: dict[str, Any] = {
        "role": normalized_role,
        "llama_cpp": llama_cpp_runtime_info(),
        "status": role_runtime_status(normalized_role),
        "load": {},
        "duration_ms": 0,
    }
    config = role_config(normalized_role)
    try:
        model_path = _resolve_model_path(config)
        lora_path = _resolve_lora_path(config)
        options = _role_runtime_options(normalized_role)
        result["load"] = {
            "ok": False,
            "model_path": model_path,
            "lora_path": lora_path,
            "options": options,
        }
        _preload_cuda_runtime()
        from llama_cpp import Llama

        kwargs: dict[str, Any] = {
            "model_path": model_path,
            "n_ctx": min(options["n_ctx"], 512),
            "n_batch": min(options["n_batch"], 128),
            "n_gpu_layers": options["n_gpu_layers"],
            "verbose": True,
        }
        if options.get("n_threads") is not None:
            kwargs["n_threads"] = options["n_threads"]
        if lora_path:
            kwargs["lora_path"] = lora_path
        Llama(**kwargs)
        result["load"]["ok"] = True
    except Exception as exc:
        result["load"]["error_type"] = type(exc).__name__
        result["load"]["error"] = str(exc).strip() or type(exc).__name__
    result["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return result
