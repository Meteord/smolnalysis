from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


ROLE_ALIASES = {
    "ckan_tool": "ckan_retrieval",
    "ckan": "ckan_retrieval",
    "retrieval": "ckan_retrieval",
    "openui": "openui_translator",
    "analysis": "data_analysis",
    "general": "general_agent",
}


@dataclass
class ModelResponse:
    content: str
    trace: dict[str, Any]


def normalize_role(role: str) -> str:
    clean = role.strip().casefold()
    return ROLE_ALIASES.get(clean, clean)


def call_role_model(role: str, messages: list[dict[str, str]], response_contract: str = "") -> ModelResponse:
    normalized_role = normalize_role(role)
    routed_messages = messages
    if response_contract:
        routed_messages = [{"role": "system", "content": response_contract}, *messages]

    backend = os.getenv("SMOLNALYSIS_MINICPM_BACKEND", "transformers").strip().casefold()
    if backend in {"llama.cpp", "llamacpp", "llama_cpp", "gguf"}:
        try:
            from .backend.minicpm_llama_cpp import generate_chat_response_with_trace
        except ImportError:
            from backend.minicpm_llama_cpp import generate_chat_response_with_trace
    else:
        try:
            from .backend.minicpm_transformers import generate_chat_response_with_trace
        except ImportError:
            from backend.minicpm_transformers import generate_chat_response_with_trace

    content, trace = generate_chat_response_with_trace(routed_messages, adapter=normalized_role)
    return ModelResponse(str(content).strip(), trace)
