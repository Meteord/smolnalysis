from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

LOCAL_ADAPTERS = {
    "ckan_retrieval": REPO_ROOT / "train" / "retrieval" / "outputs" / "tool-results-minicpm5-lora" / "checkpoint-260",
    "openui_translator": REPO_ROOT / "train" / "openui_lang" / "outputs" / "openui-translate-mini-lora" / "checkpoint-160",
}

HUB_ADAPTERS = {
    "ckan_retrieval": "build-small-hackathon/smolnalysis-generation-minicpm5-lora",
    "openui_translator": "build-small-hackathon/smolnalysis-translation-minicpm5-lora",
}

ROLE_ENV_KEYS = {
    "ckan_retrieval": "CKAN_RETRIEVAL",
    "openui_translator": "OPENUI_TRANSLATOR",
}


def _env_adapter_source(adapter_name: str) -> str:
    key = ROLE_ENV_KEYS.get(adapter_name)
    if not key:
        return ""
    return os.getenv(f"SMOLNALYSIS_{key}_ADAPTER_SOURCE", "").strip()


def adapters() -> dict[str, str]:
    return {name: adapter_source(name) for name in HUB_ADAPTERS}


def adapter_source(adapter_name: str | None) -> str:
    name = (adapter_name or "").strip().casefold()
    override = _env_adapter_source(name)
    if override:
        return override
    path = LOCAL_ADAPTERS.get(name)
    if path is not None and path.exists():
        return str(path)
    return HUB_ADAPTERS.get(name, "")
