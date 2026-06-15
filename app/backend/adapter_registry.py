from __future__ import annotations

import os


DEFAULT_CKAN_RETRIEVAL_ADAPTER_REPO_ID = "build-small-hackathon/smolnalysis-ckan-retrieval-minicpm5-lora"
ADAPTER_ENV_KEYS = {
    "ckan_retrieval": "CKAN_RETRIEVAL",
    "data_analysis": "DATA_ANALYSIS",
    "openui_translator": "OPENUI_TRANSLATOR",
}


def _clean_env_value(name: str, default: str = "") -> str:
    raw = os.getenv(name, default)
    lines = []
    for line in str(raw).splitlines():
        value = line.strip().strip('"').strip("'")
        if value and not value.startswith("#"):
            lines.append(value)
    return lines[-1] if lines else default


def adapters() -> dict[str, str]:
    return {name: adapter_source(name) for name in ADAPTER_ENV_KEYS}


def adapter_source(adapter_name: str | None) -> str:
    name = (adapter_name or "").strip().casefold()
    role_key = ADAPTER_ENV_KEYS.get(name)
    if role_key is None:
        return ""

    for suffix in ("ADAPTER_PATH", "LORA_PATH", "ADAPTER_REPO_ID", "LORA_REPO_ID"):
        value = _clean_env_value(f"SMOLNALYSIS_MINICPM_{role_key}_{suffix}")
        if value:
            return value
    if name == "ckan_retrieval":
        return _clean_env_value("SMOLNALYSIS_DEFAULT_CKAN_RETRIEVAL_ADAPTER_REPO_ID", DEFAULT_CKAN_RETRIEVAL_ADAPTER_REPO_ID)
    return ""
