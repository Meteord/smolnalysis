from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

LOCAL_ADAPTERS = {
    "ckan_retrieval": REPO_ROOT / "train" / "retrieval" / "outputs" / "tool-results-minicpm5-lora" / "checkpoint-260",
    "openui_translator": REPO_ROOT / "train" / "openui_lang" / "outputs" / "openui-translate-mini-lora" / "checkpoint-160",
}


def adapters() -> dict[str, str]:
    return {name: str(path) for name, path in LOCAL_ADAPTERS.items()}


def adapter_source(adapter_name: str | None) -> str:
    path = LOCAL_ADAPTERS.get((adapter_name or "").strip().casefold())
    return str(path) if path is not None else ""
