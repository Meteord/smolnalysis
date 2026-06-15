from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GEMMA_DIR = REPO_ROOT / "models" / "gemma"


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    path: Path
    description: str

    @property
    def exists(self) -> bool:
        return (self.path / "adapter_config.json").exists()


ADAPTERS = {
    "retrieval": AdapterSpec(
        name="retrieval",
        path=GEMMA_DIR / "checkpoints" / "gemma4_retrieval_adapter",
        description="Retrieval/tool-call adapter checkpoint used by the Gemma backend.",
    ),
    "openui_translater": AdapterSpec(
        name="openui_translater",
        path=REPO_ROOT / "train" / "openui_lang" / "ouputs" / "openui-translate-mini-lora",
        description="Adapter trained on openui_sft_train.jsonl for tool result - openui component translation.",
    ),
}

DEFAULT_ADAPTER = "auto"


def get_adapter(name: str) -> AdapterSpec:
    try:
        return ADAPTERS[name]
    except KeyError as exc:
        available = ", ".join(sorted(ADAPTERS))
        raise KeyError(f"Unknown adapter '{name}'. Available adapters: {available}") from exc


def list_adapters() -> list[AdapterSpec]:
    return list(ADAPTERS.values())
