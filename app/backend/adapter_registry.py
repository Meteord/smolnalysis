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
    "tool_json": AdapterSpec(
        name="tool_json",
        path=REPO_ROOT / "models" / "gemma4-tool-lora-adapter",
        description="Adapter trained by training/trainer.py on llm_user_queries.jsonl.",
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
