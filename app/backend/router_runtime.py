from __future__ import annotations

import importlib.util
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROUTER_DIR = REPO_ROOT / "train" / "router" / "outputs" / "router-mlp"
ROUTER_SOURCE = REPO_ROOT / "train" / "router" / "router_mlp.py"
DEFAULT_TOKENIZER_MODEL_ID = "openbmb/MiniCPM5-1B"


@dataclass(frozen=True)
class RouterPrediction:
    role: str
    confidence: float
    logits: list[float]
    source: str


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def router_enabled() -> bool:
    return _truthy(os.getenv("SMOLNALYSIS_ROUTER_ENABLED"))


def router_output_dir() -> Path:
    path = Path(os.getenv("SMOLNALYSIS_ROUTER_OUTPUT_DIR", str(DEFAULT_ROUTER_DIR))).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def router_max_length() -> int:
    return max(1, int(os.getenv("SMOLNALYSIS_ROUTER_MAX_LENGTH", "512")))


def router_min_confidence() -> float:
    raw = float(os.getenv("SMOLNALYSIS_ROUTER_MIN_CONFIDENCE", "0"))
    return max(0.0, min(raw, 1.0))


def router_tokenizer_model_id(default: str = DEFAULT_TOKENIZER_MODEL_ID) -> str:
    return os.getenv("SMOLNALYSIS_ROUTER_TOKENIZER_MODEL_ID", default).strip() or default


def _load_router_module():
    spec = importlib.util.spec_from_file_location("smolnalysis_router_mlp", ROUTER_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load router module from {ROUTER_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _load_router_runtime(model_id: str, output_dir: str):
    from transformers import AutoTokenizer

    module = _load_router_module()
    router, config = module.load_router_mlp(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return tokenizer, router, config


def _chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    lines = [f"{message['role']}: {message['content']}" for message in messages]
    lines.append("assistant:")
    return "\n".join(lines)


def _tokenize(tokenizer: Any, messages: list[dict[str, str]], max_length: int):
    import torch

    text = _chat_template(tokenizer, messages)
    encoded = tokenizer(text, add_special_tokens=False, return_tensors=None)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    input_ids = list(input_ids)[-max_length:]
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
    }


def predict_role(messages: list[dict[str, str]], *, model_id: str) -> RouterPrediction | None:
    if not router_enabled():
        return None

    output_dir = router_output_dir()
    if not (output_dir / "router_mlp.pt").exists() or not (output_dir / "config.json").exists():
        logger.warning("router enabled but artifacts are missing in %s", output_dir)
        return None

    try:
        import torch

        tokenizer, router, config = _load_router_runtime(model_id, str(output_dir))
        features = _tokenize(tokenizer, messages, router_max_length())
        with torch.inference_mode():
            output = router(**features)
            probabilities = torch.softmax(output["logits"], dim=-1)[0]
        label_index = int(probabilities.argmax().item())
        confidence = float(probabilities[label_index].item())
        role = str(config.labels[label_index])
        if confidence < router_min_confidence():
            logger.info("router prediction below threshold: role=%s confidence=%.3f", role, confidence)
            return None
        return RouterPrediction(
            role=role,
            confidence=confidence,
            logits=[float(value) for value in output["logits"][0].detach().cpu().tolist()],
            source=str(output_dir),
        )
    except Exception:
        logger.exception("router prediction failed; falling back to heuristic routing")
        return None


def runtime_status() -> dict[str, Any]:
    output_dir = router_output_dir()
    cache = _load_router_runtime.cache_info()
    return {
        "enabled": router_enabled(),
        "output_dir": str(output_dir),
        "artifacts_present": (output_dir / "router_mlp.pt").exists() and (output_dir / "config.json").exists(),
        "max_length": router_max_length(),
        "min_confidence": router_min_confidence(),
        "cache": {
            "loaded": cache.currsize > 0,
            "hits": cache.hits,
            "misses": cache.misses,
        },
    }
