from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RouterMLPConfig:
    def __init__(
        self,
        *,
        vocab_size: int | None = None,
        embedding_dim: int | None = None,
        hidden_dim: int,
        num_labels: int,
        dropout: float,
        pad_token_id: int,
        encoder_model_name: str | None = None,
        encoder_hidden_size: int | None = None,
        architecture: str = "frozen_encoder",
        labels: list[str] | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.encoder_model_name = encoder_model_name
        self.encoder_hidden_size = encoder_hidden_size
        self.architecture = architecture
        self.labels = labels or ["general_agent", "ckan_retrieval", "openui_translator"]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RouterMLPConfig":
        return cls(
            vocab_size=int(payload["vocab_size"]) if payload.get("vocab_size") is not None else None,
            embedding_dim=int(payload["embedding_dim"]) if payload.get("embedding_dim") is not None else None,
            hidden_dim=int(payload["hidden_dim"]),
            num_labels=int(payload["num_labels"]),
            dropout=float(payload["dropout"]),
            pad_token_id=int(payload["pad_token_id"]),
            encoder_model_name=payload.get("encoder_model_name"),
            encoder_hidden_size=(
                int(payload["encoder_hidden_size"]) if payload.get("encoder_hidden_size") is not None else None
            ),
            architecture=str(payload.get("architecture") or "embedding_mlp"),
            labels=list(payload.get("labels") or ["general_agent", "ckan_retrieval", "openui_translator"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "RouterMLPConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_labels": self.num_labels,
            "dropout": self.dropout,
            "pad_token_id": self.pad_token_id,
            "encoder_model_name": self.encoder_model_name,
            "encoder_hidden_size": self.encoder_hidden_size,
            "architecture": self.architecture,
            "labels": self.labels,
        }


def build_router_mlp(config: RouterMLPConfig):
    import torch
    from torch import nn

    if config.architecture == "embedding_mlp":
        if config.vocab_size is None or config.embedding_dim is None:
            raise ValueError("embedding_mlp router requires vocab_size and embedding_dim.")

        class RouterMLP(nn.Module):
            def __init__(self, cfg: RouterMLPConfig) -> None:
                super().__init__()
                self.config = cfg
                self.embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim, padding_idx=cfg.pad_token_id)
                self.net = nn.Sequential(
                    nn.LayerNorm(cfg.embedding_dim),
                    nn.Linear(cfg.embedding_dim, cfg.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.hidden_dim, cfg.num_labels),
                )

            def forward(self, input_ids, attention_mask=None, labels=None):
                input_ids = input_ids.clamp(min=0, max=self.config.vocab_size - 1)
                embeddings = self.embedding(input_ids)
                if attention_mask is None:
                    pooled = embeddings.mean(dim=1)
                else:
                    mask = attention_mask.unsqueeze(-1).to(embeddings.dtype)
                    pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                logits = self.net(pooled)
                loss = None
                if labels is not None:
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                return {"loss": loss, "logits": logits}

        return RouterMLP(config)

    if config.architecture != "frozen_encoder":
        raise ValueError(f"Unknown router architecture: {config.architecture}")

    if not config.encoder_model_name:
        raise ValueError("frozen_encoder router requires encoder_model_name.")

    def load_encoder(model_name: str):
        from transformers import AutoModel, AutoModelForCausalLM

        try:
            return AutoModel.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            return AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    class RouterMLP(nn.Module):
        def __init__(self, cfg: RouterMLPConfig) -> None:
            super().__init__()
            self.config = cfg
            self.encoder = load_encoder(cfg.encoder_model_name)
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            self.encoder.eval()

            hidden_size = cfg.encoder_hidden_size
            if hidden_size is None:
                hidden_size = int(getattr(self.encoder.config, "hidden_size", 0) or 0)
            if hidden_size <= 0:
                raise ValueError("Could not determine encoder hidden size.")

            self.net = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, cfg.num_labels),
            )

        @staticmethod
        def _last_token_pool(hidden_states, attention_mask):
            if attention_mask is None:
                return hidden_states[:, -1]
            lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
            batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
            return hidden_states[batch_indices, lengths]

        def forward(self, input_ids, attention_mask=None, labels=None):
            self.encoder.eval()
            with torch.no_grad():
                output = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = getattr(output, "last_hidden_state", None)
                if hidden_states is None:
                    hidden_states = output.hidden_states[-1]
                pooled = self._last_token_pool(hidden_states, attention_mask).detach()
                pooled = pooled.to(next(self.net.parameters()).dtype)
            logits = self.net(pooled)
            loss = None
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}

    return RouterMLP(config)


def load_router_mlp(output_dir: str | Path, *, map_location: str = "cpu"):
    import torch

    output_dir = Path(output_dir)
    config = RouterMLPConfig.from_json(output_dir / "config.json")
    model = build_router_mlp(config)
    state_dict = torch.load(output_dir / "router_mlp.pt", map_location=map_location)
    strict = config.architecture != "frozen_encoder"
    incompatible = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        missing_head_keys = [key for key in incompatible.missing_keys if not key.startswith("encoder.")]
        if missing_head_keys:
            raise RuntimeError(f"Router checkpoint is missing classifier weights: {missing_head_keys}")
    model.eval()
    return model, config
