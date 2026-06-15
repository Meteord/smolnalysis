from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RouterMLPConfig:
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_labels: int,
        dropout: float,
        pad_token_id: int,
        labels: list[str] | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.labels = labels or ["general_agent", "ckan_retrieval", "openui_translator"]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RouterMLPConfig":
        return cls(
            vocab_size=int(payload["vocab_size"]),
            embedding_dim=int(payload["embedding_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            num_labels=int(payload["num_labels"]),
            dropout=float(payload["dropout"]),
            pad_token_id=int(payload["pad_token_id"]),
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
            "labels": self.labels,
        }


def build_router_mlp(config: RouterMLPConfig):
    from torch import nn
    import torch

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


def load_router_mlp(output_dir: str | Path, *, map_location: str = "cpu"):
    import torch

    output_dir = Path(output_dir)
    config = RouterMLPConfig.from_json(output_dir / "config.json")
    model = build_router_mlp(config)
    model.load_state_dict(torch.load(output_dir / "router_mlp.pt", map_location=map_location))
    model.eval()
    return model, config
