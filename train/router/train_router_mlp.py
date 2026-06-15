from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parent / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from dataset import ID_TO_LABEL, ROUTER_LABELS, RouterDataCollator, RouterTrainingDataset  # type: ignore
from router_mlp import RouterMLPConfig, build_router_mlp  # type: ignore


DEFAULT_MODEL = "openbmb/MiniCPM5-1B"
DEFAULT_OUTPUT_DIR = Path("train/router/outputs/router-mlp")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a lightweight adapter-router MLP over tokenizer input_ids.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--train-data", type=Path, default=DATA_DIR / "train")
    parser.add_argument("--eval-data", type=Path, default=DATA_DIR / "valid")
    parser.add_argument("--test-data", type=Path, default=DATA_DIR / "test")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenizer_vocab_size(tokenizer: Any) -> int:
    value = getattr(tokenizer, "vocab_size", None)
    if isinstance(value, int) and value > 0:
        return value
    return len(tokenizer)


def build_loader(dataset: RouterTrainingDataset, tokenizer: Any, batch_size: int, *, shuffle: bool):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=RouterDataCollator(tokenizer),
    )


def evaluate(model: Any, loader: Any, device: Any) -> dict[str, Any]:
    import torch

    model.eval()
    total = 0
    correct = 0
    loss_total = 0.0
    confusion = [[0 for _ in ROUTER_LABELS] for _ in ROUTER_LABELS]
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            logits = output["logits"]
            loss = output["loss"]
            predictions = logits.argmax(dim=-1)
            labels = batch["labels"]
            total += int(labels.numel())
            correct += int((predictions == labels).sum().item())
            loss_total += float(loss.item()) * int(labels.numel())
            for gold, pred in zip(labels.tolist(), predictions.tolist(), strict=False):
                confusion[gold][pred] += 1
    return {
        "loss": loss_total / max(1, total),
        "accuracy": correct / max(1, total),
        "samples": total,
        "confusion": {
            ID_TO_LABEL[index]: {ID_TO_LABEL[j]: value for j, value in enumerate(row)}
            for index, row in enumerate(confusion)
        },
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    tokenizer = load_tokenizer(args.model_name)
    train_dataset = RouterTrainingDataset(args.train_data, tokenizer, max_length=args.max_length)
    eval_dataset = RouterTrainingDataset(args.eval_data, tokenizer, max_length=args.max_length)
    test_dataset = RouterTrainingDataset(args.test_data, tokenizer, max_length=args.max_length)

    pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0) or 0
    config = RouterMLPConfig(
        vocab_size=tokenizer_vocab_size(tokenizer),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_labels=len(ROUTER_LABELS),
        dropout=args.dropout,
        pad_token_id=int(pad_token_id),
    )
    model = build_router_mlp(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = build_loader(train_dataset, tokenizer, args.batch_size, shuffle=True)
    eval_loader = build_loader(eval_dataset, tokenizer, args.batch_size, shuffle=False)
    test_loader = build_loader(test_dataset, tokenizer, args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.dry_run:
        first = train_dataset[0]
        summary = {
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "test_samples": len(test_dataset),
            "input_ids_len": int(first["input_ids"].shape[0]),
            "label": int(first["labels"].item()),
            "label_name": ID_TO_LABEL[int(first["labels"].item())],
            "vocab_size": config.vocab_size,
            "device": str(device),
        }
        print(json.dumps(summary, indent=2))
        return summary

    best_eval = math.inf
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_total = 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            loss = output["loss"]
            loss.backward()
            optimizer.step()
            batch_size = int(batch["labels"].shape[0])
            train_loss += float(loss.item()) * batch_size
            train_total += batch_size

        eval_metrics = evaluate(model, eval_loader, device)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, train_total),
            "eval_loss": eval_metrics["loss"],
            "eval_accuracy": eval_metrics["accuracy"],
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics))
        if eval_metrics["loss"] < best_eval:
            best_eval = eval_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    eval_metrics = evaluate(model, eval_loader, device)
    metrics = {
        "eval": eval_metrics,
        "test": test_metrics,
        "history": history,
        "config": config.to_dict(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "router_mlp.pt")
    (args.output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> int:
    args = build_arg_parser().parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
