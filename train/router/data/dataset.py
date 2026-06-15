from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROUTER_LABELS = ["general_agent", "ckan_retrieval", "openui_translator"]
LABEL_TO_ID = {label: index for index, label in enumerate(ROUTER_LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def load_json_samples(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        rows = []
        for sample_path in sorted(path.glob("*.json")):
            if sample_path.name == "manifest.json":
                continue
            rows.append(_read_json(sample_path))
        return rows

    if path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
        return rows

    return [_read_json(path)]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def label_id(label: str) -> int:
    try:
        return LABEL_TO_ID[label]
    except KeyError as exc:
        available = ", ".join(ROUTER_LABELS)
        raise ValueError(f"Unknown router label {label!r}. Available labels: {available}") from exc


def extract_router_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    messages = sample.get("messages")
    if messages:
        normalized = []
        for index, message in enumerate(messages):
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant"} or not isinstance(content, str):
                raise ValueError(f"Invalid message at index {index}: {message!r}")
            normalized.append({"role": role, "content": content})
        return normalized

    prompt = sample.get("prompt") or sample.get("request") or sample.get("user_question")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Router sample must include messages, prompt, request, or user_question.")
    return [{"role": "user", "content": prompt.strip()}]


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool = True) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    lines = [f"{message['role']}: {message['content']}" for message in messages]
    if add_generation_prompt:
        lines.append("assistant:")
    return "\n".join(lines)


def _tokenize_to_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors=None,
    )
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return list(input_ids)


def _to_tensor(values: list[int] | int, return_tensors: str):
    if return_tensors == "list":
        return values
    if return_tensors != "pt":
        raise ValueError(f"Unsupported return_tensors value: {return_tensors}")
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch is required when return_tensors='pt'.") from exc
    if isinstance(values, int):
        return torch.tensor(values, dtype=torch.long)
    return torch.tensor(values, dtype=torch.long)


def make_router_features(
    tokenizer: Any,
    messages: list[dict[str, str]],
    label: str,
    *,
    max_length: int = 512,
    return_tensors: str = "pt",
) -> dict[str, Any]:
    if max_length <= 0:
        raise ValueError("max_length must be positive.")

    text = apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    input_ids = _tokenize_to_ids(tokenizer, text)
    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": _to_tensor(input_ids, return_tensors),
        "attention_mask": _to_tensor(attention_mask, return_tensors),
        "labels": _to_tensor(label_id(label), return_tensors),
    }


class RouterTrainingDataset:
    def __init__(
        self,
        data_path: str | Path,
        tokenizer: Any,
        *,
        max_length: int = 512,
        return_tensors: str = "pt",
        include_metadata: bool = False,
    ) -> None:
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tensors = return_tensors
        self.include_metadata = include_metadata
        self.samples = load_json_samples(self.data_path)

        if not self.samples:
            raise ValueError(f"No router samples found in {self.data_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        label = sample.get("label") or sample.get("adapter")
        if not isinstance(label, str):
            raise ValueError(f"Router sample is missing a string label: {sample!r}")
        messages = extract_router_messages(sample)
        features = make_router_features(
            self.tokenizer,
            messages,
            label,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
        )
        if self.include_metadata:
            features["messages"] = messages
            features["label_name"] = label
            features["sample_id"] = sample.get("id")
            features["source"] = sample.get("source")
        return features


class RouterDataCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            import torch
            from torch.nn.utils.rnn import pad_sequence
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("torch is required to collate router training batches.") from exc

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id.")

        return {
            "input_ids": pad_sequence(
                [feature["input_ids"] for feature in features],
                batch_first=True,
                padding_value=pad_token_id,
            ),
            "attention_mask": pad_sequence(
                [feature["attention_mask"] for feature in features],
                batch_first=True,
                padding_value=0,
            ),
            "labels": torch.stack([feature["labels"] for feature in features]),
        }


class _DebugTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "\n".join(f"<{message['role']}> {message['content']}" for message in messages)
        if add_generation_prompt:
            text += "\n<assistant> "
        return text

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return {"input_ids": [ord(char) % 251 + 1 for char in text]}
