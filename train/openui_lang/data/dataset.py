from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_OPENUI_SYSTEM_PROMPT = (
    "You translate smolnalysis workflow results into OpenUI-Lang. "
    "Output OpenUI-Lang only."
)


def load_json_samples(path: str | Path) -> list[dict[str, Any]]:
    """Load OpenUI samples from one JSON file, a JSONL file, or a split directory."""
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


def extract_openui_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    """Return normalized chat messages for one OpenUI sample."""
    messages = sample.get("messages")
    if messages:
        normalized = []
        for index, message in enumerate(messages):
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant"} or not isinstance(content, str):
                raise ValueError(f"Invalid message at index {index}: {message!r}")
            normalized.append({"role": role, "content": content})
        if not any(message["role"] == "assistant" for message in normalized):
            raise ValueError("OpenUI sample messages must include an assistant message.")
        return normalized

    assistant = sample.get("openui_lang")
    if not isinstance(assistant, str) or not assistant.strip():
        raise ValueError("OpenUI sample must include messages or a non-empty openui_lang target.")

    user_payload = {
        "task": sample.get("task", "render_openui"),
        "user_question": sample.get("user_question"),
        "query_result": sample.get("query_result"),
        "component_hints": sample.get("component_hints"),
        "quality_score": sample.get("quality_score"),
    }
    return [
        {"role": "system", "content": DEFAULT_OPENUI_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
        {"role": "assistant", "content": assistant},
    ]


def prompt_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return all messages before the final assistant target."""
    assistant_indexes = [index for index, message in enumerate(messages) if message["role"] == "assistant"]
    if not assistant_indexes:
        raise ValueError("Messages must include an assistant target.")
    return messages[: assistant_indexes[-1]]


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
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


def _to_tensor(values: list[int], return_tensors: str):
    if return_tensors == "list":
        return values
    if return_tensors != "pt":
        raise ValueError(f"Unsupported return_tensors value: {return_tensors}")
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch is required when return_tensors='pt'.") from exc
    return torch.tensor(values, dtype=torch.long)


def make_chat_features(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_length: int = 4096,
    return_tensors: str = "pt",
) -> dict[str, Any]:
    """Tokenize one chat sample and mask non-assistant tokens in labels."""
    if max_length <= 0:
        raise ValueError("max_length must be positive.")

    prompt_text = apply_chat_template(tokenizer, prompt_messages(messages), add_generation_prompt=True)
    full_text = apply_chat_template(tokenizer, messages, add_generation_prompt=False)

    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token is not None and not full_text.endswith(eos_token):
        full_text += eos_token

    prompt_ids = _tokenize_to_ids(tokenizer, prompt_text)
    full_ids = _tokenize_to_ids(tokenizer, full_text)

    removed = max(0, len(full_ids) - max_length)
    if removed:
        full_ids = full_ids[removed:]
    prompt_len = max(0, min(len(prompt_ids) - removed, len(full_ids)))

    labels = list(full_ids)
    labels[:prompt_len] = [-100] * prompt_len
    attention_mask = [1] * len(full_ids)

    return {
        "input_ids": _to_tensor(full_ids, return_tensors),
        "attention_mask": _to_tensor(attention_mask, return_tensors),
        "labels": _to_tensor(labels, return_tensors),
    }


class OpenUITrainingDataset:
    """Dataset for OpenUI-Lang supervised fine-tuning samples."""

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: Any,
        *,
        max_length: int = 4096,
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
            raise ValueError(f"No OpenUI samples found in {self.data_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        messages = extract_openui_messages(sample)
        features = make_chat_features(
            self.tokenizer,
            messages,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
        )
        if self.include_metadata:
            features["messages"] = messages
            features["task"] = sample.get("task")
            features["dataset_title"] = (sample.get("query_result") or {}).get("dataset_title")
        return features


class OpenUIDataCollator:
    """Pad OpenUITrainingDataset items into a causal-LM batch."""

    def __init__(self, tokenizer: Any, *, label_pad_token_id: int = -100) -> None:
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            from torch.nn.utils.rnn import pad_sequence
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("torch is required to collate OpenUI training batches.") from exc

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
            "labels": pad_sequence(
                [feature["labels"] for feature in features],
                batch_first=True,
                padding_value=self.label_pad_token_id,
            ),
        }


class _DebugTokenizer:
    eos_token = "<eos>"
    eos_token_id = 0
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "\n".join(f"<{message['role']}> {message['content']}" for message in messages)
        if add_generation_prompt:
            text += "\n<assistant> "
        return text

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return {"input_ids": [ord(char) % 251 + 1 for char in text]}


def _preview_text(value: str, max_chars: int = 500) -> str:
    value = value.replace("\n", "\\n")
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "..."


def _main() -> None:
    split_dir = Path(__file__).resolve().parent / "train"
    dataset = OpenUITrainingDataset(
        split_dir,
        _DebugTokenizer(),
        max_length=12000,
        return_tensors="list",
        include_metadata=True,
    )
    item = dataset[0]
    messages = item["messages"]
    supervised_tokens = sum(1 for label in item["labels"] if label != -100)
    masked_tokens = sum(1 for label in item["labels"] if label == -100)

    print("OpenUITrainingDataset smoke test")
    print(f"data_path: {split_dir}")
    print(f"samples: {len(dataset)}")
    print(f"first_task: {item.get('task')}")
    print(f"first_dataset_title: {item.get('dataset_title')}")
    print(f"message_roles: {[message['role'] for message in messages]}")
    print(f"user_preview: {_preview_text(messages[-2]['content'])}")
    print(f"assistant_preview: {_preview_text(messages[-1]['content'])}")
    print(f"input_ids_len: {len(item['input_ids'])}")
    print(f"attention_mask_len: {len(item['attention_mask'])}")
    print(f"labels_len: {len(item['labels'])}")
    print(f"masked_prompt_tokens: {masked_tokens}")
    print(f"supervised_assistant_tokens: {supervised_tokens}")

    assert len(item["input_ids"]) == len(item["attention_mask"]) == len(item["labels"])
    assert masked_tokens > 0
    assert supervised_tokens > 0
    print("status: ok")


if __name__ == "__main__":
    _main()
