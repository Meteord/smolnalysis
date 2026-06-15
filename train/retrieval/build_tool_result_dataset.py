from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "train" / "openui_lang" / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
TOOL_RESULT_MARKER = "\n\nTool result:\n"
SYSTEM_PROMPT = (
    "You generate the structured tool result for a smolnalysis user question. "
    "Return only valid JSON. Do not include explanations, markdown, or any label prefix."
)

SPLITS = {
    "train": SOURCE_DIR / "openui_sft_train.jsonl",
    "eval": SOURCE_DIR / "openui_sft_eval.jsonl",
    "test": SOURCE_DIR / "openui_sft_test.jsonl",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def extract_question_and_tool_result(sample: dict[str, Any], *, source_path: Path, index: int) -> tuple[str, str]:
    messages = sample.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{source_path}:{index} is missing messages")

    user_messages = [message for message in messages if message.get("role") == "user"]
    if len(user_messages) != 1:
        raise ValueError(f"{source_path}:{index} expected exactly one user message")

    user_content = user_messages[0].get("content")
    if not isinstance(user_content, str) or TOOL_RESULT_MARKER not in user_content:
        raise ValueError(f"{source_path}:{index} user content does not contain the tool-result marker")

    question, raw_tool_result = user_content.split(TOOL_RESULT_MARKER, 1)
    question = question.strip()
    raw_tool_result = raw_tool_result.strip()
    if not question:
        raise ValueError(f"{source_path}:{index} extracted an empty question")

    try:
        parsed_tool_result = json.loads(raw_tool_result)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source_path}:{index} tool result is not valid JSON") from exc

    label = json.dumps(parsed_tool_result, ensure_ascii=False, indent=2)
    if "Tool result:" in label:
        raise ValueError(f"{source_path}:{index} label still contains the tool-result prefix")
    return question, label


def convert_sample(sample: dict[str, Any], *, split: str, source_path: Path, index: int) -> dict[str, Any]:
    question, label = extract_question_and_tool_result(sample, source_path=source_path, index=index)
    metadata = dict(sample.get("metadata") or {})
    metadata.update(
        {
            "source_dataset": "openui_lang",
            "source_split": split,
            "source_index": index,
            "task": "generate_tool_result",
        }
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": label},
        ],
        "metadata": metadata,
    }


def build_split(split: str, source_path: Path, output_path: Path) -> dict[str, Any]:
    rows = read_jsonl(source_path)
    converted = [
        convert_sample(sample, split=split, source_path=source_path, index=index)
        for index, sample in enumerate(rows, start=1)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in converted:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    return {
        "split": split,
        "source": str(source_path.relative_to(ROOT)),
        "output": str(output_path.relative_to(ROOT)),
        "examples": len(converted),
    }


def build_dataset(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    split_summaries = [
        build_split(split, source_path, output_dir / f"tool_result_{split}.jsonl")
        for split, source_path in SPLITS.items()
    ]
    manifest = {
        "task": "generate_tool_result",
        "description": "Question-to-tool-result shortcut dataset derived from OpenUI-Lang SFT data.",
        "label_contract": "Assistant content is valid JSON only; it never includes the source tool-result marker.",
        "system_prompt": SYSTEM_PROMPT,
        "splits": split_summaries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build question-to-tool-result training data from OpenUI SFT JSONL.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    manifest = build_dataset(args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
