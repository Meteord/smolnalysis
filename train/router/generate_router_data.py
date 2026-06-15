from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ROUTER_DIR = Path(__file__).resolve().parent
DATA_DIR = ROUTER_DIR / "data"
OPENUI_DATA_DIR = ROOT / "train" / "openui_lang" / "data"
CKAN_DATA_DIR = ROOT / "train" / "ckan" / "data"

SPLITS = ("train", "valid", "test")
LABELS = ("general_agent", "ckan_retrieval", "openui_translator")

SMALLTALK_PROMPTS = [
    "hi",
    "hello",
    "hey there",
    "good morning",
    "good evening",
    "how are you?",
    "how is your day going?",
    "thanks",
    "thank you",
    "cool, thanks",
    "nice",
    "what can you do?",
    "tell me a joke",
    "who are you?",
    "can you help me?",
    "ok",
    "yes",
    "no",
    "sounds good",
    "please explain your role",
    "guten morgen",
    "hallo",
    "danke",
    "wie geht es dir?",
    "was kannst du?",
    "erzähl mir einen witz",
    "alles klar",
    "super, danke",
    "merci",
    "servus",
]

SMALLTALK_TEMPLATES = [
    "{prompt}",
    "{prompt}!",
    "{prompt}.",
    "{prompt} :)",
    "Quick note: {prompt}",
    "Just checking in: {prompt}",
    "No data task, just saying {prompt}",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_user_prompt_from_messages(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"].strip()
    return ""


def openui_prompt_from_sample(sample: dict[str, Any]) -> str:
    messages = sample.get("messages")
    if isinstance(messages, list):
        prompt = extract_user_prompt_from_messages(messages)
        if prompt:
            return prompt
    question = sample.get("user_question")
    query_result = sample.get("query_result")
    component_hints = sample.get("component_hints")
    if isinstance(question, str) and question.strip() and query_result is not None:
        parts = [
            question.strip(),
            "Tool_result:\n" + json.dumps(query_result, ensure_ascii=False, indent=2),
        ]
        if component_hints is not None:
            parts.append("component_hints:\n" + json.dumps(component_hints, ensure_ascii=False, indent=2))
        return "\n\n".join(parts)
    if isinstance(question, str) and question.strip():
        return question.strip()
    payload = {}
    if query_result is not None:
        payload["Tool_result"] = query_result
    if component_hints is not None:
        payload["component_hints"] = component_hints
    return json.dumps(payload or sample, ensure_ascii=False, separators=(",", ":"))


def ckan_prompt_from_row(row: dict[str, Any]) -> str:
    request = row.get("request")
    if isinstance(request, str) and request.strip():
        return request.strip()
    messages = row.get("messages")
    if isinstance(messages, list):
        user = extract_user_prompt_from_messages(messages)
        if user.startswith("Request:"):
            return user.splitlines()[0].replace("Request:", "", 1).strip()
        if user:
            return user
    return ""


def collect_openui_samples(limit: int | None = None) -> list[dict[str, Any]]:
    samples = []
    for split in SPLITS:
        for sample_path in sorted((OPENUI_DATA_DIR / split).glob("*.json")):
            if sample_path.name == "manifest.json":
                continue
            sample = read_json(sample_path)
            prompt = openui_prompt_from_sample(sample)
            if not prompt:
                continue
            samples.append(
                {
                    "id": f"openui_{split}_{sample_path.stem}",
                    "source": str(sample_path.relative_to(ROOT)),
                    "label": "openui_translator",
                    "messages": [{"role": "user", "content": prompt}],
                }
            )
            if limit is not None and len(samples) >= limit:
                return samples
    return samples


def collect_ckan_samples(limit: int | None = None) -> list[dict[str, Any]]:
    sources = [
        CKAN_DATA_DIR / "scenarios_train.jsonl",
        CKAN_DATA_DIR / "harvested_inventory_scenarios.jsonl",
        CKAN_DATA_DIR / "harvested_scenarios.jsonl",
        CKAN_DATA_DIR / "scenarios_200.jsonl",
        CKAN_DATA_DIR / "scenarios_eval_golden.jsonl",
        CKAN_DATA_DIR / "multitool_eval_golden.jsonl",
    ]
    samples = []
    seen_prompts: set[str] = set()
    for source in sources:
        for index, row in enumerate(read_jsonl(source)):
            prompt = ckan_prompt_from_row(row)
            key = prompt.casefold()
            if not prompt or key in seen_prompts:
                continue
            seen_prompts.add(key)
            samples.append(
                {
                    "id": f"ckan_{source.stem}_{index:05d}",
                    "source": str(source.relative_to(ROOT)),
                    "label": "ckan_retrieval",
                    "messages": [{"role": "user", "content": prompt}],
                }
            )
            if limit is not None and len(samples) >= limit:
                return samples
    return samples


def collect_smalltalk_samples(count: int, rng: random.Random) -> list[dict[str, Any]]:
    samples = []
    seen = set()
    attempts = 0
    while len(samples) < count and attempts < count * 20:
        attempts += 1
        prompt = rng.choice(SMALLTALK_PROMPTS)
        text = rng.choice(SMALLTALK_TEMPLATES).format(prompt=prompt)
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        samples.append(
            {
                "id": f"smalltalk_{len(samples):05d}",
                "source": "synthetic_smalltalk",
                "label": "general_agent",
                "messages": [{"role": "user", "content": text}],
            }
        )
    if len(samples) < count:
        raise RuntimeError(f"Could only generate {len(samples)} smalltalk samples, requested {count}.")
    return samples


def split_samples(samples: list[dict[str, Any]], rng: random.Random) -> dict[str, list[dict[str, Any]]]:
    by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in LABELS}
    for sample in samples:
        by_label[sample["label"]].append(sample)
    for label_samples in by_label.values():
        rng.shuffle(label_samples)

    split_rows = {split: [] for split in SPLITS}
    for label, label_samples in by_label.items():
        n = len(label_samples)
        train_end = int(n * 0.8)
        valid_end = train_end + int(n * 0.1)
        partitions = {
            "train": label_samples[:train_end],
            "valid": label_samples[train_end:valid_end],
            "test": label_samples[valid_end:],
        }
        for split, rows in partitions.items():
            split_rows[split].extend(rows)

    for rows in split_rows.values():
        rng.shuffle(rows)
    return split_rows


def write_split(split: str, samples: list[dict[str, Any]]) -> None:
    split_dir = DATA_DIR / split
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter(sample["label"] for sample in samples)
    for index, sample in enumerate(samples, start=1):
        output = split_dir / f"{index:04d}-{sample['label']}.json"
        output.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "split": split,
        "samples": len(samples),
        "labels": dict(sorted(counts.items())),
    }
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate balanced adapter-router classification data.")
    parser.add_argument("--per-label", type=int, default=240, help="Maximum samples per label before splitting.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rng = random.Random(args.seed)

    ckan = collect_ckan_samples(args.per_label)
    openui = collect_openui_samples(args.per_label)
    target = min(args.per_label, len(ckan), len(openui))
    if target <= 0:
        raise RuntimeError("Could not collect enough CKAN/OpenUI samples for router data.")

    samples = ckan[:target] + openui[:target] + collect_smalltalk_samples(target, rng)
    split_rows = split_samples(samples, rng)
    for split, rows in split_rows.items():
        write_split(split, rows)

    summary = {
        "per_label": target,
        "total": len(samples),
        "splits": {
            split: {
                "samples": len(rows),
                "labels": dict(sorted(Counter(row["label"] for row in rows).items())),
            }
            for split, rows in split_rows.items()
        },
    }
    (DATA_DIR / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
