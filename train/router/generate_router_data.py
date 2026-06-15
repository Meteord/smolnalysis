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
RETRIEVAL_DATA_DIR = ROOT / "train" / "retrieval" / "data"

SPLITS = ("train", "valid", "test")
LABELS = ("general_agent", "ckan_retrieval", "openui_translator")
OPENUI_SFT_SPLITS = {
    "train": OPENUI_DATA_DIR / "openui_sft_train.jsonl",
    "valid": OPENUI_DATA_DIR / "openui_sft_eval.jsonl",
    "test": OPENUI_DATA_DIR / "openui_sft_test.jsonl",
}
RETRIEVAL_SFT_SPLITS = {
    "train": RETRIEVAL_DATA_DIR / "tool_result_train.jsonl",
    "valid": RETRIEVAL_DATA_DIR / "tool_result_eval.jsonl",
    "test": RETRIEVAL_DATA_DIR / "tool_result_test.jsonl",
}
TOOL_RESULT_MARKERS = ("\n\nTool result:\n", "\n\nTool_result:\n", "\n\nTool result\n", "\n\nTool_result\n")

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

GENERAL_INTENTS = [
    "write a short project update for my team",
    "summarize this paragraph in plain language",
    "help me brainstorm names for a small analytics app",
    "explain what a Python virtual environment is",
    "turn these notes into a checklist",
    "draft a polite reply to a meeting invite",
    "what is the difference between precision and recall?",
    "give me three ways to debug a failing unit test",
    "translate this sentence into German: the report is ready",
    "make this sentence more concise",
    "help me plan a weekend in Munich",
    "explain how git rebase differs from merge",
    "write a regex that matches ISO dates",
    "what should I consider before choosing a database?",
    "create a simple agenda for a design review",
    "how do I make a command line script easier to use?",
    "give me a short explanation of REST APIs",
    "draft release notes for a bug fix",
    "suggest a folder structure for a small Python package",
    "explain the tradeoff between latency and throughput",
]

GENERAL_TEMPLATES = [
    "{intent}",
    "Can you {intent}?",
    "Please {intent}.",
    "I do not need data retrieval; {intent}.",
    "No chart or catalog lookup needed: {intent}.",
    "General question: {intent}.",
]

GENERAL_ACTIONS = [
    "explain",
    "summarize",
    "rewrite",
    "outline",
    "compare",
    "debug",
    "review",
    "draft",
    "translate",
    "simplify",
]

GENERAL_TOPICS = [
    "a Python function",
    "a README section",
    "a product requirements note",
    "a git workflow",
    "an API error message",
    "a test failure",
    "a deployment checklist",
    "a short email",
    "a meeting agenda",
    "a database schema idea",
    "a shell command",
    "a code comment",
    "a naming convention",
    "an architecture decision",
    "a project timeline",
    "a pull request description",
    "a plain-language definition",
    "a user story",
    "a validation rule",
    "a troubleshooting guide",
]

GENERAL_CONTEXTS = [
    "for a teammate",
    "in two sentences",
    "with concrete examples",
    "for a beginner",
    "as a checklist",
    "without using a table",
    "in German",
    "in English",
    "for documentation",
    "for a quick chat reply",
]

RETRIEVAL_REQUEST_TEMPLATES = [
    "{question}",
]

OPENUI_REQUEST_TEMPLATES = [
    "{content}",
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


def extract_prompt_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    messages = sample.get("messages")
    if not isinstance(messages, list):
        return []
    normalized = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "assistant":
            break
        if role in {"system", "user"} and isinstance(content, str) and content.strip():
            normalized.append({"role": role, "content": content.strip()})
    return normalized


def split_tool_result_prompt(content: str) -> tuple[str, str] | None:
    for marker in TOOL_RESULT_MARKERS:
        if marker in content:
            question, tool_result = content.split(marker, 1)
            question = question.strip()
            tool_result = tool_result.strip()
            if question and tool_result:
                return question, tool_result
    return None


def one_line_prompt(text: str) -> str:
    return " ".join(text.strip().split())


def make_sample(
    *,
    sample_id: str,
    source: str,
    label: str,
    messages: list[dict[str, str]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "id": sample_id,
        "source": source,
        "label": label,
        "messages": messages,
    }
    if metadata:
        sample["metadata"] = metadata
    return sample


def add_unique_sample(
    samples: list[dict[str, Any]],
    seen: set[tuple[str, str]],
    sample: dict[str, Any],
) -> None:
    text = "\n".join(message["content"] for message in sample["messages"] if message.get("role") == "user")
    key = (sample["label"], one_line_prompt(text).casefold())
    if key in seen:
        return
    seen.add(key)
    samples.append(sample)


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
    seen: set[tuple[str, str]] = set()
    for split, source_path in OPENUI_SFT_SPLITS.items():
        for index, row in enumerate(read_jsonl(source_path), start=1):
            messages = extract_prompt_messages(row)
            if not messages:
                continue
            user_text = extract_user_prompt_from_messages(messages)
            if not split_tool_result_prompt(user_text):
                continue
            for variant_index, template in enumerate(OPENUI_REQUEST_TEMPLATES):
                variant_messages = [{"role": "user", "content": template.format(content=user_text)}]
                add_unique_sample(
                    samples,
                    seen,
                    make_sample(
                        sample_id=f"openui_sft_{split}_{index:05d}_{variant_index}",
                        source=str(source_path.relative_to(ROOT)),
                        label="openui_translator",
                        messages=variant_messages,
                        metadata={
                            "router_stage": "openui_translation",
                            "source_split": split,
                            **dict(row.get("metadata") or {}),
                        },
                    ),
                )
                if limit is not None and len(samples) >= limit:
                    return samples

    for split in SPLITS:
        for sample_path in sorted((OPENUI_DATA_DIR / split).glob("*.json")):
            if sample_path.name == "manifest.json":
                continue
            sample = read_json(sample_path)
            prompt = openui_prompt_from_sample(sample)
            if not prompt:
                continue
            add_unique_sample(
                samples,
                seen,
                make_sample(
                    sample_id=f"openui_{split}_{sample_path.stem}",
                    source=str(sample_path.relative_to(ROOT)),
                    label="openui_translator",
                    messages=[{"role": "user", "content": prompt}],
                    metadata={"router_stage": "openui_translation", "source_split": split},
                ),
            )
            if limit is not None and len(samples) >= limit:
                return samples
    return samples


def collect_ckan_samples(limit: int | None = None) -> list[dict[str, Any]]:
    samples = []
    seen: set[tuple[str, str]] = set()
    for split, source_path in RETRIEVAL_SFT_SPLITS.items():
        for index, row in enumerate(read_jsonl(source_path), start=1):
            messages = extract_prompt_messages(row)
            question = extract_user_prompt_from_messages(messages)
            if not question:
                continue
            for variant_index, template in enumerate(RETRIEVAL_REQUEST_TEMPLATES):
                prompt = template.format(question=question)
                add_unique_sample(
                    samples,
                    seen,
                    make_sample(
                        sample_id=f"retrieval_sft_{split}_{index:05d}_{variant_index}",
                        source=str(source_path.relative_to(ROOT)),
                        label="ckan_retrieval",
                        messages=[{"role": "user", "content": prompt}],
                        metadata={
                            "router_stage": "initial_retrieval",
                            "source_split": split,
                            **dict(row.get("metadata") or {}),
                        },
                    ),
                )
                if limit is not None and len(samples) >= limit:
                    return samples

    sources = [
        CKAN_DATA_DIR / "scenarios_train.jsonl",
        CKAN_DATA_DIR / "harvested_inventory_scenarios.jsonl",
        CKAN_DATA_DIR / "harvested_scenarios.jsonl",
        CKAN_DATA_DIR / "scenarios_200.jsonl",
        CKAN_DATA_DIR / "scenarios_eval_golden.jsonl",
        CKAN_DATA_DIR / "multitool_eval_golden.jsonl",
    ]
    for source in sources:
        for index, row in enumerate(read_jsonl(source)):
            prompt = ckan_prompt_from_row(row)
            if not prompt:
                continue
            add_unique_sample(
                samples,
                seen,
                make_sample(
                    sample_id=f"ckan_{source.stem}_{index:05d}",
                    source=str(source.relative_to(ROOT)),
                    label="ckan_retrieval",
                    messages=[{"role": "user", "content": prompt}],
                    metadata={"router_stage": "initial_retrieval", "source_kind": "ckan_policy"},
                ),
            )
            if limit is not None and len(samples) >= limit:
                return samples
    return samples


def collect_smalltalk_samples(count: int, rng: random.Random) -> list[dict[str, Any]]:
    samples = []
    seen: set[tuple[str, str]] = set()
    for intent in GENERAL_INTENTS:
        for template in GENERAL_TEMPLATES:
            text = template.format(intent=intent)
            add_unique_sample(
                samples,
                seen,
                make_sample(
                    sample_id=f"general_task_{len(samples):05d}",
                    source="synthetic_general",
                    label="general_agent",
                    messages=[{"role": "user", "content": text}],
                    metadata={"router_stage": "base_chat"},
                ),
            )
            if len(samples) >= count:
                return samples

    for action in GENERAL_ACTIONS:
        for topic in GENERAL_TOPICS:
            for context in GENERAL_CONTEXTS:
                text = f"{action.capitalize()} {topic} {context}."
                add_unique_sample(
                    samples,
                    seen,
                    make_sample(
                        sample_id=f"general_combo_{len(samples):05d}",
                        source="synthetic_general",
                        label="general_agent",
                        messages=[{"role": "user", "content": text}],
                        metadata={"router_stage": "base_chat"},
                    ),
                )
                if len(samples) >= count:
                    return samples

    attempts = 0
    while len(samples) < count and attempts < count * 50:
        attempts += 1
        prompt = rng.choice(SMALLTALK_PROMPTS)
        text = rng.choice(SMALLTALK_TEMPLATES).format(prompt=prompt)
        add_unique_sample(
            samples,
            seen,
            make_sample(
                sample_id=f"smalltalk_{len(samples):05d}",
                source="synthetic_smalltalk",
                label="general_agent",
                messages=[{"role": "user", "content": text}],
                metadata={"router_stage": "base_chat"},
            ),
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
        "stages": dict(
            sorted(Counter((sample.get("metadata") or {}).get("router_stage", "unknown") for sample in samples).items())
        ),
    }
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate balanced adapter-router classification data.")
    parser.add_argument("--per-label", type=int, default=1200, help="Maximum samples per label before splitting.")
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
        "description": (
            "Balanced router data from the actual adapter prompt surfaces: raw initial user requests for retrieval, "
            "question plus Tool result prompts for OpenUI translation, and broad base-chat negatives."
        ),
        "splits": {
            split: {
                "samples": len(rows),
                "labels": dict(sorted(Counter(row["label"] for row in rows).items())),
                "stages": dict(sorted(Counter((row.get("metadata") or {}).get("router_stage", "unknown") for row in rows).items())),
            }
            for split, rows in split_rows.items()
        },
    }
    (DATA_DIR / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
