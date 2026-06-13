from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


CKAN_DIR = Path(__file__).resolve().parent
if str(CKAN_DIR) not in sys.path:
    sys.path.insert(0, str(CKAN_DIR))

from ckan_dataset_tools import extract_context_from_example, parse_action, read_jsonl, validate_ckan_action, write_jsonl


DEFAULT_MODEL = "openbmb/MiniCPM5-1B"
DEFAULT_ADAPTER = "/outputs/smolnalysis-ckan-retrieval-minicpm5-lora"
DEFAULT_EVAL_DATA = "train/ckan/data/generated/valid_examples_multitool_eval_160.jsonl"
DEFAULT_OUTPUT_DIR = "train/ckan/outputs/eval"
INFERENCE_SYSTEM_PROMPT = """You are the CKAN retrieval policy for smolnalysis. Emit strict JSON only.
Allowed actions: tag_search, group_list, organization_list, package_search, package_show, select_resource, finish, ask_clarification.
Use exact args schemas:
- tag_search: {"query":"string","rows":10}
- group_list: {"rows":15}
- organization_list: {"rows":15}
- package_search: {"query":"string","rows":5,"start":0}
- package_show: {"package_id":"observed-package-id"}
- select_resource: {"package_id":"observed-package-id","resource_id":"observed-resource-id","match_evidence":"why it fits"}
- finish: {"selected_candidates":[{"package_id":"observed-package-id","resource_id":"observed-resource-id"}],"rationale":"why retrieval is complete"}
- ask_clarification: {"question":"short question for the user","reason":"why the request is ambiguous"}
Use tag/group/organization discovery when catalog vocabulary is unclear. After empty or weak results, choose a different real tool or a refined package_search query. Do not invent action names. Do not output markdown."""


def prompt_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    messages = example.get("messages", [])
    return [message for message in messages if isinstance(message, dict) and message.get("role") != "assistant"]


def strict_prompt_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    messages = prompt_messages(example)
    if messages and messages[0].get("role") == "system":
        return [{"role": "system", "content": INFERENCE_SYSTEM_PROMPT}, *messages[1:]]
    return [{"role": "system", "content": INFERENCE_SYSTEM_PROMPT}, *messages]


def expected_action(example: dict[str, Any]) -> dict[str, Any] | None:
    messages = example.get("messages", [])
    assistant_messages = [message for message in messages if isinstance(message, dict) and message.get("role") == "assistant"]
    if not assistant_messages:
        return None
    content = assistant_messages[-1].get("content")
    if not isinstance(content, str):
        return None
    payload, _issues = parse_action(content)
    return payload


def normalize_prediction(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end >= start:
        return stripped[start : end + 1]
    return stripped


def generate_prediction(model: Any, tokenizer: Any, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    examples = read_jsonl(Path(args.eval_data))
    if args.limit is not None:
        examples = examples[: args.limit]

    predictions = []
    issue_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    exact_action_matches = 0
    valid_outputs = 0
    parsed_outputs = 0

    for index, example in enumerate(examples, start=1):
        messages = strict_prompt_messages(example) if args.strict_prompt else prompt_messages(example)
        raw_prediction = generate_prediction(model, tokenizer, messages, args.max_new_tokens)
        normalized = normalize_prediction(raw_prediction)
        expected = expected_action(example)
        parsed, parse_issues = parse_action(normalized)
        validation = validate_ckan_action(normalized, extract_context_from_example(example))
        for issue in [*parse_issues, *validation.issues]:
            issue_counts[issue.code] += 1
        if parsed is not None:
            parsed_outputs += 1
            action = parsed.get("action")
            if isinstance(action, str):
                action_counts[action] += 1
            if expected is not None and parsed.get("action") == expected.get("action"):
                exact_action_matches += 1
        if validation.ok:
            valid_outputs += 1
        predictions.append(
            {
                "line": index,
                "scenario_id": example.get("metadata", {}).get("scenario_id"),
                "expected": expected,
                "raw_prediction": raw_prediction,
                "normalized_prediction": normalized,
                "parsed": parsed,
                "ok": validation.ok,
                "issues": [issue.__dict__ for issue in validation.issues],
            }
        )
        if not args.quiet:
            print(f"[{index}/{len(examples)}] ok={validation.ok}", flush=True)

    total = len(examples)
    summary = {
        "total": total,
        "json_parse_rate": parsed_outputs / total if total else 0,
        "valid_action_rate": valid_outputs / total if total else 0,
        "exact_action_match_rate": exact_action_matches / total if total else 0,
        "exact_action_matches": exact_action_matches,
        "issue_counts": dict(issue_counts),
        "predicted_action_counts": dict(action_counts),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "eval_predictions.jsonl", predictions)
    (output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the CKAN retrieval LoRA adapter on golden examples.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER)
    parser.add_argument("--eval-data", default=DEFAULT_EVAL_DATA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--strict-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = evaluate(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
