from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parent / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from dataset import extract_retrieval_messages, load_json_samples, prompt_messages  # type: ignore


DEFAULT_MODEL = "openbmb/MiniCPM5-1B"
DEFAULT_ADAPTER_PATH = Path("outputs/tool-result-minicpm5-lora/checkpoint-200")
DEFAULT_DATA_PATH = DATA_DIR / "tool_result_test.jsonl"
TOOL_RESULT_MARKER = "Tool result:"


@dataclass(frozen=True)
class SampleResult:
    index: int
    question: str
    expected: Any
    raw_output: str
    cleaned_output: str
    parsed_output: Any | None
    valid_json: bool
    exact_match: bool
    no_marker: bool
    expected_keys_present: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "question": self.question,
            "expected": self.expected,
            "raw_output": self.raw_output,
            "cleaned_output": self.cleaned_output,
            "parsed_output": self.parsed_output,
            "valid_json": self.valid_json,
            "exact_match": self.exact_match,
            "no_marker": self.no_marker,
            "expected_keys_present": self.expected_keys_present,
        }


def add_bool_arg(parser: argparse.ArgumentParser, name: str, *, default: bool, help: str | None = None) -> None:
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def clean_generated_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        if text.startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return text[start : end + 1]
    return text


def expected_keys_present(expected: Any, parsed: Any | None) -> bool:
    if not isinstance(expected, dict) or not isinstance(parsed, dict):
        return expected == parsed
    return set(expected).issubset(set(parsed))


def summarize_results(results: list[SampleResult]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "samples": 0,
            "valid_json_rate": 0.0,
            "exact_match_rate": 0.0,
            "no_tool_result_marker_rate": 0.0,
            "expected_keys_present_rate": 0.0,
        }

    return {
        "samples": total,
        "valid_json_rate": sum(result.valid_json for result in results) / total,
        "exact_match_rate": sum(result.exact_match for result in results) / total,
        "no_tool_result_marker_rate": sum(result.no_marker for result in results) / total,
        "expected_keys_present_rate": sum(result.expected_keys_present for result in results) / total,
        "failed_indexes": [
            result.index
            for result in results
            if not (result.valid_json and result.no_marker and result.expected_keys_present)
        ],
    }


def build_quantization_config(load_in_4bit: bool, bf16: bool):
    if not load_in_4bit:
        return None

    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_model_and_tokenizer(args: argparse.Namespace):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = build_quantization_config(args.load_in_4bit, args.bf16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else "auto",
        device_map="auto" if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    return model, tokenizer


def generate_tool_result(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
) -> str:
    import torch

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    input_tokens = int(inputs["input_ids"].shape[-1])

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, input_tokens:], skip_special_tokens=True).strip()


def evaluate_samples(
    model: Any,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    *,
    max_samples: int | None,
    max_new_tokens: int,
) -> list[SampleResult]:
    selected = samples[:max_samples] if max_samples is not None else samples
    results: list[SampleResult] = []
    for index, sample in enumerate(selected, start=1):
        messages = extract_retrieval_messages(sample)
        prompt = prompt_messages(messages)
        question = next(message["content"] for message in messages if message["role"] == "user")
        expected = json.loads(messages[-1]["content"])

        raw_output = generate_tool_result(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        cleaned_output = clean_generated_json(raw_output)
        parsed_output = None
        valid_json = False
        try:
            parsed_output = json.loads(cleaned_output)
            valid_json = True
        except json.JSONDecodeError:
            pass

        results.append(
            SampleResult(
                index=index,
                question=question,
                expected=expected,
                raw_output=raw_output,
                cleaned_output=cleaned_output,
                parsed_output=parsed_output,
                valid_json=valid_json,
                exact_match=parsed_output == expected,
                no_marker=TOOL_RESULT_MARKER not in raw_output and TOOL_RESULT_MARKER not in cleaned_output,
                expected_keys_present=expected_keys_present(expected, parsed_output),
            )
        )
    return results


def write_report(path: Path, summary: dict[str, Any], results: list[SampleResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "results": [result.to_dict() for result in results],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sanity-check the MiniCPM tool-result LoRA adapter.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--report-path", type=Path, default=Path("train/retrieval/outputs/tool-result-sanity-report.json"))
    parser.add_argument("--show-failures", type=int, default=5)
    add_bool_arg(parser, "load-in-4bit", default=True)
    add_bool_arg(parser, "bf16", default=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    samples = load_json_samples(args.data_path)
    model, tokenizer = load_model_and_tokenizer(args)
    results = evaluate_samples(
        model,
        tokenizer,
        samples,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
    )
    summary = summarize_results(results)
    write_report(args.report_path, summary, results)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    failures = [
        result
        for result in results
        if not (result.valid_json and result.no_marker and result.expected_keys_present)
    ][: args.show_failures]
    if failures:
        print("\nFailures")
        for failure in failures:
            print(json.dumps(failure.to_dict(), ensure_ascii=False, indent=2))
    print(f"\nWrote report to {args.report_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
