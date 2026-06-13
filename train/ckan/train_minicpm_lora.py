from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "openbmb/MiniCPM5-1B"
DEFAULT_TRAIN_DATA = "train/ckan/data/generated/valid_examples_multitool_train_1600_repaired.jsonl"
DEFAULT_EVAL_DATA = "train/ckan/data/generated/valid_examples_multitool_eval_160.jsonl"
DEFAULT_OUTPUT_DIR = "train/ckan/outputs/smolnalysis-ckan-retrieval-minicpm5-lora"
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
PROTOCOL_SYSTEM_PROMPT = """You are the CKAN retrieval policy for smolnalysis. Emit strict JSON only.
Output exactly one JSON object with keys: thought, action, args, confidence.
Do not output <think> tags. Do not output markdown. Do not output prose before or after JSON.
The thought field is a short decision summary, not chain-of-thought.
Allowed actions: tag_search, group_list, organization_list, package_search, package_show, select_resource, finish, ask_clarification."""
TRAIN_CHAT_TEMPLATE = (
    "{{- bos_token }}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'user' %}"
    "{{- '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- generation %}"
    "{{- message['content'] + '<|im_end|>' }}"
    "{%- endgeneration %}"
    "{{- '\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized = [dict(message) for message in messages]
    if normalized and normalized[0].get("role") == "system":
        normalized[0]["content"] = PROTOCOL_SYSTEM_PROMPT
    else:
        normalized.insert(0, {"role": "system", "content": PROTOCOL_SYSTEM_PROMPT})
    return normalized


def prepare_dataset(path: str | Path, limit: int | None = None):
    from datasets import Dataset

    rows = load_jsonl(path, limit)
    return Dataset.from_list([{"messages": normalize_messages(row["messages"])} for row in rows])


def build_lora_config(args: argparse.Namespace):
    from peft import LoraConfig

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules.split(",") if isinstance(args.target_modules, str) else args.target_modules,
        task_type="CAUSAL_LM",
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    original_chat_template = tokenizer.chat_template
    tokenizer.chat_template = TRAIN_CHAT_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model.config.use_cache = False

    train_dataset = prepare_dataset(args.train_data, args.train_limit)
    eval_dataset = prepare_dataset(args.eval_data, args.eval_limit)
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / max(1, args.per_device_train_batch_size * args.gradient_accumulation_steps)))

    config = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_length,
        packing=args.packing,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=max(1, min(args.eval_steps, steps_per_epoch)),
        save_strategy="steps",
        save_steps=max(1, min(args.save_steps, steps_per_epoch)),
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        assistant_only_loss=args.assistant_only_loss,
        report_to=args.report_to,
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=build_lora_config(args),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.chat_template = original_chat_template
    tokenizer.save_pretrained(args.output_dir)

    metrics = trainer.evaluate()
    metrics_path = Path(args.output_dir) / "eval_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the CKAN retrieval MiniCPM5 LoRA adapter.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--train-data", default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--eval-data", default=DEFAULT_EVAL_DATA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-limit", type=int, help="Use a subset for smoke tests.")
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-train-epochs", type=float, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default=",".join(DEFAULT_TARGET_MODULES))
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--assistant-only-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    metrics = train(args)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
