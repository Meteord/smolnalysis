from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from dataset import (  # type: ignore
    ToolResultDataCollator,
    ToolResultTrainingDataset,
    apply_chat_template,
    extract_retrieval_messages,
    prompt_messages,
)


DEFAULT_MODEL = "openbmb/MiniCPM5-1B"
DEFAULT_TRAIN_DATA = DATA_DIR / "tool_result_train.jsonl"
DEFAULT_EVAL_DATA = DATA_DIR / "tool_result_eval.jsonl"
DEFAULT_TEST_DATA = DATA_DIR / "tool_result_test.jsonl"
DEFAULT_OUTPUT_DIR = Path("train/retrieval/outputs/tool-result-minicpm5-lora")
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def add_bool_arg(parser: argparse.ArgumentParser, name: str, *, default: bool, help: str | None = None) -> None:
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the MiniCPM5 LoRA adapter for question-to-tool-result JSON.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--eval-data", type=Path, default=DEFAULT_EVAL_DATA)
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-limit", type=int, help="Use a subset for smoke tests.")
    parser.add_argument("--eval-limit", type=int, help="Use a subset for smoke tests.")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--generation-eval-samples", type=int, default=0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default=",".join(DEFAULT_TARGET_MODULES))
    add_bool_arg(parser, "load-in-4bit", default=True)
    add_bool_arg(parser, "bf16", default=True)
    add_bool_arg(parser, "fp16", default=False)
    add_bool_arg(parser, "gradient-checkpointing", default=True)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Load/tokenize data and print a sample without loading the model.")
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


def limit_dataset(dataset: ToolResultTrainingDataset, limit: int | None) -> ToolResultTrainingDataset:
    if limit is not None:
        dataset.samples = dataset.samples[:limit]
    return dataset


def target_modules(args: argparse.Namespace) -> list[str]:
    if isinstance(args.target_modules, str):
        return [module.strip() for module in args.target_modules.split(",") if module.strip()]
    return list(args.target_modules)


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_lora_config(args: argparse.Namespace):
    from peft import LoraConfig

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules(args),
        task_type="CAUSAL_LM",
        bias="none",
    )


def build_quantization_config(args: argparse.Namespace):
    if not args.load_in_4bit:
        return None

    import torch
    from transformers import BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def load_model(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM

    quantization_config = build_quantization_config(args)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else "auto",
        device_map="auto" if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if quantization_config is not None:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    return model


def build_datasets(args: argparse.Namespace, tokenizer: Any) -> tuple[ToolResultTrainingDataset, ToolResultTrainingDataset]:
    for name, data_path in (("train", args.train_data), ("eval", args.eval_data)):
        if data_path.suffix != ".jsonl":
            raise ValueError(
                f"Retrieval {name} data must be a JSONL file. Use "
                f"{DEFAULT_TRAIN_DATA if name == 'train' else DEFAULT_EVAL_DATA}."
            )

    train_dataset = ToolResultTrainingDataset(
        args.train_data,
        tokenizer,
        max_length=args.max_length,
        return_tensors="pt",
    )
    eval_dataset = ToolResultTrainingDataset(
        args.eval_data,
        tokenizer,
        max_length=args.max_length,
        return_tensors="pt",
    )
    return limit_dataset(train_dataset, args.train_limit), limit_dataset(eval_dataset, args.eval_limit)


def debug_supervised_text(tokenizer: Any, dataset: ToolResultTrainingDataset, index: int = 0, window: int = 120) -> None:
    sample = dataset.samples[index]
    messages = extract_retrieval_messages(sample)

    prompt_text = apply_chat_template(
        tokenizer,
        prompt_messages(messages),
        add_generation_prompt=True,
    )
    full_text = apply_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=False,
    )

    item = dataset[index]
    labels = item["labels"]
    input_ids = item["input_ids"]

    if hasattr(labels, "tolist"):
        labels = labels.tolist()
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()

    supervised_indices = [i for i, x in enumerate(labels) if x != -100]
    first = supervised_indices[0]
    last = supervised_indices[-1]

    print("=" * 100)
    print("RETRIEVAL TOKENIZATION DEBUG")
    print("=" * 100)
    print("prompt_text tail:")
    print(repr(prompt_text[-500:]))
    print("full_text tail:")
    print(repr(full_text[-500:]))
    print("len(dataset input_ids):", len(input_ids))
    print("num masked tokens:", sum(1 for x in labels if x == -100))
    print("num supervised tokens:", len(supervised_indices))
    print("first supervised index:", first)
    print("last supervised index:", last)
    print("decoded supervised start:")
    print(tokenizer.decode(input_ids[first:first + window], skip_special_tokens=False))
    print("raw assistant JSON label:")
    print(messages[-1]["content"])


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = load_tokenizer(args.model_name)
    train_dataset, eval_dataset = build_datasets(args, tokenizer)
    first = train_dataset[0]
    supervised_tokens = int((first["labels"] != -100).sum().item())
    masked_tokens = int((first["labels"] == -100).sum().item())

    debug_supervised_text(tokenizer, train_dataset)
    summary = {
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "input_ids_len": int(first["input_ids"].shape[0]),
        "attention_mask_len": int(first["attention_mask"].shape[0]),
        "labels_len": int(first["labels"].shape[0]),
        "masked_prompt_tokens": masked_tokens,
        "supervised_assistant_tokens": supervised_tokens,
    }
    print(json.dumps(summary, indent=2))
    return summary


def _clean_generated_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return text[start : end + 1]
    return text


def evaluate_json_generation(
    model: Any,
    tokenizer: Any,
    dataset: ToolResultTrainingDataset,
    *,
    max_samples: int,
    max_new_tokens: int = 768,
) -> dict[str, Any]:
    if max_samples <= 0:
        return {"generation_eval_samples": 0}

    import torch

    total = min(max_samples, len(dataset.samples))
    valid_json = 0
    exact_match = 0
    no_marker = 0
    model.eval()

    for sample in dataset.samples[:total]:
        messages = prompt_messages(extract_retrieval_messages(sample))
        expected = json.loads(sample["messages"][-1]["content"])
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
        generated = tokenizer.decode(output[0, input_tokens:], skip_special_tokens=True)
        cleaned = _clean_generated_json(generated)
        if "Tool result:" not in cleaned:
            no_marker += 1
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            continue
        valid_json += 1
        if parsed == expected:
            exact_match += 1

    return {
        "generation_eval_samples": total,
        "valid_json_rate": valid_json / total if total else 0.0,
        "exact_match_rate": exact_match / total if total else 0.0,
        "no_tool_result_marker_rate": no_marker / total if total else 0.0,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from peft import get_peft_model
    from transformers import Trainer, TrainingArguments

    tokenizer = load_tokenizer(args.model_name)
    train_dataset, eval_dataset = build_datasets(args, tokenizer)
    debug_supervised_text(tokenizer, train_dataset)

    model = load_model(args)
    model = get_peft_model(model, build_lora_config(args))
    model.print_trainable_parameters()

    steps_per_epoch = max(
        1,
        math.ceil(len(train_dataset) / max(1, args.per_device_train_batch_size * args.gradient_accumulation_steps)),
    )
    eval_steps = max(1, min(args.eval_steps, steps_per_epoch))
    save_steps = max(1, min(args.save_steps, steps_per_epoch))

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16 and torch.cuda.is_available(),
        fp16=args.fp16 and torch.cuda.is_available(),
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=[] if args.report_to == "none" else args.report_to.split(","),
        seed=args.seed,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=ToolResultDataCollator(tokenizer),
    )
    train_result = trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    eval_metrics = trainer.evaluate()
    generation_metrics = evaluate_json_generation(
        model,
        tokenizer,
        eval_dataset,
        max_samples=args.generation_eval_samples,
    )

    metrics = dict(eval_metrics)
    metrics.update(train_result.metrics)
    metrics.update(generation_metrics)
    metrics_path = args.output_dir / "metrics.json"
    eval_metrics_path = args.output_dir / "eval_metrics.json"
    generation_metrics_path = args.output_dir / "generation_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    eval_metrics_path.write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")
    generation_metrics_path.write_text(json.dumps(generation_metrics, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Eval metrics")
    print(json.dumps(eval_metrics, indent=2))
    print("Generation metrics")
    print(json.dumps(generation_metrics, indent=2))
    print("Combined metrics")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> int:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.dry_run:
        dry_run(args)
    else:
        train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
