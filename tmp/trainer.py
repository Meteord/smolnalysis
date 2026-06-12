import argparse
import os
import random

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from training.data.dataset import load_tool_examples
except ModuleNotFoundError:
    from data.dataset import load_tool_examples


TEXT_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def get_input_device(model):
    return next(model.parameters()).device


def discover_text_targets(model):
    targets = []

    for name, module in model.named_modules():
        lname = name.lower()

        if "vision" in lname or "audio" in lname:
            continue

        if name.endswith(TEXT_SUFFIXES) and isinstance(module, nn.Linear):
            targets.append(name)

    return targets


def make_chat_features(tokenizer, user_text, assistant_text, max_length):
    prompt_messages = [{"role": "user", "content": user_text}]
    full_messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    if tokenizer.eos_token is not None and not full_text.endswith(tokenizer.eos_token):
        full_text += tokenizer.eos_token

    prompt_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"][0]
    full_tokens = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )

    input_ids = full_tokens["input_ids"][0]
    attention_mask = full_tokens["attention_mask"][0]
    labels = input_ids.clone()
    labels[: min(prompt_ids.shape[0], labels.shape[0])] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def collate(features, tokenizer, device):
    return {
        "input_ids": pad_sequence(
            [f["input_ids"] for f in features],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        ).to(device),
        "attention_mask": pad_sequence(
            [f["attention_mask"] for f in features],
            batch_first=True,
            padding_value=0,
        ).to(device),
        "labels": pad_sequence(
            [f["labels"] for f in features],
            batch_first=True,
            padding_value=-100,
        ).to(device),
    }


def generate_reply(model, tokenizer, user_text, max_new_tokens=128):
    model.eval()
    device = get_input_device(model)
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[-1]

    stop_ids = [tokenizer.eos_token_id]
    turn_end_id = tokenizer.convert_tokens_to_ids("<turn|>")
    if isinstance(turn_end_id, int) and turn_end_id >= 0:
        stop_ids.append(turn_end_id)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=False).strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/gemma-4-E4B-it")
    parser.add_argument("--data-path", default="training/data/generated/llm_user_queries.jsonl")
    parser.add_argument("--output-dir", default="models/gemma4-tool-lora-adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("CUDA available:", torch.cuda.is_available())

    raw_examples = load_tool_examples(args.data_path)
    random.shuffle(raw_examples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    #model = prepare_model_for_kbit_training(model)

    target_modules = discover_text_targets(model)
    if not target_modules:
        raise RuntimeError("No LoRA target modules found.")

    print(f"Found {len(target_modules)} LoRA target modules.")
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        ),
    )
    model.print_trainable_parameters()

    features = [
        make_chat_features(
            tokenizer,
            example["user"],
            example["assistant"],
            args.max_length,
        )
        for example in raw_examples
    ]

    print("\nFirst training example:")
    print("User:", raw_examples[0]["user"])
    print("Assistant:", raw_examples[0]["assistant"])

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    device = get_input_device(model)
    global_step = 0

    print("\nInitial generation:")
    print(generate_reply(model, tokenizer, raw_examples[0]["user"]))

    for epoch in range(args.epochs):
        model.train()

        for start in range(0, len(features), args.batch_size):
            batch_features = features[start : start + args.batch_size]
            batch = collate(batch_features, tokenizer, device)

            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()

            global_step += 1
            print(f"epoch={epoch + 1} step={global_step} loss={loss.item():.4f}")

            if args.eval_every and global_step % args.eval_every == 0:
                print("\n--- eval generation ---")
                print("Prompt:", raw_examples[0]["user"])
                print(generate_reply(model, tokenizer, raw_examples[0]["user"]))
                print("-----------------------\n")

            if args.save_every and global_step % args.save_every == 0:
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                print(f"Saved checkpoint to {ckpt_dir}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved final LoRA adapter to: {args.output_dir}")

    print("\nFinal generation:")
    print(generate_reply(model, tokenizer, raw_examples[0]["user"]))


if __name__ == "__main__":
    main()
