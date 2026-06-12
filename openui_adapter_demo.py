#!/usr/bin/env python3
"""Interactive Gradio playground for the synthetic OpenUI SFT adapter."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "openbmb/MiniCPM5-1B"
DEFAULT_ADAPTER = Path("train/openui_lang/outputs/openui-sft-stats-components-lora")
DEFAULT_DATASET = Path("train/openui_lang/data/openui_sft_eval.jsonl")
SYSTEM_PROMPT = (
    "You generate raw OpenUI component code from a user query and a structured tool result. "
    "Use only the values from the tool result. Do not invent data. Return only the OpenUI "
    "component code, without explanations or markdown."
)

MODEL: Any | None = None
TOKENIZER: Any | None = None
ACTIVE_MODEL_KEY: tuple[str, str, bool] | None = None


@dataclass
class DemoExample:
    user_query: str
    tool_result: dict[str, Any]
    expected: str
    label: str


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch an interactive OpenUI adapter test app.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--max-new-tokens", type=int, default=1600)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    return parser


def parse_user_content(content: str) -> tuple[str, dict[str, Any]]:
    match = re.search(r"User query:\s*(.*?)\n\nTool result:\n(.*)\s*$", content, flags=re.DOTALL)
    if not match:
        raise ValueError("Sample user content does not match the generated dataset format.")
    user_query = match.group(1).strip()
    tool_result = json.loads(match.group(2))
    return user_query, tool_result


def load_examples(path: Path, limit: int = 12) -> list[DemoExample]:
    examples = []
    if not path.exists():
        return examples
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row["messages"]
            user_query, tool_result = parse_user_content(messages[1]["content"])
            metadata = row.get("metadata", {})
            label = " / ".join(
                str(part)
                for part in [
                    metadata.get("domain", "unknown"),
                    metadata.get("data_shape", "unknown"),
                    metadata.get("component", "unknown"),
                ]
            )
            examples.append(DemoExample(user_query, tool_result, messages[2]["content"], label))
            if len(examples) >= limit:
                break
    return examples


def make_user_message(user_query: str, tool_result_text: str) -> str:
    parsed = json.loads(tool_result_text)
    return "User query: " + user_query.strip() + "\n\nTool result:\n" + json.dumps(parsed, ensure_ascii=False, indent=2)


def clean_component_output(output: str) -> str:
    output = output.strip()
    fence = re.search(r"```(?:jsx|xml|openui|text)?\s*(.*?)```", output, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        output = fence.group(1).strip()
    tag = re.search(r"<[A-Za-z][A-Za-z0-9]*(?:\s|>|/)", output)
    if tag and tag.start() > 0:
        output = output[tag.start() :].strip()
    return output


def ensure_adapter_ready(adapter: Path) -> None:
    if not adapter.exists():
        raise FileNotFoundError(
            f"Adapter directory does not exist yet: {adapter}. Wait for training to create it, or pass --adapter."
        )
    if not (adapter / "adapter_config.json").exists():
        checkpoints = sorted(adapter.glob("checkpoint-*/adapter_config.json"))
        if checkpoints:
            return
        raise FileNotFoundError(
            f"No adapter_config.json found in {adapter}. Wait until a checkpoint is saved, or pass a checkpoint path."
        )


def resolve_adapter_path(adapter: Path) -> Path:
    if (adapter / "adapter_config.json").exists():
        return adapter
    checkpoints = sorted(
        [path.parent for path in adapter.glob("checkpoint-*/adapter_config.json")],
        key=lambda path: int(re.search(r"checkpoint-(\d+)$", path.name).group(1)) if re.search(r"checkpoint-(\d+)$", path.name) else -1,
    )
    if checkpoints:
        return checkpoints[-1]
    return adapter


def load_model_once(model_name: str, adapter: Path, load_in_4bit: bool) -> tuple[Any, Any]:
    global ACTIVE_MODEL_KEY, MODEL, TOKENIZER

    adapter = resolve_adapter_path(adapter)
    key = (model_name, str(adapter.resolve()), load_in_4bit)
    if MODEL is not None and TOKENIZER is not None and ACTIVE_MODEL_KEY == key:
        return MODEL, TOKENIZER

    ensure_adapter_ready(adapter)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(str(adapter) if (adapter / "tokenizer_config.json").exists() else model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if load_in_4bit else "auto",
        device_map="auto",
        quantization_config=quantization_config,
    )
    model = PeftModel.from_pretrained(base_model, adapter)
    model.eval()

    MODEL = model
    TOKENIZER = tokenizer
    ACTIVE_MODEL_KEY = key
    return model, tokenizer


def generate_component(
    user_query: str,
    tool_result_text: str,
    model_name: str,
    adapter: Path,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    load_in_4bit: bool,
) -> tuple[str, str, str]:
    try:
        user_content = make_user_message(user_query, tool_result_text)
    except Exception as exc:
        message = f"Invalid tool result JSON: {exc}"
        return "", render_error(message), message

    try:
        model, tokenizer = load_model_once(model_name, adapter, load_in_4bit)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
        input_len = inputs["input_ids"].shape[-1]
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        import torch

        with torch.no_grad():
            generated = model.generate(**inputs, **generation_kwargs)
        output = clean_component_output(tokenizer.decode(generated[0, input_len:], skip_special_tokens=True))
        return output, render_component_preview(output), "OK"
    except Exception as exc:
        message = str(exc)
        return "", render_error(message), message


def extract_prop(component: str, name: str) -> str | None:
    patterns = [
        rf'{name}\s*=\s*"([^"]*)"',
        rf"{name}\s*=\s*'([^']*)'",
        rf"{name}\s*=\s*\{{([^{{}}]+)\}}",
    ]
    for pattern in patterns:
        match = re.search(pattern, component, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def extract_component_name(component: str) -> str:
    match = re.search(r"<([A-Za-z][A-Za-z0-9]*)", component)
    return match.group(1) if match else "OpenUI"


def extract_stat_cards(component: str) -> list[dict[str, str]]:
    cards = []
    for match in re.finditer(r"<StatCard\b(.*?)/>", component, flags=re.DOTALL):
        raw = match.group(0)
        cards.append(
            {
                "title": extract_prop(raw, "title") or "Stat",
                "value": extract_prop(raw, "value") or "",
                "unit": extract_prop(raw, "unit") or "",
            }
        )
    return cards


def extract_data_rows(component: str, limit: int = 10) -> list[tuple[str, str]]:
    rows = []
    for label_key in ["label", "district", "month", "date", "office"]:
        pattern = rf'{label_key}\s*:\s*"([^"]+)".*?value\s*:\s*(-?\d+(?:\.\d+)?)'
        for label, value in re.findall(pattern, component, flags=re.DOTALL):
            rows.append((label, value))
            if len(rows) >= limit:
                return rows
    return rows


def render_component_preview(component: str) -> str:
    if not component.strip():
        return render_error("No model output.")

    component_name = extract_component_name(component)
    title = extract_prop(component, "title") or component_name
    value = extract_prop(component, "value")
    unit = extract_prop(component, "unit") or ""
    severity = extract_prop(component, "severity")
    stat_cards = extract_stat_cards(component)
    rows = extract_data_rows(component)

    if component_name == "DashboardGrid" and stat_cards:
        body = "".join(
            f"""
            <div class="stat">
              <span>{escape(card["title"])}</span>
              <strong>{escape(card["value"])}</strong>
              <small>{escape(card["unit"])}</small>
            </div>
            """
            for card in stat_cards
        )
        return f'<section class="preview"><h2>{escape(title)}</h2><div class="grid">{body}</div></section>'

    if component_name in {"LineChartCard", "BarChartCard", "HorizontalBarChartCard", "DistrictMapCard", "DistrictBarChartCard"} and rows:
        max_value = max(abs(float(value)) for _, value in rows) or 1.0
        bars = []
        for label, raw_value in rows:
            value_number = float(raw_value)
            width = max(2.0, abs(value_number) / max_value * 100.0)
            bars.append(
                f"""
                <div class="bar-row">
                  <span>{escape(label)}</span>
                  <div><i style="width:{width:.1f}%"></i></div>
                  <b>{escape(raw_value)} {escape(unit)}</b>
                </div>
                """
            )
        return f'<section class="preview"><h2>{escape(title)}</h2><div class="bars">{"".join(bars)}</div></section>'

    if component_name == "AlertCard":
        tone = "danger" if severity == "danger" else "warning" if severity == "warning" else "success"
        threshold = extract_prop(component, "threshold")
        description = extract_prop(component, "description") or ""
        return (
            f'<section class="preview alert {tone}"><h2>{escape(title)}</h2>'
            f'<p><strong>{escape(value)}</strong> {escape(unit)} / Grenzwert {escape(threshold)}</p>'
            f'<span>{escape(description)}</span></section>'
        )

    if component_name == "ProgressCard":
        number = float(value or 0)
        width = max(0.0, min(100.0, number))
        description = extract_prop(component, "description") or ""
        return (
            f'<section class="preview"><h2>{escape(title)}</h2><div class="progress"><i style="width:{width:.1f}%"></i></div>'
            f'<p><strong>{escape(value)}</strong>{escape(unit)} {escape(description)}</p></section>'
        )

    if component_name == "TableCard":
        return f'<section class="preview"><h2>{escape(title)}</h2><p>TableCard generated. Inspect raw component code for columns and rows.</p></section>'

    return (
        f'<section class="preview"><h2>{escape(title)}</h2>'
        f'<p><strong>{escape(value)}</strong> {escape(unit)}</p>'
        f'<small>{escape(component_name)}</small></section>'
    )


def render_error(message: str) -> str:
    return f'<section class="preview error"><h2>Error</h2><pre>{escape(message)}</pre></section>'


def escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def sample_to_gradio(example: DemoExample) -> list[str]:
    return [
        example.user_query,
        json.dumps(example.tool_result, ensure_ascii=False, indent=2),
        example.expected,
        render_component_preview(example.expected),
        example.label,
    ]


def build_app(args: argparse.Namespace):
    import gradio as gr

    examples = load_examples(args.dataset)
    initial = examples[0] if examples else DemoExample(
        "Wie hoch war Sonnenstunden in München im Jahr 2022?",
        {
            "domain": "weather",
            "metric": "Sonnenstunden",
            "location": "München",
            "year": 2022,
            "aggregation": "sum",
            "value": 2000,
            "unit": "h",
        },
        "",
        "manual",
    )

    css = """
    .preview { border: 1px solid #d7dde8; border-radius: 8px; padding: 14px; background: #fff; color: #111827; }
    .preview h2 { margin: 0 0 10px; font-size: 18px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; }
    .stat { border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; display: grid; gap: 3px; }
    .stat span, .stat small { color: #4b5563; font-size: 12px; }
    .stat strong { font-size: 20px; }
    .bar-row { display: grid; grid-template-columns: minmax(72px, 150px) 1fr minmax(72px, 120px); gap: 8px; align-items: center; margin: 8px 0; }
    .bar-row span, .bar-row b { font-size: 12px; overflow-wrap: anywhere; }
    .bar-row div, .progress { height: 14px; background: #e5e7eb; border-radius: 999px; overflow: hidden; }
    .bar-row i, .progress i { display: block; height: 100%; background: #2563eb; border-radius: 999px; }
    .alert.warning { border-color: #f59e0b; background: #fffbeb; }
    .alert.danger { border-color: #ef4444; background: #fef2f2; }
    .alert.success { border-color: #10b981; background: #ecfdf5; }
    .error { border-color: #ef4444; background: #fef2f2; }
    .error pre { white-space: pre-wrap; }
    """

    with gr.Blocks(title="OpenUI Adapter Playground") as demo:
        gr.Markdown("# OpenUI Adapter Playground")
        gr.Markdown(f"Adapter: `{args.adapter}`")
        with gr.Row():
            with gr.Column(scale=1):
                query = gr.Textbox(label="User query", value=initial.user_query, lines=3)
                tool_result = gr.Code(
                    label="Tool result JSON",
                    value=json.dumps(initial.tool_result, ensure_ascii=False, indent=2),
                    language="json",
                    lines=18,
                )
                with gr.Row():
                    max_new_tokens = gr.Slider(128, 3000, value=args.max_new_tokens, step=64, label="Max new tokens")
                    temperature = gr.Slider(0.0, 1.0, value=args.temperature, step=0.05, label="Temperature")
                generate = gr.Button("Generate", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=1):
                output = gr.Code(label="Model OpenUI component code", language="javascript", lines=18)
                preview = gr.HTML(label="Preview")

        if examples:
            gr.Examples(
                examples=[sample_to_gradio(example) for example in examples],
                inputs=[query, tool_result, output, preview, status],
                label="Eval examples",
            )

        generate.click(
            fn=lambda user_query, tool_json, max_tokens, temp: generate_component(
                user_query=user_query,
                tool_result_text=tool_json,
                model_name=args.model_name,
                adapter=args.adapter,
                max_new_tokens=int(max_tokens),
                temperature=float(temp),
                top_p=args.top_p,
                load_in_4bit=args.load_in_4bit,
            ),
            inputs=[query, tool_result, max_new_tokens, temperature],
            outputs=[output, preview, status],
        )

    return demo, css


def main() -> int:
    args = build_arg_parser().parse_args()
    app, css = build_app(args)
    app.launch(server_name=args.server_name, server_port=args.server_port, share=args.share, css=css)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
