from __future__ import annotations

import argparse
import copy
import html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_MODEL = "openbmb/MiniCPM5-1B"
DEFAULT_ADAPTER_ROOT = Path("train/openui_lang/outputs/smolnalysis-openui-minicpm5-lora")
DEFAULT_REPORT = Path("train/openui_lang/outputs/adapter_demo.html")

if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from dataset import extract_openui_messages  # type: ignore


@dataclass
class DemoResult:
    name: str
    adapter_path: str
    output: str
    rendered_html: str
    error: str | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one OpenUI test sample through one or more LoRA adapters.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", action="append", default=[], help="Adapter path. Repeat for multiple adapters.")
    parser.add_argument("--adapter-root", type=Path, default=DEFAULT_ADAPTER_ROOT, help="Used when --adapter is omitted.")
    parser.add_argument("--include-base", action="store_true", help="Also generate with the base model without an adapter.")
    parser.add_argument("--sample-file", type=Path, help="Specific JSON sample file to use.")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-new-tokens", type=int, default=1800)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--question",
        help="Override the sample user_question in memory. The JSON sample file is not modified.",
    )
    parser.add_argument("--load-in-4bit", action="store_true", help="Use bitsandbytes 4-bit loading.")
    parser.add_argument("--no-open", action="store_true", help="Do not print a browser-open hint.")
    return parser


def discover_adapters(adapter_root: Path) -> list[Path]:
    if not adapter_root.exists():
        return []
    candidates = []
    if (adapter_root / "adapter_config.json").exists():
        candidates.append(adapter_root)
    candidates.extend(
        path
        for path in sorted(adapter_root.glob("checkpoint-*"), key=_checkpoint_sort_key)
        if (path / "adapter_config.json").exists()
    )
    return candidates


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return (int(match.group(1)) if match else -1, path.name)


def select_sample(sample_file: Path | None, split: str, sample_index: int) -> tuple[Path, dict[str, Any]]:
    if sample_file is None:
        split_dir = DATA_DIR / split
        files = sorted(path for path in split_dir.glob("*.json") if path.name != "manifest.json")
        if not files:
            raise FileNotFoundError(f"No JSON samples found in {split_dir}")
        if sample_index < 0 or sample_index >= len(files):
            raise IndexError(f"sample_index {sample_index} is outside 0..{len(files) - 1}")
        sample_file = files[sample_index]

    with sample_file.open(encoding="utf-8") as file:
        return sample_file, json.load(file)


def apply_question_override(sample: dict[str, Any], question: str | None) -> dict[str, Any]:
    if question is None:
        return normalize_message_content(sample)

    sample = copy.deepcopy(sample)
    if isinstance(sample.get("user_question"), str):
        sample["user_question"] = question

    messages = sample.get("messages")
    if isinstance(messages, list):
        user_indexes = [index for index, message in enumerate(messages) if message.get("role") == "user"]
        if not user_indexes:
            raise ValueError("Sample messages do not contain a user message.")

        user_message = messages[user_indexes[-1]]
        content = user_message.get("content")
        if isinstance(content, str):
            user_message["content"] = replace_user_question_in_content(content, question)
        elif isinstance(content, dict):
            content["user_question"] = question
        else:
            raise ValueError(f"Unsupported user message content type: {type(content).__name__}")
    return normalize_message_content(sample)


def normalize_message_content(sample: dict[str, Any]) -> dict[str, Any]:
    messages = sample.get("messages")
    if not isinstance(messages, list):
        return sample

    sample = copy.deepcopy(sample)
    for message in sample["messages"]:
        content = message.get("content")
        if isinstance(content, dict):
            message["content"] = json.dumps(content, ensure_ascii=False, indent=2)
    return sample


def replace_user_question_in_content(content: str, question: str) -> str:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return question

    if isinstance(payload, dict):
        payload["user_question"] = question
        return json.dumps(payload, ensure_ascii=False, indent=2)
    return question


def prompt_messages_for_generation(sample: dict[str, Any]) -> list[dict[str, str]]:
    messages = extract_openui_messages(sample)
    assistant_indexes = [index for index, message in enumerate(messages) if message["role"] == "assistant"]
    if not assistant_indexes:
        raise ValueError("Sample messages do not contain an assistant target.")
    return messages[: assistant_indexes[-1]]


def load_tokenizer(model_name: str, adapter_paths: list[Path]):
    from transformers import AutoTokenizer

    tokenizer_path = next((path for path in adapter_paths if (path / "tokenizer_config.json").exists()), None)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path or model_name), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name: str, load_in_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if load_in_4bit else "auto",
        device_map="auto",
        quantization_config=quantization_config,
    )


def load_adapters(model: Any, adapter_paths: list[Path]):
    if not adapter_paths:
        return model

    from peft import PeftModel

    model = PeftModel.from_pretrained(model, adapter_paths[0], adapter_name=adapter_paths[0].name)
    for adapter_path in adapter_paths[1:]:
        model.load_adapter(adapter_path, adapter_name=adapter_path.name)
    return model


def generate_with_active_model(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    input_len = inputs["input_ids"].shape[-1]
    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            **generation_kwargs,
        )
    return clean_model_output(tokenizer.decode(generated[0, input_len:], skip_special_tokens=True))


def clean_model_output(output: str) -> str:
    output = output.strip()
    fence = re.search(r"```(?:openui-lang|text|json)?\s*(.*?)```", output, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        output = fence.group(1).strip()
    root_index = output.find("root =")
    if root_index > 0:
        output = output[root_index:].strip()
    return output


def run_generation(
    args: argparse.Namespace,
    messages: list[dict[str, str]],
    adapter_paths: list[Path],
) -> list[DemoResult]:
    tokenizer = load_tokenizer(args.model_name, adapter_paths)
    model = load_model(args.model_name, args.load_in_4bit)
    model = load_adapters(model, adapter_paths)
    model.eval()

    results = []
    if args.include_base:
        try:
            if hasattr(model, "disable_adapter"):
                with model.disable_adapter():
                    output = generate_with_active_model(
                        model,
                        tokenizer,
                        messages,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
            else:
                output = generate_with_active_model(
                    model,
                    tokenizer,
                    messages,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            results.append(render_result("base", "base model", output))
        except Exception as exc:
            results.append(DemoResult("base", "base model", "", render_error(str(exc)), str(exc)))

    for adapter_path in adapter_paths:
        name = adapter_path.name
        try:
            model.set_adapter(name)
            output = generate_with_active_model(
                model,
                tokenizer,
                messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            results.append(render_result(name, str(adapter_path), output))
        except Exception as exc:
            results.append(DemoResult(name, str(adapter_path), "", render_error(str(exc)), str(exc)))
    return results


def render_result(name: str, adapter_path: str, output: str) -> DemoResult:
    try:
        return DemoResult(name, adapter_path, output, render_openui_preview(output))
    except Exception as exc:
        return DemoResult(name, adapter_path, output, render_error(str(exc)), str(exc))


def _split_args(args_text: str) -> list[str]:
    args = []
    start = 0
    depth = 0
    quote = None
    escaped = False
    for index, char in enumerate(args_text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote:
            escaped = True
            continue
        if char in {'"', "'"}:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
            continue
        if quote:
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            args.append(args_text[start:index].strip())
            start = index + 1
    tail = args_text[start:].strip()
    if tail:
        args.append(tail)
    return args


def _parse_value(value: str) -> Any:
    value = value.strip()
    if value in {"None", "null"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [_parse_value(part) for part in _split_args(inner)]
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        return {"$ref": value}
    return json.loads(value)


def parse_openui_assignments(openui_lang: str) -> dict[str, dict[str, Any]]:
    components = {}
    for raw_line in openui_lang.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)", line)
        if not match:
            raise ValueError(f"Invalid OpenUI line: {line}")
        identifier, component_type, args_text = match.groups()
        components[identifier] = {
            "type": component_type,
            "args": [_parse_value(part) for part in _split_args(args_text)],
        }
    if "root" not in components:
        raise ValueError("Missing `root = ...` component.")
    return components


def render_openui_preview(openui_lang: str) -> str:
    components = parse_openui_assignments(openui_lang)
    return render_component("root", components)


def render_component(ref: str | dict[str, str], components: dict[str, dict[str, Any]]) -> str:
    if isinstance(ref, dict):
        ref = ref["$ref"]
    component = components.get(ref)
    if not component:
        return f'<div class="missing">Missing component: {escape(str(ref))}</div>'
    ctype = component["type"]
    args = component["args"]
    if ctype == "Card":
        children = args[0] if args else []
        return f'<section class="card">{"".join(render_component(child, components) for child in children)}</section>'
    if ctype == "CardHeader":
        title = args[0] if args else ""
        subtitle = args[1] if len(args) > 1 else ""
        return f'<header class="card-header"><h2>{escape(title)}</h2><p>{escape(subtitle)}</p></header>'
    if ctype == "TextContent":
        text = args[0] if args else ""
        return f'<p class="text-content">{escape(text)}</p>'
    if ctype == "ListBlock":
        items = args[0] if args else []
        return f'<div class="list-block">{"".join(render_component(item, components) for item in items)}</div>'
    if ctype == "ListItem":
        title = args[0] if args else ""
        body = args[1] if len(args) > 1 else ""
        return f'<div class="list-item"><strong>{escape(title)}</strong><span>{escape(body)}</span></div>'
    if ctype == "Callout":
        tone = args[0] if args else "info"
        title = args[1] if len(args) > 1 else ""
        body = args[2] if len(args) > 2 else ""
        return f'<div class="callout {escape(tone)}"><strong>{escape(title)}</strong><p>{escape(body)}</p></div>'
    if ctype == "FollowUpBlock":
        items = args[0] if args else []
        return f'<div class="followups">{"".join(render_component(item, components) for item in items)}</div>'
    if ctype == "FollowUpItem":
        return f'<button type="button">{escape(args[0] if args else "")}</button>'
    if ctype == "Table":
        columns = [components[col["$ref"]] for col in (args[0] if args else []) if isinstance(col, dict) and col.get("$ref") in components]
        return render_table(columns)
    if ctype == "BarChart":
        labels = args[0] if args else []
        series_refs = args[1] if len(args) > 1 else []
        series = [components[item["$ref"]] for item in series_refs if isinstance(item, dict) and item.get("$ref") in components]
        x_label = args[3] if len(args) > 3 else ""
        y_label = args[4] if len(args) > 4 else ""
        return render_bar_chart(labels, series, x_label, y_label)
    if ctype == "Series":
        label = args[0] if args else ""
        values = args[1] if len(args) > 1 else []
        return f'<span data-series="{escape(label)}">{escape(values)}</span>'
    return f'<div class="unsupported"><strong>{escape(ctype)}</strong><pre>{escape(json.dumps(args, ensure_ascii=False))}</pre></div>'


def render_table(columns: list[dict[str, Any]]) -> str:
    headers = [column["args"][0] for column in columns]
    values = [column["args"][1] if len(column["args"]) > 1 else [] for column in columns]
    rows = max([len(column_values) for column_values in values] or [0])
    body = []
    for row_index in range(rows):
        cells = []
        for column_values in values:
            value = column_values[row_index] if row_index < len(column_values) else ""
            cells.append(f"<td>{escape(value)}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + "".join(f"<th>{escape(header)}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def render_bar_chart(labels: list[Any], series: list[dict[str, Any]], x_label: str, y_label: str) -> str:
    first_series = series[0] if series else {"args": ["Value", []]}
    series_name = first_series["args"][0] if first_series["args"] else "Value"
    values = first_series["args"][1] if len(first_series["args"]) > 1 else []
    numeric_values = [float(value) for value in values if isinstance(value, (int, float))]
    max_value = max(numeric_values or [1.0])
    bars = []
    for label, value in zip(labels, values):
        number = float(value) if isinstance(value, (int, float)) else 0.0
        width = max(2.0, (number / max_value) * 100.0) if max_value else 2.0
        bars.append(
            '<div class="bar-row">'
            f'<span class="bar-label">{escape(label)}</span>'
            f'<span class="bar-track"><span style="width:{width:.2f}%"></span></span>'
            f'<span class="bar-value">{escape(value)}</span>'
            "</div>"
        )
    return (
        '<section class="chart">'
        f'<h3>{escape(series_name)}</h3><p>{escape(x_label)} -> {escape(y_label)}</p>'
        + "".join(bars)
        + "</section>"
    )


def render_error(message: str) -> str:
    return f'<div class="render-error"><strong>Render error</strong><pre>{escape(message)}</pre></div>'


def escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def write_report(
    *,
    output_path: Path,
    sample_path: Path,
    sample: dict[str, Any],
    prompt_messages: list[dict[str, str]],
    results: list[DemoResult],
) -> None:
    expected = extract_openui_messages(sample)[-1]["content"]
    sections = []
    for result in results:
        sections.append(
            f"""
            <section class="result">
              <div class="result-head">
                <h2>{escape(result.name)}</h2>
                <span>{escape(result.adapter_path)}</span>
              </div>
              <div class="columns">
                <div>
                  <h3>1. LLM Output</h3>
                  <pre>{escape(result.output)}</pre>
                </div>
                <div>
                  <h3>2. Rendered UI Components</h3>
                  {result.rendered_html}
                </div>
              </div>
            </section>
            """
        )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OpenUI Adapter Demo</title>
  <style>{REPORT_CSS}</style>
</head>
<body>
  <main>
    <h1>OpenUI Adapter Demo</h1>
    <section class="sample">
      <h2>3. Test Sample</h2>
      <p><strong>sample:</strong> {escape(sample_path)}</p>
      <p><strong>dataset:</strong> {escape((sample.get("query_result") or {}).get("dataset_title"))}</p>
      <details>
        <summary>Prompt messages sent to model</summary>
        <pre>{escape(json.dumps(prompt_messages, ensure_ascii=False, indent=2))}</pre>
      </details>
      <details>
        <summary>Expected target from sample</summary>
        <pre>{escape(expected)}</pre>
      </details>
    </section>
    {"".join(sections)}
  </main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_doc, encoding="utf-8")


REPORT_CSS = """
body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f8fafc; color: #0f172a; }
main { width: min(1400px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 48px; }
h1 { margin: 0 0 18px; font-size: 28px; }
.sample, .result { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin: 14px 0; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05); }
.result-head { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; border-bottom: 1px solid #e2e8f0; padding-bottom: 10px; margin-bottom: 14px; }
.result-head h2 { margin: 0; font-size: 20px; }
.result-head span { color: #64748b; font-size: 12px; overflow-wrap: anywhere; }
.columns { display: grid; grid-template-columns: minmax(0, 1fr) minmax(360px, 0.9fr); gap: 16px; align-items: start; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #0f172a; color: #e2e8f0; border-radius: 8px; padding: 12px; max-height: 620px; overflow: auto; font-size: 12px; line-height: 1.45; }
.card { display: grid; gap: 12px; border: 1px solid #dbeafe; border-radius: 8px; padding: 16px; background: #fff; }
.card-header h2 { margin: 0 0 4px; font-size: 19px; }
.card-header p, .text-content, .chart p, .callout p { margin: 0; color: #475569; }
.list-block { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; }
.list-item { display: grid; gap: 4px; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px; background: #f8fafc; }
.list-item span { color: #475569; font-size: 13px; }
.callout { border-radius: 8px; padding: 12px; border: 1px solid #bae6fd; background: #f0f9ff; }
.callout.warning { border-color: #fde68a; background: #fffbeb; }
.followups { display: flex; flex-wrap: wrap; gap: 8px; }
.followups button { border: 1px solid #cbd5e1; border-radius: 999px; background: #fff; padding: 7px 11px; color: #0f172a; }
.table-wrap { overflow: auto; border: 1px solid #e2e8f0; border-radius: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { border-bottom: 1px solid #e2e8f0; padding: 8px; text-align: left; vertical-align: top; }
th { background: #f8fafc; }
.chart { border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; }
.chart h3 { margin: 0 0 4px; font-size: 15px; }
.bar-row { display: grid; grid-template-columns: minmax(72px, 130px) minmax(0, 1fr) minmax(40px, 80px); gap: 8px; align-items: center; margin-top: 8px; }
.bar-label, .bar-value { font-size: 12px; color: #334155; overflow-wrap: anywhere; }
.bar-track { height: 14px; border-radius: 999px; background: #e2e8f0; overflow: hidden; }
.bar-track span { display: block; height: 100%; border-radius: 999px; background: linear-gradient(90deg, #2563eb, #0891b2); }
.render-error, .unsupported, .missing { border: 1px solid #fecaca; background: #fef2f2; color: #7f1d1d; border-radius: 8px; padding: 12px; }
@media (max-width: 900px) { .columns { grid-template-columns: 1fr; } }
"""


def main() -> int:
    args = build_arg_parser().parse_args()
    adapter_paths = [Path(path) for path in args.adapter] or discover_adapters(args.adapter_root)
    if not adapter_paths and not args.include_base:
        raise SystemExit(f"No adapters found. Pass --adapter or use --include-base. Searched: {args.adapter_root}")

    sample_path, sample = select_sample(args.sample_file, args.split, args.sample_index)
    sample = apply_question_override(sample, args.question)
    messages = prompt_messages_for_generation(sample)
    results = run_generation(args, messages, adapter_paths)
    write_report(
        output_path=args.output_html,
        sample_path=sample_path,
        sample=sample,
        prompt_messages=messages,
        results=results,
    )

    print(f"sample: {sample_path}")
    for result in results:
        print(f"\n=== {result.name} ===")
        if result.error:
            print(f"render/generation error: {result.error}")
        print(result.output[:2000])
    print(f"\nHTML report: {args.output_html.resolve()}")
    if not args.no_open:
        print(f"Open in browser: {args.output_html.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
