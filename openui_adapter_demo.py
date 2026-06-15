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
DEFAULT_ADAPTER = Path("train/openui_lang/outputs/openui-translate-mini-lora")
DEFAULT_DATASET = Path("train/openui_lang/data/openui_sft_train.jsonl")
SYSTEM_PROMPT = (
    "You generate OpenUI Lang from a user query and a structured tool result. "
    "Use only the values from the tool result. Do not invent data. "
    "Return only OpenUI Lang assignment statements, without explanations or markdown. "
    "Start with root = Root([...])."
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
    match = re.search(r"(?:User query:\s*)?(.*?)\n\nTool result:\n(.*)\s*$", content, flags=re.DOTALL)
    if not match:
        raise ValueError("Sample user content does not match the generated dataset format.")
    user_query = match.group(1).strip()
    tool_result = json.loads(match.group(2))
    return user_query, tool_result


def load_examples(path: Path, limit: int = 12) -> list[DemoExample]:
    examples_by_shape: dict[str, DemoExample] = {}
    examples: list[DemoExample] = []
    if not path.exists():
        return examples

    def iter_rows() -> list[dict[str, Any]]:
        if path.is_dir():
            rows = []
            for sample_path in sorted(path.glob("*.json")):
                if sample_path.name == "manifest.json":
                    continue
                rows.append(json.loads(sample_path.read_text(encoding="utf-8")))
            return rows

        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    preferred_shapes = [
        "scalar",
        "comparison",
        "time_series_daily",
        "time_series_monthly",
        "ranking",
        "threshold",
        "percentage",
        "table",
        "multi_kpi",
        "geo_values",
    ]
    for row in iter_rows():
        try:
            messages = row["messages"]
            user_query, tool_result = parse_user_content(messages[1]["content"])
            expected = messages[2]["content"]
        except Exception:
            continue

        metadata = row.get("metadata", {})
        data_shape = str(metadata.get("data_shape", "unknown"))
        label = " / ".join(
            str(part)
            for part in [
                metadata.get("domain", "unknown"),
                data_shape,
                metadata.get("component", "unknown"),
            ]
        )
        example = DemoExample(user_query, tool_result, expected, label)
        if data_shape not in examples_by_shape:
            examples_by_shape[data_shape] = example
        if len(examples) < limit:
            examples.append(example)

    diverse = [
        examples_by_shape[shape]
        for shape in preferred_shapes
        if shape in examples_by_shape
    ]
    if len(diverse) >= min(limit, len(examples_by_shape)):
        seen = {id(example) for example in diverse}
        diverse.extend(
            example
            for example in examples
            if id(example) not in seen
        )
        return diverse[:limit]

    return examples[:limit]


def make_user_message(user_query: str, tool_result_text: str) -> str:
    parsed = json.loads(tool_result_text)
    return user_query.strip() + "\n\nTool result:\n" + json.dumps(parsed, ensure_ascii=False, indent=2)


def clean_component_output(output: str) -> str:
    output = output.strip()
    fence = re.search(r"```(?:jsx|xml|openui|text)?\s*(.*?)```", output, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        output = fence.group(1).strip()
    root = re.search(r"(?m)^\s*root\s*=\s*Root\s*\(", output)
    if root and root.start() > 0:
        output = output[root.start() :].strip()
    tag = re.search(r"<[A-Za-z][A-Za-z0-9]*(?:\s|>|/)", output)
    if not root and tag and tag.start() > 0:
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
        #quantization_config=quantization_config,
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
            "eos_token_id": [tokenizer.eos_token_id, 130073],
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


def extract_table_rows(component: str, limit: int = 8) -> list[dict[str, str]]:
    rows = []
    data_match = re.search(r"data\s*=\s*\{\s*\[(.*?)\]\s*\}", component, flags=re.DOTALL)
    if not data_match:
        return rows
    for row_match in re.finditer(r"\{(.*?)\}", data_match.group(1), flags=re.DOTALL):
        raw = row_match.group(1)
        label = ""
        for key in ["office", "label", "district", "month", "date"]:
            prop = re.search(rf'{key}\s*:\s*"([^"]*)"', raw)
            if prop:
                label = prop.group(1)
                break
        value = re.search(r"value\s*:\s*(-?\d+(?:\.\d+)?)", raw)
        unit = re.search(r'unit\s*:\s*"([^"]*)"', raw)
        if label or value:
            rows.append(
                {
                    "label": label or "Row",
                    "value": value.group(1) if value else "",
                    "unit": unit.group(1) if unit else "",
                }
            )
        if len(rows) >= limit:
            break
    return rows


def split_openui_args(args_text: str) -> list[str]:
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


def parse_openui_value(value: str) -> Any:
    value = value.strip()
    if value in {"None", "null"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [parse_openui_value(part) for part in split_openui_args(inner)]
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        return {"$ref": value}
    return json.loads(value)


def parse_openui_assignments(openui_lang: str) -> dict[str, dict[str, Any]]:
    components: dict[str, dict[str, Any]] = {}
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
            "args": [parse_openui_value(part) for part in split_openui_args(args_text)],
        }
    if "root" not in components:
        raise ValueError("Missing `root = Root([...])` component.")
    return components


def render_openui_lang_preview(openui_lang: str) -> str:
    components = parse_openui_assignments(openui_lang)
    return render_openui_ref("root", components)


def render_openui_ref(ref: str | dict[str, str], components: dict[str, dict[str, Any]]) -> str:
    if isinstance(ref, dict):
        ref = ref["$ref"]
    component = components.get(ref)
    if not component:
        return f'<div class="missing">Missing component: {escape(str(ref))}</div>'

    ctype = component["type"]
    args = component["args"]
    if ctype == "Root":
        children = args[0] if args else []
        return f'<section class="preview openui-lang">{"".join(render_openui_ref(child, components) for child in children)}</section>'
    if ctype == "InsightCard":
        title = args[0] if args else "Insight"
        body = args[1] if len(args) > 1 else ""
        return f'<article class="insight"><h2>{escape(title)}</h2><p>{escape(body)}</p></article>'
    if ctype == "Notice":
        message = args[0] if args else ""
        tone = args[1] if len(args) > 1 else "info"
        return f'<article class="alert {escape(tone)}"><p>{escape(message)}</p></article>'
    if ctype == "MetricGrid":
        items = args[0] if args else []
        return f'<div class="grid">{"".join(render_openui_ref(item, components) for item in items)}</div>'
    if ctype == "Metric":
        label = args[0] if args else "Metric"
        value = args[1] if len(args) > 1 else ""
        caption = args[2] if len(args) > 2 else ""
        return (
            '<div class="stat">'
            f"<span>{escape(label)}</span>"
            f"<strong>{escape(value)}</strong>"
            f"<small>{escape(caption)}</small>"
            "</div>"
        )
    if ctype == "DataTable":
        title = args[0] if args else "Table"
        rows = args[1] if len(args) > 1 and isinstance(args[1], list) else []
        return render_openui_data_table(title, rows)
    if ctype == "BarChart":
        title = args[0] if args else "Chart"
        x_column = args[1] if len(args) > 1 else "label"
        y_column = args[2] if len(args) > 2 else "value"
        rows = args[3] if len(args) > 3 and isinstance(args[3], list) else []
        return render_openui_bar_chart(title, x_column, y_column, rows)
    if ctype == "Histogram":
        title = args[0] if args else "Histogram"
        column = args[1] if len(args) > 1 else "value"
        values = args[2] if len(args) > 2 and isinstance(args[2], list) else []
        return render_openui_histogram(title, column, values)
    return f'<div class="unsupported"><strong>{escape(ctype)}</strong><pre>{escape(json.dumps(args, ensure_ascii=False))}</pre></div>'


def render_openui_data_table(title: Any, rows: list[Any]) -> str:
    columns = list(rows[0].keys()) if rows and isinstance(rows[0], dict) else []
    head = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        body.append("<tr>" + "".join(f"<td>{escape(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    return f'<article class="table-preview"><h2>{escape(title)}</h2><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></article>'


def render_openui_bar_chart(title: Any, x_column: Any, y_column: Any, rows: list[Any]) -> str:
    pairs = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_value = row.get(str(y_column), 0)
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError):
            numeric = 0.0
        pairs.append((str(row.get(str(x_column), "")), numeric, str(raw_value)))
    max_value = max([abs(value) for _, value, _ in pairs] or [1.0]) or 1.0
    bars = []
    for label, numeric, raw_value in pairs:
        width = max(2.0, abs(numeric) / max_value * 100.0)
        bars.append(
            f"""
            <div class="bar-row">
              <span>{escape(label)}</span>
              <div><i style="width:{width:.1f}%"></i></div>
              <b>{escape(raw_value)}</b>
            </div>
            """
        )
    return f'<article class="chart-preview"><h2>{escape(title)}</h2><div class="bars">{"".join(bars)}</div></article>'


def render_openui_histogram(title: Any, column: Any, values: list[Any]) -> str:
    numeric_values = []
    for value in values:
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not numeric_values:
        return f'<article class="chart-preview"><h2>{escape(title)}</h2><p>{escape(column)}: no numeric values</p></article>'
    buckets = [0] * min(12, max(1, len(numeric_values)))
    minimum = min(numeric_values)
    span = max(numeric_values) - minimum or 1.0
    for value in numeric_values:
        index = min(len(buckets) - 1, int(((value - minimum) / span) * len(buckets)))
        buckets[index] += 1
    top = max(buckets) or 1
    bars = "".join(
        f'<div class="histogram-bar" title="{escape(count)}" style="height:{max(4.0, count / top * 100.0):.1f}%"></div>'
        for count in buckets
    )
    return f'<article class="chart-preview"><h2>{escape(title)}</h2><p>{escape(column)}</p><div class="histogram">{bars}</div></article>'


def render_component_preview(component: str) -> str:
    if not component.strip():
        return render_error("No model output.")
    if re.search(r"(?m)^\s*root\s*=\s*Root\s*\(", component):
        try:
            return render_openui_lang_preview(component)
        except Exception as exc:
            return render_error(str(exc))

    component_name = extract_component_name(component)
    title = extract_prop(component, "title") or component_name
    value = extract_prop(component, "value")
    unit = extract_prop(component, "unit") or ""
    severity = extract_prop(component, "severity")
    stat_cards = extract_stat_cards(component)
    rows = extract_data_rows(component)
    table_rows = extract_table_rows(component)

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

    if component_name == "ComparisonCard":
        current_label = extract_prop(component, "currentLabel") or "Current"
        previous_label = extract_prop(component, "previousLabel") or "Previous"
        current_value = extract_prop(component, "currentValue") or ""
        previous_value = extract_prop(component, "previousValue") or ""
        delta_value = extract_prop(component, "deltaValue") or ""
        direction = extract_prop(component, "deltaDirection") or ""
        return f"""
        <section class="preview">
          <h2>{escape(title)}</h2>
          <div class="grid">
            <div class="stat"><span>{escape(current_label)}</span><strong>{escape(current_value)}</strong><small>{escape(unit)}</small></div>
            <div class="stat"><span>{escape(previous_label)}</span><strong>{escape(previous_value)}</strong><small>{escape(unit)}</small></div>
            <div class="stat"><span>Delta {escape(direction)}</span><strong>{escape(delta_value)}</strong><small>{escape(unit)}</small></div>
          </div>
        </section>
        """

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
        maximum = float(extract_prop(component, "max") or 100)
        width = max(0.0, min(100.0, number / max(1.0, maximum) * 100.0))
        description = extract_prop(component, "description") or ""
        return (
            f'<section class="preview"><h2>{escape(title)}</h2><div class="progress"><i style="width:{width:.1f}%"></i></div>'
            f'<p><strong>{escape(value)}</strong>{escape(unit)} {escape(description)}</p></section>'
        )

    if component_name == "TableCard":
        body = "".join(
            f"<tr><td>{escape(row['label'])}</td><td>{escape(row['value'])}</td><td>{escape(row['unit'] or unit)}</td></tr>"
            for row in table_rows
        )
        if not body:
            body = '<tr><td colspan="3">Inspect raw component code for columns and rows.</td></tr>'
        return f'<section class="preview"><h2>{escape(title)}</h2><table><tbody>{body}</tbody></table></section>'

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
    .preview, .preview * { color: #111827; }
    .preview h2 { margin: 0 0 10px; font-size: 18px; color: #0f172a; }
    .preview p { color: #334155; }
    .insight, .chart-preview, .table-preview { background: #fff; color: #111827; }
    .insight { border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; margin-bottom: 10px; }
    .insight p, .chart-preview p { color: #334155; margin: 0 0 10px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; }
    .stat { border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; display: grid; gap: 3px; background: #f8fafc; }
    .stat span, .stat small { color: #475569; font-size: 12px; }
    .stat strong { color: #0f172a; font-size: 20px; }
    .bar-row { display: grid; grid-template-columns: minmax(72px, 150px) 1fr minmax(72px, 120px); gap: 8px; align-items: center; margin: 8px 0; }
    .bar-row span, .bar-row b { color: #1f2937; font-size: 12px; overflow-wrap: anywhere; }
    .bar-row div, .progress { height: 14px; background: #e5e7eb; border-radius: 999px; overflow: hidden; }
    .bar-row i, .progress i { display: block; height: 100%; background: #2563eb; border-radius: 999px; }
    .alert.warning { border-color: #f59e0b; background: #fffbeb; }
    .alert.danger { border-color: #ef4444; background: #fef2f2; }
    .alert.success { border-color: #10b981; background: #ecfdf5; }
    .preview table { width: 100%; border-collapse: collapse; }
    .preview th { color: #111827; border-bottom: 1px solid #cbd5e1; padding: 7px 6px; font-size: 13px; text-align: left; }
    .preview td { color: #1f2937; border-top: 1px solid #e5e7eb; padding: 7px 6px; font-size: 13px; }
    .preview td:nth-child(2), .preview td:nth-child(3) { text-align: right; }
    .histogram { display: flex; align-items: end; gap: 4px; height: 160px; padding-top: 8px; }
    .histogram-bar { flex: 1; min-width: 8px; background: #2563eb; border-radius: 4px 4px 0 0; }
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
