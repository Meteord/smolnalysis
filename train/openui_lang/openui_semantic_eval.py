from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any

import torch

try:
    from data.dataset import apply_chat_template, extract_openui_messages, prompt_messages
except ModuleNotFoundError:
    from dataset import apply_chat_template, extract_openui_messages, prompt_messages  # type: ignore


NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")
COMPONENT_RE = re.compile(r"^\s*<([A-Z][A-Za-z0-9_]*)\b")
OPENUI_COMPONENT_RE = re.compile(
    r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*([A-Z][A-Za-z0-9_]*)\s*\(",
    re.MULTILINE,
)


def _raw_samples(eval_dataset: Any, max_samples: int) -> list[dict[str, Any]]:
    samples = getattr(eval_dataset, "samples", None)
    if samples is not None:
        return list(samples[:max_samples])

    rows = []
    for index in range(min(max_samples, len(eval_dataset))):
        item = eval_dataset[index]
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _assistant_label(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message["role"] == "assistant":
            return message["content"]
    raise ValueError("Messages must include an assistant target.")


def _first_component(output: str) -> str | None:
    match = COMPONENT_RE.search(output or "")
    if match:
        return match.group(1)
    for component in OPENUI_COMPONENT_RE.findall(output or ""):
        if component not in {"Root", "Card"}:
            return component
    return None


def _components(output: str) -> set[str]:
    components = set(OPENUI_COMPONENT_RE.findall(output or ""))
    match = COMPONENT_RE.search(output or "")
    if match:
        components.add(match.group(1))
    return components


def _is_openui_like(output: str) -> bool:
    stripped = (output or "").strip()
    if stripped.startswith("root = ") and bool(OPENUI_COMPONENT_RE.search(stripped)):
        return True
    return bool(
        stripped.startswith("<")
        and _first_component(stripped)
        and (
            "/>" in stripped
            or re.search(r"</\s*[A-Z][A-Za-z0-9_]*\s*>", stripped) is not None
        )
    )


def _canonical_number(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if decimal == decimal.to_integral_value():
        return str(int(decimal))
    return format(decimal.normalize(), "f").rstrip("0").rstrip(".")


def _numbers_from_text(text: str) -> set[str]:
    numbers = set()
    for match in NUMBER_RE.finditer(text or ""):
        canonical = _canonical_number(match.group(0))
        if canonical is not None:
            numbers.add(canonical)
    return numbers


def _numeric_values(value: Any) -> set[str]:
    numbers: set[str] = set()
    if isinstance(value, dict):
        for child in value.values():
            numbers.update(_numeric_values(child))
    elif isinstance(value, list):
        for child in value:
            numbers.update(_numeric_values(child))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        canonical = _canonical_number(value)
        if canonical is not None:
            numbers.add(canonical)
    elif isinstance(value, str):
        numbers.update(_numbers_from_text(value))
    return numbers


def _parse_tool_result(sample: dict[str, Any], messages: list[dict[str, str]]) -> dict[str, Any]:
    query_result = sample.get("query_result")
    if isinstance(query_result, dict):
        return query_result

    user_text = "\n\n".join(message["content"] for message in messages if message["role"] == "user")
    marker = "Tool result:"
    if marker not in user_text:
        return {}
    after_marker = user_text.split(marker, 1)[1].lstrip()
    try:
        parsed, _ = json.JSONDecoder().raw_decode(after_marker)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _numbers_at_paths(value: Any, keys: set[str]) -> set[str]:
    numbers: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in keys:
                numbers.update(_numeric_values(child))
            if isinstance(child, (dict, list)):
                numbers.update(_numbers_at_paths(child, keys))
    elif isinstance(value, list):
        for child in value:
            numbers.update(_numbers_at_paths(child, keys))
    return numbers


def _expected_tool_numbers(tool_result: dict[str, Any], data_shape: str) -> set[str]:
    if not tool_result:
        return set()

    common = _numbers_at_paths(tool_result, {"year"})
    if data_shape in {"scalar", "percentage"}:
        return common | _numbers_at_paths(tool_result, {"value", "max"})
    if data_shape == "threshold":
        return common | _numbers_at_paths(tool_result, {"value", "threshold"})
    if data_shape == "comparison":
        return common | _numbers_at_paths(tool_result, {"year", "value"})
    if data_shape in {"time_series_daily", "time_series_monthly", "ranking", "geo_values"}:
        return common | _numbers_at_paths(tool_result, {"date", "month", "value"})
    if data_shape == "table":
        return common | _numbers_at_paths(tool_result, {"value"})
    if data_shape == "multi_kpi":
        return common | _numbers_at_paths(tool_result, {"value"})
    return _numeric_values(tool_result)


def _allowed_derived_numbers(tool_result: dict[str, Any], data_shape: str) -> set[str]:
    if data_shape != "comparison":
        return set()
    try:
        current = tool_result["current"]["value"]
        previous = tool_result["previous"]["value"]
    except (KeyError, TypeError):
        return set()
    delta = _canonical_number(Decimal(str(current)) - Decimal(str(previous)))
    return {delta} if delta is not None else set()


def _contains_all(available: set[str], required: set[str]) -> bool:
    return required.issubset(available)


def _sample_metrics(sample: dict[str, Any], generated: str) -> dict[str, float]:
    messages = extract_openui_messages(sample)
    metadata = sample.get("metadata") or {}
    target = _assistant_label(messages)
    tool_result = _parse_tool_result(sample, messages)
    data_shape = str(metadata.get("data_shape") or "")

    generated_numbers = _numbers_from_text(generated)
    target_numbers = _numbers_from_text(target)
    tool_numbers = _numeric_values(tool_result)
    expected_numbers = _expected_tool_numbers(tool_result, data_shape)
    allowed_numbers = tool_numbers | _allowed_derived_numbers(tool_result, data_shape)
    hallucinated_numbers = generated_numbers - allowed_numbers

    return {
        "component_accuracy": float(str(metadata.get("component") or "") in _components(generated)),
        "exact_match_rate": float((generated or "").strip() == target.strip()),
        "required_value_accuracy": float(_contains_all(generated_numbers, target_numbers)),
        "tool_value_accuracy": float(_contains_all(generated_numbers, expected_numbers)),
        "hallucinated_number_rate": float(len(hallucinated_numbers) / max(1, len(generated_numbers))),
        "valid_openui_like_rate": float(_is_openui_like(generated)),
    }


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _generate_batch(
    model: Any,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    max_new_tokens: int,
) -> list[str]:
    prompts = []
    for sample in samples:
        messages = extract_openui_messages(sample)
        prompts.append(apply_chat_template(tokenizer, prompt_messages(messages), add_generation_prompt=True))

    encoded = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt")
    device = _model_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
        )

    prompt_width = encoded["input_ids"].shape[1]
    return [
        tokenizer.decode(output_ids[index, prompt_width:], skip_special_tokens=True)
        for index in range(output_ids.shape[0])
    ]


def evaluate_openui_semantic(
    model: Any,
    tokenizer: Any,
    eval_dataset: Any,
    max_samples: int = 200,
    batch_size: int = 4,
    max_new_tokens: int = 512,
) -> dict[str, float]:
    """Generate OpenUI outputs for eval prompts and compute semantic analysis metrics."""
    samples = _raw_samples(eval_dataset, max_samples)
    totals = {
        "component_accuracy": 0.0,
        "exact_match_rate": 0.0,
        "required_value_accuracy": 0.0,
        "tool_value_accuracy": 0.0,
        "hallucinated_number_rate": 0.0,
        "valid_openui_like_rate": 0.0,
    }
    evaluated = 0
    failed = 0
    was_training = bool(getattr(model, "training", False))
    model.eval()

    try:
        for start in range(0, len(samples), max(1, batch_size)):
            batch = samples[start:start + max(1, batch_size)]
            try:
                generated_outputs = _generate_batch(model, tokenizer, batch, max_new_tokens)
            except Exception:
                failed += len(batch)
                evaluated += len(batch)
                continue

            for sample, generated in zip(batch, generated_outputs):
                try:
                    metrics = _sample_metrics(sample, generated)
                except Exception:
                    failed += 1
                    evaluated += 1
                    continue
                for key in totals:
                    totals[key] += metrics[key]
                evaluated += 1
    finally:
        if was_training:
            model.train()

    if evaluated == 0:
        result = {key: 0.0 for key in totals}
        result["semantic_score"] = 0.0
        result["semantic_eval_samples"] = 0.0
        result["semantic_eval_failed_samples"] = 0.0
        return result

    denominator = max(1, evaluated)
    result = {key: value / denominator for key, value in totals.items()}
    result["semantic_score"] = (
        0.25 * result["component_accuracy"]
        + 0.25 * result["required_value_accuracy"]
        + 0.20 * result["tool_value_accuracy"]
        + 0.15 * result["valid_openui_like_rate"]
        + 0.15 * (1.0 - result["hallucinated_number_rate"])
    )
    result["semantic_eval_samples"] = float(evaluated)
    result["semantic_eval_failed_samples"] = float(failed)
    return result
