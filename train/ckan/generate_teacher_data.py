from __future__ import annotations

import argparse
import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ckan_dataset_tools import DEFAULT_SYSTEM_PROMPT, read_jsonl, validate_training_example, write_jsonl


TEACHER_BASE_URL_ENV = "SMOLNALYSIS_TEACHER_BASE_URL"
TEACHER_API_KEY_ENV = "SMOLNALYSIS_TEACHER_API_KEY"
TEACHER_MODEL_ENV = "SMOLNALYSIS_TEACHER_MODEL"
TEACHER_TIMEOUT_ENV = "SMOLNALYSIS_TEACHER_TIMEOUT_SECONDS"

TEACHER_SYSTEM_PROMPT = """You generate one supervised fine-tuning example for a CKAN retrieval policy.

The assistant must emit strict JSON only:
{
  "thought": "short decision summary, no long chain-of-thought",
  "action": "package_search | package_show | select_resource | reject_result | finish",
  "args": {},
  "confidence": 0.0
}

Rules:
- package_show must use an observed package.
- select_resource must use an observed resource.
- finish only if enough evidence is true.
- package_search queries must not contain URLs, credentials, or API keys.
- Use these exact args schemas:
  - package_search args: {"query": "string", "rows": 5, "start": 0}
  - package_show args: {"package_id": "observed-package-id"}
  - select_resource args: {"package_id": "observed-package-id", "resource_id": "observed-resource-id-without-package-prefix-if-possible", "reason": "why this resource fits"}
  - reject_result args: {"reason": "why current result is unsuitable", "next_query": "better search query"}
  - finish args: {"selected_candidates": [{"package_id": "observed-package-id", "resource_id": "observed-resource-id"}], "rationale": "why retrieval is complete"}
- Do not use CKAN API-native names like q, fq, f, id, or package_name in args. Use the exact schemas above.
- output strict JSON only."""


UrlOpen = Callable[..., Any]


@dataclass(frozen=True)
class TeacherConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 30.0
    temperature: float = 0.4

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def teacher_config_from_env(temperature: float, env_file: str | None = ".env") -> TeacherConfig:
    if env_file:
        load_env_file(Path(env_file))

    base_url = os.environ.get(TEACHER_BASE_URL_ENV, "").strip()
    api_key = os.environ.get(TEACHER_API_KEY_ENV, "").strip()
    model = os.environ.get(TEACHER_MODEL_ENV, "").strip()
    timeout_raw = os.environ.get(TEACHER_TIMEOUT_ENV, "30").strip()

    missing = [
        name
        for name, value in [
            (TEACHER_BASE_URL_ENV, base_url),
            (TEACHER_API_KEY_ENV, api_key),
            (TEACHER_MODEL_ENV, model),
        ]
        if not value
    ]
    if missing:
        raise ValueError(f"Missing teacher model environment variables: {', '.join(missing)}")

    try:
        timeout_seconds = float(timeout_raw)
    except ValueError as exc:
        raise ValueError(f"{TEACHER_TIMEOUT_ENV} must be a number.") from exc

    return TeacherConfig(base_url, api_key, model, timeout_seconds, temperature)


def build_user_content(scenario: dict[str, Any]) -> str:
    observed_packages = scenario.get("observed_packages", [])
    observed_resources = scenario.get("observed_resources", [])
    lines = [
        f"Request: {scenario.get('request', '')}",
        f"Endpoint: {scenario.get('endpoint', 'https://opendata.muenchen.de/')}",
        f"Current state: {scenario.get('state', 'No prior retrieval state.')}",
        f"Observed packages: {json.dumps(observed_packages, ensure_ascii=False)}",
        f"Observed resources: {json.dumps(observed_resources, ensure_ascii=False)}",
        f"Enough evidence: {bool(scenario.get('has_enough_evidence', False))}",
    ]
    if scenario.get("target_action"):
        lines.append(f"Desired next action type: {scenario['target_action']}")
        lines.append(f"Required args schema: {required_args_schema(str(scenario['target_action']))}")
    if scenario.get("notes"):
        lines.append(f"Notes: {scenario['notes']}")
    return "\n".join(lines)


def required_args_schema(action: str) -> str:
    schemas = {
        "package_search": '{"query":"search terms","rows":5,"start":0}',
        "package_show": '{"package_id":"one observed package id"}',
        "select_resource": '{"package_id":"one observed package id","resource_id":"one observed resource id","reason":"selection reason"}',
        "reject_result": '{"reason":"rejection reason","next_query":"better search terms"}',
        "finish": '{"selected_candidates":[{"package_id":"observed package id","resource_id":"observed resource id"}],"rationale":"completion rationale"}',
    }
    return schemas.get(action, "{}")


def build_teacher_messages(scenario: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(scenario)},
    ]


def build_training_example_from_teacher(scenario: dict[str, Any], assistant_content: str) -> dict[str, Any]:
    example = {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_content(scenario)},
            {"role": "assistant", "content": assistant_content.strip()},
        ],
        "metadata": {
            "scenario_id": scenario.get("id"),
            "teacher_generated": True,
            "ckan_context": {
                "observed_packages": scenario.get("observed_packages", []),
                "observed_resources": scenario.get("observed_resources", []),
                "has_enough_evidence": bool(scenario.get("has_enough_evidence", False)),
            },
        },
    }
    return example


def call_teacher(config: TeacherConfig, messages: list[dict[str, str]], urlopen: UrlOpen = urllib.request.urlopen) -> str:
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
    }
    request = urllib.request.Request(
        config.chat_completions_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=config.timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
        raise RuntimeError(f"Teacher request failed: {exc}") from exc

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Teacher returned invalid JSON.") from exc

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Teacher response did not include choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Teacher response did not include message content.")
    return content


def generate_examples(
    scenarios: list[dict[str, Any]],
    config: TeacherConfig,
    limit: int | None = None,
    delay_seconds: float = 0.0,
    urlopen: UrlOpen = urllib.request.urlopen,
    show_progress: bool = False,
    retries: int = 2,
    retry_delay_seconds: float = 2.0,
    existing_examples: list[dict[str, Any]] | None = None,
    incremental_output: Path | None = None,
) -> list[dict[str, Any]]:
    selected = scenarios[:limit] if limit is not None else scenarios
    examples = list(existing_examples or [])
    completed_ids = {
        str(example.get("metadata", {}).get("scenario_id"))
        for example in examples
        if example.get("metadata", {}).get("scenario_id")
    }
    total = len(selected)
    started_at = time.monotonic()
    for index, scenario in enumerate(selected, start=1):
        scenario_id = str(scenario.get("id") or f"scenario-{index}")
        if scenario_id in completed_ids:
            if show_progress:
                print(f"[{index}/{total}] Skipping existing {scenario_id}.", flush=True)
            continue
        if show_progress:
            print(f"[{index}/{total}] Generating {scenario_id}...", flush=True)
        assistant_content = call_teacher_with_retries(config, build_teacher_messages(scenario), retries, retry_delay_seconds, urlopen)
        example = build_training_example_from_teacher(scenario, assistant_content)
        examples.append(example)
        completed_ids.add(scenario_id)
        if incremental_output:
            write_jsonl(incremental_output, examples)
        if show_progress:
            elapsed = time.monotonic() - started_at
            print(f"[{index}/{total}] Generated in {elapsed:.1f}s elapsed.", flush=True)
        if delay_seconds and index < total:
            time.sleep(delay_seconds)
    return examples


def call_teacher_with_retries(
    config: TeacherConfig,
    messages: list[dict[str, str]],
    retries: int,
    retry_delay_seconds: float,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> str:
    attempts = max(1, retries + 1)
    last_error: RuntimeError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call_teacher(config, messages, urlopen=urlopen)
        except RuntimeError as exc:
            last_error = exc
            if attempt >= attempts:
                break
            print(f"Teacher call failed on attempt {attempt}/{attempts}: {exc}. Retrying...", flush=True)
            time.sleep(retry_delay_seconds)
    raise last_error or RuntimeError("Teacher request failed.")


def _command_generate(args: argparse.Namespace) -> int:
    try:
        config = teacher_config_from_env(args.temperature, args.env_file)
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        return 2
    scenarios = [row for row in read_jsonl(Path(args.scenarios)) if not row.get("_invalid_jsonl")]
    try:
        existing_examples = read_existing_examples(Path(args.output)) if args.resume else []
        examples = generate_examples(
            scenarios,
            config,
            args.limit,
            args.delay_seconds,
            show_progress=not args.quiet,
            retries=args.retries,
            retry_delay_seconds=args.retry_delay_seconds,
            existing_examples=existing_examples,
            incremental_output=Path(args.output) if args.incremental or args.resume else None,
        )
    except RuntimeError as exc:
        print(f"Generation error: {exc}")
        return 1
    write_jsonl(Path(args.output), examples)

    valid_examples = []
    report_rows = []
    if args.validate or args.valid_output or args.report:
        for index, example in enumerate(examples, start=1):
            result = validate_training_example(example)
            report_rows.append({"line": index, "scenario_id": example.get("metadata", {}).get("scenario_id"), **result.to_dict()})
            if result.ok:
                valid_examples.append(example)
            if not args.quiet:
                print(
                    f"[validate {index}/{len(examples)}] valid={len(valid_examples)} rejected={index - len(valid_examples)}",
                    flush=True,
                )
        if args.valid_output:
            write_jsonl(Path(args.valid_output), valid_examples)
        if args.report:
            write_jsonl(Path(args.report), report_rows)

    print(f"Wrote {len(examples)} teacher-generated examples to {args.output}.")
    if args.validate or args.valid_output or args.report:
        print(f"Validation: {len(valid_examples)} valid, {len(examples) - len(valid_examples)} rejected.")
    return 0


def read_existing_examples(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [row for row in read_jsonl(path) if not row.get("_invalid_jsonl")]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CKAN retrieval training examples with an OpenAI-compatible teacher model.")
    parser.add_argument("--scenarios", required=True, help="Input scenario JSONL.")
    parser.add_argument("--output", required=True, help="Raw generated training JSONL.")
    parser.add_argument("--limit", type=int, help="Maximum number of scenarios to process.")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--delay-seconds", type=float, default=0.0, help="Optional delay between teacher calls.")
    parser.add_argument("--retries", type=int, default=2, help="Retries per teacher call after timeout/network failure.")
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    parser.add_argument("--env-file", default=".env", help="Optional env file to load before reading teacher settings.")
    parser.add_argument("--validate", action="store_true", help="Validate generated examples after writing raw output.")
    parser.add_argument("--valid-output", help="Optional JSONL path for valid generated examples.")
    parser.add_argument("--report", help="Optional JSONL path for validation report.")
    parser.add_argument("--quiet", action="store_true", help="Disable per-scenario progress output.")
    parser.add_argument("--incremental", action="store_true", help="Write raw output after every successful teacher call.")
    parser.add_argument("--resume", action="store_true", help="Load existing raw output and skip already generated scenario ids.")
    args = parser.parse_args()
    return _command_generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
