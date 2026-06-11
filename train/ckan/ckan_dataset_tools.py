from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ALLOWED_ACTIONS = {"package_search", "package_show", "select_resource", "reject_result", "finish"}
DEFAULT_SYSTEM_PROMPT = "You are the CKAN retrieval policy for smolnalysis. Emit strict JSON actions only."


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    issues: list[ValidationIssue]
    action: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": [issue.__dict__ for issue in self.issues],
            "action": self.action,
        }


def parse_action(content: str) -> tuple[dict[str, Any] | None, list[ValidationIssue]]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        return None, [ValidationIssue("invalid_json", f"Assistant content is not valid JSON: {exc.msg}.")]

    if not isinstance(payload, dict):
        return None, [ValidationIssue("not_object", "Assistant content must be a JSON object.")]
    return payload, []


def validate_ckan_action(content: str, context: dict[str, Any] | None = None) -> ValidationResult:
    payload, issues = parse_action(content)
    if payload is None:
        return ValidationResult(False, issues)

    context = context or {}
    action = payload.get("action")
    args = payload.get("args")
    confidence = payload.get("confidence")
    thought = payload.get("thought")

    if not isinstance(thought, str) or not thought.strip():
        issues.append(ValidationIssue("missing_thought", "`thought` must be a non-empty string."))
    elif len(thought.split()) > 40:
        issues.append(ValidationIssue("long_thought", "`thought` should be a short decision summary."))

    if action not in ALLOWED_ACTIONS:
        issues.append(ValidationIssue("invalid_action", f"`action` must be one of {sorted(ALLOWED_ACTIONS)}."))

    if not isinstance(args, dict):
        issues.append(ValidationIssue("invalid_args", "`args` must be an object."))
        args = {}

    if not isinstance(confidence, int | float) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
        issues.append(ValidationIssue("invalid_confidence", "`confidence` must be a number between 0.0 and 1.0."))

    if action == "package_search":
        _validate_package_search(args, issues)
    elif action == "package_show":
        _validate_package_show(args, context, issues)
    elif action == "select_resource":
        _validate_select_resource(args, context, issues)
    elif action == "reject_result":
        _validate_reject_result(args, issues)
    elif action == "finish":
        _validate_finish(args, context, issues)

    return ValidationResult(not issues, issues, payload)


def build_training_example(user_content: str, assistant_action: dict[str, Any], system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(assistant_action, ensure_ascii=False, separators=(",", ":"))},
        ]
    }


def extract_context_from_example(example: dict[str, Any]) -> dict[str, Any]:
    metadata = example.get("metadata")
    if isinstance(metadata, dict):
        context = metadata.get("ckan_context")
        if isinstance(context, dict):
            return context
    return {}


def validate_training_example(example: dict[str, Any]) -> ValidationResult:
    messages = example.get("messages")
    if not isinstance(messages, list) or not messages:
        return ValidationResult(False, [ValidationIssue("missing_messages", "Example must contain a non-empty `messages` list.")])

    assistant_messages = [message for message in messages if isinstance(message, dict) and message.get("role") == "assistant"]
    if not assistant_messages:
        return ValidationResult(False, [ValidationIssue("missing_assistant", "Example must contain an assistant message.")])

    content = assistant_messages[-1].get("content")
    if not isinstance(content, str):
        return ValidationResult(False, [ValidationIssue("invalid_assistant_content", "Assistant content must be a string.")])

    return validate_ckan_action(content, extract_context_from_example(example))


def seed_examples() -> list[dict[str, Any]]:
    contexts = [
        {
            "observed_packages": [],
            "observed_resources": [],
            "has_enough_evidence": False,
        },
        {
            "observed_packages": ["population-indicators"],
            "observed_resources": [],
            "has_enough_evidence": False,
        },
        {
            "observed_packages": ["mobility-counts"],
            "observed_resources": ["mobility-counts:traffic_csv"],
            "has_enough_evidence": True,
        },
    ]
    examples = [
        build_training_example(
            "Request: Find population data for Munich districts.\nEndpoint: https://opendata.muenchen.de/\nState: no searches yet.",
            {
                "thought": "Need an initial package search for the main topic.",
                "action": "package_search",
                "args": {"query": "population districts Munich", "rows": 5, "start": 0},
                "confidence": 0.82,
            },
        ),
        build_training_example(
            "Request: Show details for the strongest population candidate.\nEndpoint: https://opendata.muenchen.de/\nObserved packages: population-indicators.",
            {
                "thought": "A promising package is known, so inspect its resources.",
                "action": "package_show",
                "args": {"package_id": "population-indicators"},
                "confidence": 0.78,
            },
        ),
        build_training_example(
            "Request: Compare mobility counts by year.\nEndpoint: https://opendata.muenchen.de/\nObserved package mobility-counts includes resource traffic_csv.",
            {
                "thought": "The CSV resource matches the request and can be selected.",
                "action": "select_resource",
                "args": {
                    "package_id": "mobility-counts",
                    "resource_id": "traffic_csv",
                    "reason": "CSV resource is suitable for yearly mobility comparison.",
                },
                "confidence": 0.86,
            },
        ),
    ]
    for example, context in zip(examples, contexts, strict=True):
        example["metadata"] = {"ckan_context": context}
    return examples


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                rows.append({"_invalid_jsonl": True, "_line_number": line_number, "_error": exc.msg})
            else:
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def balanced_sample(rows: list[dict[str, Any]], limit: int, key: str = "target_action", seed: int = 13) -> list[dict[str, Any]]:
    if limit >= len(rows):
        return rows
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row.get(key) or "unknown"), []).append(row)
    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)

    selected = []
    bucket_names = sorted(buckets)
    while len(selected) < limit and bucket_names:
        progressed = False
        for bucket_name in list(bucket_names):
            bucket_rows = buckets[bucket_name]
            if bucket_rows:
                selected.append(bucket_rows.pop())
                progressed = True
                if len(selected) >= limit:
                    break
            else:
                bucket_names.remove(bucket_name)
        if not progressed:
            break
    rng.shuffle(selected)
    return selected


def scenario_package_ids(row: dict[str, Any]) -> set[str]:
    package_ids = set()
    for package_id in row.get("observed_packages") or []:
        if isinstance(package_id, str) and package_id:
            package_ids.add(package_id)
    package_summary = row.get("package_summary")
    if isinstance(package_summary, dict):
        package_id = package_summary.get("id")
        if isinstance(package_id, str) and package_id:
            package_ids.add(package_id)
    return package_ids


def split_by_package(
    rows: list[dict[str, Any]],
    eval_size: int,
    train_size: int | None = None,
    seed: int = 17,
    key: str = "target_action",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    package_to_rows: dict[str, list[dict[str, Any]]] = {}
    ungrouped_rows = []
    for row in rows:
        package_ids = sorted(scenario_package_ids(row))
        if not package_ids:
            ungrouped_rows.append(row)
            continue
        package_to_rows.setdefault(package_ids[0], []).append(row)

    package_ids = list(package_to_rows)
    rng.shuffle(package_ids)
    eval_rows = []
    eval_packages = set()
    for package_id in package_ids:
        if len(eval_rows) >= eval_size:
            break
        eval_packages.add(package_id)
        eval_rows.extend(package_to_rows[package_id])

    train_rows = []
    for package_id, package_rows in package_to_rows.items():
        if package_id not in eval_packages:
            train_rows.extend(package_rows)
    train_rows.extend(ungrouped_rows)

    eval_sample = balanced_sample(eval_rows, eval_size, key, seed)
    train_sample = balanced_sample(train_rows, train_size, key, seed + 1) if train_size is not None else train_rows
    return train_sample, eval_sample


def package_overlap(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> set[str]:
    left_ids = {package_id for row in left for package_id in scenario_package_ids(row)}
    right_ids = {package_id for row in right for package_id in scenario_package_ids(row)}
    return left_ids & right_ids


def _validate_package_search(args: dict[str, Any], issues: list[ValidationIssue]) -> None:
    query = args.get("query")
    rows = args.get("rows")
    start = args.get("start")
    if not isinstance(query, str) or len(query.strip()) < 3:
        issues.append(ValidationIssue("invalid_query", "`package_search.args.query` must be a useful string."))
    if not isinstance(rows, int) or not 1 <= rows <= 25:
        issues.append(ValidationIssue("invalid_rows", "`package_search.args.rows` must be an integer from 1 to 25."))
    if not isinstance(start, int) or start < 0:
        issues.append(ValidationIssue("invalid_start", "`package_search.args.start` must be a non-negative integer."))
    if any(token in str(query).casefold() for token in ["@", "://", "api_key", "password"]):
        issues.append(ValidationIssue("unsafe_query", "`package_search.args.query` must not contain credentials or URLs."))


def _validate_package_show(args: dict[str, Any], context: dict[str, Any], issues: list[ValidationIssue]) -> None:
    package_id = args.get("package_id")
    observed = set(context.get("observed_packages") or [])
    if not isinstance(package_id, str) or not package_id.strip():
        issues.append(ValidationIssue("invalid_package_id", "`package_show.args.package_id` must be a non-empty string."))
    elif observed and package_id not in observed:
        issues.append(ValidationIssue("unobserved_package", "`package_id` must refer to an observed package."))


def _validate_select_resource(args: dict[str, Any], context: dict[str, Any], issues: list[ValidationIssue]) -> None:
    package_id = args.get("package_id")
    resource_id = args.get("resource_id")
    reason = args.get("reason")
    observed_packages = set(context.get("observed_packages") or [])
    observed_resources = set(context.get("observed_resources") or [])
    combined_resource_id = f"{package_id}:{resource_id}"

    if not isinstance(package_id, str) or not package_id.strip():
        issues.append(ValidationIssue("invalid_package_id", "`select_resource.args.package_id` must be a non-empty string."))
    elif observed_packages and package_id not in observed_packages:
        issues.append(ValidationIssue("unobserved_package", "`package_id` must refer to an observed package."))
    if not isinstance(resource_id, str) or not resource_id.strip():
        issues.append(ValidationIssue("invalid_resource_id", "`select_resource.args.resource_id` must be a non-empty string."))
    elif observed_resources and combined_resource_id not in observed_resources and resource_id not in observed_resources:
        issues.append(ValidationIssue("unobserved_resource", "`resource_id` must refer to an observed resource."))
    if not isinstance(reason, str) or not reason.strip():
        issues.append(ValidationIssue("missing_reason", "`select_resource.args.reason` must explain the selection."))


def _validate_reject_result(args: dict[str, Any], issues: list[ValidationIssue]) -> None:
    reason = args.get("reason")
    next_query = args.get("next_query")
    if not isinstance(reason, str) or not reason.strip():
        issues.append(ValidationIssue("missing_reason", "`reject_result.args.reason` must be a non-empty string."))
    if not isinstance(next_query, str) or len(next_query.strip()) < 3:
        issues.append(ValidationIssue("invalid_next_query", "`reject_result.args.next_query` must be a useful string."))


def _validate_finish(args: dict[str, Any], context: dict[str, Any], issues: list[ValidationIssue]) -> None:
    selected_candidates = args.get("selected_candidates")
    rationale = args.get("rationale")
    if not isinstance(selected_candidates, list) or not selected_candidates:
        issues.append(ValidationIssue("missing_candidates", "`finish.args.selected_candidates` must be a non-empty list."))
    if not isinstance(rationale, str) or not rationale.strip():
        issues.append(ValidationIssue("missing_rationale", "`finish.args.rationale` must explain why retrieval is done."))
    if context and context.get("has_enough_evidence") is not True:
        issues.append(ValidationIssue("finish_too_early", "`finish` requires enough evidence in the current context."))


def _command_seed(args: argparse.Namespace) -> int:
    rows = seed_examples()
    write_jsonl(Path(args.output), rows)
    print(f"Wrote {len(rows)} seed examples to {args.output}.")
    return 0


def _command_validate(args: argparse.Namespace) -> int:
    rows = read_jsonl(Path(args.input))
    valid_rows = []
    report_rows = []
    for index, row in enumerate(rows, start=1):
        if row.get("_invalid_jsonl"):
            result = ValidationResult(False, [ValidationIssue("invalid_jsonl", row.get("_error", "Invalid JSONL row."))])
        else:
            result = validate_training_example(row)
        report_rows.append({"line": index, **result.to_dict()})
        if result.ok:
            valid_rows.append(row)

    if args.valid_output:
        write_jsonl(Path(args.valid_output), valid_rows)
    if args.report:
        write_jsonl(Path(args.report), report_rows)

    print(f"Validated {len(rows)} examples: {len(valid_rows)} valid, {len(rows) - len(valid_rows)} rejected.")
    return 0 if len(valid_rows) == len(rows) else 1


def _command_sample(args: argparse.Namespace) -> int:
    rows = [row for row in read_jsonl(Path(args.input)) if not row.get("_invalid_jsonl")]
    sampled = balanced_sample(rows, args.limit, args.key, args.seed)
    write_jsonl(Path(args.output), sampled)
    print(f"Wrote {len(sampled)} sampled rows to {args.output}.")
    return 0


def _command_split(args: argparse.Namespace) -> int:
    rows = [row for row in read_jsonl(Path(args.input)) if not row.get("_invalid_jsonl")]
    train_rows, eval_rows = split_by_package(rows, args.eval_size, args.train_size, args.seed, args.key)
    overlap = package_overlap(train_rows, eval_rows)
    if overlap:
        print(f"Split error: train/eval package overlap detected: {sorted(overlap)[:5]}")
        return 1
    write_jsonl(Path(args.train_output), train_rows)
    write_jsonl(Path(args.eval_output), eval_rows)
    print(f"Wrote {len(train_rows)} train scenarios to {args.train_output}.")
    print(f"Wrote {len(eval_rows)} eval scenarios to {args.eval_output}.")
    print("Package overlap: 0")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="CKAN retrieval dataset utilities.")
    subparsers = parser.add_subparsers(required=True)

    seed_parser = subparsers.add_parser("seed", help="Write a tiny seed JSONL dataset.")
    seed_parser.add_argument("--output", default="train/ckan/data/seed_examples.jsonl")
    seed_parser.set_defaults(func=_command_seed)

    validate_parser = subparsers.add_parser("validate", help="Validate CKAN retrieval training JSONL.")
    validate_parser.add_argument("--input", required=True)
    validate_parser.add_argument("--valid-output")
    validate_parser.add_argument("--report")
    validate_parser.set_defaults(func=_command_validate)

    sample_parser = subparsers.add_parser("sample", help="Write a deterministic balanced sample from JSONL rows.")
    sample_parser.add_argument("--input", required=True)
    sample_parser.add_argument("--output", required=True)
    sample_parser.add_argument("--limit", type=int, required=True)
    sample_parser.add_argument("--key", default="target_action")
    sample_parser.add_argument("--seed", type=int, default=13)
    sample_parser.set_defaults(func=_command_sample)

    split_parser = subparsers.add_parser("split", help="Split scenario JSONL into package-disjoint train/eval files.")
    split_parser.add_argument("--input", required=True)
    split_parser.add_argument("--train-output", required=True)
    split_parser.add_argument("--eval-output", required=True)
    split_parser.add_argument("--train-size", type=int)
    split_parser.add_argument("--eval-size", type=int, required=True)
    split_parser.add_argument("--key", default="target_action")
    split_parser.add_argument("--seed", type=int, default=17)
    split_parser.set_defaults(func=_command_split)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
