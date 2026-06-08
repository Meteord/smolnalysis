from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("training/data/generated/simple_query_context.jsonl")
DEFAULT_CATALOG = Path("training/data/raw/munich_catalog_sample.jsonl")
DEFAULT_OUTPUT = Path("training/data/generated/llm_user_queries.jsonl")
DEFAULT_MODEL = "gpt-4o-mini"


SYSTEM_PROMPT = """You generate realistic user questions for a Munich open-data assistant.

The user is not a data engineer. They ask natural questions in German, English, or mixed German/English.
The assistant will later:
1. search the Munich CKAN catalog,
2. fetch the relevant resource as a dataframe,
3. generate dataframe filter/aggregation parameters from the dataframe columns.

Generate questions that are answerable from the provided dataset description and columns.
Do not copy the dataset title verbatim in every question.
Do not invent columns, values, years, districts, categories, or facts that are not implied by the provided context.
Prefer concrete civic-data questions over generic "show me this dataset" requests.

Return JSON only.
"""


USER_PROMPT_TEMPLATE = """Create {count} realistic user questions for this Munich open-data resource.

Dataset context:
{context_json}

Return exactly this JSON shape:
{{
  "queries": [
    {{
      "query": "natural user question",
      "language": "de|en|mixed",
      "intent": "short description of what the user wants",
      "filter_intent": {{
        "filters": [
          {{"column": "provided column name", "operator": "provided operator", "value": "provided example value or numeric/date range"}}
        ],
        "group_by": ["provided column name"],
        "aggregate": [
          {{"column": "provided numeric column name", "function": "count|sum|mean|median|min|max"}}
        ],
        "sort": [
          {{"column": "provided column or aggregate alias", "direction": "asc|desc"}}
        ],
        "limit": 10
      }}
    }}
  ]
}}

Rules:
- `filter_intent` can be empty if the question is only about finding or previewing the dataset.
- Use only provided column names.
- Use only provided operators for filters.
- Use only example values shown in the column metadata, unless using a numeric min/max range.
- Make at least half of the questions German.
- Include a mix of simple filtering, top-k, comparisons, year/date filtering if possible, and overview questions.
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_catalog_notes(path: Path) -> dict[str, dict[str, Any]]:
    catalog = {}
    for entry in read_jsonl(path):
        package_name = entry.get("name")
        if package_name:
            catalog[package_name] = entry
    return catalog


def compact_text(value: str | None, max_chars: int = 900) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()[:max_chars]


def compact_columns(entry: dict[str, Any], max_columns: int = 18, max_examples: int = 8) -> list[dict[str, Any]]:
    if entry.get("columns"):
        return [
            {
                "name": column.get("name"),
                "dtype": column.get("dtype"),
                "kind": column.get("kind"),
                "operators": column.get("operators", []),
                "examples": column.get("examples", [])[:max_examples],
                "min": column.get("min"),
                "max": column.get("max"),
            }
            for column in entry.get("columns", [])[:max_columns]
            if column.get("name")
        ]

    columns = []
    for param in entry.get("filter_parameters", []):
        column = param.get("column")
        if not column:
            continue
        columns.append(
            {
                "name": column,
                "dtype": param.get("dtype"),
                "kind": param.get("kind"),
                "operators": param.get("operators", []),
                "examples": param.get("example_values", [])[:max_examples],
                "min": param.get("min"),
                "max": param.get("max"),
            }
        )
        if len(columns) >= max_columns:
            break
    return columns


def resource_context(entry: dict[str, Any], catalog_by_name: dict[str, dict[str, Any]]) -> dict[str, Any]:
    catalog_entry = catalog_by_name.get(entry.get("package_name"), {})
    return {
        "package_title": entry.get("package_title"),
        "dataset_description": compact_text(entry.get("description") or catalog_entry.get("notes") or entry.get("package_title")),
        "resource_name": entry.get("resource_name"),
        "resource_format": entry.get("resource_format"),
        "filter_mode": entry.get("filter_mode"),
        "server_filter_supported": entry.get("server_filter_supported"),
        "columns": compact_columns(entry),
        "example_row": entry.get("example_row"),
    }


def strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_llm_json(text: str) -> dict[str, Any]:
    text = strip_json_fence(text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict) or not isinstance(payload.get("queries"), list):
        raise ValueError("LLM response must be an object with a `queries` list.")
    return payload


def validate_filter_intent(filter_intent: dict[str, Any], columns: list[dict[str, Any]]) -> dict[str, Any]:
    column_map = {column["name"]: column for column in columns}
    cleaned: dict[str, Any] = {}

    filters = []
    for condition in filter_intent.get("filters", []) or []:
        column = condition.get("column")
        op = condition.get("operator")
        if column not in column_map:
            continue
        if op not in column_map[column].get("operators", []):
            continue
        filters.append(
            {
                "column": column,
                "operator": op,
                "value": condition.get("value"),
            }
        )
    if filters:
        cleaned["filters"] = filters

    group_by = [column for column in filter_intent.get("group_by", []) or [] if column in column_map]
    if group_by:
        cleaned["group_by"] = group_by

    aggregates = []
    for aggregate in filter_intent.get("aggregate", []) or []:
        column = aggregate.get("column")
        function = aggregate.get("function")
        if column in column_map and function in {"count", "sum", "mean", "median", "min", "max"}:
            aggregates.append({"column": column, "function": function})
    if aggregates:
        cleaned["aggregate"] = aggregates

    sort = []
    for item in filter_intent.get("sort", []) or []:
        column = item.get("column")
        direction = item.get("direction", "desc")
        if column in column_map or any(column == f"{agg['function']}_{agg['column']}" for agg in aggregates):
            sort.append({"column": column, "direction": "desc" if direction == "desc" else "asc"})
    if sort:
        cleaned["sort"] = sort

    limit = filter_intent.get("limit")
    if isinstance(limit, int) and 1 <= limit <= 1000:
        cleaned["limit"] = limit

    return cleaned


def normalize_query_row(
    *,
    entry: dict[str, Any],
    query: dict[str, Any],
    columns: list[dict[str, Any]],
    index: int,
    model: str,
) -> dict[str, Any] | None:
    text = compact_text(str(query.get("query") or ""), max_chars=300)
    if len(text) < 8:
        return None

    filter_intent = query.get("filter_intent") or {}
    if not isinstance(filter_intent, dict):
        filter_intent = {}

    return {
        "query_id": f"{entry['package_name']}::{entry['resource_id']}::{index}",
        "query": text,
        "language": query.get("language") if query.get("language") in {"de", "en", "mixed"} else "de",
        "intent": compact_text(str(query.get("intent") or ""), max_chars=260),
        "search_target": {
            "tool_name": "search_open_data",
            "arguments": {
                "query": search_query_for_entry(entry),
                "limit": 5,
            },
        },
        "filter_intent": validate_filter_intent(filter_intent, columns),
        "package_name": entry.get("package_name"),
        "package_title": entry.get("package_title"),
        "resource_id": entry.get("resource_id"),
        "resource_name": entry.get("resource_name"),
        "resource_format": entry.get("resource_format"),
        "filter_mode": entry.get("filter_mode"),
        "server_filter_supported": entry.get("server_filter_supported"),
        "columns": columns,
        "generated_by": model,
    }


def search_query_for_entry(entry: dict[str, Any]) -> str:
    title = entry.get("package_title") or entry.get("package_name") or ""
    resource = entry.get("resource_name") or ""
    if resource and resource.casefold() not in title.casefold():
        return f"{title} {resource}".strip()
    return title.strip()


def completion(model: str, messages: list[dict[str, str]], temperature: float) -> str:
    try:
        from litellm import completion as litellm_completion
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency `litellm`. Install it with `uv add litellm` or `pip install litellm`."
        ) from exc

    response = litellm_completion(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


def generate_queries_for_entry(
    *,
    entry: dict[str, Any],
    catalog_by_name: dict[str, dict[str, Any]],
    model: str,
    count: int,
    temperature: float,
) -> list[dict[str, Any]]:
    context = resource_context(entry, catalog_by_name)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                count=count,
                context_json=json.dumps(context, ensure_ascii=False, indent=2),
            ),
        },
    ]
    text = completion(model, messages, temperature)
    payload = parse_llm_json(text)
    columns = context["columns"]
    rows = []
    for index, query in enumerate(payload.get("queries", []), start=1):
        if not isinstance(query, dict):
            continue
        row = normalize_query_row(entry=entry, query=query, columns=columns, index=index, model=model)
        if row:
            rows.append(row)
    return rows


def already_done(path: Path) -> set[tuple[str, str]]:
    done = set()
    for row in read_jsonl(path):
        package_name = row.get("package_name")
        resource_id = row.get("resource_id")
        if package_name and resource_id:
            done.add((package_name, resource_id))
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate realistic user queries for Munich CKAN resources with LiteLLM.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--queries-per-resource", type=int, default=8)
    parser.add_argument("--max-resources", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.overwrite and args.output.exists():
        args.output.unlink()

    entries = [
        entry
        for entry in read_jsonl(args.input)
        if entry.get("columns") or (entry.get("ok") and entry.get("filter_parameters"))
    ]
    rng = random.Random(args.seed)
    rng.shuffle(entries)
    if args.max_resources:
        entries = entries[: args.max_resources]

    catalog_by_name = load_catalog_notes(args.catalog)
    done = already_done(args.output) if args.resume else set()
    summary = {
        "input": str(args.input),
        "catalog": str(args.catalog),
        "output": str(args.output),
        "model": args.model,
        "requested_resources": len(entries),
        "processed_resources": 0,
        "generated_queries": 0,
        "errors": [],
    }

    for entry in entries:
        key = (entry.get("package_name"), entry.get("resource_id"))
        if key in done:
            continue
        try:
            rows = generate_queries_for_entry(
                entry=entry,
                catalog_by_name=catalog_by_name,
                model=args.model,
                count=args.queries_per_resource,
                temperature=args.temperature,
            )
            append_jsonl(args.output, rows)
            summary["processed_resources"] += 1
            summary["generated_queries"] += len(rows)
            print(f"{entry.get('package_name')} / {entry.get('resource_name')}: {len(rows)} queries")
        except Exception as exc:
            error = {
                "package_name": entry.get("package_name"),
                "resource_id": entry.get("resource_id"),
                "error": str(exc),
            }
            summary["errors"].append(error)
            print(f"ERROR {error}")
        if args.sleep:
            time.sleep(args.sleep)

    write_json(args.output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
