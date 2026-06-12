from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from truststore import inject_into_ssl
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

inject_into_ssl()
load_dotenv()

DEFAULT_INPUT = Path("training/data/generated/simple_query_context.jsonl")
DEFAULT_CATALOG = Path("training/data/raw/munich_catalog_sample.jsonl")
DEFAULT_OUTPUT = Path("training/data/generated/llm_user_queries.jsonl")
DEFAULT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")


SYSTEM_PROMPT = """You generate realistic user questions for a Munich open-data assistant.

The user is not a data engineer. They ask natural questions in German, English, or mixed German/English.
The assistant will later:
1. search the Munich CKAN catalog,
2. fetch the relevant resource as a dataframe,
3. answer questions using the dataframe columns.

Generate questions that are answerable from the provided dataset description and columns.
Do not copy the dataset title verbatim in every question.
Do not invent columns, values, years, districts, categories, or facts that are not implied by the provided context.
Prefer concrete civic-data questions over generic "show me this dataset" requests.
"""


USER_PROMPT_TEMPLATE = """Create {count} realistic user questions for this Munich open-data resource.

Dataset context:
{context_json}

Rules:
- Return exactly {count} questions.
- Make at least half of the questions German.
- Use only provided column names and example values when referring to concrete fields or values.
- Include a mix of overview, filtering, comparison, top-k, and year/date questions where the columns support them.
"""


class GeneratedQuery(BaseModel):
    query: str = Field(description="Natural user question.")
    language: Literal["de", "en", "mixed"] = Field(description="Question language.")
    intent: str = Field(description="Short description of what the user wants.")


class GeneratedQueries(BaseModel):
    queries: list[GeneratedQuery] = Field(description="Generated natural user questions.")


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


def compact_columns(entry: dict[str, Any], max_columns: int = 12, max_examples: int = 5) -> list[dict[str, Any]]:
    if entry.get("columns"):
        return [
            {
                "name": column.get("name"),
                "dtype": column.get("dtype"),
                "kind": column.get("kind"),
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


def normalize_query_row(
    *,
    entry: dict[str, Any],
    query: GeneratedQuery,
    columns: list[dict[str, Any]],
    index: int,
    model: str,
) -> dict[str, Any] | None:
    text = compact_text(query.query, max_chars=300)
    if len(text) < 8:
        return None

    return {
        "query_id": f"{entry['package_name']}::{entry['resource_id']}::{index}",
        "query": text,
        "language": query.language,
        "intent": compact_text(query.intent, max_chars=260),
        "search_target": {
            "tool_name": "search_open_data",
            "arguments": {
                "query": search_query_for_entry(entry),
                "limit": 5,
            },
        },
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


def generate_structured_queries(model: str, messages: list[tuple[str, str]], temperature: float) -> GeneratedQueries:
    try:
        from langchain_openai.chat_models import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency `langchain-openai`. Install it with `uv add langchain-openai`."
        ) from exc

    chat_model = ChatOpenAI(
        model=model,
        temperature=temperature,
        timeout=int(os.getenv("LLM_TIMEOUT", "30")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
    )
    structured_model = chat_model.with_structured_output(GeneratedQueries, method="json_schema")
    result = structured_model.invoke(messages)
    if isinstance(result, GeneratedQueries):
        return result
    return GeneratedQueries.model_validate(result)


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
        ("system", SYSTEM_PROMPT),
        (
            "user",
            USER_PROMPT_TEMPLATE.format(
                count=count,
                context_json=json.dumps(context, ensure_ascii=False, indent=2),
            ),
        ),
    ]
    payload = generate_structured_queries(model, messages, temperature)
    columns = context["columns"]
    rows = []
    for index, query in enumerate(payload.queries[:count], start=1):
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
    parser = argparse.ArgumentParser(description="Generate realistic user queries for Munich CKAN resources.")
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
