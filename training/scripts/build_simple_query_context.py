from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_FILTER_PARAMETERS = Path("training/data/raw/munich_filter_parameters.jsonl")
DEFAULT_CATALOG = Path("training/data/raw/munich_catalog_sample.jsonl")
DEFAULT_OUTPUT_JSONL = Path("training/data/generated/simple_query_context.jsonl")
DEFAULT_OUTPUT_CSV = Path("training/data/generated/simple_query_context.csv")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def compact_text(value: str | None, max_chars: int = 1200) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()[:max_chars]


def simplify_column(param: dict[str, Any]) -> dict[str, Any]:
    simplified = {
        "name": param.get("column"),
        "dtype": param.get("dtype"),
        "kind": param.get("kind"),
    }
    examples = param.get("example_values") or []
    if examples:
        simplified["examples"] = examples[:5]
    if param.get("min") is not None:
        simplified["min"] = param.get("min")
    if param.get("max") is not None:
        simplified["max"] = param.get("max")
    return simplified


def simplify_example_row(entry: dict[str, Any], max_columns: int = 20) -> dict[str, Any] | None:
    sample_rows = entry.get("sample_rows") or []
    if not sample_rows or not isinstance(sample_rows[0], dict):
        return None

    simplified = {}
    for index, (key, value) in enumerate(sample_rows[0].items()):
        if index >= max_columns:
            break
        if isinstance(value, str):
            value = compact_text(value, max_chars=180)
        simplified[str(key)] = value
    return simplified


def build_context(
    *,
    filter_entries: list[dict[str, Any]],
    catalog_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    catalog_by_name = {entry.get("name"): entry for entry in catalog_entries if entry.get("name")}
    catalog_by_id = {entry.get("id"): entry for entry in catalog_entries if entry.get("id")}
    rows = []

    for entry in filter_entries:
        if not entry.get("ok") or not entry.get("filter_parameters"):
            continue
        if not entry.get("package_name") or not entry.get("resource_id"):
            continue

        catalog = catalog_by_name.get(entry.get("package_name")) or catalog_by_id.get(entry.get("package_id")) or {}
        rows.append(
            {
                "package_name": entry.get("package_name"),
                "package_title": entry.get("package_title"),
                "description": compact_text(catalog.get("notes") or entry.get("package_title")),
                "resource_id": entry.get("resource_id"),
                "resource_name": entry.get("resource_name"),
                "resource_format": entry.get("resource_format"),
                "filter_mode": entry.get("filter_mode"),
                "server_filter_supported": entry.get("server_filter_supported"),
                "columns": [
                    simplify_column(param)
                    for param in entry.get("filter_parameters", [])
                    if param.get("column")
                ],
                "example_row": simplify_example_row(entry),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build simplified context rows for LLM user-query generation.")
    parser.add_argument("--filter-parameters", type=Path, default=DEFAULT_FILTER_PARAMETERS)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()

    rows = build_context(
        filter_entries=read_jsonl(args.filter_parameters),
        catalog_entries=read_jsonl(args.catalog),
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    preview = pd.DataFrame(
        [
            {
                "package_name": row["package_name"],
                "package_title": row["package_title"],
                "resource_name": row["resource_name"],
                "resource_format": row["resource_format"],
                "columns": len(row["columns"]),
                "has_example_row": row["example_row"] is not None,
                "description": row["description"],
            }
            for row in rows
        ]
    )
    preview.to_csv(args.output_csv, index=False)
    print(json.dumps({"rows": len(rows), "jsonl": str(args.output_jsonl), "csv": str(args.output_csv)}, indent=2))


if __name__ == "__main__":
    main()
