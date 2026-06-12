from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("training/data/raw/munich_filter_parameters.jsonl")
DEFAULT_OUTPUT_DIR = Path("training/data/generated")
SYSTEM_RETRIEVAL = (
    "You generate valid JSON arguments for the search_open_data tool. "
    "Convert the user's natural-language request into a concise Munich CKAN search query. "
    "Do not invent package IDs, resource IDs, or dataframe filters."
)
SYSTEM_FILTER = (
    "You generate valid JSON arguments for the query_dataframe tool. "
    "Use only the provided dataframe_id, columns, operators, and example values. "
    "Do not write pandas code."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_splits(items: list[dict[str, Any]], seed: int) -> dict[str, set[str]]:
    package_names = sorted({item["package_name"] for item in items if item.get("package_name")})
    rng = random.Random(seed)
    rng.shuffle(package_names)
    n = len(package_names)
    train_cut = max(1, int(n * 0.8))
    validation_cut = max(train_cut + 1, int(n * 0.9)) if n > 2 else train_cut
    return {
        "train": set(package_names[:train_cut]),
        "validation": set(package_names[train_cut:validation_cut]),
        "test": set(package_names[validation_cut:]),
    }


def split_name(package_name: str, splits: dict[str, set[str]]) -> str:
    for name, package_names in splits.items():
        if package_name in package_names:
            return name
    return "train"


def dataframe_id(entry: dict[str, Any]) -> str:
    package = entry["package_name"].replace("-", "_")
    resource = entry["resource_id"].replace("-", "_")
    return f"df_{package}_{resource}"


def compact_columns(entry: dict[str, Any]) -> list[dict[str, Any]]:
    columns = []
    for param in entry.get("filter_parameters", []):
        columns.append(
            {
                "name": param["column"],
                "dtype": param["dtype"],
                "kind": param["kind"],
                "operators": param["operators"],
                "examples": param.get("example_values", [])[:8],
                "min": param.get("min"),
                "max": param.get("max"),
            }
        )
    return columns


def retrieval_user_questions(entry: dict[str, Any]) -> list[str]:
    title = entry["package_title"]
    resource = entry.get("resource_name") or "the tabular resource"
    category = first_param(entry, {"category", "string"}, {"eq", "in", "contains"})
    year = first_param(entry, {"year"}, {"eq", "between", "gte"})
    metric = best_metric(entry)

    questions = [
        f"Ich brauche Daten zu {title}.",
        f"Welche offenen Daten gibt es in Muenchen zu {title}?",
        f"Show me Munich open data about {title}.",
        f"Find data from Munich related to {resource}.",
    ]
    if category and category.get("example_values"):
        questions.append(f"Zeige mir Informationen zu {title} fuer {category['example_values'][0]}.")
    if year and year.get("example_values"):
        questions.append(f"Was zeigt der Datensatz {title} im Jahr {year['example_values'][0]}?")
    if metric:
        questions.append(f"Wie hoch ist {metric['column']} im Datensatz {title}?")
    return questions


def retrieval_search_query(entry: dict[str, Any], question: str) -> str:
    title = entry["package_title"]
    resource = entry.get("resource_name") or ""
    query_parts = [title]
    if resource and resource.casefold() not in title.casefold():
        query_parts.append(resource)

    for param in entry.get("filter_parameters", []):
        if param.get("kind") in {"year", "category", "string"} and param.get("example_values"):
            value = str(param["example_values"][0])
            if value.casefold() in question.casefold() and value.casefold() not in " ".join(query_parts).casefold():
                query_parts.append(value)

    return " ".join(query_parts)


def make_retrieval_examples(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    seen = set()
    for entry in entries:
        key = (entry["package_name"], entry["resource_id"])
        if key in seen:
            continue
        seen.add(key)
        metadata = {
            "package_name": entry["package_name"],
            "package_title": entry["package_title"],
            "resource_id": entry["resource_id"],
            "resource_name": entry.get("resource_name"),
            "resource_format": entry.get("resource_format"),
            "datastore_active": entry.get("datastore_active"),
        }
        for question in retrieval_user_questions(entry):
            target = {
                "tool_name": "search_open_data",
                "arguments": {
                    "query": retrieval_search_query(entry, question),
                    "limit": 5,
                },
            }
            examples.append(
                {
                    "task": "search_open_data",
                    "messages": [
                        {"role": "system", "content": SYSTEM_RETRIEVAL},
                        {"role": "user", "content": f"Question: {question}"},
                        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
                    ],
                    "metadata": metadata,
                }
            )
    return examples


def first_param(entry: dict[str, Any], kinds: set[str], operators: set[str] | None = None) -> dict[str, Any] | None:
    for param in entry.get("filter_parameters", []):
        if param.get("column") == "_id":
            continue
        if param.get("kind") not in kinds:
            continue
        if operators and not operators.intersection(param.get("operators", [])):
            continue
        if not param.get("example_values") and param.get("min") is None:
            continue
        return param
    return None


def best_metric(entry: dict[str, Any]) -> dict[str, Any] | None:
    for param in entry.get("filter_parameters", []):
        if param.get("column") == "_id":
            continue
        if param.get("kind") == "numeric":
            return param
    return None


def output_columns(entry: dict[str, Any], *preferred: str | None) -> list[str]:
    columns = [column for column in preferred if column]
    for param in entry.get("filter_parameters", []):
        column = param["column"]
        if column not in columns and column != "_id":
            columns.append(column)
        if len(columns) >= 6:
            break
    return columns


def filter_prompt(entry: dict[str, Any], question: str) -> str:
    profile = {
        "dataframe_id": dataframe_id(entry),
        "package_name": entry["package_name"],
        "package_title": entry["package_title"],
        "resource_id": entry["resource_id"],
        "resource_name": entry.get("resource_name"),
        "filter_mode": entry.get("filter_mode"),
        "server_filter_supported": entry.get("server_filter_supported"),
        "columns": compact_columns(entry),
    }
    return f"Question: {question}\n\nDataframe profile:\n{json.dumps(profile, ensure_ascii=False)}"


def filter_example(
    entry: dict[str, Any],
    question: str,
    operation_spec: dict[str, Any],
) -> dict[str, Any]:
    target = {
        "tool_name": "query_dataframe",
        "arguments": {
            "dataframe_id": dataframe_id(entry),
            "operation_spec": operation_spec,
        },
    }
    return {
        "task": "query_dataframe",
        "messages": [
            {"role": "system", "content": SYSTEM_FILTER},
            {"role": "user", "content": filter_prompt(entry, question)},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
        "metadata": {
            "package_name": entry["package_name"],
            "resource_id": entry["resource_id"],
            "dataframe_id": dataframe_id(entry),
        },
    }


def make_filter_examples_for_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    examples = []
    category = first_param(entry, {"category", "string", "boolean"}, {"eq", "in", "contains"})
    year = first_param(entry, {"year"}, {"eq", "between", "gte"})
    date = first_param(entry, {"date"}, {"eq", "between", "gte"})
    metric = best_metric(entry)
    title = entry["package_title"]

    if category and category.get("example_values"):
        value = category["example_values"][0]
        column = category["column"]
        select = output_columns(entry, column, metric["column"] if metric else None)
        examples.append(
            filter_example(
                entry,
                f"Zeige mir die Daten zu {title} fuer {value}.",
                {
                    "filters": [{"column": column, "operator": "eq", "value": value}],
                    "select": select,
                    "limit": 50,
                },
            )
        )
        if len(category["example_values"]) >= 2:
            values = category["example_values"][:2]
            examples.append(
                filter_example(
                    entry,
                    f"Vergleiche {title} fuer {', '.join(values)}.",
                    {
                        "filters": [{"column": column, "operator": "in", "value": values}],
                        "select": select,
                        "limit": 50,
                    },
                )
            )
        if metric:
            examples.append(
                filter_example(
                    entry,
                    f"Was sind die wichtigsten Werte in {title} nach {column}?",
                    {
                        "group_by": [column],
                        "aggregate": [
                            {
                                "column": metric["column"],
                                "function": "sum",
                                "alias": f"sum_{metric['column']}",
                            }
                        ],
                        "sort": [{"column": f"sum_{metric['column']}", "direction": "desc"}],
                        "limit": 10,
                    },
                )
            )

    temporal = year or date
    if temporal:
        column = temporal["column"]
        if temporal["kind"] == "year" and temporal.get("min") is not None and temporal.get("max") is not None:
            low = int(temporal["min"])
            high = int(temporal["max"])
            if high - low > 3:
                low = high - 3
            value: Any = [low, high]
            question = f"Zeige mir {title} zwischen {low} und {high}."
            operator = "between"
        else:
            value = temporal.get("example_values", [""])[0]
            question = f"Zeige mir {title} fuer {value}."
            operator = "eq"
        operation = {
            "filters": [{"column": column, "operator": operator, "value": value}],
            "select": output_columns(entry, column, metric["column"] if metric else None),
            "limit": 100,
        }
        if metric:
            operation["sort"] = [{"column": metric["column"], "direction": "desc"}]
        examples.append(filter_example(entry, question, operation))

    if metric:
        column = metric["column"]
        examples.append(
            filter_example(
                entry,
                f"Wo sind die hoechsten Werte in {title}?",
                {
                    "select": output_columns(entry, column),
                    "sort": [{"column": column, "direction": "desc"}],
                    "limit": 10,
                },
            )
        )
        if metric.get("min") is not None and metric.get("max") is not None:
            threshold = round((float(metric["min"]) + float(metric["max"])) / 2, 2)
            examples.append(
                filter_example(
                    entry,
                    f"Zeige auffaellige hohe Werte in {title}.",
                    {
                        "filters": [{"column": column, "operator": "gte", "value": threshold}],
                        "select": output_columns(entry, column),
                        "sort": [{"column": column, "direction": "desc"}],
                        "limit": 50,
                    },
                )
            )

    examples.append(
        filter_example(
            entry,
            f"Gib mir einen kompakten Ueberblick ueber {title}.",
            {
                "select": output_columns(entry),
                "limit": 20,
            },
        )
    )
    return examples


def make_filter_examples(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    for entry in entries:
        examples.extend(make_filter_examples_for_entry(entry))
    return examples


def write_split_files(
    examples: list[dict[str, Any]],
    *,
    output_dir: Path,
    prefix: str,
    splits: dict[str, set[str]],
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        package_name = example["metadata"]["package_name"]
        grouped[split_name(package_name, splits)].append(example)

    for name in ["train", "validation", "test"]:
        write_jsonl(output_dir / f"{prefix}.{name}.jsonl", grouped[name])
    write_jsonl(output_dir / f"{prefix}.all.jsonl", examples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-resources", type=int, default=0)
    args = parser.parse_args()

    raw_entries = [entry for entry in read_jsonl(args.input) if entry.get("ok") and entry.get("filter_parameters")]
    if args.max_resources:
        raw_entries = raw_entries[: args.max_resources]

    splits = stable_splits(raw_entries, seed=args.seed)
    retrieval_examples = make_retrieval_examples(raw_entries)
    filter_examples = make_filter_examples(raw_entries)

    write_split_files(retrieval_examples, output_dir=args.output_dir, prefix="retrieval_query_tool", splits=splits)
    write_split_files(filter_examples, output_dir=args.output_dir, prefix="filter_tool", splits=splits)

    summary = {
        "source": str(args.input),
        "resources": len(raw_entries),
        "retrieval_examples": len(retrieval_examples),
        "filter_examples": len(filter_examples),
        "splits": {name: len(package_names) for name, package_names in splits.items()},
        "outputs": {
            "retrieval": str(args.output_dir / "retrieval_query_tool.{train,validation,test,all}.jsonl"),
            "filter": str(args.output_dir / "filter_tool.{train,validation,test,all}.jsonl"),
        },
    }
    (args.output_dir / "get_filter_dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
