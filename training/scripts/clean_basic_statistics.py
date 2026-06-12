from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("training/data/raw/munich_dataset_basic_statistics.jsonl")
DEFAULT_OUTPUT_JSONL = Path("training/data/generated/openui_statistics_samples.jsonl")
DEFAULT_OUTPUT_CSV = Path("training/data/generated/openui_statistics_samples.csv")

SUPPORTED_COMPONENTS = {"InsightCard", "MetricGrid", "DataTable", "BarChart", "Histogram", "Notice"}
UNSUPPORTED_TABLE_FORMATS = {"html", "wms", "geojson", "shape", "zip", "json", ""}
IDENTIFIER_NAMES = {
    "_id",
    "id",
    "fid",
    "uuid",
    "identifier",
    "record.identifier",
    "resource_id",
    "package_id",
}
LEGAL_BOILERPLATE_PATTERNS = [
    r"Das Referat .+? Schäden jeglicher Art\.",
    r"Eine Gewähr .+? übernommen\.",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def compact_text(value: str | None, max_chars: int = 500) -> str:
    if not value:
        return ""

    text = value
    for pattern in LEGAL_BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars].rstrip()


def normalize_format(value: str | None) -> str:
    return (value or "").strip().lower()


def safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def truncate_value(value: Any, max_chars: int = 160) -> Any:
    if isinstance(value, str):
        return compact_text(value, max_chars=max_chars)
    return value


@dataclass(frozen=True)
class CleanerConfig:
    max_samples: int | None = None
    max_columns: int = 12
    max_examples_per_column: int = 5
    include_notice_samples: bool = False
    min_quality_score: float = 3.0


class MunichBasicStatisticsCleaner:
    """Build compact OpenUI training inputs from raw Munich dataset statistics."""

    def __init__(self, config: CleanerConfig | None = None) -> None:
        self.config = config or CleanerConfig()

    def generate_samples(self, packages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        samples = []
        for package in packages:
            samples.extend(self.clean_package(package))

        samples.sort(key=lambda row: row["quality_score"], reverse=True)
        if self.config.max_samples is not None:
            return samples[: self.config.max_samples]
        return samples

    def clean_package(self, package: dict[str, Any]) -> list[dict[str, Any]]:
        samples = []
        for resource in package.get("resources") or []:
            sample = self.clean_resource(package, resource)
            if sample:
                samples.append(sample)
        return samples

    def clean_resource(self, package: dict[str, Any], resource: dict[str, Any]) -> dict[str, Any] | None:
        table = resource.get("table")
        resource_format = normalize_format(resource.get("format"))

        if not table:
            if self.config.include_notice_samples and resource.get("profile_error"):
                return self._build_notice_sample(package, resource)
            return None

        columns = [
            self._simplify_column(column)
            for column in table.get("columns") or []
            if column.get("name")
        ]
        columns = [column for column in columns if column["kind"] != "identifier"]
        ranked_columns = sorted(columns, key=self._column_score, reverse=True)
        selected_columns = ranked_columns[: self.config.max_columns]

        quality_score = self._resource_score(table, resource_format, selected_columns)
        if quality_score < self.config.min_quality_score:
            return None

        component_hints = self._component_hints(selected_columns)
        return {
            "task": "render_openui",
            "user_question": self._question_for_components(package, resource, component_hints),
            "query_result": {
                "dataset_title": package.get("title") or package.get("name"),
                "package_id": package.get("id"),
                "package_name": package.get("name"),
                "resource_id": resource.get("id"),
                "resource_name": resource.get("name"),
                "resource_format": resource_format,
                "description": compact_text(package.get("description")),
                "organization": package.get("organization"),
                "tags": (package.get("tags") or [])[:8],
                "groups": (package.get("groups") or [])[:5],
                "row_count": table.get("entry_count"),
                "sampled_rows": table.get("sampled_rows"),
                "column_count": table.get("column_count"),
                "columns": selected_columns,
            },
            "component_hints": component_hints,
            "quality_score": round(quality_score, 3),
            "cleaning_notes": {
                "dropped_columns": max(0, len(columns) - len(selected_columns)),
                "dropped_identifier_columns": sum(
                    1
                    for column in table.get("columns") or []
                    if self._infer_kind(column) == "identifier"
                ),
                "source_profile_error": resource.get("profile_error"),
            },
        }

    def _build_notice_sample(self, package: dict[str, Any], resource: dict[str, Any]) -> dict[str, Any] | None:
        resource_format = normalize_format(resource.get("format"))
        if resource_format not in UNSUPPORTED_TABLE_FORMATS:
            return None

        return {
            "task": "render_openui",
            "user_question": f"Explain why {resource.get('name') or package.get('title')} cannot be charted.",
            "query_result": {
                "dataset_title": package.get("title") or package.get("name"),
                "package_id": package.get("id"),
                "package_name": package.get("name"),
                "resource_id": resource.get("id"),
                "resource_name": resource.get("name"),
                "resource_format": resource_format,
                "description": compact_text(package.get("description")),
                "profile_error": compact_text(resource.get("profile_error"), max_chars=220),
                "columns": [],
            },
            "component_hints": {
                "recommended_components": ["Notice", "DataTable"],
                "reason": "Resource has no sampled table profile and should render an explanatory notice.",
            },
            "quality_score": 1.0,
            "cleaning_notes": {"notice_sample": True},
        }

    def _simplify_column(self, column: dict[str, Any]) -> dict[str, Any]:
        simplified = {
            "name": column.get("name"),
            "dtype": column.get("dtype"),
            "kind": self._infer_kind(column),
            "non_null_count": column.get("non_null_count"),
            "null_count": column.get("null_count"),
        }

        examples = [truncate_value(value) for value in column.get("example_values") or []]
        if examples:
            simplified["examples"] = examples[: self.config.max_examples_per_column]

        numeric = self._simplify_numeric_stats(column.get("numeric") or {})
        if numeric:
            simplified["numeric"] = numeric

        null_rate = self._null_rate(column)
        if null_rate is not None:
            simplified["null_rate"] = round(null_rate, 4)

        return {key: value for key, value in simplified.items() if value is not None}

    def _simplify_numeric_stats(self, numeric: dict[str, Any]) -> dict[str, Any]:
        stats = {}
        for key in ("count", "min", "max", "mean", "median"):
            number = safe_float(numeric.get(key))
            if number is not None:
                stats[key] = round(number, 4)
        return stats

    def _infer_kind(self, column: dict[str, Any]) -> str:
        name = str(column.get("name") or "").strip()
        lower_name = name.lower()
        dtype = str(column.get("dtype") or "").lower()
        examples = column.get("example_values") or []

        if lower_name in IDENTIFIER_NAMES or lower_name.endswith("_id") or lower_name.endswith(".identifier"):
            return "identifier"
        if "date" in lower_name or "datum" in lower_name or re.search(r"\b(jahr|year)\b", lower_name):
            return "year" if re.search(r"\b(jahr|year)\b", lower_name) else "date"
        if "uhrzeit" in lower_name or lower_name in {"time", "start_time", "end_time"}:
            return "time"
        if any(token in lower_name for token in ("lat", "lon", "koord", "shape", "point", "pos")):
            return "geospatial"
        if dtype.startswith(("int", "float")) or column.get("numeric"):
            return "numeric"
        if len(examples) <= self.config.max_examples_per_column and examples:
            return "category"
        return "string"

    def _column_score(self, column: dict[str, Any]) -> float:
        kind = column.get("kind")
        score_by_kind = {
            "numeric": 5.0,
            "date": 4.5,
            "year": 4.5,
            "category": 4.0,
            "time": 3.0,
            "geospatial": 2.5,
            "string": 2.0,
        }
        score = score_by_kind.get(str(kind), 0.0)

        null_rate = column.get("null_rate")
        if isinstance(null_rate, (int, float)):
            score -= float(null_rate) * 3.0

        if kind == "category" and len(column.get("examples") or []) < 2:
            score -= 2.0
        if column.get("examples"):
            score += 0.5
        if column.get("numeric"):
            score += 0.5
        return score

    def _resource_score(
        self,
        table: dict[str, Any],
        resource_format: str,
        selected_columns: list[dict[str, Any]],
    ) -> float:
        kinds = {column.get("kind") for column in selected_columns}
        score = 0.0
        if resource_format == "csv":
            score += 1.5
        elif resource_format in {"xml", "txt"}:
            score += 0.5

        row_count = safe_float(table.get("entry_count")) or safe_float(table.get("sampled_rows")) or 0.0
        if row_count >= 100:
            score += 1.5
        elif row_count > 0:
            score += 0.75

        if "numeric" in kinds:
            score += 2.0
        if kinds.intersection({"date", "year"}):
            score += 1.5
        if "category" in kinds:
            score += 1.0
        if len(selected_columns) >= 3:
            score += 1.0
        return score

    def _component_hints(self, columns: list[dict[str, Any]]) -> dict[str, Any]:
        numeric_columns = [column["name"] for column in columns if column.get("kind") == "numeric"]
        category_columns = [column["name"] for column in columns if column.get("kind") == "category"]
        time_columns = [
            column["name"]
            for column in columns
            if column.get("kind") in {"date", "year", "time"}
        ]

        recommended = ["InsightCard"]
        chart: dict[str, Any] | None = None
        if numeric_columns:
            recommended.append("MetricGrid")
            if category_columns:
                recommended.append("BarChart")
                chart = {
                    "type": "BarChart",
                    "x": category_columns[0],
                    "y": numeric_columns[0],
                }
            else:
                recommended.append("Histogram")
                chart = {"type": "Histogram", "x": numeric_columns[0]}
        recommended.append("DataTable")

        hints: dict[str, Any] = {
            "recommended_components": [component for component in recommended if component in SUPPORTED_COMPONENTS],
            "numeric_columns": numeric_columns[:4],
            "category_columns": category_columns[:4],
            "time_columns": time_columns[:3],
        }
        if chart:
            hints["primary_chart"] = chart
        return hints

    def _question_for_components(
        self,
        package: dict[str, Any],
        resource: dict[str, Any],
        component_hints: dict[str, Any],
    ) -> str:
        title = package.get("title") or package.get("name") or "this dataset"
        resource_name = resource.get("name")
        chart = component_hints.get("primary_chart") or {}
        if chart.get("type") == "BarChart":
            return f"Summarize {title} and compare {chart['y']} by {chart['x']}."
        if chart.get("type") == "Histogram":
            return f"Show the distribution of {chart['x']} in {title}."
        if resource_name:
            return f"Create a compact OpenUI summary for {resource_name} from {title}."
        return f"Create a compact OpenUI summary for {title}."

    def _null_rate(self, column: dict[str, Any]) -> float | None:
        null_count = safe_float(column.get("null_count"))
        non_null_count = safe_float(column.get("non_null_count"))
        if null_count is None or non_null_count is None:
            return None
        total = null_count + non_null_count
        if total <= 0:
            return None
        return null_count / total


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_preview_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "quality_score",
        "dataset_title",
        "resource_name",
        "resource_format",
        "row_count",
        "columns",
        "components",
        "question",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "quality_score": row["quality_score"],
                    "dataset_title": row["query_result"].get("dataset_title"),
                    "resource_name": row["query_result"].get("resource_name"),
                    "resource_format": row["query_result"].get("resource_format"),
                    "row_count": row["query_result"].get("row_count"),
                    "columns": len(row["query_result"].get("columns") or []),
                    "components": ",".join(row["component_hints"].get("recommended_components") or []),
                    "question": row["user_question"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw Munich statistics into compact OpenUI samples.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-columns", type=int, default=CleanerConfig.max_columns)
    parser.add_argument("--include-notice-samples", action="store_true")
    parser.add_argument("--min-quality-score", type=float, default=CleanerConfig.min_quality_score)
    args = parser.parse_args()

    cleaner = MunichBasicStatisticsCleaner(
        CleanerConfig(
            max_samples=args.max_samples,
            max_columns=args.max_columns,
            include_notice_samples=args.include_notice_samples,
            min_quality_score=args.min_quality_score,
        )
    )
    samples = cleaner.generate_samples(read_jsonl(args.input))
    write_jsonl(args.output_jsonl, samples)
    write_preview_csv(args.output_csv, samples)
    print(json.dumps({"rows": len(samples), "jsonl": str(args.output_jsonl), "csv": str(args.output_csv)}, indent=2))


if __name__ == "__main__":
    main()
