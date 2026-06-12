from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("training/data/raw/munich_dataset_basic_statistics.jsonl")
DEFAULT_OUTPUT_JSONL = Path("training/data/generated/openui_statistics_samples.jsonl")
DEFAULT_OUTPUT_CSV = Path("training/data/generated/openui_statistics_samples.csv")
DEFAULT_TRAIN_JSONL = Path("training/data/generated/openui_statistics_train.jsonl")
DEFAULT_VALIDATION_JSONL = Path("training/data/generated/openui_statistics_validation.jsonl")
DEFAULT_TEST_JSONL = Path("training/data/generated/openui_statistics_test.jsonl")
DEFAULT_MANIFEST = Path("training/data/generated/openui_statistics_dataset.manifest.json")

SUPPORTED_COMPONENTS = {"InsightCard", "MetricGrid", "DataTable", "BarChart", "Histogram", "Notice"}
OPENUI_SYSTEM_PROMPT = (
    "You translate smolnalysis workflow results into OpenUI-Lang. "
    "Output OpenUI-Lang only."
)
CHAT_COMPONENTS = {
    "BarChart",
    "Callout",
    "Card",
    "CardHeader",
    "Col",
    "FollowUpBlock",
    "FollowUpItem",
    "ListBlock",
    "ListItem",
    "Series",
    "Table",
    "TextContent",
}
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


def json_arg(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def make_identifier(prefix: str, index: int) -> str:
    return f"{prefix}{index}"


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
        sample = {
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
        return attach_training_messages(sample)

    def _build_notice_sample(self, package: dict[str, Any], resource: dict[str, Any]) -> dict[str, Any] | None:
        resource_format = normalize_format(resource.get("format"))
        if resource_format not in UNSUPPORTED_TABLE_FORMATS:
            return None

        sample = {
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
        return attach_training_messages(sample)

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


def attach_training_messages(sample: dict[str, Any]) -> dict[str, Any]:
    openui_lang = build_openui_lang(sample)
    validate_openui_chat_lang(openui_lang)
    user_payload = {
        "task": sample["task"],
        "user_question": sample["user_question"],
        "query_result": sample["query_result"],
        "component_hints": sample["component_hints"],
    }
    sample["openui_lang"] = openui_lang
    sample["messages"] = [
        {"role": "system", "content": OPENUI_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
        {"role": "assistant", "content": openui_lang},
    ]
    sample["validation"] = {
        "openui_chat_components": True,
        "root_component": "Card",
        "assistant_output": "openui_lang_only",
    }
    return sample


def build_openui_lang(sample: dict[str, Any]) -> str:
    query_result = sample["query_result"]
    hints = sample["component_hints"]
    title = compact_text(query_result.get("dataset_title") or "Dataset", max_chars=90)
    resource_name = compact_text(query_result.get("resource_name") or "Selected resource", max_chars=90)
    resource_format = str(query_result.get("resource_format") or "unknown").upper()
    row_count = query_result.get("row_count")
    column_count = query_result.get("column_count") or len(query_result.get("columns") or [])
    columns = query_result.get("columns") or []

    if not columns:
        error_text = query_result.get("profile_error") or "No tabular profile was available for this resource."
        return "\n".join(
            [
                "root = Card([header, summary, callout, followups])",
                f"header = CardHeader({json_arg('OpenUI resource summary')}, {json_arg(resource_name)})",
                f"summary = TextContent({json_arg(title)}, \"default\")",
                f"callout = Callout(\"warning\", \"No chartable table profile\", {json_arg(compact_text(error_text, max_chars=220))})",
                "followups = FollowUpBlock([f1, f2])",
                'f1 = FollowUpItem("Search for a CSV resource")',
                'f2 = FollowUpItem("Inspect dataset metadata")',
            ]
        )

    lines = [
        "root = Card([header, summary, metrics, schema, chart, quality, followups])",
        f"header = CardHeader({json_arg(title)}, {json_arg(resource_name)})",
        f"summary = TextContent({json_arg(_summary_text(query_result, hints))}, \"default\")",
        "metrics = ListBlock([m1, m2, m3, m4], \"number\")",
        f"m1 = ListItem(\"Rows\", {json_arg(_format_count(row_count) if row_count is not None else 'unknown')})",
        f"m2 = ListItem(\"Columns\", {json_arg(_format_count(column_count))})",
        f"m3 = ListItem(\"Format\", {json_arg(resource_format)})",
        f"m4 = ListItem(\"Quality\", {json_arg(str(sample.get('quality_score', 'n/a')))})",
    ]
    lines.extend(_schema_table_lines(columns[:8]))
    lines.extend(_chart_lines(hints, columns))
    lines.extend(
        [
            f"quality = Callout(\"info\", \"Training signal\", {json_arg(_quality_text(sample))})",
            "followups = FollowUpBlock([f1, f2, f3])",
            'f1 = FollowUpItem("Show a bar chart")',
            'f2 = FollowUpItem("Check missing values")',
            'f3 = FollowUpItem("Summarize another resource")',
        ]
    )
    return "\n".join(lines)


def _summary_text(query_result: dict[str, Any], hints: dict[str, Any]) -> str:
    organization = query_result.get("organization") or "unknown organization"
    numeric = hints.get("numeric_columns") or []
    category = hints.get("category_columns") or []
    parts = [
        f"{query_result.get('resource_name') or 'This resource'} belongs to {organization}.",
        f"It exposes {len(query_result.get('columns') or [])} profiled columns for OpenUI rendering.",
    ]
    if numeric:
        parts.append(f"Numeric fields include {', '.join(numeric[:3])}.")
    if category:
        parts.append(f"Categorical fields include {', '.join(category[:3])}.")
    return compact_text(" ".join(parts), max_chars=420)


def _format_count(value: Any) -> str:
    number = safe_float(value)
    if number is None:
        return str(value)
    return f"{int(number):,}" if number.is_integer() else f"{number:,.2f}"


def _schema_table_lines(columns: list[dict[str, Any]]) -> list[str]:
    names = [column.get("name") for column in columns]
    kinds = [column.get("kind") for column in columns]
    dtypes = [column.get("dtype") for column in columns]
    missing = [_format_count(column.get("null_count", 0)) for column in columns]
    return [
        f"c1 = Col(\"Column\", {json_arg(names)}, \"string\")",
        f"c2 = Col(\"Kind\", {json_arg(kinds)}, \"string\")",
        f"c3 = Col(\"Type\", {json_arg(dtypes)}, \"string\")",
        f"c4 = Col(\"Missing\", {json_arg(missing)}, \"string\")",
        "schema = Table([c1, c2, c3, c4])",
    ]


def _chart_lines(hints: dict[str, Any], columns: list[dict[str, Any]]) -> list[str]:
    chart = hints.get("primary_chart") or {}
    if chart.get("type") != "BarChart":
        numeric_name = (hints.get("numeric_columns") or [None])[0]
        numeric_column = next((column for column in columns if column.get("name") == numeric_name), None)
        values = _numeric_examples(numeric_column)
        if not values:
            values = [0.0]
        labels = [f"v{i + 1}" for i in range(len(values))]
        return [
            f"series = Series({json_arg(numeric_name or 'value')}, {json_arg(values)})",
            f"chart = BarChart({json_arg(labels)}, [series], \"grouped\", \"Example\", {json_arg(numeric_name or 'Value')})",
        ]

    x_name = chart.get("x")
    y_name = chart.get("y")
    x_column = next((column for column in columns if column.get("name") == x_name), None)
    y_column = next((column for column in columns if column.get("name") == y_name), None)
    labels = [str(value) for value in (x_column or {}).get("examples") or []]
    values = _numeric_examples(y_column)
    size = min(len(labels), len(values), 8)
    if size == 0:
        labels = ["sample"]
        values = [0.0]
    else:
        labels = labels[:size]
        values = values[:size]
    return [
        f"series = Series({json_arg(str(y_name or 'value'))}, {json_arg(values)})",
        f"chart = BarChart({json_arg(labels)}, [series], \"grouped\", {json_arg(str(x_name or 'category'))}, {json_arg(str(y_name or 'value'))})",
    ]


def _numeric_examples(column: dict[str, Any] | None) -> list[float]:
    values = []
    for value in (column or {}).get("examples") or []:
        number = safe_float(value)
        if number is not None:
            values.append(round(number, 4))
    numeric = (column or {}).get("numeric") or {}
    if not values:
        for key in ("min", "median", "mean", "max"):
            number = safe_float(numeric.get(key))
            if number is not None:
                values.append(round(number, 4))
    return values[:8]


def _quality_text(sample: dict[str, Any]) -> str:
    notes = sample.get("cleaning_notes") or {}
    dropped = notes.get("dropped_columns", 0)
    dropped_ids = notes.get("dropped_identifier_columns", 0)
    return (
        f"The cleaner kept the most useful columns, dropped {dropped} lower-ranked columns, "
        f"and removed {dropped_ids} identifier columns before creating this OpenUI target."
    )


def validate_openui_chat_lang(openui_lang: str) -> None:
    components: dict[str, tuple[str, list[Any]]] = {}
    references: set[str] = set()
    for raw_line in openui_lang.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)", line)
        if not match:
            raise ValueError(f"Invalid OpenUI statement: {line}")
        identifier, component_type, args_text = match.groups()
        if component_type not in CHAT_COMPONENTS:
            raise ValueError(f"Unsupported OpenUI chat component: {component_type}")
        args = [_parse_openui_value(part) for part in _split_openui_args(args_text)]
        components[identifier] = (component_type, args)
        references.update(_collect_refs(args))

    if "root" not in components:
        raise ValueError("OpenUI-Lang must include root.")
    root_type, root_args = components["root"]
    if root_type != "Card" or not root_args:
        raise ValueError("root must be a Card([...]) component.")

    missing = sorted(ref for ref in references if ref not in components)
    if missing:
        raise ValueError(f"Missing OpenUI definitions: {', '.join(missing)}")

    for identifier, (component_type, args) in components.items():
        _validate_component_args(identifier, component_type, args)


def _split_openui_args(args_text: str) -> list[str]:
    args: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
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


def _parse_openui_value(value: str) -> Any:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_openui_value(part) for part in _split_openui_args(inner)]
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        return {"$ref": value}
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid OpenUI argument: {value}") from exc


def _collect_refs(values: Any) -> set[str]:
    if isinstance(values, dict) and "$ref" in values:
        return {values["$ref"]}
    if isinstance(values, list):
        refs: set[str] = set()
        for value in values:
            refs.update(_collect_refs(value))
        return refs
    return set()


def _validate_component_args(identifier: str, component_type: str, args: list[Any]) -> None:
    min_args = {
        "BarChart": 5,
        "Callout": 3,
        "Card": 1,
        "CardHeader": 2,
        "Col": 3,
        "FollowUpBlock": 1,
        "FollowUpItem": 1,
        "ListBlock": 2,
        "ListItem": 2,
        "Series": 2,
        "Table": 1,
        "TextContent": 2,
    }[component_type]
    if len(args) < min_args:
        raise ValueError(f"{identifier}={component_type} has too few arguments.")
    if component_type in {"Card", "FollowUpBlock", "ListBlock", "Table"} and not isinstance(args[0], list):
        raise ValueError(f"{identifier}={component_type} must receive a reference list.")
    if component_type == "Series" and not all(isinstance(value, (int, float)) for value in args[1]):
        raise ValueError(f"{identifier}=Series values must be numeric.")
    if component_type == "BarChart" and not isinstance(args[0], list):
        raise ValueError(f"{identifier}=BarChart labels must be a list.")
    if component_type == "Col" and not isinstance(args[1], list):
        raise ValueError(f"{identifier}=Col values must be a list.")


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


def write_split_jsonl(
    rows: list[dict[str, Any]],
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    manifest_path: Path,
    seed: int = 42,
) -> dict[str, Any]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * 0.8)
    validation_end = train_end + int(len(shuffled) * 0.1)
    splits = {
        "train": (train_path, shuffled[:train_end]),
        "validation": (validation_path, shuffled[train_end:validation_end]),
        "test": (test_path, shuffled[validation_end:]),
    }
    for path, split_rows in splits.values():
        write_jsonl(path, split_rows)

    manifest = {
        "source": str(DEFAULT_OUTPUT_JSONL),
        "seed": seed,
        "format": "chat-style JSONL; each row includes messages plus trace fields",
        "assistant_target": "messages[-1].content",
        "rows": len(rows),
        "splits": {
            split_name: {"path": str(path), "count": len(split_rows)}
            for split_name, (path, split_rows) in splits.items()
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw Munich statistics into compact OpenUI samples.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--validation-jsonl", type=Path, default=DEFAULT_VALIDATION_JSONL)
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST_JSONL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-seed", type=int, default=42)
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
    manifest = write_split_jsonl(
        samples,
        args.train_jsonl,
        args.validation_jsonl,
        args.test_jsonl,
        args.manifest,
        seed=args.split_seed,
    )
    print(
        json.dumps(
            {
                "rows": len(samples),
                "jsonl": str(args.output_jsonl),
                "csv": str(args.output_csv),
                "manifest": str(args.manifest),
                "splits": manifest["splits"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
