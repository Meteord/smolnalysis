from __future__ import annotations

import json
import operator
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


BASE_URL = "https://opendata.muenchen.de/api/3/action"
USER_AGENT = "smolnalysis-open-data-tools/0.1"
PARSABLE_FORMATS = {"csv", "txt", "xml", "rdf", "rss", "atom", "html", "htm"}


class ToolValidationError(ValueError):
    def __init__(
        self,
        error_type: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error_type": self.error_type,
            "message": self.message,
            **self.details,
        }


@dataclass
class DataframeStore:
    root: Path
    index: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, dataframe_id: str, df: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, Any]:
        path = self.root / f"{dataframe_id}.parquet"
        fallback_path = self.root / f"{dataframe_id}.csv"
        try:
            df.to_parquet(path, index=False)
            storage_path = path
            storage_format = "parquet"
        except Exception:
            df.to_csv(fallback_path, index=False)
            storage_path = fallback_path
            storage_format = "csv"

        profile = dataframe_profile(dataframe_id, df, metadata)
        profile["storage_path"] = str(storage_path)
        profile["storage_format"] = storage_format
        self.index[dataframe_id] = profile
        return profile

    def load(self, dataframe_id: str) -> pd.DataFrame:
        if dataframe_id not in self.index:
            raise ToolValidationError(
                "unknown_dataframe",
                f"Unknown dataframe_id: {dataframe_id}",
                details={"known_dataframes": sorted(self.index)},
            )

        profile = self.index[dataframe_id]
        path = Path(profile["storage_path"])
        if profile.get("storage_format") == "parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)


class SimpleHTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []
            self._in_cell = True

    def handle_data(self, data: str) -> None:
        if self._in_cell and self._cell is not None:
            text = data.strip()
            if text:
                self._cell.append(text)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._row is not None and self._cell is not None:
            self._row.append(" ".join(self._cell).strip())
            self._cell = None
            self._in_cell = False
        elif tag == "tr" and self._table is not None and self._row is not None:
            if self._row:
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            if self._table:
                self.tables.append(self._table)
            self._table = None


def ckan_action(action: str, params: dict[str, Any] | None = None, sleep_s: float = 0.0) -> Any:
    params = params or {}
    query = f"?{urlencode(params, doseq=True)}" if params else ""
    url = f"{BASE_URL}/{action}{query}"
    request = Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError({"url": url, "status": exc.code, "body": body[:2000]}) from exc

    if not payload.get("success"):
        raise RuntimeError({"action": action, "params": params, "error": payload.get("error")})

    if sleep_s:
        time.sleep(sleep_s)

    return payload["result"]


def fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=45) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def normalize_format(resource: dict[str, Any]) -> str:
    fmt = (resource.get("format") or "").strip().lower()
    url = (resource.get("url") or "").lower()
    if fmt:
        return fmt
    for suffix in PARSABLE_FORMATS:
        if url.endswith(f".{suffix}"):
            return suffix
    return ""


def flatten_xml_element(element: ET.Element) -> dict[str, Any]:
    row: dict[str, Any] = dict(element.attrib)
    for child in list(element):
        key = child.tag.split("}")[-1]
        value = (child.text or "").strip()
        if list(child):
            for nested_key, nested_value in flatten_xml_element(child).items():
                row[f"{key}.{nested_key}"] = nested_value
        elif value:
            row[key] = value
    return row


def parse_xml(url: str, limit: int) -> pd.DataFrame:
    root = ET.fromstring(fetch_text(url))
    rows = []
    for element in root.iter():
        if list(element):
            row = flatten_xml_element(element)
            if row:
                rows.append(row)
        if len(rows) >= limit:
            break
    if not rows:
        rows = [flatten_xml_element(root)]
    return pd.DataFrame(rows).head(limit)


def parse_html_table(url: str, limit: int, table_index: int = 0) -> pd.DataFrame:
    parser = SimpleHTMLTableParser()
    parser.feed(fetch_text(url))
    if not parser.tables:
        raise ToolValidationError("unparseable_resource", "No HTML tables found in resource.")

    table = parser.tables[table_index]
    header, rows = table[0], table[1:]
    width = len(header)
    normalized = [row[:width] + [None] * max(0, width - len(row)) for row in rows]
    return pd.DataFrame(normalized, columns=header).head(limit)


def fetch_resource_dataframe(resource: dict[str, Any], limit: int = 500) -> tuple[pd.DataFrame, dict[str, Any]]:
    if resource.get("datastore_active"):
        result = ckan_action("datastore_search", {"resource_id": resource["id"], "limit": limit})
        return pd.DataFrame(result.get("records", [])), {
            "filter_mode": "ckan_datastore",
            "server_filter_supported": True,
            "source": "datastore_search",
        }

    url = resource.get("url")
    fmt = normalize_format(resource)
    if not url:
        raise ToolValidationError("missing_resource_url", "Resource has no URL.")

    metadata = {
        "filter_mode": "local_dataframe",
        "server_filter_supported": False,
        "source": "file",
    }
    if fmt in {"csv", "txt"}:
        return pd.read_csv(url, sep=None, engine="python", nrows=limit), metadata
    if fmt in {"xml", "rdf", "rss", "atom"} or url.lower().endswith(".xml"):
        return parse_xml(url, limit=limit), metadata
    if fmt in {"html", "htm"} or url.lower().endswith((".html", ".htm")):
        return parse_html_table(url, limit=limit), metadata

    raise ToolValidationError(
        "unsupported_resource_format",
        f"Unsupported format: {resource.get('format')}",
        details={"url": url},
    )


def retrieve_open_data(
    *,
    package_id_or_name: str,
    resource_id: str | None = None,
    limit: int = 500,
    store: DataframeStore,
) -> dict[str, Any]:
    package = ckan_action("package_show", {"id": package_id_or_name})
    resources = package.get("resources", [])
    if resource_id:
        matches = [resource for resource in resources if resource.get("id") == resource_id]
        if not matches:
            raise ToolValidationError(
                "unknown_resource",
                f"Resource {resource_id} not found in package {package_id_or_name}.",
                details={"available_resources": [resource.get("id") for resource in resources]},
            )
        resource = matches[0]
    else:
        sampleable = [
            resource
            for resource in resources
            if resource.get("datastore_active") or normalize_format(resource) in PARSABLE_FORMATS
        ]
        if not sampleable:
            raise ToolValidationError("no_parseable_resource", "Package has no parseable resource.")
        resource = sampleable[0]

    df, fetch_metadata = fetch_resource_dataframe(resource, limit=limit)
    dataframe_id = f"df_{package.get('name')}_{resource.get('id')}".replace("-", "_")
    metadata = {
        "package_id": package.get("id"),
        "package_name": package.get("name"),
        "package_title": package.get("title"),
        "resource_id": resource.get("id"),
        "resource_name": resource.get("name"),
        "resource_format": normalize_format(resource),
        **fetch_metadata,
    }
    return store.save(dataframe_id, df, metadata)


def search_open_data(query: str, limit: int = 5) -> dict[str, Any]:
    """Search Munich CKAN and return parseable dataset/resource candidates."""
    result = ckan_action("package_search", {"q": query, "rows": limit})
    candidates = []
    for package in result.get("results", []):
        parseable_resources = [
            resource
            for resource in package.get("resources", [])
            if resource.get("datastore_active") or normalize_format(resource) in PARSABLE_FORMATS
        ]
        candidates.append(
            {
                "package_id": package.get("id"),
                "package_name": package.get("name"),
                "package_title": package.get("title"),
                "notes": compact_text(package.get("notes")),
                "resources": [
                    {
                        "resource_id": resource.get("id"),
                        "resource_name": resource.get("name"),
                        "resource_format": normalize_format(resource),
                        "datastore_active": bool(resource.get("datastore_active")),
                    }
                    for resource in parseable_resources
                ],
            }
        )
    return {"ok": True, "query": query, "count": result.get("count"), "candidates": candidates}


def dataframe_profile(dataframe_id: str, df: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, Any]:
    columns = []
    for column in df.columns:
        series = df[column]
        examples = series.dropna().astype(str).drop_duplicates().head(10).tolist()
        columns.append(
            {
                "name": str(column),
                "dtype": str(series.dtype),
                "nullable": bool(series.isna().any()),
                "examples": examples,
            }
        )
    return {
        "dataframe_id": dataframe_id,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": columns,
        "metadata": metadata,
        "sample_rows": df.head(5).where(pd.notna(df.head(5)), None).to_dict(orient="records"),
    }


def _ensure_columns(df: pd.DataFrame, columns: list[str], field: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ToolValidationError(
            "unknown_column",
            f"Unknown column(s) in {field}: {', '.join(missing)}",
            details={"available_columns": list(df.columns), "missing_columns": missing},
        )


def _coerce_comparable(series: pd.Series, value: Any) -> tuple[pd.Series, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.8:
        if isinstance(value, list):
            return numeric, [float(item) for item in value]
        return numeric, float(value)
    return series.astype(str), value


def _apply_filter(df: pd.DataFrame, condition: dict[str, Any]) -> pd.DataFrame:
    column = condition.get("column")
    op = condition.get("operator")
    value = condition.get("value")
    if not isinstance(column, str):
        raise ToolValidationError("invalid_filter", "Filter column must be a string.")
    if column not in df.columns:
        raise ToolValidationError(
            "unknown_column",
            f"Column does not exist: {column}",
            details={"available_columns": list(df.columns), "missing_columns": [column]},
        )

    allowed = {"eq", "neq", "gt", "gte", "lt", "lte", "contains", "in", "between"}
    if op not in allowed:
        raise ToolValidationError(
            "invalid_operator",
            f"Invalid operator: {op}",
            details={"allowed_operators": sorted(allowed)},
        )

    series = df[column]
    if op in {"gt", "gte", "lt", "lte", "between"}:
        series, value = _coerce_comparable(series, value)

    comparisons = {
        "eq": operator.eq,
        "neq": operator.ne,
        "gt": operator.gt,
        "gte": operator.ge,
        "lt": operator.lt,
        "lte": operator.le,
    }
    if op in comparisons:
        mask = comparisons[op](series, value)
    elif op == "contains":
        mask = series.astype(str).str.contains(str(value), case=False, na=False, regex=False)
    elif op == "in":
        if not isinstance(value, list):
            raise ToolValidationError("invalid_filter_value", "Operator 'in' requires a list value.")
        mask = series.isin(value) | series.astype(str).isin([str(item) for item in value])
    elif op == "between":
        if not isinstance(value, list) or len(value) != 2:
            raise ToolValidationError("invalid_filter_value", "Operator 'between' requires a two-item list.")
        low, high = value
        mask = series.between(low, high)
    else:
        raise AssertionError(op)
    return df[mask]


def filter_dataframe(df: pd.DataFrame, operation_spec: dict[str, Any]) -> pd.DataFrame:
    result = df.copy()

    for condition in operation_spec.get("filters", []) or []:
        result = _apply_filter(result, condition)

    group_by = operation_spec.get("group_by") or []
    aggregate = operation_spec.get("aggregate") or []
    if group_by or aggregate:
        _ensure_columns(result, group_by, "group_by")
        if not aggregate:
            result = result.groupby(group_by, dropna=False).size().reset_index(name="count")
        else:
            agg_spec: dict[str, list[str]] = {}
            rename_map: dict[str, str] = {}
            for item in aggregate:
                column = item.get("column")
                function = item.get("function")
                alias = item.get("alias")
                if column not in result.columns:
                    raise ToolValidationError(
                        "unknown_column",
                        f"Aggregate column does not exist: {column}",
                        details={"available_columns": list(result.columns), "missing_columns": [column]},
                    )
                if function not in {"count", "sum", "mean", "median", "min", "max"}:
                    raise ToolValidationError("invalid_aggregation", f"Invalid aggregation: {function}")
                if function != "count" and not pd.api.types.is_numeric_dtype(pd.to_numeric(result[column], errors="coerce")):
                    raise ToolValidationError("invalid_aggregation", f"Aggregation {function} requires numeric column {column}.")
                agg_spec.setdefault(column, []).append(function)
                if alias:
                    rename_map[f"{column}_{function}"] = alias
            grouped = result.groupby(group_by, dropna=False).agg(agg_spec).reset_index()
            grouped.columns = [
                "_".join(part for part in column if part) if isinstance(column, tuple) else column
                for column in grouped.columns
            ]
            result = grouped.rename(columns=rename_map)

    select = operation_spec.get("select") or []
    if select:
        _ensure_columns(result, select, "select")
        result = result[select]

    sort = operation_spec.get("sort") or []
    if sort:
        sort_columns = [item.get("column") for item in sort]
        _ensure_columns(result, sort_columns, "sort")
        ascending = [item.get("direction", "asc") != "desc" for item in sort]
        result = result.sort_values(sort_columns, ascending=ascending)

    limit = operation_spec.get("limit", 100)
    if not isinstance(limit, int) or limit < 1 or limit > 1000:
        raise ToolValidationError("invalid_limit", "Limit must be an integer between 1 and 1000.")

    return result.head(limit).reset_index(drop=True)


def query_dataframe(
    *,
    dataframe_id: str,
    operation_spec: dict[str, Any],
    store: DataframeStore,
    result_dataframe_id: str | None = None,
) -> dict[str, Any]:
    df = store.load(dataframe_id)
    result = filter_dataframe(df, operation_spec)
    if result.empty:
        raise ToolValidationError("empty_result", "The dataframe query returned no rows.")

    result_dataframe_id = result_dataframe_id or f"{dataframe_id}_result"
    metadata = {
        **store.index[dataframe_id].get("metadata", {}),
        "source_dataframe_id": dataframe_id,
        "operation_spec": operation_spec,
    }
    return store.save(result_dataframe_id, result, metadata)


def compact_text(value: str | None, max_chars: int = 220) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()[:max_chars]
