from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable


APP_DIR = Path(__file__).resolve().parents[2] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ckan_support import DEFAULT_CKAN_ENDPOINT, ckan_api_base, normalize_ckan_base_url

from ckan_dataset_tools import write_jsonl


DEFAULT_QUERIES = [
    "population",
    "mobility",
    "traffic",
    "environment",
    "budget",
    "district",
    "schools",
    "housing",
]
REQUEST_TEMPLATES = [
    "Find {topic} datasets that could be analyzed.",
    "Search CKAN for {topic} data and pick useful tabular resources.",
    "I need open data about {topic}; inspect likely CKAN candidates.",
    "Find a dataset for a chart about {topic}.",
]
INVENTORY_REQUEST_TEMPLATES_EN = [
    "Do you have data about {topic}?",
    "Can you find data about {topic}?",
    "I want to analyze {topic}.",
    "Is there open data for {topic}?",
    "Show me useful data about {topic}.",
]
INVENTORY_REQUEST_TEMPLATES_DE = [
    "Hast du Daten zu {topic}?",
    "Kannst du Daten zu {topic} finden?",
    "Ich möchte {topic} analysieren.",
    "Gibt es offene Daten zu {topic}?",
    "Zeig mir nützliche Daten zu {topic}.",
]
GERMAN_HINTS = {
    "ä",
    "ö",
    "ü",
    "ß",
    "fahrrad",
    "bevölkerung",
    "stadtbezirk",
    "verkehr",
    "haushalt",
    "schule",
    "bildung",
    "sitzplätze",
    "wohn",
    "münchen",
}
TABULAR_FORMATS = {"csv", "xlsx", "xls", "json", "geojson"}
DOCUMENT_FORMATS = {"pdf", "html", "htm", "doc", "docx"}
UrlOpen = Callable[..., Any]


def read_ckan_action(api_base: str, action: str, params: dict[str, Any] | None = None, timeout_seconds: float = 10, urlopen: UrlOpen = urllib.request.urlopen) -> Any:
    query = urllib.parse.urlencode(params or {})
    url = f"{api_base.rstrip('/')}/{action}"
    if query:
        url = f"{url}?{query}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"CKAN request failed for {action}: {exc}") from exc
    payload = json.loads(body)
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise RuntimeError(f"CKAN action {action} did not return success.")
    return payload.get("result")


def search_packages(api_base: str, query: str, rows: int, filters: dict[str, str] | None = None, timeout_seconds: float = 10, urlopen: UrlOpen = urllib.request.urlopen) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"q": query, "rows": rows}
    fq = build_filter_query(filters or {})
    if fq:
        params["fq"] = fq
    result = read_ckan_action(api_base, "package_search", params, timeout_seconds, urlopen)
    if not isinstance(result, dict):
        return []
    packages = result.get("results", [])
    return [package for package in packages if isinstance(package, dict)]


def fetch_package_page(api_base: str, rows: int, start: int, timeout_seconds: float = 10, urlopen: UrlOpen = urllib.request.urlopen) -> tuple[list[dict[str, Any]], int | None]:
    result = read_ckan_action(api_base, "package_search", {"rows": rows, "start": start}, timeout_seconds, urlopen)
    if not isinstance(result, dict):
        return [], None
    packages = result.get("results", [])
    count = result.get("count")
    return [package for package in packages if isinstance(package, dict)], count if isinstance(count, int) else None


def fetch_dataset_inventory(endpoint: str, rows_per_page: int = 100, max_datasets: int | None = None, timeout_seconds: float = 10, urlopen: UrlOpen = urllib.request.urlopen) -> list[dict[str, Any]]:
    base_url = normalize_ckan_base_url(endpoint)
    api_base = ckan_api_base(base_url)
    inventory = []
    start = 0
    while True:
        packages, total_count = fetch_package_page(api_base, rows_per_page, start, timeout_seconds, urlopen)
        if not packages:
            break
        for package in packages:
            inventory.append(package)
            if max_datasets is not None and len(inventory) >= max_datasets:
                return inventory
        start += len(packages)
        if total_count is not None and start >= total_count:
            break
    return inventory


def list_named_entities(api_base: str, action: str, limit: int | None = None, timeout_seconds: float = 10, urlopen: UrlOpen = urllib.request.urlopen) -> list[str]:
    result = read_ckan_action(api_base, action, {}, timeout_seconds, urlopen)
    names = [name for name in result if isinstance(name, str)] if isinstance(result, list) else []
    return names[:limit] if limit is not None else names


def build_filter_query(filters: dict[str, str]) -> str:
    parts = []
    group = filters.get("group", "").strip()
    organization = filters.get("organization", "").strip()
    if group:
        parts.append(f"groups:{group}")
    if organization:
        parts.append(f"organization:{organization}")
    return " ".join(parts)


def resource_identifier(package_id: str, resource: dict[str, Any]) -> str:
    resource_id = str(resource.get("id") or resource.get("name") or "").strip()
    return f"{package_id}:{resource_id}" if resource_id else ""


def summarize_package(package: dict[str, Any]) -> dict[str, Any]:
    package_id = str(package.get("name") or package.get("id") or "").strip()
    resources = [resource for resource in package.get("resources", []) if isinstance(resource, dict)]
    summarized_resources = []
    for resource in resources[:8]:
        summarized_resources.append(
            {
                "id": str(resource.get("id") or "").strip(),
                "name": str(resource.get("name") or "").strip(),
                "format": str(resource.get("format") or resource.get("mimetype") or "").strip(),
                "url": str(resource.get("url") or "").strip(),
            }
        )
    return {
        "id": package_id,
        "title": str(package.get("title") or package_id).strip(),
        "notes": str(package.get("notes") or "").strip()[:500],
        "organization": _organization_name(package),
        "groups": _group_names(package),
        "tags": _tag_names(package),
        "resources": summarized_resources,
    }


def classify_resources(package: dict[str, Any]) -> tuple[list[str], list[str]]:
    package_id = str(package.get("name") or package.get("id") or "").strip()
    tabular = []
    unsuitable = []
    for resource in package.get("resources", []):
        if not isinstance(resource, dict):
            continue
        identifier = resource_identifier(package_id, resource)
        if not identifier:
            continue
        resource_format = str(resource.get("format") or resource.get("mimetype") or "").casefold().strip(". ")
        if resource_format in TABULAR_FORMATS:
            tabular.append(identifier)
        elif resource_format in DOCUMENT_FORMATS:
            unsuitable.append(identifier)
    return tabular, unsuitable


def _organization_name(package: dict[str, Any]) -> str:
    organization = package.get("organization")
    if isinstance(organization, dict):
        return str(organization.get("name") or organization.get("title") or "").strip()
    owner_org = package.get("owner_org")
    return str(owner_org or "").strip()


def _group_names(package: dict[str, Any]) -> list[str]:
    groups = package.get("groups")
    if not isinstance(groups, list):
        return []
    names = []
    for group in groups:
        if isinstance(group, dict):
            name = str(group.get("name") or group.get("title") or "").strip()
            if name:
                names.append(name)
    return names


def _tag_names(package: dict[str, Any]) -> list[str]:
    tags = package.get("tags")
    if not isinstance(tags, list):
        return []
    names = []
    for tag in tags:
        if isinstance(tag, dict):
            name = str(tag.get("name") or tag.get("display_name") or "").strip()
            if name:
                names.append(name)
    return names[:8]


def inventory_topic(package: dict[str, Any], fallback: str = "open data") -> str:
    tags = _tag_names(package)
    if tags:
        return tags[0]
    groups = _group_names(package)
    if groups:
        return groups[0]
    title = str(package.get("title") or package.get("name") or fallback).strip()
    words = [word.strip(".,:;()[]").casefold() for word in title.split() if len(word.strip(".,:;()[]")) > 3]
    return " ".join(words[:3]) or fallback


def inventory_filters(package: dict[str, Any]) -> dict[str, str]:
    filters = {}
    groups = _group_names(package)
    organization = _organization_name(package)
    if groups:
        filters["group"] = groups[0]
    if organization:
        filters["organization"] = organization
    return filters


def scenario_base(endpoint: str, query: str, package: dict[str, Any], index: int, filters: dict[str, str] | None = None) -> dict[str, Any]:
    topic = query.replace("_", " ")
    base = {
        "endpoint": endpoint,
        "request": REQUEST_TEMPLATES[index % len(REQUEST_TEMPLATES)].format(topic=topic),
        "topic": topic,
        "package_summary": summarize_package(package),
    }
    filters = {key: value for key, value in (filters or {}).items() if value}
    if filters:
        base["filters"] = filters
        filter_text = ", ".join(f"{key}={value}" for key, value in filters.items())
        base["request"] = f"{base['request']} Limit the search to {filter_text}."
    return base


def inventory_scenario_base(endpoint: str, package: dict[str, Any], index: int, filters: dict[str, str]) -> dict[str, Any]:
    topic = inventory_topic(package)
    templates = INVENTORY_REQUEST_TEMPLATES_DE if is_probably_german(topic) else INVENTORY_REQUEST_TEMPLATES_EN
    request = templates[index % len(templates)].format(topic=topic)
    return {
        "endpoint": endpoint,
        "request": request,
        "topic": topic,
        "filters": filters or None,
        "package_summary": summarize_package(package),
    }


def build_scenarios_for_package(endpoint: str, query: str, package: dict[str, Any], index: int, filters: dict[str, str] | None = None) -> list[dict[str, Any]]:
    package_id = str(package.get("name") or package.get("id") or "").strip()
    if not package_id:
        return []

    tabular_resources, unsuitable_resources = classify_resources(package)
    scenarios = []
    base = scenario_base(endpoint, query, package, index, filters)
    suffix = filter_suffix(filters or {})

    scenarios.append(
        {
            **base,
            "id": f"{query}_{package_id}{suffix}_search",
            "state": "No searches have been run yet.",
            "observed_packages": [],
            "observed_resources": [],
            "has_enough_evidence": False,
            "target_action": "package_search",
            "notes": "Start with a concise query based on the user request and include CKAN filters when provided.",
        }
    )
    scenarios.append(
        {
            **base,
            "id": f"{query}_{package_id}{suffix}_show",
            "state": f"Search results include package {package_id}. Inspect it before selecting resources.",
            "observed_packages": [package_id],
            "observed_resources": [],
            "has_enough_evidence": False,
            "target_action": "package_show",
            "notes": "Use package_show on the observed package id.",
        }
    )

    if tabular_resources:
        scenarios.append(
            {
                **base,
                "id": f"{query}_{package_id}{suffix}_select",
                "state": f"Package details for {package_id} include tabular resources.",
                "observed_packages": [package_id],
                "observed_resources": tabular_resources[:3],
                "has_enough_evidence": True,
                "target_action": "select_resource",
                "notes": "Select one observed tabular resource for analysis.",
            }
        )
        scenarios.append(
            {
                **base,
                "id": f"{query}_{package_id}{suffix}_finish",
                "state": f"A suitable tabular resource from {package_id} is already selected.",
                "observed_packages": [package_id],
                "observed_resources": tabular_resources[:3],
                "has_enough_evidence": True,
                "target_action": "finish",
                "notes": "Finish retrieval with selected candidates.",
            }
        )
    elif unsuitable_resources:
        scenarios.append(
            {
                **base,
                "id": f"{query}_{package_id}{suffix}_reject",
                "state": f"Package details for {package_id} only expose unsuitable document-like resources.",
                "observed_packages": [package_id],
                "observed_resources": unsuitable_resources[:3],
                "has_enough_evidence": False,
                "target_action": "reject_result",
                "notes": "Reject this result and suggest a better search query.",
            }
        )

    return scenarios


def build_inventory_scenarios_for_package(endpoint: str, package: dict[str, Any], index: int) -> list[dict[str, Any]]:
    package_id = str(package.get("name") or package.get("id") or "").strip()
    if not package_id:
        return []

    filters = inventory_filters(package)
    tabular_resources, unsuitable_resources = classify_resources(package)
    base = inventory_scenario_base(endpoint, package, index, filters)
    topic = inventory_topic(package)
    topic_slug = slugify(topic)
    suffix = filter_suffix(filters)
    topic_matches_package = package_matches_topic(package, topic)

    scenarios = [
        {
            **base,
            "id": f"inventory_{topic_slug}_{index}{suffix}_search",
            "state": "No searches have been run yet.",
            "observed_packages": [],
            "observed_resources": [],
            "has_enough_evidence": False,
            "target_action": "package_search",
            "notes": _policy_hint(filters, "Use a general query from the topic. Apply available group or organization context internally, not because the user said CKAN."),
        },
        {
            **base,
            "id": f"inventory_{topic_slug}_{index}{suffix}_show",
            "state": _filtered_state(filters, "Search results include one promising package. Inspect it before selecting resources."),
            "observed_packages": [package_id],
            "observed_resources": [],
            "has_enough_evidence": False,
            "target_action": "package_show",
            "notes": "Inspect the observed package id. This teaches the package_show step, not this exact dataset.",
        },
    ]

    if tabular_resources and topic_matches_package:
        scenarios.extend(
            [
                {
                    **base,
                    "id": f"inventory_{topic_slug}_{index}{suffix}_select",
                    "state": _filtered_state(filters, "Package details include one or more tabular resources suitable for analysis."),
                    "observed_packages": [package_id],
                    "observed_resources": tabular_resources[:3],
                    "has_enough_evidence": True,
                    "target_action": "select_resource",
                    "notes": "Select an observed tabular resource and explain the general suitability.",
                },
                {
                    **base,
                    "id": f"inventory_{topic_slug}_{index}{suffix}_finish",
                    "state": "A suitable resource has already been identified and retrieval can stop.",
                    "observed_packages": [package_id],
                    "observed_resources": tabular_resources[:3],
                    "has_enough_evidence": True,
                    "target_action": "finish",
                    "notes": "Finish retrieval with selected candidates. Do not search further.",
                },
            ]
        )
        scenarios.append(
            {
                **base,
                "id": f"inventory_{topic_slug}_{index}{suffix}_choose",
                "state": _filtered_state(filters, "A filtered result set is available. Choose the best candidate resource for the user request."),
                "observed_packages": [package_id],
                "observed_resources": tabular_resources[:3],
                "has_enough_evidence": True,
                "target_action": "select_resource",
                "notes": "Choose from already-filtered candidates. The user request should stay natural; filters are retrieval context.",
            }
        )
    if (unsuitable_resources and not tabular_resources) or (tabular_resources and not topic_matches_package):
        observed_resources = unsuitable_resources[:3] or tabular_resources[:3]
        scenarios.append(
            {
                **base,
                "id": f"inventory_{topic_slug}_{index}{suffix}_reject",
                "state": _filtered_state(filters, "Package details do not provide a suitable match for the natural-language request."),
                "observed_packages": [package_id],
                "observed_resources": observed_resources,
                "has_enough_evidence": False,
                "target_action": "reject_result",
                "notes": "Reject the current result when the filtered candidates do not match the topic well enough.",
            }
        )
    return scenarios


def _policy_hint(filters: dict[str, str], base: str) -> str:
    if not filters:
        return base
    filter_text = ", ".join(f"{key}={value}" for key, value in filters.items())
    return f"{base} Retrieval context has filters: {filter_text}."


def _filtered_state(filters: dict[str, str], base: str) -> str:
    if not filters:
        return base
    filter_text = ", ".join(f"{key}={value}" for key, value in filters.items())
    return f"{base} Current retrieval context is constrained by {filter_text}."


def package_matches_topic(package: dict[str, Any], topic: str) -> bool:
    topic_tokens = meaningful_tokens(topic)
    if not topic_tokens:
        return True
    haystack_parts = [
        str(package.get("name") or ""),
        str(package.get("title") or ""),
        str(package.get("notes") or ""),
        " ".join(_group_names(package)),
        _organization_name(package),
    ]
    haystack_tokens = set(meaningful_tokens(" ".join(haystack_parts)))
    return bool(topic_tokens & haystack_tokens)


def meaningful_tokens(value: str) -> set[str]:
    stopwords = {
        "der",
        "die",
        "das",
        "und",
        "oder",
        "mit",
        "für",
        "fuer",
        "von",
        "zur",
        "zum",
        "data",
        "daten",
        "open",
        "muenchen",
        "münchen",
    }
    tokens = set()
    for raw in value.casefold().replace("_", " ").replace("-", " ").split():
        token = raw.strip(".,:;()[]{}!?\"'")
        if len(token) >= 4 and token not in stopwords:
            tokens.add(token)
    return tokens


def filter_suffix(filters: dict[str, str]) -> str:
    parts = []
    if filters.get("group"):
        parts.append(f"group_{filters['group']}")
    if filters.get("organization"):
        parts.append(f"org_{filters['organization']}")
    return f"_{'_'.join(parts)}" if parts else ""


def slugify(value: str) -> str:
    chars = []
    for char in value.casefold():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "-":
            chars.append("-")
    return "".join(chars).strip("-")[:48] or "dataset"


def is_probably_german(value: str) -> bool:
    lower = value.casefold()
    return any(hint in lower for hint in GERMAN_HINTS)


def harvest_scenarios(
    endpoint: str,
    queries: list[str],
    rows_per_query: int,
    max_scenarios: int | None = None,
    timeout_seconds: float = 10,
    urlopen: UrlOpen = urllib.request.urlopen,
    groups: list[str] | None = None,
    organizations: list[str] | None = None,
) -> list[dict[str, Any]]:
    base_url = normalize_ckan_base_url(endpoint)
    api_base = ckan_api_base(base_url)
    scenarios: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    filter_sets = build_filter_sets(groups or [], organizations or [])
    for query in queries:
        for filters in filter_sets:
            packages = search_packages(api_base, query, rows_per_query, filters, timeout_seconds, urlopen)
            for index, package in enumerate(packages):
                for scenario in build_scenarios_for_package(base_url, query, package, index, filters):
                    scenario_id = str(scenario.get("id"))
                    if scenario_id in seen_ids:
                        continue
                    seen_ids.add(scenario_id)
                    scenarios.append(scenario)
                    if max_scenarios is not None and len(scenarios) >= max_scenarios:
                        return scenarios
    return scenarios


def build_filter_sets(groups: list[str], organizations: list[str]) -> list[dict[str, str]]:
    filter_sets: list[dict[str, str]] = [{}]
    filter_sets.extend({"group": group} for group in groups)
    filter_sets.extend({"organization": organization} for organization in organizations)
    return filter_sets


def parse_queries(values: list[str] | None, query_file: str | None) -> list[str]:
    queries = []
    if values:
        queries.extend(values)
    if query_file:
        with Path(query_file).open("r", encoding="utf-8") as handle:
            queries.extend(line.strip() for line in handle if line.strip() and not line.strip().startswith("#"))
    return queries or DEFAULT_QUERIES


def build_inventory_scenarios(endpoint: str, packages: list[dict[str, Any]], max_scenarios: int | None = None) -> list[dict[str, Any]]:
    base_url = normalize_ckan_base_url(endpoint)
    scenarios = []
    seen_ids = set()
    for index, package in enumerate(packages):
        for scenario in build_inventory_scenarios_for_package(base_url, package, index):
            scenario_id = str(scenario.get("id"))
            if scenario_id in seen_ids:
                continue
            seen_ids.add(scenario_id)
            scenarios.append(scenario)
            if max_scenarios is not None and len(scenarios) >= max_scenarios:
                return scenarios
    return scenarios


def _command_harvest(args: argparse.Namespace) -> int:
    try:
        scenarios = harvest_scenarios(
            args.endpoint,
            parse_queries(args.query, args.query_file),
            args.rows_per_query,
            args.max_scenarios,
            args.timeout_seconds,
            groups=args.group or [],
            organizations=args.organization or [],
        )
    except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Harvest error: {exc}")
        return 1
    write_jsonl(Path(args.output), scenarios)
    print(f"Wrote {len(scenarios)} CKAN-grounded scenarios to {args.output}.")
    return 0


def _command_inventory(args: argparse.Namespace) -> int:
    try:
        packages = fetch_dataset_inventory(args.endpoint, args.rows_per_page, args.max_datasets, args.timeout_seconds)
        inventory_rows = [summarize_package(package) for package in packages]
        scenarios = build_inventory_scenarios(args.endpoint, packages, args.max_scenarios)
    except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Inventory harvest error: {exc}")
        return 1
    if args.inventory_output:
        write_jsonl(Path(args.inventory_output), inventory_rows)
        print(f"Wrote {len(inventory_rows)} inventory rows to {args.inventory_output}.")
    write_jsonl(Path(args.scenarios_output), scenarios)
    print(f"Wrote {len(scenarios)} inventory-grounded scenarios to {args.scenarios_output}.")
    return 0


def _command_list_entities(args: argparse.Namespace, action: str, label: str) -> int:
    try:
        base_url = normalize_ckan_base_url(args.endpoint)
        names = list_named_entities(ckan_api_base(base_url), action, args.limit, args.timeout_seconds)
    except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"List {label} error: {exc}")
        return 1
    for name in names:
        print(name)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Harvest real CKAN metadata into teacher-generation scenarios.")
    subparsers = parser.add_subparsers(dest="command")

    harvest_parser = subparsers.add_parser("harvest", help="Harvest CKAN package/resource metadata into scenarios.")
    harvest_parser.add_argument("--endpoint", default=DEFAULT_CKAN_ENDPOINT, help="Public CKAN base URL.")
    harvest_parser.add_argument("--query", action="append", help="Search query. Can be provided multiple times.")
    harvest_parser.add_argument("--query-file", help="Optional file with one query per line.")
    harvest_parser.add_argument("--group", action="append", help="CKAN group name filter. Can be provided multiple times.")
    harvest_parser.add_argument("--organization", action="append", help="CKAN organization name filter. Can be provided multiple times.")
    harvest_parser.add_argument("--rows-per-query", type=int, default=3)
    harvest_parser.add_argument("--max-scenarios", type=int)
    harvest_parser.add_argument("--timeout-seconds", type=float, default=10)
    harvest_parser.add_argument("--output", default="train/ckan/data/harvested_scenarios.jsonl")
    harvest_parser.set_defaults(func=_command_harvest)

    inventory_parser = subparsers.add_parser("inventory", help="Page through the dataset index and generate broad scenarios.")
    inventory_parser.add_argument("--endpoint", default=DEFAULT_CKAN_ENDPOINT, help="Public CKAN base URL.")
    inventory_parser.add_argument("--rows-per-page", type=int, default=100)
    inventory_parser.add_argument("--max-datasets", type=int)
    inventory_parser.add_argument("--max-scenarios", type=int)
    inventory_parser.add_argument("--timeout-seconds", type=float, default=10)
    inventory_parser.add_argument("--inventory-output", default="train/ckan/data/dataset_inventory.jsonl")
    inventory_parser.add_argument("--scenarios-output", default="train/ckan/data/harvested_inventory_scenarios.jsonl")
    inventory_parser.set_defaults(func=_command_inventory)

    groups_parser = subparsers.add_parser("list-groups", help="Print CKAN group names.")
    groups_parser.add_argument("--endpoint", default=DEFAULT_CKAN_ENDPOINT, help="Public CKAN base URL.")
    groups_parser.add_argument("--limit", type=int)
    groups_parser.add_argument("--timeout-seconds", type=float, default=10)
    groups_parser.set_defaults(func=lambda args: _command_list_entities(args, "group_list", "groups"))

    orgs_parser = subparsers.add_parser("list-organizations", help="Print CKAN organization names.")
    orgs_parser.add_argument("--endpoint", default=DEFAULT_CKAN_ENDPOINT, help="Public CKAN base URL.")
    orgs_parser.add_argument("--limit", type=int)
    orgs_parser.add_argument("--timeout-seconds", type=float, default=10)
    orgs_parser.set_defaults(func=lambda args: _command_list_entities(args, "organization_list", "organizations"))

    if len(sys.argv) > 1 and sys.argv[1] not in {"harvest", "inventory", "list-groups", "list-organizations", "-h", "--help"}:
        sys.argv.insert(1, "harvest")
    elif len(sys.argv) == 1:
        sys.argv.append("harvest")

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
