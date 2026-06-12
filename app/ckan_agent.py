from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

try:
    from .ckan_support import DEFAULT_CKAN_ENDPOINT, group_list, normalize_ckan_base_url, organization_list, package_search, package_show, tag_search
    from .model_roles import ModelResponse, call_role_model
    from .openui_support import OpenUIValidationError, _json_arg, parse_openui_lang
except ImportError:
    from ckan_support import DEFAULT_CKAN_ENDPOINT, group_list, normalize_ckan_base_url, organization_list, package_search, package_show, tag_search
    from model_roles import ModelResponse, call_role_model
    from openui_support import OpenUIValidationError, _json_arg, parse_openui_lang


AgentStatus = Literal["selected", "finished", "needs_clarification", "max_tool_calls", "error"]
AgentEventType = Literal["model_action", "tool_call", "tool_result", "selection", "retry", "error", "done"]

ALLOWED_ACTIONS = {"tag_search", "group_list", "organization_list", "package_search", "package_show", "select_resource", "finish", "ask_clarification"}
MAX_TOOL_CALLS = 8
CKAN_ACTION_CONTRACT = (
    "You are the smolnalysis CKAN retrieval specialist. Choose the next action to find the correct dataset/resource. "
    "Return exactly one JSON object and nothing else. No markdown, no explanation outside JSON. Shape: "
    '{"action":"tag_search|group_list|organization_list|package_search|package_show|select_resource|finish|ask_clarification","args":{},"reason":"short reason","confidence":0.0}. '
    "Use tag_search first when the user's terms may not match the catalog language. "
    "Use group_list or organization_list to discover catalog vocabulary when package_search returns weak or empty results. "
    "Use package_show only with observed package_id values. Use select_resource only with observed resource_id values. "
    "For bicycle topics, prefer CKAN vocabulary like fahrrad, radverkehr, zaehlstelle, verkehr. Do not invent URLs."
)
OPENUI_CONTRACT = "You translate structured smolnalysis results into OpenUI-Lang. Output OpenUI-Lang only. No markdown, no prose."


@dataclass
class AgentAction:
    action: str
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    confidence: float = 0.0
    source: str = "model"


@dataclass
class AgentEvent:
    type: AgentEventType
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResource:
    package_id: str
    package_title: str
    resource_id: str
    name: str
    format: str
    url: str


@dataclass
class ToolResult:
    tool: str
    ok: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class AgentSession:
    prompt: str
    endpoint: str
    messages: list[dict[str, str]] = field(default_factory=list)
    packages: dict[str, dict[str, Any]] = field(default_factory=dict)
    resources: dict[str, RetrievalResource] = field(default_factory=dict)
    events: list[AgentEvent] = field(default_factory=list)
    tool_calls: int = 0
    searched_queries: list[str] = field(default_factory=list)
    failed_searches: dict[str, int] = field(default_factory=dict)
    consecutive_service_errors: int = 0
    discovered_tags: list[str] = field(default_factory=list)
    discovered_groups: list[dict[str, Any]] = field(default_factory=list)
    discovered_organizations: list[dict[str, Any]] = field(default_factory=list)
    catalog_tools_run: set[str] = field(default_factory=set)


@dataclass
class AgentResult:
    status: AgentStatus
    endpoint: str
    prompt: str
    messages: list[dict[str, str]]
    events: list[AgentEvent]
    packages: list[dict[str, Any]]
    resources: list[RetrievalResource]
    selected_package: dict[str, Any] | None = None
    selected_resource: RetrievalResource | None = None
    confidence: float = 0.0
    clarification: str = ""


def run_ckan_agent(
    prompt: str,
    endpoint: str | None = None,
    history: list[dict[str, str]] | None = None,
    on_event: Callable[[AgentEvent], None] | None = None,
    model_caller: Callable[[str, list[dict[str, str]], str], ModelResponse] = call_role_model,
    max_tool_calls: int = MAX_TOOL_CALLS,
) -> AgentResult:
    prompt = _clean_prompt(prompt)
    session = AgentSession(
        prompt=prompt.strip() or "Find a dataset",
        endpoint=_normalize_endpoint_or_default(endpoint),
        messages=_initial_messages(prompt, endpoint, history),
    )

    while session.tool_calls < max_tool_calls:
        action = _next_model_action(session, model_caller, on_event)
        valid_action, error = validate_action(action, session)
        if error:
            _record_event(session, AgentEvent("retry", error, {"action": asdict(action)}), on_event)
            valid_action = fallback_action(session)
        _record_event(session, AgentEvent("model_action", f"{valid_action.action}: {valid_action.reason}", {"action": asdict(valid_action)}), on_event)

        if valid_action.action == "ask_clarification":
            return _finish(session, "needs_clarification", clarification=str(valid_action.args.get("question") or valid_action.reason))
        if valid_action.action == "finish":
            return _finish(session, "finished", confidence=valid_action.confidence)
        if valid_action.action == "select_resource":
            resource_id = str(valid_action.args.get("resource_id", ""))
            resource = session.resources.get(resource_id)
            package = session.packages.get(resource.package_id) if resource else None
            if resource:
                _record_event(session, AgentEvent("selection", f"Selected {resource.name} from {resource.package_title}.", {"resource": asdict(resource)}), on_event)
                return _finish(session, "selected", selected_package=package, selected_resource=resource, confidence=valid_action.confidence or _score_resource(resource, session.prompt))
            _record_event(session, AgentEvent("retry", "Model selected an unobserved resource; falling back.", {"resource_id": resource_id}), on_event)
            continue

        result = execute_action(valid_action, session, on_event)
        session.tool_calls += 1
        _append_tool_result(session, valid_action, result)
        _record_event(session, AgentEvent("tool_result", result.summary, {"tool_result": asdict(result)}), on_event)
        if _should_stop_for_tool_failure(session, result):
            _record_event(session, AgentEvent("error", result.summary, {"tool_result": asdict(result)}), on_event)
            return _finish(session, "error", clarification=result.summary)

        auto_resource = _best_observed_resource(session)
        if auto_resource and _score_resource(auto_resource, session.prompt) >= 0.86:
            _record_event(session, AgentEvent("selection", f"Selected {auto_resource.name} from {auto_resource.package_title}.", {"resource": asdict(auto_resource)}), on_event)
            return _finish(
                session,
                "selected",
                selected_package=session.packages.get(auto_resource.package_id),
                selected_resource=auto_resource,
                confidence=_score_resource(auto_resource, session.prompt),
            )

    return _finish(session, "max_tool_calls")


def parse_agent_action(raw: str) -> AgentAction:
    text = raw.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        payload = _first_json_object(text)
        if payload is None:
            raise ValueError("Model did not return JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Model action must be a JSON object.")
    return _agent_action_from_payload(payload)


def _agent_action_from_payload(payload: dict[str, Any]) -> AgentAction:
    action = str(payload.get("action", "")).strip()
    args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
    reason = str(payload.get("reason", "")).strip()
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return AgentAction(action, args, reason, max(0.0, min(confidence, 1.0)))


def _first_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            payload, _index = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def validate_action(action: AgentAction, session: AgentSession) -> tuple[AgentAction, str]:
    if action.action not in ALLOWED_ACTIONS:
        return action, f"Unknown action '{action.action}'."
    if action.action == "package_search":
        query = str(action.args.get("query", "")).strip()
        if not query or "://" in query:
            return action, "package_search requires a plain query string."
        if session.failed_searches.get(query, 0) > 0:
            return action, f"package_search query already failed in this run: {query}"
        rows = int(action.args.get("rows", 5) or 5)
        action.args["rows"] = max(1, min(rows, 10))
    if action.action == "tag_search":
        query = str(action.args.get("query", "")).strip()
        if not query or "://" in query:
            return action, "tag_search requires a plain query string."
        rows = int(action.args.get("rows", 10) or 10)
        action.args["rows"] = max(1, min(rows, 25))
    if action.action in {"group_list", "organization_list"}:
        rows = int(action.args.get("rows", 10) or 10)
        action.args["rows"] = max(1, min(rows, 25))
    if action.action == "package_show":
        package_id = str(action.args.get("package_id", "")).strip()
        if package_id not in session.packages:
            return action, f"package_show package_id was not observed: {package_id}"
    if action.action == "select_resource":
        resource_id = str(action.args.get("resource_id", "")).strip()
        if resource_id not in session.resources:
            return action, f"select_resource resource_id was not observed: {resource_id}"
    return action, ""


def fallback_action(session: AgentSession) -> AgentAction:
    if session.consecutive_service_errors >= 2:
        return AgentAction("finish", {}, "CKAN service is currently unavailable; stop instead of repeating failing calls.", 0.0, "fallback")
    for package_id, package in session.packages.items():
        if package_id and not package.get("_shown"):
            return AgentAction("package_show", {"package_id": package_id}, "Inspect the next observed package.", 0.45, "fallback")
    if "tag_search" not in session.catalog_tools_run:
        query = _catalog_query(session.prompt)
        return AgentAction("tag_search", {"query": query, "rows": 10}, "Discover matching CKAN tags before package search.", 0.45, "fallback")
    if "group_list" not in session.catalog_tools_run:
        return AgentAction("group_list", {"rows": 15}, "Inspect catalog groups for topic vocabulary.", 0.35, "fallback")
    query = _fallback_query(session.prompt, session.searched_queries, session.discovered_tags, session.discovered_groups)
    return AgentAction("package_search", {"query": query, "rows": 5}, "Fallback search query.", 0.35, "fallback")


def execute_action(action: AgentAction, session: AgentSession, on_event: Callable[[AgentEvent], None] | None = None) -> ToolResult:
    if action.action == "tag_search":
        query = str(action.args.get("query", "")).strip()
        rows = int(action.args.get("rows", 10) or 10)
        session.catalog_tools_run.add("tag_search")
        _record_event(session, AgentEvent("tool_call", f"tag_search: {query}", {"query": query, "rows": rows}), on_event)
        try:
            result = tag_search(session.endpoint, query, rows=rows)
        except Exception as exc:
            return ToolResult("tag_search", False, f"tag_search failed for '{query}': {_short_error(exc)}", error=_short_error(exc))
        tags = _names_from_ckan_list_result(result)
        for tag in tags:
            if tag not in session.discovered_tags:
                session.discovered_tags.append(tag)
        return ToolResult("tag_search", True, f"Found {len(tags)} tags matching '{query}'.", {"query": query, "tags": tags[:12]})

    if action.action == "group_list":
        rows = int(action.args.get("rows", 10) or 10)
        session.catalog_tools_run.add("group_list")
        _record_event(session, AgentEvent("tool_call", f"group_list: {rows}", {"rows": rows}), on_event)
        try:
            result = group_list(session.endpoint, rows=rows)
        except Exception as exc:
            return ToolResult("group_list", False, f"group_list failed: {_short_error(exc)}", error=_short_error(exc))
        groups = _dicts_from_ckan_list_result(result)
        session.discovered_groups = groups
        return ToolResult("group_list", True, f"Found {len(groups)} catalog groups.", {"groups": [_compact_named_item(group) for group in groups[:12]]})

    if action.action == "organization_list":
        rows = int(action.args.get("rows", 10) or 10)
        session.catalog_tools_run.add("organization_list")
        _record_event(session, AgentEvent("tool_call", f"organization_list: {rows}", {"rows": rows}), on_event)
        try:
            result = organization_list(session.endpoint, rows=rows)
        except Exception as exc:
            return ToolResult("organization_list", False, f"organization_list failed: {_short_error(exc)}", error=_short_error(exc))
        organizations = _dicts_from_ckan_list_result(result)
        session.discovered_organizations = organizations
        return ToolResult("organization_list", True, f"Found {len(organizations)} organizations.", {"organizations": [_compact_named_item(org) for org in organizations[:12]]})

    if action.action == "package_search":
        query = str(action.args.get("query", "")).strip()
        rows = int(action.args.get("rows", 5) or 5)
        if query not in session.searched_queries:
            session.searched_queries.append(query)
        _record_event(session, AgentEvent("tool_call", f"package_search: {query}", {"query": query, "rows": rows}), on_event)
        try:
            result = package_search(session.endpoint, query, rows=rows)
        except Exception as exc:
            short_error = _short_error(exc)
            session.failed_searches[query] = session.failed_searches.get(query, 0) + 1
            if _is_service_unavailable_error(short_error):
                session.consecutive_service_errors += 1
            return ToolResult("package_search", False, f"package_search failed for '{query}': {_short_error(exc)}", error=_short_error(exc))
        session.consecutive_service_errors = 0
        packages = [package for package in result.get("results", []) if isinstance(package, dict)]
        for package in packages:
            package_id = _package_id(package)
            if package_id:
                session.packages[package_id] = package
        return ToolResult("package_search", True, f"Found {len(packages)} package candidates for '{query}'.", {"query": query, "packages": [_compact_package(package) for package in packages]})

    if action.action == "package_show":
        package_id = str(action.args.get("package_id", "")).strip()
        _record_event(session, AgentEvent("tool_call", f"package_show: {package_id}", {"package_id": package_id}), on_event)
        try:
            package = package_show(session.endpoint, package_id)
        except Exception as exc:
            return ToolResult("package_show", False, f"package_show failed for '{package_id}': {_short_error(exc)}", error=_short_error(exc))
        package["_shown"] = True
        session.packages[package_id] = package
        resources = _resources_from_package(package)
        for resource in resources:
            session.resources[resource.resource_id] = resource
        return ToolResult("package_show", True, f"Inspected {package.get('title') or package_id}: {len(resources)} resources.", {"package": _compact_package(package), "resources": [asdict(resource) for resource in resources]})

    return ToolResult(action.action, False, f"Action {action.action} is not executable as a tool.")


def build_retrieval_openui(result: AgentResult) -> str:
    event_ids = []
    lines = ['root = Card([header, progress, candidates, callout, followups])']
    lines.append('header = CardHeader("Dataset search", "CKAN retrieval result")')
    for index, event in enumerate(result.events[:12]):
        ident = f"event{index + 1}"
        event_ids.append(ident)
        lines.append(f"{ident} = ListItem({_json_arg(event.type)}, {_json_arg(event.detail)})")
    if not event_ids:
        lines.append('event1 = ListItem("start", "No retrieval steps were recorded.")')
        event_ids.append("event1")
    lines.append(f"progress = ListBlock([{', '.join(event_ids)}], \"number\")")

    rows = [_compact_package(package) for package in result.packages[:8]]
    if not rows:
        rows = [{"title": "No candidates", "name": result.prompt, "resources": 0}]
    columns = list(rows[0].keys())[:4]
    for index, column in enumerate(columns):
        lines.append(f"col{index + 1} = Col({_json_arg(column)}, {_json_arg([row.get(column) for row in rows])}, \"string\")")
    lines.append(f"candidates = Table([{', '.join(f'col{index + 1}' for index in range(len(columns)))}])")

    if result.selected_resource:
        message = f"Selected {result.selected_resource.name} from {result.selected_resource.package_title}."
    elif result.clarification:
        message = result.clarification
    else:
        message = "No final resource selected yet."
    lines.append(f'callout = Callout("info", "Result", {_json_arg(message)})')
    lines.extend(
        [
            'f1 = FollowUpItem("Try another CKAN search")',
            'f2 = FollowUpItem("Inspect the selected dataset")',
            'followups = FollowUpBlock([f1, f2])',
        ]
    )
    return "\n".join(lines)


def generate_openui_for_result(
    result: AgentResult,
    model_caller: Callable[[str, list[dict[str, str]], str], ModelResponse] = call_role_model,
) -> str:
    payload = {
        "status": result.status,
        "selected_package": result.selected_package,
        "selected_resource": asdict(result.selected_resource) if result.selected_resource else None,
        "events": [asdict(event) for event in result.events],
    }
    messages = [{"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)}]
    first_content = ""
    try:
        response = model_caller("openui_translator", messages, OPENUI_CONTRACT)
        first_content = response.content
        parse_openui_lang(response.content)
        return response.content
    except Exception:
        try:
            repair_messages = [*messages, {"role": "assistant", "content": first_content}, {"role": "user", "content": "Repair this as valid OpenUI-Lang only."}]
            repaired = model_caller("openui_translator", repair_messages, OPENUI_CONTRACT)
            parse_openui_lang(repaired.content)
            return repaired.content
        except Exception:
            fallback = build_retrieval_openui(result)
            parse_openui_lang(fallback)
            return fallback


def _next_model_action(
    session: AgentSession,
    model_caller: Callable[[str, list[dict[str, str]], str], ModelResponse],
    on_event: Callable[[AgentEvent], None] | None,
) -> AgentAction:
    try:
        response = model_caller("ckan_retrieval", session.messages, CKAN_ACTION_CONTRACT)
        action = parse_agent_action(response.content)
        session.messages.append({"role": "assistant", "content": json.dumps(asdict(action), ensure_ascii=False)})
        return action
    except Exception as exc:
        _record_event(session, AgentEvent("retry", f"Model action failed: {_short_error(exc)}"), on_event)
        return fallback_action(session)


def _initial_messages(prompt: str, endpoint: str | None, history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    messages = []
    for message in history or []:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if role == "user":
            content = _clean_prompt(content)
        if role and content:
            messages.append({"role": role, "content": content})
    messages.append(
        {
            "role": "user",
            "content": f"User request: {prompt.strip() or 'Find a dataset'}\nCKAN endpoint: {_normalize_endpoint_or_default(endpoint)}",
        }
    )
    return messages


def _append_tool_result(session: AgentSession, action: AgentAction, result: ToolResult) -> None:
    session.messages.append(
        {
            "role": "user",
            "content": "Tool observation:\n" + json.dumps({"action": asdict(action), "result": asdict(result)}, ensure_ascii=False, default=str),
        }
    )


def _record_event(session: AgentSession, event: AgentEvent, on_event: Callable[[AgentEvent], None] | None = None) -> None:
    session.events.append(event)
    if on_event:
        on_event(event)


def _finish(
    session: AgentSession,
    status: AgentStatus,
    selected_package: dict[str, Any] | None = None,
    selected_resource: RetrievalResource | None = None,
    confidence: float = 0.0,
    clarification: str = "",
) -> AgentResult:
    result = AgentResult(
        status=status,
        endpoint=session.endpoint,
        prompt=session.prompt,
        messages=session.messages,
        events=session.events,
        packages=list(session.packages.values()),
        resources=list(session.resources.values()),
        selected_package=selected_package,
        selected_resource=selected_resource,
        confidence=confidence,
        clarification=clarification,
    )
    session.events.append(AgentEvent("done", f"Agent stopped with status {status}.", {"status": status}))
    return result


def _resources_from_package(package: dict[str, Any]) -> list[RetrievalResource]:
    package_id = _package_id(package)
    package_title = str(package.get("title") or package.get("name") or package_id or "CKAN package")
    resources: list[RetrievalResource] = []
    for resource in package.get("resources") or []:
        if not isinstance(resource, dict):
            continue
        resource_id = str(resource.get("id") or "")
        if not resource_id:
            continue
        resources.append(
            RetrievalResource(
                package_id=package_id,
                package_title=package_title,
                resource_id=resource_id,
                name=str(resource.get("name") or resource.get("description") or resource.get("url") or "resource"),
                format=str(resource.get("format") or resource.get("mimetype") or "").lower(),
                url=str(resource.get("url") or ""),
            )
        )
    return resources


def _best_observed_resource(session: AgentSession) -> RetrievalResource | None:
    best: RetrievalResource | None = None
    best_score = 0.0
    for resource in session.resources.values():
        score = _score_resource(resource, session.prompt)
        if score > best_score:
            best = resource
            best_score = score
    return best


def _score_resource(resource: RetrievalResource, prompt: str) -> float:
    haystack = f"{resource.package_title} {resource.name} {resource.format} {resource.url}".casefold()
    terms = [term for term in re.findall(r"[\w-]+", prompt.casefold()) if len(term) > 2]
    score = 0.15
    if "csv" in resource.format or resource.url.casefold().endswith(".csv"):
        score += 0.45
    if any(term in haystack for term in terms):
        score += 0.25
    if "metadata" in haystack or "metadaten" in haystack:
        score -= 0.2
    return max(0.0, min(score, 1.0))


def _compact_package(package: dict[str, Any]) -> dict[str, Any]:
    return {
        "package_id": _package_id(package),
        "name": str(package.get("name") or ""),
        "title": str(package.get("title") or package.get("name") or ""),
        "resources": len(package.get("resources") or []),
    }


def _package_id(package: dict[str, Any]) -> str:
    return str(package.get("id") or package.get("name") or "")


def _fallback_query(
    prompt: str,
    searched_queries: list[str] | int | None = None,
    discovered_tags: list[str] | None = None,
    discovered_groups: list[dict[str, Any]] | None = None,
) -> str:
    if isinstance(searched_queries, int):
        searched = []
        index = searched_queries
    else:
        searched = list(searched_queries or [])
        index = len(searched)
    terms = _search_terms(prompt)
    expanded_terms = _expand_domain_terms(terms)
    base = " ".join(expanded_terms[:4]) or "open data"
    short = " ".join(terms[:2]) or base
    discovered = _matching_discovered_queries(expanded_terms, discovered_tags or [], discovered_groups or [])
    variants = _unique_queries([*discovered, base, short, *expanded_terms[:4], f"{base} csv", f"{short} csv", " ".join(expanded_terms[:3]) or base])
    for query in variants:
        if query not in searched:
            return query
    return variants[min(index, len(variants) - 1)]


def _matching_discovered_queries(expanded_terms: list[str], tags: list[str], groups: list[dict[str, Any]]) -> list[str]:
    candidates: list[str] = []
    for tag in tags:
        clean = str(tag).strip()
        if clean and any(term in clean.casefold() or clean.casefold() in term for term in expanded_terms):
            candidates.append(clean)
    for group in groups:
        label = str(group.get("title") or group.get("display_name") or group.get("name") or "").strip()
        if label and any(term in label.casefold() or label.casefold() in term for term in expanded_terms):
            candidates.append(label)
    return _unique_queries(candidates)


def _catalog_query(prompt: str) -> str:
    terms = _expand_domain_terms(_search_terms(prompt))
    return terms[0] if terms else "open data"


def _search_terms(prompt: str) -> list[str]:
    prompt = _clean_prompt(prompt)
    stopwords = {
        "api",
        "assistant",
        "catalog",
        "ckan",
        "content",
        "context",
        "data",
        "dataset",
        "datasets",
        "endpoint",
        "find",
        "https",
        "know",
        "knwo",
        "muenchen",
        "open",
        "opendata",
        "request",
        "resource",
        "resources",
        "search",
        "tell",
        "what",
        "user",
        "yo",
        "you",
    }
    terms = []
    for term in re.findall(r"[\w-]+", prompt.casefold()):
        clean = term.strip("-_")
        if len(clean) <= 2 or clean in stopwords or clean.isdigit():
            continue
        if clean not in terms:
            terms.append(clean)
    return terms


def _expand_domain_terms(terms: list[str]) -> list[str]:
    expanded: list[str] = []
    for term in terms:
        for candidate in _domain_synonyms(term):
            if candidate not in expanded:
                expanded.append(candidate)
    return expanded


def _domain_synonyms(term: str) -> list[str]:
    bicycle_terms = {"bike", "bikes", "bicycle", "bicycles", "bycycle", "bycycles", "cycle", "cycles", "cycling", "fahrrad", "rad", "radverkehr"}
    if term in bicycle_terms:
        return ["fahrrad", "radverkehr", "bike", "bicycle"]
    counter_terms = {"counter", "counters", "count", "counts", "zaehler", "zaehlstelle", "zählstelle", "zählstellen"}
    if term in counter_terms:
        return ["zaehlstelle", "zaehlung", "counter"]
    traffic_terms = {"traffic", "verkehr", "mobility", "mobilitaet", "mobilität"}
    if term in traffic_terms:
        return ["verkehr", "mobilitaet", "traffic"]
    return [term]


def _unique_queries(queries: list[str]) -> list[str]:
    unique = []
    for query in queries:
        clean = " ".join(query.split()).strip()
        if clean and clean not in unique:
            unique.append(clean)
    return unique or ["open data"]


def _clean_prompt(prompt: str) -> str:
    value = str(prompt or "").strip()
    content_match = re.search(r"<content>(.*?)</content>", value, re.IGNORECASE | re.DOTALL)
    if content_match:
        value = content_match.group(1).strip()
    value = re.sub(r"<context>.*?</context>", " ", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _names_from_ckan_list_result(result: dict[str, Any]) -> list[str]:
    values = result.get("results") or result.get("value") or []
    if not isinstance(values, list):
        return []
    names: list[str] = []
    for item in values:
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("display_name") or "").strip()
        else:
            name = str(item).strip()
        if name and name not in names:
            names.append(name)
    return names


def _dicts_from_ckan_list_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    values = result.get("results") or result.get("value") or []
    if not isinstance(values, list):
        return []
    return [item for item in values if isinstance(item, dict)]


def _compact_named_item(item: dict[str, Any]) -> dict[str, str]:
    return {
        "name": str(item.get("name") or ""),
        "title": str(item.get("title") or item.get("display_name") or item.get("name") or ""),
    }


def _normalize_endpoint_or_default(endpoint: str | None) -> str:
    if not endpoint:
        return DEFAULT_CKAN_ENDPOINT
    try:
        return normalize_ckan_base_url(endpoint)
    except ValueError:
        return DEFAULT_CKAN_ENDPOINT


def _short_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc).strip() or type(exc).__name__}"


def _is_service_unavailable_error(error: str) -> bool:
    return "503" in error or "service unavailable" in error.casefold()


def _should_stop_for_tool_failure(session: AgentSession, result: ToolResult) -> bool:
    if result.ok:
        return False
    if session.consecutive_service_errors >= 2:
        return True
    return any(count >= 2 for count in session.failed_searches.values())
