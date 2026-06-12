from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

try:
    from .ckan_support import DEFAULT_CKAN_ENDPOINT, normalize_ckan_base_url, package_search, package_show
    from .model_roles import ModelResponse, call_role_model
    from .openui_support import OpenUIValidationError, _json_arg, parse_openui_lang
except ImportError:
    from ckan_support import DEFAULT_CKAN_ENDPOINT, normalize_ckan_base_url, package_search, package_show
    from model_roles import ModelResponse, call_role_model
    from openui_support import OpenUIValidationError, _json_arg, parse_openui_lang


AgentStatus = Literal["selected", "finished", "needs_clarification", "max_tool_calls", "error"]
AgentEventType = Literal["model_action", "tool_call", "tool_result", "selection", "retry", "error", "done"]

ALLOWED_ACTIONS = {"package_search", "package_show", "select_resource", "finish", "ask_clarification"}
MAX_TOOL_CALLS = 8
CKAN_ACTION_CONTRACT = (
    "You are the smolnalysis CKAN retrieval specialist. Choose the next action to find the correct dataset/resource. "
    "Return strict JSON only. Shape: "
    '{"action":"package_search|package_show|select_resource|finish|ask_clarification","args":{},"reason":"short reason","confidence":0.0}. '
    "Use only observed package_id/resource_id values for package_show and select_resource. Do not invent URLs."
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
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("Model did not return JSON.") from exc
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Model action must be a JSON object.")
    action = str(payload.get("action", "")).strip()
    args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
    reason = str(payload.get("reason", "")).strip()
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return AgentAction(action, args, reason, max(0.0, min(confidence, 1.0)))


def validate_action(action: AgentAction, session: AgentSession) -> tuple[AgentAction, str]:
    if action.action not in ALLOWED_ACTIONS:
        return action, f"Unknown action '{action.action}'."
    if action.action == "package_search":
        query = str(action.args.get("query", "")).strip()
        if not query or "://" in query:
            return action, "package_search requires a plain query string."
        rows = int(action.args.get("rows", 5) or 5)
        action.args["rows"] = max(1, min(rows, 10))
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
    for package_id, package in session.packages.items():
        if package_id and not package.get("_shown"):
            return AgentAction("package_show", {"package_id": package_id}, "Inspect the next observed package.", 0.45, "fallback")
    query = _fallback_query(session.prompt, len([event for event in session.events if "package_search" in event.detail]))
    return AgentAction("package_search", {"query": query, "rows": 5}, "Fallback search query.", 0.35, "fallback")


def execute_action(action: AgentAction, session: AgentSession, on_event: Callable[[AgentEvent], None] | None = None) -> ToolResult:
    if action.action == "package_search":
        query = str(action.args.get("query", "")).strip()
        rows = int(action.args.get("rows", 5) or 5)
        _record_event(session, AgentEvent("tool_call", f"package_search: {query}", {"query": query, "rows": rows}), on_event)
        try:
            result = package_search(session.endpoint, query, rows=rows)
        except Exception as exc:
            return ToolResult("package_search", False, f"package_search failed for '{query}': {_short_error(exc)}", error=_short_error(exc))
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
    messages = list(history or [])
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


def _fallback_query(prompt: str, index: int = 0) -> str:
    terms = [term for term in re.findall(r"[\w-]+", prompt.casefold()) if len(term) > 2 and term not in {"find", "search", "dataset", "datasets", "ckan", "data"}]
    base = " ".join(terms[:4]) or "open data"
    variants = [base, " ".join(terms[:2]) or base, f"{base} csv"]
    return variants[min(index, len(variants) - 1)]


def _normalize_endpoint_or_default(endpoint: str | None) -> str:
    if not endpoint:
        return DEFAULT_CKAN_ENDPOINT
    try:
        return normalize_ckan_base_url(endpoint)
    except ValueError:
        return DEFAULT_CKAN_ENDPOINT


def _short_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc).strip() or type(exc).__name__}"
