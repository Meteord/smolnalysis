from __future__ import annotations

from typing import Any, TypedDict

try:
    from .ckan_agent import AgentResult, build_retrieval_openui, run_ckan_agent
    from .ckan_support import DEFAULT_CKAN_ENDPOINT
    from .openui_support import _json_arg
except ImportError:
    from ckan_agent import AgentResult, build_retrieval_openui, run_ckan_agent
    from ckan_support import DEFAULT_CKAN_ENDPOINT
    from openui_support import _json_arg


class AgentWorkflowState(TypedDict, total=False):
    prompt: str
    ckan_endpoint: str
    dataset_path: str
    retrieval_result: dict[str, Any]
    openui_lang: str
    steps: list[dict[str, str]]
    intent: dict[str, str]


def run_agent_workflow(prompt: str, ckan_endpoint: str | None = None, dataset_path: str | None = None) -> AgentWorkflowState:
    if dataset_path:
        openui_lang = _demo_dataset_openui(prompt)
        return {
            "prompt": prompt,
            "ckan_endpoint": ckan_endpoint or DEFAULT_CKAN_ENDPOINT,
            "dataset_path": dataset_path,
            "intent": {"task_type": "analysis"},
            "steps": [{"node": "demo_dataset", "title": "analysis", "detail": "Rendered demo dataset placeholder through the compatibility workflow."}],
            "openui_lang": openui_lang,
        }

    result = run_ckan_agent(prompt, ckan_endpoint)
    return {
        "prompt": prompt,
        "ckan_endpoint": ckan_endpoint or DEFAULT_CKAN_ENDPOINT,
        "intent": {"task_type": "dataset_retrieval"},
        "steps": [{"node": event.type, "title": event.type, "detail": event.detail} for event in result.events],
        "retrieval_result": _agent_result_dict(result),
        "openui_lang": build_retrieval_openui(result),
    }


def _agent_result_dict(result: AgentResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "endpoint": result.endpoint,
        "packages": result.packages,
        "resources": [resource.__dict__ for resource in result.resources],
        "selected": result.selected_resource.__dict__ if result.selected_resource else None,
        "confidence": result.confidence,
        "events": [event.__dict__ for event in result.events],
    }


def _demo_dataset_openui(prompt: str) -> str:
    return "\n".join(
        [
            "root = Card([header, callout, followups])",
            'header = CardHeader("Dataset analysis", "Demo dataset path")',
            f'callout = Callout("info", "Next step", {_json_arg(f"Retrieval is now handled by the clean CKAN agent. Dataset analysis/OpenUI detail comes next. Request: {prompt}")})',
            'f1 = FollowUpItem("Find CKAN population datasets")',
            'f2 = FollowUpItem("Find CKAN bike counter datasets")',
            "followups = FollowUpBlock([f1, f2])",
        ]
    )
