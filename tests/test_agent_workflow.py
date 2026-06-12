from __future__ import annotations

import json
import importlib
import sys
from pathlib import Path
from typing import Any
from unittest import TestCase, main
from unittest.mock import patch

from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ckan_agent import (
    AgentAction,
    AgentSession,
    ModelResponse,
    fallback_action,
    generate_openui_for_result,
    parse_agent_action,
    run_ckan_agent,
    validate_action,
)
from app.openui_support import parse_openui_lang

app_module = importlib.import_module("app.app")
app = app_module.app


class CkanAgentWorkflowTests(TestCase):
    def test_valid_json_action_parses(self) -> None:
        action = parse_agent_action('{"action":"package_search","args":{"query":"population","rows":5},"reason":"start","confidence":0.7}')

        self.assertEqual(action.action, "package_search")
        self.assertEqual(action.args["query"], "population")
        self.assertEqual(action.confidence, 0.7)

    def test_invalid_json_falls_back_to_search(self) -> None:
        session = AgentSession("Find population datasets", "https://opendata.muenchen.de/")

        fallback = fallback_action(session)

        self.assertEqual(fallback.action, "package_search")
        self.assertIn("population", fallback.args["query"])

    def test_fallback_search_ignores_chat_wrapper_words(self) -> None:
        session = AgentSession("content population content context", "https://opendata.muenchen.de/")

        fallback = fallback_action(session)

        self.assertEqual(fallback.args["query"], "population")

    def test_fallback_search_extracts_openui_clicked_content(self) -> None:
        session = AgentSession(
            '<content>Find CKAN bike counter datasets</content><context>["User clicked: Find CKAN bike counter datasets",{}]</context>',
            "https://opendata.muenchen.de/",
        )

        fallback = fallback_action(session)

        self.assertEqual(fallback.args["query"], "bike counter")

    def test_unknown_action_is_rejected(self) -> None:
        session = AgentSession("Find data", "https://opendata.muenchen.de/")
        action = AgentAction("delete_everything", {}, "bad", 1)

        _validated, error = validate_action(action, session)

        self.assertIn("Unknown action", error)

    def test_unobserved_package_show_is_rejected(self) -> None:
        session = AgentSession("Find data", "https://opendata.muenchen.de/")
        action = AgentAction("package_show", {"package_id": "missing"}, "inspect", 0.5)

        _validated, error = validate_action(action, session)

        self.assertIn("not observed", error)

    def test_unobserved_resource_selection_is_rejected(self) -> None:
        session = AgentSession("Find data", "https://opendata.muenchen.de/")
        action = AgentAction("select_resource", {"resource_id": "missing"}, "select", 0.5)

        _validated, error = validate_action(action, session)

        self.assertIn("not observed", error)

    @patch("app.ckan_agent.package_show")
    @patch("app.ckan_agent.package_search")
    def test_agent_loop_searches_inspects_and_selects(self, package_search_mock, package_show_mock) -> None:
        package = _ckan_package()
        package_search_mock.return_value = {"count": 1, "results": [package]}
        package_show_mock.return_value = package
        model = _ModelScript(
            [
                {"action": "package_search", "args": {"query": "population", "rows": 5}, "reason": "search", "confidence": 0.6},
                {"action": "package_show", "args": {"package_id": "pkg-1"}, "reason": "inspect", "confidence": 0.7},
                {"action": "select_resource", "args": {"resource_id": "res-1"}, "reason": "best csv", "confidence": 0.9},
            ]
        )

        result = run_ckan_agent("Find population datasets", "https://opendata.muenchen.de/", model_caller=model)

        self.assertEqual(result.status, "selected")
        self.assertEqual(result.selected_resource.resource_id, "res-1")
        self.assertEqual([event.type for event in result.events if event.type == "tool_call"], ["tool_call", "tool_call"])
        self.assertTrue(any("Tool observation" in message["content"] for message in result.messages if message["role"] == "user"))

    @patch("app.ckan_agent.package_show")
    @patch("app.ckan_agent.package_search")
    def test_agent_loop_refines_after_weak_results(self, package_search_mock, package_show_mock) -> None:
        weak_package = {**_ckan_package(), "resources": [{"id": "meta", "name": "Metadata XML", "format": "XML", "url": "https://example.org/meta.xml"}]}
        strong_package = _ckan_package()
        package_search_mock.side_effect = [
            {"count": 1, "results": [weak_package]},
            {"count": 1, "results": [strong_package]},
        ]
        package_show_mock.side_effect = [weak_package, strong_package]
        model = _ModelScript(
            [
                {"action": "package_search", "args": {"query": "population metadata", "rows": 5}, "reason": "start", "confidence": 0.5},
                {"action": "package_show", "args": {"package_id": "pkg-1"}, "reason": "inspect weak", "confidence": 0.5},
                {"action": "package_search", "args": {"query": "population csv", "rows": 5}, "reason": "refine", "confidence": 0.6},
                {"action": "package_show", "args": {"package_id": "pkg-1"}, "reason": "inspect strong", "confidence": 0.7},
                {"action": "select_resource", "args": {"resource_id": "res-1"}, "reason": "csv", "confidence": 0.9},
            ]
        )

        result = run_ckan_agent("Find population datasets", "https://opendata.muenchen.de/", model_caller=model)

        self.assertEqual(package_search_mock.call_count, 2)
        self.assertEqual(result.selected_resource.resource_id, "res-1")

    @patch("app.ckan_agent.package_search")
    def test_model_failure_falls_back_to_deterministic_query(self, package_search_mock) -> None:
        package_search_mock.return_value = {"count": 0, "results": []}

        result = run_ckan_agent("Find schools", "https://opendata.muenchen.de/", model_caller=_FailingModel(), max_tool_calls=1)

        self.assertEqual(result.status, "max_tool_calls")
        self.assertEqual(package_search_mock.call_args.args[1], "schools")
        self.assertTrue(any(event.type == "retry" for event in result.events))

    @patch("app.ckan_agent.package_search")
    def test_loop_stops_at_max_tool_calls(self, package_search_mock) -> None:
        package_search_mock.return_value = {"count": 0, "results": []}
        model = _ModelScript([{"action": "package_search", "args": {"query": "nothing", "rows": 5}, "reason": "again", "confidence": 0.4}] * 4)

        result = run_ckan_agent("Find nothing", "https://opendata.muenchen.de/", model_caller=model, max_tool_calls=2)

        self.assertEqual(result.status, "max_tool_calls")
        self.assertEqual(package_search_mock.call_count, 2)

    @patch("app.ckan_agent.package_search")
    def test_repeated_503_stops_without_hammering_endpoint(self, package_search_mock) -> None:
        package_search_mock.side_effect = RuntimeError("HTTP Error 503: Service Unavailable")
        model = _ModelScript(
            [
                {"action": "package_search", "args": {"query": "population", "rows": 5}, "reason": "search", "confidence": 0.4},
                {"action": "package_search", "args": {"query": "population csv", "rows": 5}, "reason": "refine", "confidence": 0.4},
                {"action": "package_search", "args": {"query": "population data", "rows": 5}, "reason": "repeat", "confidence": 0.4},
            ]
        )

        result = run_ckan_agent("Find population datasets", "https://opendata.muenchen.de/", model_caller=model, max_tool_calls=8)

        self.assertEqual(result.status, "error")
        self.assertEqual(package_search_mock.call_count, 2)
        self.assertTrue(any(event.type == "error" for event in result.events))

    def test_openui_generation_accepts_valid_model_output(self) -> None:
        result = _selected_result()
        model = _TextModel('root = Card([header])\nheader = CardHeader("Dataset", "Selected")')

        openui_lang = generate_openui_for_result(result, model_caller=model)

        parse_openui_lang(openui_lang)
        self.assertIn("Dataset", openui_lang)

    def test_openui_generation_repairs_once(self) -> None:
        result = _selected_result()
        model = _TextSequence(["not openui", 'root = Card([header])\nheader = CardHeader("Fixed", "OK")'])

        openui_lang = generate_openui_for_result(result, model_caller=model)

        self.assertIn("Fixed", openui_lang)
        self.assertEqual(model.calls, 2)

    def test_openui_generation_falls_back_after_invalid_output(self) -> None:
        result = _selected_result()
        model = _TextSequence(["not openui", "still not openui"])

        openui_lang = generate_openui_for_result(result, model_caller=model)

        self.assertIn("Dataset search", openui_lang)
        parse_openui_lang(openui_lang)

    @patch("app.ckan_agent.package_show")
    @patch("app.ckan_agent.package_search")
    def test_chat_route_streams_retrieval_progress(self, package_search_mock, package_show_mock) -> None:
        package = _ckan_package()
        package_search_mock.return_value = {"count": 1, "results": [package]}
        package_show_mock.return_value = package
        model = _ModelScript(
            [
                {"action": "package_search", "args": {"query": "population", "rows": 5}, "reason": "search", "confidence": 0.6},
                {"action": "package_show", "args": {"package_id": "pkg-1"}, "reason": "inspect", "confidence": 0.7},
                {"action": "select_resource", "args": {"resource_id": "res-1"}, "reason": "select", "confidence": 0.9},
            ]
        )
        client = TestClient(app)

        with patch.object(app_module, "call_role_model", side_effect=model):
            response = client.post(
                "/api/chat",
                json={
                    "messages": [{"role": "user", "content": "Find CKAN population datasets"}],
                    "ckan": {"connected": True, "base_url": "https://opendata.muenchen.de/"},
                },
            )

        chunks = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
        payloads = [json.loads(chunk) for chunk in chunks]
        content_chunks = [payload["choices"][0]["delta"].get("content", "") for payload in payloads]
        content = "".join(content_chunks)

        self.assertGreaterEqual(len([chunk for chunk in content_chunks if chunk]), 4)
        self.assertIn("Finding dataset", content)
        self.assertIn("progress1 = ListItem", content)
        self.assertIn("Selected Population CSV", content)
        parse_openui_lang(content)

    @patch("app.ckan_agent.package_search")
    def test_chat_route_cleans_clicked_followup_payload(self, package_search_mock) -> None:
        package_search_mock.return_value = {"count": 0, "results": []}
        client = TestClient(app)

        with patch.object(app_module, "call_role_model", side_effect=_FailingModel()):
            response = client.post(
                "/api/chat",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": '<content>Find CKAN bike counter datasets</content><context>["User clicked: Find CKAN bike counter datasets",{}]</context>',
                        }
                    ]
                },
            )

        queries = [call.args[1] for call in package_search_mock.call_args_list]
        self.assertEqual(queries[0], "bike counter")
        self.assertTrue(all("content" not in query and "context" not in query for query in queries))
        self.assertIn("Request: Find CKAN bike counter datasets", response.text)
        self.assertNotIn("content bike counter content", response.text)

    @patch("app.ckan_agent.package_search")
    def test_chat_route_treats_bycycles_as_retrieval(self, package_search_mock) -> None:
        package_search_mock.return_value = {"count": 0, "results": []}
        client = TestClient(app)

        with patch.object(app_module, "call_role_model", side_effect=_FailingModel()):
            response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "what about bycycles"}]})

        self.assertIsNotNone(package_search_mock.call_args)
        self.assertIn("Finding dataset", response.text)
        self.assertNotIn("Dataset analysis", response.text)

    @patch("app.ckan_agent.package_show")
    @patch("app.ckan_agent.package_search")
    def test_chat_trace_records_model_and_tool_events(self, package_search_mock, package_show_mock) -> None:
        package = _ckan_package()
        package_search_mock.return_value = {"count": 1, "results": [package]}
        package_show_mock.return_value = package
        client = TestClient(app)

        with patch.object(app_module, "call_role_model", side_effect=_ModelScript([
            {"action": "package_search", "args": {"query": "population", "rows": 5}, "reason": "search", "confidence": 0.6},
            {"action": "package_show", "args": {"package_id": "pkg-1"}, "reason": "inspect", "confidence": 0.7},
            {"action": "select_resource", "args": {"resource_id": "res-1"}, "reason": "select", "confidence": 0.9},
        ])):
            response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "Find CKAN population datasets"}]})

        trace_id = response.headers["x-smolnalysis-trace-id"]
        trace = client.get(f"/api/traces/{trace_id}").json()

        self.assertEqual(trace["backend"], "simple_ckan_agent")
        self.assertTrue(any(event["name"] == "model_action" for event in trace["events"]))
        self.assertTrue(any(event["name"] == "tool_call" for event in trace["events"]))
        self.assertEqual(trace["retrieval"]["selected"], "Population CSV")


class _ModelScript:
    def __init__(self, actions: list[dict[str, Any]]) -> None:
        self.actions = list(actions)

    def __call__(self, role: str, messages: list[dict[str, str]], response_contract: str = "") -> ModelResponse:
        if not self.actions:
            action = {"action": "finish", "args": {}, "reason": "done", "confidence": 0.1}
        else:
            action = self.actions.pop(0)
        return ModelResponse(json.dumps(action), {"role": role, "events": [{"name": "generate", "detail": "ok"}]})


class _FailingModel:
    def __call__(self, role: str, messages: list[dict[str, str]], response_contract: str = "") -> ModelResponse:
        raise RuntimeError("model unavailable")


class _TextModel:
    def __init__(self, text: str) -> None:
        self.text = text

    def __call__(self, role: str, messages: list[dict[str, str]], response_contract: str = "") -> ModelResponse:
        return ModelResponse(self.text, {"role": role})


class _TextSequence:
    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.calls = 0

    def __call__(self, role: str, messages: list[dict[str, str]], response_contract: str = "") -> ModelResponse:
        self.calls += 1
        return ModelResponse(self.texts.pop(0), {"role": role})


def _selected_result():
    return run_ckan_agent(
        "Find population datasets",
        "https://opendata.muenchen.de/",
        model_caller=_ModelScript([{"action": "finish", "args": {}, "reason": "done", "confidence": 0.1}]),
        max_tool_calls=0,
    )


def _ckan_package() -> dict[str, Any]:
    return {
        "id": "pkg-1",
        "name": "population",
        "title": "Population by district",
        "resources": [
            {
                "id": "res-1",
                "name": "Population CSV",
                "format": "CSV",
                "url": "https://example.org/population.csv",
            }
        ],
    }


if __name__ == "__main__":
    main()
