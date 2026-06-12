from __future__ import annotations

import json
import importlib
import os
import sys
from pathlib import Path
from typing import Any, cast
from unittest import TestCase, main
from unittest.mock import patch

from fastapi.testclient import TestClient


os.environ["SMOLNALYSIS_WORKFLOW_DISABLE_DELAYS"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent_workflow import DEFAULT_CKAN_ENDPOINT, build_agent_workflow, run_agent_workflow

app_module = importlib.import_module("app.app")
app = app_module.app


class AgentWorkflowTests(TestCase):
    def test_graph_compiles_and_returns_openui_lang(self) -> None:
        graph = build_agent_workflow()
        result = graph.invoke({"prompt": "Find population data", "ckan_endpoint": DEFAULT_CKAN_ENDPOINT, "steps": []})

        self.assertIn("openui_lang", result)
        self.assertIn("root = Card", result["openui_lang"])

    def test_graph_invokes_nodes_in_order(self) -> None:
        result = cast(dict[str, Any], run_agent_workflow("Find population data", "https://opendata.muenchen.de/"))
        step_nodes = [step["node"] for step in result["steps"]]

        self.assertEqual(step_nodes[0], "react_agent")
        self.assertEqual(step_nodes[-1], "translate_openui")
        self.assertIn("retrieve_ckan", step_nodes)
        self.assertIn("analyze_data", step_nodes)
        self.assertLess(step_nodes.index("retrieve_ckan"), step_nodes.index("analyze_data"))
        self.assertLess(step_nodes.index("analyze_data"), step_nodes.index("translate_openui"))

    def test_agent_can_rerun_stub_tools(self) -> None:
        result = cast(dict[str, Any], run_agent_workflow("Compare more population quality chart data", "https://opendata.muenchen.de/"))
        step_nodes = [step["node"] for step in result["steps"]]

        self.assertGreaterEqual(step_nodes.count("retrieve_ckan"), 2)
        self.assertGreaterEqual(step_nodes.count("analyze_data"), 2)
        self.assertEqual(result["ckan_result"]["attempt"], 2)
        self.assertEqual(result["analysis_result"]["attempt"], 2)
        self.assertIn("Thought:", result["openui_lang"])

    def test_ckan_endpoint_is_reflected_in_output(self) -> None:
        result = cast(dict[str, Any], run_agent_workflow("Find population data", "https://example.org/ckan"))

        self.assertEqual(result["ckan_endpoint"], "https://example.org/ckan/")
        self.assertIn("https://example.org/ckan/", result["openui_lang"])

    def test_empty_ckan_endpoint_uses_default(self) -> None:
        result = cast(dict[str, Any], run_agent_workflow("Find population data", ""))

        self.assertEqual(result["ckan_endpoint"], DEFAULT_CKAN_ENDPOINT)

    def test_chat_route_keeps_openai_compatible_sse_shape(self) -> None:
        client = TestClient(app)
        trace = {"backend": "llama.cpp", "role": "general_agent", "events": [{"name": "generate", "detail": "ok"}]}
        with patch.object(app_module, "generate_chat_response_with_trace", return_value=("MiniCPM backend response", trace)) as generate:
            response = client.post(
                "/api/chat",
                json={
                    "messages": [{"role": "user", "content": "Find population data"}],
                    "ckan": {"connected": True, "base_url": "https://example.org/ckan"},
                },
            )

        self.assertEqual(response.status_code, 200)
        generate.assert_called_once_with([{"role": "user", "content": "Find population data"}], adapter="auto")
        self.assertTrue(response.headers.get("x-smolnalysis-trace-id"))
        chunks = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
        payloads = [json.loads(chunk) for chunk in chunks]
        self.assertEqual(payloads[0]["choices"][0]["delta"]["role"], "assistant")
        content = "".join(payload["choices"][0]["delta"].get("content", "") for payload in payloads)
        self.assertIn("root = Card", content)
        self.assertIn('header = CardHeader("MiniCPM", "Backend response")', content)
        self.assertIn("MiniCPM backend response", content)

    def test_chat_route_preserves_model_openui_lang(self) -> None:
        client = TestClient(app)
        model_openui = 'root = Card([TextContent("Already OpenUI", "default")])'
        with patch.object(app_module, "generate_chat_response_with_trace", return_value=(model_openui, {"backend": "llama.cpp", "events": []})):
            response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "Render UI"}]})

        chunks = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
        payloads = [json.loads(chunk) for chunk in chunks]
        content = "".join(payload["choices"][0]["delta"].get("content", "") for payload in payloads)
        self.assertEqual(content, model_openui)

    def test_chat_trace_can_be_read_back(self) -> None:
        client = TestClient(app)
        trace = {"backend": "llama.cpp", "role": "data_analysis", "events": [{"name": "route_role", "detail": "auto -> data_analysis"}]}
        with patch.object(app_module, "generate_chat_response_with_trace", return_value=("Traceable response", trace)):
            response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "Analyze trends"}]})

        trace_id = response.headers["x-smolnalysis-trace-id"]
        trace_response = client.get(f"/api/traces/{trace_id}")

        self.assertEqual(trace_response.status_code, 200)
        payload = trace_response.json()
        self.assertEqual(payload["request_id"], trace_id)
        self.assertEqual(payload["role"], "data_analysis")
        self.assertEqual(payload["events"][0]["name"], "route_role")

    def test_backend_generation_failure_does_not_use_stub_fallback_by_default(self) -> None:
        def failing_backend():
            raise RuntimeError("backend exploded")

        with patch.object(app_module, "ENABLE_STUB_CHAT_FALLBACK", False), patch.object(app_module, "_backend_generate_function", side_effect=failing_backend):
            response, trace = app_module.generate_chat_response_with_trace(
                [{"role": "user", "content": "Find population data"}],
                adapter="auto",
            )

        self.assertIn("MiniCPM backend unavailable", response)
        self.assertNotIn("ReAct-style LangGraph agent", response)
        self.assertEqual(trace["backend"], "transformers")
        self.assertEqual(trace["model_family"], "MiniCPM")
        self.assertEqual(trace["role"], "backend_error")
        self.assertFalse(trace["stub_fallback_enabled"])
        self.assertIn("backend exploded", trace["fallback_detail"])

    def test_backend_generation_failure_can_opt_into_stub_fallback(self) -> None:
        def failing_backend():
            raise RuntimeError("backend exploded")

        with patch.object(app_module, "ENABLE_STUB_CHAT_FALLBACK", True), patch.object(app_module, "_backend_generate_function", side_effect=failing_backend):
            response, trace = app_module.generate_chat_response_with_trace(
                [{"role": "user", "content": "Find population data"}],
                adapter="auto",
            )

        self.assertIn("ReAct-style LangGraph agent", response)
        self.assertEqual(trace["backend"], "langgraph_fallback")
        self.assertEqual(trace["model_family"], "stub")
        self.assertEqual(trace["role"], "fallback")


if __name__ == "__main__":
    main()
