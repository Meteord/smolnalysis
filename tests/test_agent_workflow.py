from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest import TestCase, main

from fastapi.testclient import TestClient


os.environ["SMOLNALYSIS_WORKFLOW_DISABLE_DELAYS"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from agent_workflow import DEFAULT_CKAN_ENDPOINT, build_agent_workflow, run_agent_workflow
from app import app


class AgentWorkflowTests(TestCase):
    def test_graph_compiles_and_returns_openui_lang(self) -> None:
        graph = build_agent_workflow()
        result = graph.invoke({"prompt": "Find population data", "ckan_endpoint": DEFAULT_CKAN_ENDPOINT, "steps": []})

        self.assertIn("openui_lang", result)
        self.assertIn("root = Card", result["openui_lang"])

    def test_graph_invokes_nodes_in_order(self) -> None:
        result = run_agent_workflow("Find population data", "https://opendata.muenchen.de/")
        step_nodes = [step["node"] for step in result["steps"]]

        self.assertEqual(step_nodes[0], "react_agent")
        self.assertEqual(step_nodes[-1], "translate_openui")
        self.assertIn("retrieve_ckan", step_nodes)
        self.assertIn("analyze_data", step_nodes)
        self.assertLess(step_nodes.index("retrieve_ckan"), step_nodes.index("analyze_data"))
        self.assertLess(step_nodes.index("analyze_data"), step_nodes.index("translate_openui"))

    def test_agent_can_rerun_stub_tools(self) -> None:
        result = run_agent_workflow("Compare more population quality chart data", "https://opendata.muenchen.de/")
        step_nodes = [step["node"] for step in result["steps"]]

        self.assertGreaterEqual(step_nodes.count("retrieve_ckan"), 2)
        self.assertGreaterEqual(step_nodes.count("analyze_data"), 2)
        self.assertEqual(result["ckan_result"]["attempt"], 2)
        self.assertEqual(result["analysis_result"]["attempt"], 2)
        self.assertIn("Thought:", result["openui_lang"])

    def test_ckan_endpoint_is_reflected_in_output(self) -> None:
        result = run_agent_workflow("Find population data", "https://example.org/ckan")

        self.assertEqual(result["ckan_endpoint"], "https://example.org/ckan/")
        self.assertIn("https://example.org/ckan/", result["openui_lang"])

    def test_empty_ckan_endpoint_uses_default(self) -> None:
        result = run_agent_workflow("Find population data", "")

        self.assertEqual(result["ckan_endpoint"], DEFAULT_CKAN_ENDPOINT)

    def test_chat_route_keeps_openai_compatible_sse_shape(self) -> None:
        client = TestClient(app)
        response = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "Find population data"}],
                "ckan": {"connected": True, "base_url": "https://example.org/ckan"},
            },
        )

        self.assertEqual(response.status_code, 200)
        chunks = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
        payloads = [json.loads(chunk) for chunk in chunks]
        self.assertEqual(payloads[0]["choices"][0]["delta"]["role"], "assistant")
        content = "".join(payload["choices"][0]["delta"].get("content", "") for payload in payloads)
        self.assertIn("ReAct-style LangGraph agent", content)
        self.assertIn("react_agent", content)
        self.assertIn("https://example.org/ckan/", content)


if __name__ == "__main__":
    main()
