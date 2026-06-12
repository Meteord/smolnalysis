from __future__ import annotations

import json
import importlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, cast
from unittest import TestCase, main
from unittest.mock import patch

from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent_workflow import (
    DEFAULT_CKAN_ENDPOINT,
    build_agent_workflow,
    build_openui_lang,
    classify_intent,
    run_agent_workflow,
)
from app.openui_support import OpenUIValidationError, parse_openui_lang

app_module = importlib.import_module("app.app")
app = app_module.app


class AgentWorkflowTests(TestCase):
    def test_graph_compiles_and_returns_valid_openui_lang(self) -> None:
        graph = build_agent_workflow()
        result = graph.invoke({"prompt": "Summarize this dataset", "ckan_endpoint": DEFAULT_CKAN_ENDPOINT, "dataset_path": _csv_fixture(), "steps": []})

        self.assertIn("openui_lang", result)
        self.assertIn("root = Card", result["openui_lang"])
        parse_openui_lang(result["openui_lang"])

    def test_deterministic_router_selects_retrieval(self) -> None:
        intent = classify_intent("Find CKAN population datasets", has_dataset=False)

        self.assertEqual(intent.task_type, "dataset_retrieval")

    def test_deterministic_router_selects_analysis_when_dataset_exists(self) -> None:
        intent = classify_intent("List the columns and missing values", has_dataset=True)

        self.assertEqual(intent.task_type, "analysis")
        self.assertEqual(intent.desired_visualization, "schema")

    def test_deterministic_router_selects_openui_without_data(self) -> None:
        intent = classify_intent("Render this as OpenUI cards", has_dataset=False)

        self.assertEqual(intent.task_type, "openui_generation")

    @patch("app.agent_workflow.package_show")
    @patch("app.agent_workflow.package_search")
    def test_ckan_workflow_uses_tool_results(self, package_search_mock, package_show_mock) -> None:
        package = _ckan_package()
        package_search_mock.return_value = {"count": 1, "results": [package]}
        package_show_mock.return_value = package

        result = cast(dict[str, Any], run_agent_workflow("Find CKAN population datasets", "https://opendata.muenchen.de/"))

        retrieval = result["retrieval_result"]
        self.assertEqual(retrieval["query"], "population")
        self.assertEqual(retrieval["selected"]["resource_id"], "res-1")
        self.assertTrue(result["analysis_result"]["errors"])
        parse_openui_lang(result["openui_lang"])

    @patch("app.agent_workflow.package_search")
    def test_ckan_workflow_handles_no_suitable_resource(self, package_search_mock) -> None:
        package_search_mock.return_value = {"count": 1, "results": [{**_ckan_package(), "resources": []}]}

        result = cast(dict[str, Any], run_agent_workflow("Find CKAN budget datasets", "https://opendata.muenchen.de/"))

        self.assertIsNone(result["retrieval_result"]["selected"])
        self.assertIn("No CSV-like dataset", result["analysis_result"]["errors"][0])
        parse_openui_lang(result["openui_lang"])

    def test_analysis_uses_csv_fixture_for_schema(self) -> None:
        result = cast(dict[str, Any], run_agent_workflow("List the columns and missing values", dataset_path=_csv_fixture()))

        analysis = result["analysis_result"]
        self.assertGreaterEqual(analysis["rows"], 3)
        self.assertGreaterEqual(analysis["columns"], 3)
        self.assertIn("city", [column["column"] for column in analysis["schema"]])
        self.assertIn("table = Table", result["openui_lang"])
        parse_openui_lang(result["openui_lang"])

    def test_analysis_builds_bar_chart_data(self) -> None:
        result = cast(dict[str, Any], run_agent_workflow("Show a bar chart of population by city", dataset_path=_csv_fixture()))

        analysis = result["analysis_result"]
        self.assertEqual(analysis["chart_x"], "city")
        self.assertEqual(analysis["chart_y"], "population")
        self.assertIn("chart = BarChart", result["openui_lang"])

    def test_analysis_handles_no_numeric_columns(self) -> None:
        path = _temp_csv("name,type\nAlpha,A\nBeta,B\n")

        result = cast(dict[str, Any], run_agent_workflow("Show a histogram", dataset_path=path))

        self.assertEqual(result["analysis_result"]["numeric_column"], "")
        parse_openui_lang(result["openui_lang"])

    def test_openui_parser_rejects_undefined_references(self) -> None:
        with self.assertRaises(OpenUIValidationError):
            parse_openui_lang('root = Card([missing])\nheader = CardHeader("x", "y")')

    def test_template_openui_parses(self) -> None:
        result = cast(dict[str, Any], run_agent_workflow("Summarize this dataset", dataset_path=_csv_fixture()))

        parse_openui_lang(result["openui_lang"])

    def test_chat_route_streams_workflow_openui_lang(self) -> None:
        client = TestClient(app)
        response = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "List the columns and missing values"}],
                "ckan": {"connected": False, "base_url": "https://opendata.muenchen.de/"},
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers.get("x-smolnalysis-trace-id"))
        chunks = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
        payloads = [json.loads(chunk) for chunk in chunks]
        content = "".join(payload["choices"][0]["delta"].get("content", "") for payload in payloads)
        self.assertIn("root = Card", content)
        self.assertIn("table = Table", content)
        parse_openui_lang(content)

    def test_chat_trace_can_be_read_back(self) -> None:
        client = TestClient(app)
        response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "Show a histogram of median_age"}]})
        trace_id = response.headers["x-smolnalysis-trace-id"]

        trace_response = client.get(f"/api/traces/{trace_id}")

        self.assertEqual(trace_response.status_code, 200)
        payload = trace_response.json()
        self.assertEqual(payload["request_id"], trace_id)
        self.assertEqual(payload["backend"], "deterministic_workflow")
        self.assertEqual(payload["role"], "analysis")


def _csv_fixture() -> str:
    path = Path(__file__).resolve().parents[1] / "app" / "examples" / "demo_cities.csv"
    if path.exists():
        return str(path)
    return _temp_csv("city,population,median_age\nBerlin,3677000,42.6\nHamburg,1906000,42.1\nMunich,1512000,41.5\n")


def _temp_csv(content: str) -> str:
    handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
    handle.write(content)
    handle.close()
    return handle.name


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
