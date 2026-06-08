from __future__ import annotations

import json
import os
import sys
import urllib.error
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import MagicMock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import llm_support


class LlmSupportTests(TestCase):
    def setUp(self) -> None:
        self._old_env = os.environ.copy()
        for key in list(os.environ):
            if key.startswith("SMOLNALYSIS_LLM_"):
                os.environ.pop(key)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._old_env)

    def test_missing_global_key_marks_roles_missing(self) -> None:
        os.environ["SMOLNALYSIS_LLM_BASE_URL"] = "https://llm.example.test"
        _set_models()

        status = llm_support.llm_status()

        self.assertEqual(len(status["roles"]), 4)
        self.assertTrue(all(not role["configured"] for role in status["roles"]))
        self.assertTrue(all("API key" in role["message"] for role in status["roles"]))

    def test_shared_defaults_configure_all_roles(self) -> None:
        os.environ["SMOLNALYSIS_LLM_BASE_URL"] = "https://llm.example.test"
        os.environ["SMOLNALYSIS_LLM_API_KEY"] = "secret-key"
        _set_models()

        status = llm_support.llm_status()

        self.assertTrue(all(role["configured"] for role in status["roles"]))
        self.assertEqual({role["base_url_display"] for role in status["roles"]}, {"https://llm.example.test"})

    def test_per_role_model_override(self) -> None:
        os.environ["SMOLNALYSIS_LLM_BASE_URL"] = "https://llm.example.test"
        os.environ["SMOLNALYSIS_LLM_API_KEY"] = "secret-key"
        _set_models()
        os.environ["SMOLNALYSIS_LLM_CKAN_TOOL_MODEL"] = "tool-model-special"

        roles = {role["key"]: role for role in llm_support.llm_status()["roles"]}

        self.assertEqual(roles["ckan_tool"]["model"], "tool-model-special")

    def test_api_keys_are_not_in_status_output(self) -> None:
        os.environ["SMOLNALYSIS_LLM_BASE_URL"] = "https://llm.example.test"
        os.environ["SMOLNALYSIS_LLM_API_KEY"] = "super-secret"
        _set_models()

        body = json.dumps(llm_support.llm_status())

        self.assertNotIn("super-secret", body)
        self.assertNotIn("api_key", body.casefold())

    @patch("llm_support.urllib.request.urlopen")
    def test_validation_success(self, urlopen: MagicMock) -> None:
        os.environ["SMOLNALYSIS_LLM_BASE_URL"] = "https://llm.example.test"
        os.environ["SMOLNALYSIS_LLM_API_KEY"] = "secret-key"
        _set_models("shared-model")
        urlopen.return_value = _response({"data": [{"id": "shared-model"}]})

        result = llm_support.validate_llms()

        self.assertTrue(all(role["validation_status"] == "valid" for role in result["roles"]))

    @patch("llm_support.urllib.request.urlopen")
    def test_validation_models_endpoint_unsupported(self, urlopen: MagicMock) -> None:
        os.environ["SMOLNALYSIS_LLM_BASE_URL"] = "https://llm.example.test"
        os.environ["SMOLNALYSIS_LLM_API_KEY"] = "secret-key"
        _set_models()
        urlopen.side_effect = urllib.error.HTTPError("url", 404, "not found", {}, None)

        result = llm_support.validate_llms()

        self.assertTrue(all(role["validation_status"] == "unvalidated" for role in result["roles"]))

    @patch("llm_support.urllib.request.urlopen")
    def test_validation_timeout(self, urlopen: MagicMock) -> None:
        os.environ["SMOLNALYSIS_LLM_BASE_URL"] = "https://llm.example.test"
        os.environ["SMOLNALYSIS_LLM_API_KEY"] = "secret-key"
        _set_models()
        urlopen.side_effect = TimeoutError()

        result = llm_support.validate_llms()

        self.assertTrue(all(role["validation_status"] == "error" for role in result["roles"]))

    @patch("llm_support.urllib.request.urlopen")
    def test_validation_invalid_json(self, urlopen: MagicMock) -> None:
        os.environ["SMOLNALYSIS_LLM_BASE_URL"] = "https://llm.example.test"
        os.environ["SMOLNALYSIS_LLM_API_KEY"] = "secret-key"
        _set_models()
        urlopen.return_value = _response_bytes(b"{nope")

        result = llm_support.validate_llms()

        self.assertTrue(all(role["validation_status"] == "error" for role in result["roles"]))


def _set_models(model: str | None = None) -> None:
    os.environ["SMOLNALYSIS_LLM_GENERAL_AGENT_MODEL"] = model or "general-model"
    os.environ["SMOLNALYSIS_LLM_CKAN_TOOL_MODEL"] = model or "ckan-model"
    os.environ["SMOLNALYSIS_LLM_DATA_ANALYSIS_MODEL"] = model or "analysis-model"
    os.environ["SMOLNALYSIS_LLM_OPENUI_TRANSLATOR_MODEL"] = model or "openui-model"


def _response(payload: dict):
    return _response_bytes(json.dumps(payload).encode("utf-8"))


def _response_bytes(body: bytes):
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = body
    return response


if __name__ == "__main__":
    main()
