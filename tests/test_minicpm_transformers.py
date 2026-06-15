from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))


class MiniCpmTransformersTests(TestCase):
    def setUp(self) -> None:
        self._old_env = os.environ.copy()
        os.environ["SMOLNALYSIS_MINICPM_TRANSFORMERS_EAGER_LOAD"] = "false"

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._old_env)

    def test_runtime_status_does_not_load_model(self) -> None:
        module = importlib.reload(importlib.import_module("backend.minicpm_transformers"))

        status = module.runtime_status()

        self.assertEqual(status["backend"], "transformers")
        self.assertEqual(status["model_family"], "MiniCPM")
        self.assertEqual(status["model"], "openbmb/MiniCPM5-1B")
        self.assertEqual(status["model_hub_url"], "https://huggingface.co/openbmb/MiniCPM5-1B")
        self.assertFalse(status["eager_load"]["enabled"])
        self.assertEqual(status["cache"]["loaded_models"], 0)
        self.assertFalse(status["router"]["enabled"])
        self.assertIn("ckan_retrieval", status["roles"])
        self.assertEqual(status["roles"]["ckan_retrieval"]["temperature"], 0.0)
        self.assertEqual(
            status["roles"]["ckan_retrieval"]["adapter_repo_id"],
            "build-small-hackathon/smolnalysis-ckan-retrieval-minicpm5-lora",
        )
        self.assertTrue(status["roles"]["ckan_retrieval"]["adapter_configured"])

    def test_role_config_reads_peft_adapter_repo(self) -> None:
        os.environ["SMOLNALYSIS_MINICPM_CKAN_RETRIEVAL_ADAPTER_REPO_ID"] = "org/adapter"
        module = importlib.reload(importlib.import_module("backend.minicpm_transformers"))

        config = module.role_config("ckan_retrieval")
        status = module.runtime_status()

        self.assertEqual(config.adapter_repo_id, "org/adapter")
        self.assertEqual(status["roles"]["ckan_retrieval"]["adapter_hub_url"], "https://huggingface.co/org/adapter")

    def test_generate_trace_uses_cached_runtime_metadata(self) -> None:
        module = importlib.reload(importlib.import_module("backend.minicpm_transformers"))

        with patch.object(module, "_generate", return_value=("ok", {
            "cache": {"hit": True, "loaded_models": 1, "hits": 1, "misses": 1},
            "adapter_source": "",
            "device": "cuda:0",
            "input_tokens": 8,
            "output_tokens": 2,
        })):
            response, trace = module.generate_chat_response_with_trace(
                [{"role": "user", "content": "hello"}],
                adapter="auto",
            )

        self.assertEqual(response, "ok")
        self.assertEqual(trace["backend"], "transformers")
        self.assertEqual(trace["cache"]["hit"], True)
        self.assertIn("cuda:0", trace["events"][-1]["detail"])

    def test_auto_role_uses_router_prediction_when_enabled(self) -> None:
        os.environ["SMOLNALYSIS_ROUTER_ENABLED"] = "true"
        module = importlib.reload(importlib.import_module("backend.minicpm_transformers"))

        with patch(
            "backend.router_runtime.predict_role",
            return_value=SimpleNamespace(role="openui_translator", confidence=0.91),
        ) as predict_role:
            role = module.route_role([{"role": "user", "content": "Find a dataset about bikes"}], adapter="auto")

        self.assertEqual(role, "openui_translator")
        predict_role.assert_called_once()

    def test_auto_role_falls_back_to_heuristic_without_router_prediction(self) -> None:
        os.environ["SMOLNALYSIS_ROUTER_ENABLED"] = "true"
        module = importlib.reload(importlib.import_module("backend.minicpm_transformers"))

        with patch("backend.router_runtime.predict_role", return_value=None):
            role = module.route_role([{"role": "user", "content": "Find a dataset about bikes"}], adapter="auto")

        self.assertEqual(role, "ckan_retrieval")


if __name__ == "__main__":
    main()
