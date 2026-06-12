from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import TestCase, main


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from backend import minicpm_llama_cpp


class MiniCpmLlamaCppTests(TestCase):
    def setUp(self) -> None:
        self._old_env = os.environ.copy()
        for key in list(os.environ):
            if key.startswith("SMOLNALYSIS_MINICPM_") or key in {"MODEL_PATH", "MODEL_REPO_ID", "MODEL_FILENAME"}:
                os.environ.pop(key)
        minicpm_llama_cpp._load_llama_cached.cache_clear()

    def tearDown(self) -> None:
        minicpm_llama_cpp._load_llama_cached.cache_clear()
        os.environ.clear()
        os.environ.update(self._old_env)

    def test_normalizes_role_aliases(self) -> None:
        self.assertEqual(minicpm_llama_cpp.normalize_role("ckan_tool"), "ckan_retrieval")
        self.assertEqual(minicpm_llama_cpp.normalize_role("retrieval"), "ckan_retrieval")
        self.assertEqual(minicpm_llama_cpp.normalize_role("analysis"), "data_analysis")
        self.assertEqual(minicpm_llama_cpp.normalize_role("openui"), "openui_translator")
        self.assertEqual(minicpm_llama_cpp.normalize_role("base"), "general_agent")

    def test_auto_routes_by_latest_user_message(self) -> None:
        self.assertEqual(_route("Search CKAN for population datasets"), "ckan_retrieval")
        self.assertEqual(_route("Analyze missing values and trends"), "data_analysis")
        self.assertEqual(_route("Render this as OpenUI cards"), "openui_translator")
        self.assertEqual(_route("Hello there"), "general_agent")

    def test_role_config_uses_shared_model_defaults(self) -> None:
        os.environ["MODEL_REPO_ID"] = "org/minicpm-gguf"
        os.environ["MODEL_FILENAME"] = "minicpm.Q4_K_M.gguf"

        config = minicpm_llama_cpp.role_config("general_agent")

        self.assertEqual(config.model_repo_id, "org/minicpm-gguf")
        self.assertEqual(config.model_filename, "minicpm.Q4_K_M.gguf")

    def test_role_config_uses_role_specific_lora(self) -> None:
        os.environ["MODEL_PATH"] = "/models/minicpm.gguf"
        os.environ["SMOLNALYSIS_MINICPM_CKAN_RETRIEVAL_LORA_PATH"] = "/models/ckan.gguf"

        config = minicpm_llama_cpp.role_config("ckan_retrieval")

        self.assertEqual(config.model_path, "/models/minicpm.gguf")
        self.assertEqual(config.lora_path, "/models/ckan.gguf")

    def test_role_config_accepts_hub_lora_files(self) -> None:
        os.environ["MODEL_REPO_ID"] = "org/minicpm-gguf"
        os.environ["MODEL_FILENAME"] = "minicpm.Q4_K_M.gguf"
        os.environ["SMOLNALYSIS_MINICPM_OPENUI_TRANSLATOR_LORA_REPO_ID"] = "org/smolnalysis-loras"
        os.environ["SMOLNALYSIS_MINICPM_OPENUI_TRANSLATOR_LORA_FILENAME"] = "openui-adapter.gguf"

        config = minicpm_llama_cpp.role_config("openui_translator")

        self.assertEqual(config.lora_repo_id, "org/smolnalysis-loras")
        self.assertEqual(config.lora_filename, "openui-adapter.gguf")

    def test_role_system_prompt_is_added_once(self) -> None:
        messages = [{"role": "user", "content": "Search CKAN"}]

        routed = minicpm_llama_cpp._with_role_system_prompt(messages, "ckan_retrieval")

        self.assertEqual(routed[0]["role"], "system")
        self.assertIn("CKAN retrieval", routed[0]["content"])
        self.assertEqual(routed[1:], messages)

        with_existing_system = [{"role": "system", "content": "Use this persona."}, *messages]
        self.assertIs(
            minicpm_llama_cpp._with_role_system_prompt(with_existing_system, "ckan_retrieval"),
            with_existing_system,
        )


def _route(prompt: str) -> str:
    return minicpm_llama_cpp.route_role([{"role": "user", "content": prompt}], "auto")


if __name__ == "__main__":
    main()
