from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))


class FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 0
    pad_token = None

    def __len__(self) -> int:
        return 128

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_dict,
        return_tensors,
    ):
        count = len(messages)
        return {
            "input_ids": torch.arange(1, count + 1, dtype=torch.long).unsqueeze(0),
            "attention_mask": torch.ones((1, count), dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(int(token)) for token in token_ids)


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.generated_input_ids = None
        self.generated_input_history = []
        self.active_adapter = None

    def generate(self, **kwargs):
        self.generated_input_ids = kwargs["input_ids"].detach().clone()
        self.generated_input_history.append(self.generated_input_ids)
        extra = torch.tensor([[99, 100]], dtype=torch.long)
        return torch.cat([kwargs["input_ids"], extra], dim=-1)

    def set_adapter(self, adapter_name):
        self.active_adapter = adapter_name

    def load_adapter(self, path, adapter_name):
        self.loaded_path = path
        self.loaded_adapter_name = adapter_name


class FakePeftModel:
    @staticmethod
    def from_pretrained(model, path, adapter_name):
        model.loaded_path = path
        model.loaded_adapter_name = adapter_name
        return model


class SmolnalysisMoETests(TestCase):
    def build_wrapper(self, router_role=None):
        import backend.smolnalysis_model_wrapper as wrapper

        fake_model = FakeModel()
        peft_module = SimpleNamespace(PeftModel=FakePeftModel)

        def fake_adapter_source(role):
            return SimpleNamespace(name=role, source=f"/tmp/{role}", is_path=False)

        patches = [
            patch.object(wrapper.AutoTokenizer, "from_pretrained", return_value=FakeTokenizer()),
            patch.object(wrapper.AutoModelForCausalLM, "from_pretrained", return_value=fake_model),
            patch.object(wrapper, "_adapter_source_for_role", side_effect=fake_adapter_source),
            patch.dict(sys.modules, {"peft": peft_module}),
        ]
        for active_patch in patches:
            active_patch.start()
            self.addCleanup(active_patch.stop)
        model = wrapper.SmolnalysisMoE(
            load_in_4bit=False,
        )
        if router_role is not None:
            model._router_decision = lambda messages: wrapper.RouterDecision(
                role=router_role,
                adapter=model._adapter_from_label(router_role),
                confidence=1.0,
                logits=[],
                source="test",
            )
        return model, fake_model

    def test_forward_uses_explicit_adapter_and_supplied_messages(self) -> None:
        model, fake_model = self.build_wrapper()

        output_ids = model(
            [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "render this as OpenUI"},
            ],
            adapter="openui_translator",
            temperature=0.0,
        )

        self.assertEqual(model.active_adapter, "openui_translator")
        self.assertEqual(fake_model.active_adapter, "openui_translator")
        self.assertEqual(fake_model.generated_input_ids.tolist(), [[1, 2, 3]])
        self.assertEqual(output_ids.tolist(), [[99, 100]])

    def test_forward_keeps_history_with_no_adapter(self) -> None:
        model, fake_model = self.build_wrapper()

        model(
            [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "hello"},
            ],
            adapter=None,
            temperature=0.0,
        )

        self.assertIsNone(model.active_adapter)
        self.assertEqual(fake_model.generated_input_ids.tolist(), [[1, 2, 3]])

    def test_forward_accepts_preprocessed_inputs(self) -> None:
        model, fake_model = self.build_wrapper()

        model(
            {
                "input_ids": torch.tensor([[7, 8, 9]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            },
            adapter="openui_translator",
            temperature=0.0,
        )

        self.assertEqual(model.active_adapter, "openui_translator")
        self.assertEqual(fake_model.generated_input_ids.tolist(), [[7, 8, 9]])

    def test_forward_accepts_wrapper_preprocess_output(self) -> None:
        model, fake_model = self.build_wrapper()
        preprocessed = model._preprocess(
            [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "render this as OpenUI"},
            ]
        )

        model(preprocessed, adapter="openui_translator", temperature=0.0)

        self.assertEqual(model.active_adapter, "openui_translator")
        self.assertEqual(fake_model.generated_input_ids.tolist(), [[1, 2, 3]])

    def test_generate_chat_returns_hardcoded_hi_response(self) -> None:
        model, fake_model = self.build_wrapper()

        result = model.generate_chat([{"role": "user", "content": "hi"}], adapter=None, temperature=0.0)

        self.assertEqual(result["content"], "hi, there how can i help you?")
        self.assertEqual(result["tool_result"], "")
        self.assertEqual(result["stages"], [{"adapter": None, "input": "hardcoded_greeting"}])
        self.assertEqual(fake_model.generated_input_history, [])

    def test_generate_chat_runs_retrieval_then_openui_with_adapter_inputs(self) -> None:
        model, fake_model = self.build_wrapper(router_role="ckan_retrieval")

        result = model.generate_chat([{"role": "user", "content": "Show temperature in Munich"}], temperature=0.0)

        self.assertEqual(result["content"], "99 100")
        self.assertEqual(result["tool_result"], "99 100")
        self.assertEqual([stage["adapter"] for stage in result["stages"]], ["ckan_retrieval", "openui_translator"])
        self.assertEqual(fake_model.generated_input_history[0].tolist(), [[1]])
        self.assertEqual(fake_model.generated_input_history[1].tolist(), [[1, 2]])

    def test_generate_chat_can_force_retrieval_without_openui_followup(self) -> None:
        model, fake_model = self.build_wrapper()

        result = model.generate_chat(
            [{"role": "user", "content": "Show temperature in Munich"}],
            adapter="ckan_retrieval",
            render_openui_after_retrieval=False,
            temperature=0.0,
        )

        self.assertEqual(result["content"], "99 100")
        self.assertEqual(result["tool_result"], "99 100")
        self.assertEqual(result["stages"], [{"adapter": "ckan_retrieval", "input": "user_message"}])
        self.assertEqual(len(fake_model.generated_input_history), 1)
        self.assertEqual(fake_model.active_adapter, "ckan_retrieval")

    def test_generate_chat_uses_openui_directly_when_tool_result_is_present(self) -> None:
        model, fake_model = self.build_wrapper(router_role="openui_translator")

        result = model.generate_chat(
            [{"role": "user", "content": "Show temperature\n\nTool result:\n{\"value\": 12}"}],
            temperature=0.0,
        )

        self.assertEqual(result["content"], "99 100")
        self.assertEqual(result["tool_result"], "")
        self.assertEqual(result["stages"], [{"adapter": "openui_translator", "input": "user_message_and_tool_result"}])
        self.assertEqual(len(fake_model.generated_input_history), 1)
        self.assertEqual(fake_model.generated_input_history[0].tolist(), [[1, 2]])
        self.assertEqual(fake_model.active_adapter, "openui_translator")

    def test_generate_chat_can_force_general_agent(self) -> None:
        model, fake_model = self.build_wrapper()

        result = model.generate_chat(
            [{"role": "user", "content": "Explain virtualenvs"}],
            adapter="general_agent",
            temperature=0.0,
        )

        self.assertEqual(result["content"], "99 100")
        self.assertEqual(result["tool_result"], "")
        self.assertEqual(result["stages"], [{"adapter": None, "input": "messages"}])
        self.assertEqual(model.active_adapter, None)
        self.assertEqual(len(fake_model.generated_input_history), 1)

    def test_adapter_choice_uses_exact_router_labels(self) -> None:
        model, _fake_model = self.build_wrapper()

        self.assertIsNone(model._adapter_from_label("general_agent"))
        self.assertEqual(model._adapter_from_label("ckan_retrieval"), "ckan_retrieval")
        self.assertEqual(model._adapter_from_label("openui_translator"), "openui_translator")

        with self.assertRaises(KeyError):
            model._adapter_from_label("ckan")
        with self.assertRaises(KeyError):
            model._adapter_from_label("openui")
        with self.assertRaises(KeyError):
            model._adapter_from_label("data_analysis")

    def test_default_adapter_sources_use_local_registry(self) -> None:
        import backend.smolnalysis_model_wrapper as wrapper

        source = wrapper._adapter_source_for_role("ckan_retrieval")

        self.assertIsNotNone(source)
        assert source is not None
        self.assertEqual(source.name, "ckan_retrieval")
        self.assertTrue(source.is_path)
        self.assertTrue(Path(source.source).exists())
        self.assertTrue(source.source.endswith("train/retrieval/outputs/tool-results-minicpm5-lora/checkpoint-260"))

        openui_source = wrapper._adapter_source_for_role("openui_translator")
        self.assertIsNotNone(openui_source)
        assert openui_source is not None
        self.assertEqual(openui_source.name, "openui_translator")
        self.assertTrue(openui_source.is_path)
        self.assertTrue(Path(openui_source.source).exists())
        self.assertTrue(openui_source.source.endswith("train/openui_lang/outputs/openui-translate-mini-lora/checkpoint-160"))

    def test_registry_ignores_env_repo_ids(self) -> None:
        import backend.smolnalysis_model_wrapper as wrapper

        with patch.dict(
            "os.environ",
            {"SMOLNALYSIS_MINICPM_OPENUI_TRANSLATOR_ADAPTER_REPO_ID": "org/openui-adapter"},
        ):
            source = wrapper._adapter_source_for_role("openui_translator")

        self.assertIsNotNone(source)
        assert source is not None
        self.assertEqual(source.name, "openui_translator")
        self.assertTrue(source.is_path)
        self.assertTrue(source.source.endswith("train/openui_lang/outputs/openui-translate-mini-lora/checkpoint-160"))


if __name__ == "__main__":
    main()
