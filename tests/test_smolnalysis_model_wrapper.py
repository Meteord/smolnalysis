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
        self.active_adapter = None

    def generate(self, **kwargs):
        self.generated_input_ids = kwargs["input_ids"].detach().clone()
        extra = torch.tensor([[99, 100]], dtype=torch.long)
        return torch.cat([kwargs["input_ids"], extra], dim=-1)

    def set_adapter(self, adapter_name):
        self.active_adapter = adapter_name

    def load_adapter(self, path, adapter_name):
        self.loaded_path = path
        self.loaded_adapter_name = adapter_name


class FakeRouter(torch.nn.Module):
    def __init__(self, label_index: int, num_labels: int = 3) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.label_index = label_index
        self.num_labels = num_labels
        self.seen_input_ids = None

    def forward(self, **kwargs):
        self.seen_input_ids = kwargs["input_ids"].detach().clone()
        logits = torch.zeros((1, self.num_labels), dtype=torch.float)
        logits[0, self.label_index] = 10
        return {"logits": logits}


class FakePeftModel:
    @staticmethod
    def from_pretrained(model, path, adapter_name):
        model.loaded_path = path
        model.loaded_adapter_name = adapter_name
        return model


class SmolnalysisMoETests(TestCase):
    def build_wrapper(self, router: FakeRouter):
        import backend.smolnalysis_model_wrapper as wrapper

        fake_model = FakeModel()
        fake_adapter = SimpleNamespace(name="openui_translator", source="org/openui-adapter", is_path=False)
        peft_module = SimpleNamespace(PeftModel=FakePeftModel)
        patches = [
            patch.object(wrapper.AutoTokenizer, "from_pretrained", return_value=FakeTokenizer()),
            patch.object(wrapper.AutoModelForCausalLM, "from_pretrained", return_value=fake_model),
            patch.object(wrapper, "_adapter_source_for_role", return_value=fake_adapter),
            patch.dict(sys.modules, {"peft": peft_module}),
        ]
        for active_patch in patches:
            active_patch.start()
            self.addCleanup(active_patch.stop)
        model = wrapper.SmolnalysisMoE(
            task_router=router,
            router_labels=["general_agent", "ckan_retrieval", "openui_translator"],
            load_in_4bit=False,
        )
        return model, fake_model

    def test_forward_routes_adapter_and_uses_latest_user_message_only(self) -> None:
        router = FakeRouter(label_index=2)
        model, fake_model = self.build_wrapper(router)

        output_ids = model(
            [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "render this as OpenUI"},
            ],
            adapter="auto",
            temperature=0.0,
        )

        self.assertEqual(model.active_adapter, "openui_translator")
        self.assertEqual(fake_model.active_adapter, "openui_translator")
        self.assertEqual(router.seen_input_ids.tolist(), [[1]])
        self.assertEqual(fake_model.generated_input_ids.tolist(), [[1]])
        self.assertEqual(output_ids.tolist(), [[99, 100]])

    def test_route_records_router_decision(self) -> None:
        router = FakeRouter(label_index=2)
        model, _fake_model = self.build_wrapper(router)

        _preprocessed, decision = model.route("render this as OpenUI")

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.role, "openui_translator")
        self.assertEqual(decision.adapter, "openui_translator")
        self.assertGreater(decision.confidence, 0.9)
        self.assertEqual(len(decision.logits), 3)

    def test_forward_keeps_history_when_router_selects_base(self) -> None:
        router = FakeRouter(label_index=0)
        model, fake_model = self.build_wrapper(router)

        model(
            [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "hello"},
            ],
            adapter="auto",
            temperature=0.0,
        )

        self.assertIsNone(model.active_adapter)
        self.assertEqual(fake_model.generated_input_ids.tolist(), [[1, 2, 3]])

    def test_forward_accepts_preprocessed_inputs(self) -> None:
        router = FakeRouter(label_index=2)
        model, fake_model = self.build_wrapper(router)

        model(
            {
                "input_ids": torch.tensor([[7, 8, 9]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            },
            adapter="auto",
            temperature=0.0,
        )

        self.assertEqual(model.active_adapter, "openui_translator")
        self.assertEqual(router.seen_input_ids.tolist(), [[7, 8, 9]])
        self.assertEqual(fake_model.generated_input_ids.tolist(), [[7, 8, 9]])

    def test_forward_accepts_wrapper_preprocess_output(self) -> None:
        router = FakeRouter(label_index=2)
        model, fake_model = self.build_wrapper(router)
        preprocessed = model._preprocess(
            [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "render this as OpenUI"},
            ]
        )

        model(preprocessed, adapter="ROUTER", temperature=0.0)

        self.assertEqual(model.active_adapter, "openui_translator")
        self.assertEqual(router.seen_input_ids.tolist(), [[1]])
        self.assertEqual(fake_model.generated_input_ids.tolist(), [[1]])

    def test_default_adapter_sources_match_available_minicpm_adapter(self) -> None:
        import backend.smolnalysis_model_wrapper as wrapper

        self.assertIsNone(wrapper._adapter_source_for_role("ckan_retrieval"))
        source = wrapper._adapter_source_for_role("openui_translator")

        self.assertIsNotNone(source)
        assert source is not None
        self.assertEqual(source.name, "openui_translator")
        self.assertTrue(source.is_path)
        self.assertTrue((Path(source.source) / "adapter_config.json").exists())


if __name__ == "__main__":
    main()
