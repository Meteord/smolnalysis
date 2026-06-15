from __future__ import annotations

import sys
from pathlib import Path
from unittest import TestCase, main


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train" / "openui_lang"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train" / "openui_lang" / "data"))

import openui_semantic_eval


class _FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "\n".join(f"<{message['role']}> {message['content']}" for message in messages)
        if add_generation_prompt:
            text += "\n<assistant> "
        return text


class _FakeModel:
    training = True

    def __init__(self):
        self.eval_called = False
        self.train_called = False

    def eval(self):
        self.eval_called = True

    def train(self):
        self.train_called = True


class OpenUISemanticEvalTests(TestCase):
    def test_sample_metrics_accepts_openui_lang_component_assignments(self) -> None:
        sample = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": 'Q\n\nTool result:\n{"value": 12, "year": 2024}'},
                {
                    "role": "assistant",
                    "content": (
                        "root = Root([summary, metrics])\n"
                        "summary = InsightCard(\"Value\", \"Year 2024\")\n"
                        "metrics = MetricGrid([m1])\n"
                        "m1 = Metric(\"Value\", \"12\", \"2024\")"
                    ),
                },
            ],
            "metadata": {"data_shape": "scalar", "component": "MetricGrid"},
        }

        metrics = openui_semantic_eval._sample_metrics(sample, sample["messages"][-1]["content"])

        self.assertEqual(metrics["component_accuracy"], 1.0)
        self.assertEqual(metrics["valid_openui_like_rate"], 1.0)

    def test_sample_metrics_accepts_comparison_delta_as_derived_number(self) -> None:
        sample = {
            "messages": [
                {"role": "system", "content": "sys"},
                {
                    "role": "user",
                    "content": (
                        "Vergleiche.\n\nTool result:\n"
                        '{"current":{"year":2024,"value":120},'
                        '"previous":{"year":2023,"value":100},"unit":"x"}'
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "<ComparisonCard currentLabel=\"2024\" currentValue={120} "
                        "previousLabel=\"2023\" previousValue={100} deltaValue={20} />"
                    ),
                },
            ],
            "metadata": {"data_shape": "comparison", "component": "ComparisonCard"},
        }

        metrics = openui_semantic_eval._sample_metrics(sample, sample["messages"][-1]["content"])

        self.assertEqual(metrics["component_accuracy"], 1.0)
        self.assertEqual(metrics["required_value_accuracy"], 1.0)
        self.assertEqual(metrics["tool_value_accuracy"], 1.0)
        self.assertEqual(metrics["hallucinated_number_rate"], 0.0)

    def test_sample_metrics_detects_hallucinated_numbers(self) -> None:
        sample = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": 'Q\n\nTool result:\n{"value": 12, "year": 2024}'},
                {"role": "assistant", "content": "<StatCard value={12} />"},
            ],
            "metadata": {"data_shape": "scalar", "component": "StatCard"},
        }

        metrics = openui_semantic_eval._sample_metrics(sample, "<StatCard value={99} />")

        self.assertGreater(metrics["hallucinated_number_rate"], 0.0)
        self.assertEqual(metrics["required_value_accuracy"], 0.0)

    def test_generation_prompt_excludes_assistant_label(self) -> None:
        sample = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": 'input\n\nTool result:\n{"value": 42}'},
                {"role": "assistant", "content": "<SecretCard value={42} />"},
            ],
            "metadata": {"data_shape": "scalar", "component": "SecretCard"},
        }
        tokenizer = _FakeTokenizer()

        prompts: list[str] = []

        def fake_generate_batch(model, tokenizer, samples, max_new_tokens):
            messages = openui_semantic_eval.extract_openui_messages(samples[0])
            prompts.append(
                openui_semantic_eval.apply_chat_template(
                    tokenizer,
                    openui_semantic_eval.prompt_messages(messages),
                    add_generation_prompt=True,
                )
            )
            return ["<SecretCard value={42} />"]

        original = openui_semantic_eval._generate_batch
        openui_semantic_eval._generate_batch = fake_generate_batch
        try:
            metrics = openui_semantic_eval.evaluate_openui_semantic(
                _FakeModel(),
                tokenizer,
                _Dataset([sample]),
                max_samples=1,
            )
        finally:
            openui_semantic_eval._generate_batch = original

        self.assertNotIn("<SecretCard value={42} />", prompts[0])
        self.assertIn("<assistant>", prompts[0])
        self.assertEqual(metrics["semantic_eval_samples"], 1.0)
        self.assertEqual(metrics["semantic_score"], 1.0)

    def test_evaluator_counts_parse_failures_without_raising(self) -> None:
        bad_sample = {"messages": [{"role": "user", "content": "missing assistant"}], "metadata": {}}

        metrics = openui_semantic_eval.evaluate_openui_semantic(
            _FakeModel(),
            _FakeTokenizer(),
            _Dataset([bad_sample]),
            max_samples=1,
        )

        self.assertEqual(metrics["semantic_eval_samples"], 1.0)
        self.assertEqual(metrics["semantic_eval_failed_samples"], 1.0)


class _Dataset:
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


if __name__ == "__main__":
    main()
