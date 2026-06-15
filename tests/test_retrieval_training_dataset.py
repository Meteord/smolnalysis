from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import TestCase, main


from train.retrieval.data.dataset import (
    ToolResultTrainingDataset,
    extract_retrieval_messages,
    make_chat_features,
)


class _FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 0
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "\n".join(f"<{message['role']}> {message['content']}" for message in messages)
        if add_generation_prompt:
            text += "\n<assistant> "
        return text

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return {"input_ids": [ord(char) % 251 + 1 for char in text]}


class RetrievalTrainingDatasetTests(TestCase):
    def test_extracts_question_to_json_messages(self) -> None:
        sample = {
            "messages": [
                {"role": "system", "content": "json only"},
                {"role": "user", "content": "Wie hoch war Temperatur?"},
                {"role": "assistant", "content": '{"value": 12}'},
            ]
        }

        messages = extract_retrieval_messages(sample)

        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant"])
        self.assertEqual(json.loads(messages[-1]["content"]), {"value": 12})

    def test_rejects_tool_result_marker_in_label(self) -> None:
        sample = {
            "messages": [
                {"role": "system", "content": "json only"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": 'Tool result: {"value": 12}'},
            ]
        }

        with self.assertRaisesRegex(ValueError, "marker"):
            extract_retrieval_messages(sample)

    def test_make_chat_features_masks_prompt_tokens(self) -> None:
        tokenizer = _FakeTokenizer()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "input"},
            {"role": "assistant", "content": '{"value": 1}'},
        ]

        features = make_chat_features(tokenizer, messages, max_length=512, return_tensors="list")

        self.assertEqual(set(features), {"input_ids", "attention_mask", "labels"})
        self.assertIn(-100, features["labels"])
        self.assertTrue(any(label != -100 for label in features["labels"]))

    def test_dataset_reads_real_train_sample(self) -> None:
        train_data = Path("train/retrieval/data/tool_result_train.jsonl")
        if not train_data.exists():
            self.skipTest("Retrieval train data is not present.")

        dataset = ToolResultTrainingDataset(
            train_data,
            _FakeTokenizer(),
            max_length=256,
            return_tensors="list",
        )
        item = dataset[0]
        first_sample = dataset.samples[0]

        self.assertGreater(len(dataset), 0)
        self.assertNotIn("Tool result:", first_sample["messages"][1]["content"])
        self.assertNotIn("Tool result:", first_sample["messages"][2]["content"])
        self.assertEqual(set(item), {"input_ids", "attention_mask", "labels"})
        self.assertTrue(any(label != -100 for label in item["labels"]))

    def test_dataset_reads_json_directory(self) -> None:
        sample = {
            "messages": [
                {"role": "system", "content": "json only"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": '{"value": 1}'},
            ],
            "metadata": {"task": "generate_tool_result"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "0001.json").write_text(json.dumps(sample), encoding="utf-8")
            (path / "manifest.json").write_text("{}", encoding="utf-8")
            dataset = ToolResultTrainingDataset(
                path,
                _FakeTokenizer(),
                return_tensors="list",
                include_metadata=True,
            )
            item = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertEqual(item["metadata"]["task"], "generate_tool_result")


if __name__ == "__main__":
    main()
