from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import TestCase, main


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train" / "openui_lang" / "data"))

from dataset import OpenUITrainingDataset, extract_openui_messages, make_chat_features


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


class OpenUITrainingDatasetTests(TestCase):
    def test_extracts_existing_messages(self) -> None:
        sample = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "input"},
                {"role": "assistant", "content": "root = Card([])"},
            ]
        }

        messages = extract_openui_messages(sample)

        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant"])
        self.assertEqual(messages[-1]["content"], "root = Card([])")

    def test_make_chat_features_masks_prompt_tokens(self) -> None:
        tokenizer = _FakeTokenizer()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "input"},
            {"role": "assistant", "content": "root = Card([])"},
        ]

        features = make_chat_features(tokenizer, messages, max_length=512, return_tensors="list")

        self.assertEqual(set(features), {"input_ids", "attention_mask", "labels"})
        self.assertEqual(len(features["input_ids"]), len(features["attention_mask"]))
        self.assertEqual(len(features["input_ids"]), len(features["labels"]))
        self.assertIn(-100, features["labels"])
        self.assertTrue(any(label != -100 for label in features["labels"]))

    def test_dataset_reads_json_directory_and_returns_training_features(self) -> None:
        sample = {
            "task": "render_openui",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "{}"},
                {"role": "assistant", "content": "root = Card([])"},
            ],
            "query_result": {"dataset_title": "Dataset"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "0001-sample.json").write_text(json.dumps(sample), encoding="utf-8")
            (path / "manifest.json").write_text("{}", encoding="utf-8")

            dataset = OpenUITrainingDataset(
                path,
                _FakeTokenizer(),
                max_length=128,
                return_tensors="list",
                include_metadata=True,
            )

            item = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertEqual(item["task"], "render_openui")
        self.assertEqual(item["dataset_title"], "Dataset")
        self.assertIn("input_ids", item)
        self.assertIn("attention_mask", item)
        self.assertIn("labels", item)
        self.assertTrue(any(label != -100 for label in item["labels"]))

    def test_dataset_handles_real_openui_sft_train_sample(self) -> None:
        train_data = Path("train/openui_lang/data/openui_sft_train.jsonl")
        if not train_data.exists():
            self.skipTest("OpenUI SFT train data is not present.")

        dataset = OpenUITrainingDataset(
            train_data,
            _FakeTokenizer(),
            max_length=256,
            return_tensors="list",
        )
        item = dataset[0]

        self.assertGreater(len(dataset), 0)
        self.assertEqual(set(item), {"input_ids", "attention_mask", "labels"})
        self.assertLessEqual(len(item["input_ids"]), 256)
        self.assertTrue(any(label != -100 for label in item["labels"]))


if __name__ == "__main__":
    main()
