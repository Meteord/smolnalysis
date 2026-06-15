from __future__ import annotations

import sys
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train" / "openui_lang"))

import train_minicpm_lora


class _FakeTensor(list):
    def __ne__(self, other):
        return _FakeTensor([value != other for value in self])

    def __eq__(self, other):
        return _FakeTensor([value == other for value in self])

    def sum(self):
        return _FakeScalar(sum(1 for value in self if value))

    @property
    def shape(self):
        return (len(self),)


class _FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _FakeTokenizer:
    def decode(self, input_ids):
        return "".join(str(value) for value in input_ids)


class _FakeDataset:
    def __init__(self, size=3):
        self.tokenizer = _FakeTokenizer()
        self.samples = [
            {
                "task": "render_openui",
                "query_result": {"dataset_title": "Dataset"},
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "input"},
                    {"role": "assistant", "content": "root = Card([])"},
                ],
            }
            for _ in range(size)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return {
            "input_ids": _FakeTensor([1, 2, 3, 4]),
            "attention_mask": _FakeTensor([1, 1, 1, 1]),
            "labels": _FakeTensor([-100, -100, 3, 4]),
        }


class OpenUIMiniCPMTrainerTests(TestCase):
    def test_parser_defaults_target_minicpm_openui_paths(self) -> None:
        args = train_minicpm_lora.build_arg_parser().parse_args([])

        self.assertEqual(args.model_name, train_minicpm_lora.DEFAULT_MODEL)
        self.assertEqual(args.lora_r, 16)
        self.assertEqual(args.lora_alpha, 32)
        self.assertTrue(args.load_in_4bit)
        self.assertEqual(args.train_data, train_minicpm_lora.DEFAULT_TRAIN_DATA)

    def test_target_modules_are_parsed_from_csv(self) -> None:
        args = train_minicpm_lora.build_arg_parser().parse_args(["--target-modules", "q_proj,v_proj"])

        self.assertEqual(train_minicpm_lora.target_modules(args), ["q_proj", "v_proj"])

    def test_limit_dataset_mutates_samples_for_smoke_runs(self) -> None:
        dataset = _FakeDataset(size=5)

        limited = train_minicpm_lora.limit_dataset(dataset, 2)

        self.assertIs(limited, dataset)
        self.assertEqual(len(limited), 2)

    def test_build_datasets_rejects_openui_split_directories(self) -> None:
        args = train_minicpm_lora.build_arg_parser().parse_args(
            ["--train-data", "train/openui_lang/data/train"]
        )

        with self.assertRaisesRegex(ValueError, "must be a JSONL file"):
            train_minicpm_lora.build_datasets(args, object())

    @patch.object(train_minicpm_lora, "load_tokenizer", return_value=object())
    @patch.object(train_minicpm_lora, "build_datasets", return_value=(_FakeDataset(), _FakeDataset(size=1)))
    @patch.object(train_minicpm_lora, "debug_supervised_text")
    def test_dry_run_reports_masked_and_supervised_tokens(self, _debug, _datasets, _tokenizer) -> None:
        args = train_minicpm_lora.build_arg_parser().parse_args(["--dry-run"])

        summary = train_minicpm_lora.dry_run(args)

        self.assertEqual(summary["train_samples"], 3)
        self.assertEqual(summary["eval_samples"], 1)
        self.assertEqual(summary["masked_prompt_tokens"], 2)
        self.assertEqual(summary["supervised_assistant_tokens"], 2)


if __name__ == "__main__":
    main()
