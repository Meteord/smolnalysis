from __future__ import annotations

import sys
from pathlib import Path
from unittest import TestCase, main


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train" / "ckan"))

from train_minicpm_lora import DEFAULT_MODEL, build_arg_parser, load_jsonl, messages_to_text


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


class TrainMiniCpmLoraTests(TestCase):
    def test_parser_defaults_target_minicpm(self) -> None:
        args = build_arg_parser().parse_args([])

        self.assertEqual(args.model_name, DEFAULT_MODEL)
        self.assertEqual(args.lora_r, 16)
        self.assertEqual(args.lora_alpha, 32)
        self.assertTrue(args.bf16)

    def test_load_jsonl_limit(self) -> None:
        rows = load_jsonl("train/ckan/data/generated/valid_eval_golden_60_repaired.jsonl", limit=3)

        self.assertEqual(len(rows), 3)
        self.assertIn("messages", rows[0])

    def test_messages_to_text_uses_chat_template(self) -> None:
        text = messages_to_text(
            _FakeTokenizer(),
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "{}"},
            ],
        )

        self.assertIn("system: sys", text)
        self.assertIn("assistant: {}", text)


if __name__ == "__main__":
    main()
