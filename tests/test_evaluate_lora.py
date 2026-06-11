from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from unittest import TestCase, main


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train" / "ckan"))

from evaluate_lora import INFERENCE_SYSTEM_PROMPT, expected_action, normalize_prediction, prompt_messages, strict_prompt_messages
from ckan_dataset_tools import read_jsonl, validate_training_example


class EvaluateLoraTests(TestCase):
    def test_prompt_messages_remove_assistant_label(self) -> None:
        example = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "user"},
                {"role": "assistant", "content": "{}"},
            ]
        }

        self.assertEqual([message["role"] for message in prompt_messages(example)], ["system", "user"])

    def test_strict_prompt_replaces_system_prompt(self) -> None:
        example = {
            "messages": [
                {"role": "system", "content": "old"},
                {"role": "user", "content": "user"},
                {"role": "assistant", "content": "{}"},
            ]
        }

        messages = strict_prompt_messages(example)

        self.assertEqual(messages[0]["content"], INFERENCE_SYSTEM_PROMPT)
        self.assertEqual([message["role"] for message in messages], ["system", "user"])

    def test_expected_action_parses_assistant_json(self) -> None:
        example = {
            "messages": [
                {"role": "assistant", "content": json.dumps({"action": "package_search", "args": {}, "thought": "x", "confidence": 1})}
            ]
        }

        self.assertEqual(expected_action(example)["action"], "package_search")

    def test_normalize_prediction_extracts_json_from_fence(self) -> None:
        prediction = """```json
{"action":"finish","args":{}}
```"""

        self.assertEqual(normalize_prediction(prediction), '{"action":"finish","args":{}}')

    def test_challenge_eval_file_is_valid_and_balanced(self) -> None:
        rows = read_jsonl(Path("train/ckan/data/generated/challenge_eval_30.jsonl"))
        actions = Counter(expected_action(row)["action"] for row in rows)

        self.assertEqual(len(rows), 30)
        self.assertTrue(all(validate_training_example(row).ok for row in rows))
        self.assertEqual(actions["package_search"], 6)
        self.assertEqual(actions["package_show"], 6)
        self.assertEqual(actions["select_resource"], 6)
        self.assertEqual(actions["reject_result"], 6)
        self.assertEqual(actions["finish"], 6)


if __name__ == "__main__":
    main()
