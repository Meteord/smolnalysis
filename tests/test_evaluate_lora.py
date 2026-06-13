from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from unittest import TestCase, main


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train" / "ckan"))

from evaluate_lora import INFERENCE_SYSTEM_PROMPT, expected_action, extract_first_json_object, normalize_prediction, prompt_messages, strict_prompt_messages
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

    def test_extract_first_json_object_ignores_trailing_json(self) -> None:
        prediction = '{"action":"tag_search","args":{"query":"Fahrrad"}} {"action":"package_search","args":{}}'

        self.assertEqual(extract_first_json_object(prediction), '{"action":"tag_search","args":{"query":"Fahrrad"}}')

    def test_extract_first_json_object_handles_braces_in_strings(self) -> None:
        prediction = '{"thought":"Use {literal} text.","action":"finish","args":{}}\nDone.'

        self.assertEqual(extract_first_json_object(prediction), '{"thought":"Use {literal} text.","action":"finish","args":{}}')

    def test_multitool_eval_file_is_valid_and_balanced(self) -> None:
        rows = read_jsonl(Path("train/ckan/data/multitool_eval_golden.jsonl"))
        actions = Counter(expected_action(row)["action"] for row in rows)

        self.assertEqual(len(rows), 8)
        self.assertTrue(all(validate_training_example(row).ok for row in rows))
        for action in [
            "tag_search",
            "group_list",
            "organization_list",
            "package_search",
            "package_show",
            "select_resource",
            "finish",
            "ask_clarification",
        ]:
            self.assertEqual(actions[action], 1)

    def test_current_contract_accepts_new_catalog_actions(self) -> None:
        for action, args in [
            ("tag_search", {"query": "Fahrrad", "rows": 10}),
            ("group_list", {"rows": 15}),
            ("organization_list", {"rows": 15}),
        ]:
            example = {
                "messages": [
                    {
                        "role": "assistant",
                        "content": json.dumps({"action": action, "args": args, "thought": "Discover catalog context.", "confidence": 0.8}),
                    }
                ]
            }

            self.assertTrue(validate_training_example(example).ok)


if __name__ == "__main__":
    main()
