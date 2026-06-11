from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import TestCase, main


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train" / "ckan"))

from evaluate_lora import expected_action, normalize_prediction, prompt_messages


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


if __name__ == "__main__":
    main()
