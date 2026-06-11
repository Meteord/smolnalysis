from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import TestCase, main


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train" / "ckan"))

from ckan_dataset_tools import (
    balanced_sample,
    build_training_example,
    package_overlap,
    repair_training_example,
    scenario_package_ids,
    seed_examples,
    split_by_package,
    validate_ckan_action,
    validate_training_example,
)


class CkanDatasetToolsTests(TestCase):
    def test_valid_package_search_action(self) -> None:
        result = validate_ckan_action(
            json.dumps(
                {
                    "thought": "Need a topic search first.",
                    "action": "package_search",
                    "args": {"query": "population Munich", "rows": 5, "start": 0},
                    "confidence": 0.8,
                }
            )
        )

        self.assertTrue(result.ok, result.issues)

    def test_rejects_invalid_json(self) -> None:
        result = validate_ckan_action("{not json")

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "invalid_json")

    def test_rejects_unobserved_package_show(self) -> None:
        result = validate_ckan_action(
            json.dumps(
                {
                    "thought": "Inspect package.",
                    "action": "package_show",
                    "args": {"package_id": "invented"},
                    "confidence": 0.7,
                }
            ),
            {"observed_packages": ["known"]},
        )

        self.assertFalse(result.ok)
        self.assertIn("unobserved_package", {issue.code for issue in result.issues})

    def test_rejects_finish_without_evidence(self) -> None:
        result = validate_ckan_action(
            json.dumps(
                {
                    "thought": "Stop now.",
                    "action": "finish",
                    "args": {"selected_candidates": [{"package_id": "known"}], "rationale": "Enough."},
                    "confidence": 0.9,
                }
            ),
            {"has_enough_evidence": False},
        )

        self.assertFalse(result.ok)
        self.assertIn("finish_too_early", {issue.code for issue in result.issues})

    def test_seed_examples_are_valid(self) -> None:
        for example in seed_examples():
            result = validate_training_example(example)
            self.assertTrue(result.ok, result.issues)

    def test_training_example_builder_uses_json_string_content(self) -> None:
        example = build_training_example(
            "Request: find schools.",
            {
                "thought": "Search schools.",
                "action": "package_search",
                "args": {"query": "schools", "rows": 5, "start": 0},
                "confidence": 0.75,
            },
        )

        assistant = example["messages"][-1]
        self.assertEqual(assistant["role"], "assistant")
        self.assertIsInstance(assistant["content"], str)
        self.assertTrue(validate_training_example(example).ok)

    def test_balanced_sample_spreads_across_target_actions(self) -> None:
        rows = [{"target_action": "a", "i": i} for i in range(10)] + [{"target_action": "b", "i": i} for i in range(10)]

        sampled = balanced_sample(rows, 6, seed=1)

        self.assertEqual(len(sampled), 6)
        self.assertEqual({row["target_action"] for row in sampled}, {"a", "b"})

    def test_scenario_package_ids_reads_observed_and_summary_ids(self) -> None:
        ids = scenario_package_ids({"observed_packages": ["a"], "package_summary": {"id": "b"}})

        self.assertEqual(ids, {"a", "b"})

    def test_split_by_package_has_no_package_overlap(self) -> None:
        rows = []
        for package_index in range(10):
            for action in ["package_search", "package_show", "select_resource"]:
                rows.append(
                    {
                        "target_action": action,
                        "observed_packages": [f"package-{package_index}"],
                        "package_summary": {"id": f"package-{package_index}"},
                    }
                )

        train_rows, eval_rows = split_by_package(rows, eval_size=6, train_size=12, seed=3)

        self.assertEqual(len(train_rows), 12)
        self.assertEqual(len(eval_rows), 6)
        self.assertEqual(package_overlap(train_rows, eval_rows), set())

    def test_repair_training_example_trims_long_thought(self) -> None:
        long_thought = " ".join(["word"] * 45)
        example = build_training_example(
            "Request: find schools.",
            {
                "thought": long_thought,
                "action": "package_search",
                "args": {"query": "schools", "rows": 5, "start": 0},
                "confidence": 0.75,
            },
        )

        repaired = repair_training_example(example, max_thought_words=12)

        self.assertFalse(validate_training_example(example).ok)
        self.assertTrue(validate_training_example(repaired).ok)
        payload = json.loads(repaired["messages"][-1]["content"])
        self.assertLessEqual(len(payload["thought"].split()), 12)


if __name__ == "__main__":
    main()
