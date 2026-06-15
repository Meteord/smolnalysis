from __future__ import annotations

import json
from unittest import TestCase, main

from train.retrieval.sanity_check_adapter import (
    SampleResult,
    clean_generated_json,
    expected_keys_present,
    summarize_results,
)


class RetrievalSanityCheckTests(TestCase):
    def test_clean_generated_json_extracts_first_object(self) -> None:
        raw = 'notes\n```json\n{"value": 12}\n```\ntrailing'

        cleaned = clean_generated_json(raw)

        self.assertEqual(json.loads(cleaned), {"value": 12})

    def test_expected_keys_present_accepts_extra_keys(self) -> None:
        self.assertTrue(expected_keys_present({"domain": "weather"}, {"domain": "weather", "value": 12}))
        self.assertFalse(expected_keys_present({"domain": "weather", "value": 12}, {"domain": "weather"}))

    def test_summarize_results_reports_rates_and_failures(self) -> None:
        results = [
            SampleResult(
                index=1,
                question="q",
                expected={"value": 1},
                raw_output='{"value": 1}',
                cleaned_output='{"value": 1}',
                parsed_output={"value": 1},
                valid_json=True,
                exact_match=True,
                no_marker=True,
                expected_keys_present=True,
            ),
            SampleResult(
                index=2,
                question="q",
                expected={"value": 2},
                raw_output="not json",
                cleaned_output="not json",
                parsed_output=None,
                valid_json=False,
                exact_match=False,
                no_marker=True,
                expected_keys_present=False,
            ),
        ]

        summary = summarize_results(results)

        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["valid_json_rate"], 0.5)
        self.assertEqual(summary["exact_match_rate"], 0.5)
        self.assertEqual(summary["failed_indexes"], [2])


if __name__ == "__main__":
    main()
