from __future__ import annotations

import sys
from pathlib import Path
from unittest import TestCase, main


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.scripts.clean_basic_statistics import CleanerConfig, MunichBasicStatisticsCleaner


class MunichBasicStatisticsCleanerTests(TestCase):
    def test_generates_compact_openui_sample_from_table_profile(self) -> None:
        package = {
            "id": "pkg-1",
            "name": "bike-counts",
            "title": "Bike counts",
            "description": "Daily bike counts in Munich.",
            "organization": "Mobility department",
            "tags": ["bike", "traffic"],
            "groups": ["tran"],
            "resources": [
                {
                    "id": "res-1",
                    "name": "Daily values",
                    "format": "CSV",
                    "table": {
                        "entry_count": 1000,
                        "sampled_rows": 500,
                        "column_count": 5,
                        "columns": [
                            {
                                "name": "_id",
                                "dtype": "int64",
                                "non_null_count": 500,
                                "null_count": 0,
                                "example_values": [1, 2, 3],
                                "numeric": {"count": 500, "min": 1, "max": 500, "mean": 250.5, "median": 250.5},
                            },
                            {
                                "name": "datum",
                                "dtype": "object",
                                "non_null_count": 500,
                                "null_count": 0,
                                "example_values": ["2026.01.01", "2026.01.02"],
                            },
                            {
                                "name": "zaehlstelle",
                                "dtype": "object",
                                "non_null_count": 500,
                                "null_count": 0,
                                "example_values": ["Arnulf", "Olympia"],
                            },
                            {
                                "name": "gesamt",
                                "dtype": "int64",
                                "non_null_count": 500,
                                "null_count": 0,
                                "example_values": [205, 272],
                                "numeric": {"count": 500, "min": 29, "max": 3180, "mean": 825.2516, "median": 595},
                            },
                        ],
                    },
                }
            ],
        }

        samples = MunichBasicStatisticsCleaner().generate_samples([package])

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["task"], "render_openui")
        self.assertGreaterEqual(sample["quality_score"], 7.0)
        self.assertIn("BarChart", sample["component_hints"]["recommended_components"])
        self.assertEqual(sample["component_hints"]["primary_chart"], {"type": "BarChart", "x": "zaehlstelle", "y": "gesamt"})
        column_names = [column["name"] for column in sample["query_result"]["columns"]]
        self.assertNotIn("_id", column_names)
        self.assertIn("gesamt", column_names)
        self.assertEqual(sample["query_result"]["columns"][0]["kind"], "numeric")

    def test_skips_unsupported_resource_without_notice_mode(self) -> None:
        package = {
            "id": "pkg-1",
            "name": "geo-layer",
            "title": "Geo layer",
            "resources": [
                {
                    "id": "res-1",
                    "name": "WMS",
                    "format": "WMS",
                    "table": None,
                    "profile_error": "No sampler implemented for resource format=WMS",
                }
            ],
        }

        samples = MunichBasicStatisticsCleaner().generate_samples([package])

        self.assertEqual(samples, [])

    def test_can_generate_notice_samples_for_unsupported_resources(self) -> None:
        package = {
            "id": "pkg-1",
            "name": "geo-layer",
            "title": "Geo layer",
            "resources": [
                {
                    "id": "res-1",
                    "name": "WMS",
                    "format": "WMS",
                    "table": None,
                    "profile_error": "No sampler implemented for resource format=WMS",
                }
            ],
        }

        cleaner = MunichBasicStatisticsCleaner(CleanerConfig(include_notice_samples=True, min_quality_score=0))
        samples = cleaner.generate_samples([package])

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["component_hints"]["recommended_components"][0], "Notice")
        self.assertEqual(samples[0]["query_result"]["resource_format"], "wms")


if __name__ == "__main__":
    main()
