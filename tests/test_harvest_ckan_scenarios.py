from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import TestCase, main


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train" / "ckan"))

from harvest_ckan_scenarios import (
    build_filter_query,
    build_inventory_scenarios,
    build_scenarios_for_package,
    fetch_dataset_inventory,
    harvest_scenarios,
    inventory_scenario_base,
    is_probably_german,
    list_named_entities,
    parse_queries,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class HarvestCkanScenariosTests(TestCase):
    def test_builds_select_and_finish_scenarios_for_tabular_resource(self) -> None:
        scenarios = build_scenarios_for_package(
            "https://opendata.muenchen.de/",
            "population",
            {
                "name": "population-indicators",
                "title": "Population indicators",
                "resources": [{"id": "csv-1", "name": "CSV", "format": "CSV"}],
            },
            0,
        )
        target_actions = {scenario["target_action"] for scenario in scenarios}

        self.assertIn("package_search", target_actions)
        self.assertIn("package_show", target_actions)
        self.assertIn("select_resource", target_actions)
        self.assertIn("finish", target_actions)
        select = next(scenario for scenario in scenarios if scenario["target_action"] == "select_resource")
        self.assertEqual(select["observed_resources"], ["population-indicators:csv-1"])

    def test_builds_retry_scenario_for_document_only_package(self) -> None:
        scenarios = build_scenarios_for_package(
            "https://opendata.muenchen.de/",
            "budget",
            {
                "name": "budget-report",
                "title": "Budget report",
                "resources": [{"id": "pdf-1", "name": "PDF", "format": "PDF"}],
            },
            0,
        )

        retry = next(scenario for scenario in scenarios if scenario["id"].endswith("_retry"))
        self.assertEqual(retry["target_action"], "package_search")
        self.assertFalse(retry["has_enough_evidence"])
        self.assertTrue(retry["tool_errors"])

    def test_harvest_scenarios_uses_ckan_search_results(self) -> None:
        captured_urls = []

        def fake_urlopen(_request, timeout):
            captured_urls.append(_request.full_url)
            return _FakeResponse(
                {
                    "success": True,
                    "result": {
                        "results": [
                            {
                                "name": "mobility-counts",
                                "title": "Mobility counts",
                                "resources": [{"id": "traffic_csv", "format": "CSV"}],
                            }
                        ]
                    },
                }
            )

        scenarios = harvest_scenarios("https://opendata.muenchen.de/", ["mobility"], 1, urlopen=fake_urlopen, groups=["transport"])

        self.assertGreaterEqual(len(scenarios), 3)
        self.assertEqual(scenarios[0]["endpoint"], "https://opendata.muenchen.de/")
        self.assertIn("mobility-counts", {package for scenario in scenarios for package in scenario["observed_packages"]})
        self.assertTrue(any("fq=groups%3Atransport" in url for url in captured_urls))
        self.assertTrue(any(scenario.get("filters", {}).get("group") == "transport" for scenario in scenarios))

    def test_parse_queries_defaults_or_combines_sources(self) -> None:
        self.assertIn("population", parse_queries(None, None))
        self.assertEqual(parse_queries(["schools"], None), ["schools"])

    def test_build_filter_query_supports_group_and_organization(self) -> None:
        self.assertEqual(build_filter_query({"group": "transport"}), "groups:transport")
        self.assertEqual(build_filter_query({"organization": "referat"}), "organization:referat")
        self.assertEqual(build_filter_query({"group": "transport", "organization": "referat"}), "groups:transport organization:referat")

    def test_list_named_entities_returns_names(self) -> None:
        def fake_urlopen(_request, timeout):
            return _FakeResponse({"success": True, "result": ["transport", "environment"]})

        names = list_named_entities("https://opendata.muenchen.de/api/3/action", "group_list", limit=1, urlopen=fake_urlopen)

        self.assertEqual(names, ["transport"])

    def test_fetch_dataset_inventory_pages_until_count(self) -> None:
        calls = []

        def fake_urlopen(request, timeout):
            calls.append(request.full_url)
            if "start=0" in request.full_url:
                return _FakeResponse({"success": True, "result": {"count": 2, "results": [{"name": "one"}]}})
            return _FakeResponse({"success": True, "result": {"count": 2, "results": [{"name": "two"}]}})

        packages = fetch_dataset_inventory("https://opendata.muenchen.de/", rows_per_page=1, urlopen=fake_urlopen)

        self.assertEqual([package["name"] for package in packages], ["one", "two"])
        self.assertEqual(len(calls), 2)

    def test_build_inventory_scenarios_uses_metadata_without_only_one_action(self) -> None:
        packages = [
            {
                "name": "dataset-a",
                "title": "Traffic counts",
                "organization": {"name": "org-a"},
                "groups": [{"name": "tran"}],
                "tags": [{"name": "traffic"}],
                "resources": [{"id": "csv-a", "format": "CSV"}],
            },
            {
                "name": "dataset-b",
                "title": "Dataset B",
                "organization": {"name": "org-b"},
                "groups": [{"name": "gove"}],
                "tags": [{"name": "report"}],
                "resources": [{"id": "pdf-b", "format": "PDF"}],
            },
        ]

        scenarios = build_inventory_scenarios("https://opendata.muenchen.de/", packages)
        target_actions = {scenario["target_action"] for scenario in scenarios}

        self.assertIn("package_search", target_actions)
        self.assertIn("package_show", target_actions)
        self.assertIn("select_resource", target_actions)
        self.assertIn("finish", target_actions)
        self.assertIn("package_search", target_actions)
        self.assertIn("tag_search", target_actions)
        self.assertIn("group_list", target_actions)
        self.assertIn("organization_list", target_actions)
        self.assertTrue(any(scenario.get("filters", {}).get("group") == "tran" for scenario in scenarios))
        self.assertFalse(any("group=" in scenario["request"] or "organization=" in scenario["request"] for scenario in scenarios))
        self.assertTrue(any("group=tran" in scenario["state"] for scenario in scenarios if scenario.get("filters")))

    def test_inventory_scenarios_retry_topic_mismatches(self) -> None:
        scenarios = build_inventory_scenarios(
            "https://opendata.muenchen.de/",
            [
                {
                    "name": "carsharing-stations",
                    "title": "Carsharing stations",
                    "organization": {"name": "mobilitaetsreferat"},
                    "groups": [{"name": "tran"}],
                    "tags": [{"name": "Digitaler Zwilling München"}],
                    "resources": [{"id": "csv-a", "format": "CSV"}],
                    "notes": "Station locations for carsharing vehicles.",
                }
            ],
        )
        actions = {scenario["target_action"] for scenario in scenarios}

        self.assertIn("package_search", actions)
        self.assertTrue(any(scenario["id"].endswith("_retry") for scenario in scenarios))
        self.assertNotIn("finish", actions)

    def test_inventory_request_language_stays_consistent(self) -> None:
        german = inventory_scenario_base(
            "https://opendata.muenchen.de/",
            {"name": "bike", "tags": [{"name": "Fahrrad"}]},
            0,
            {},
        )
        german_umlaut = inventory_scenario_base(
            "https://opendata.muenchen.de/",
            {"name": "seats", "tags": [{"name": "Sitzplätze"}]},
            0,
            {},
        )
        english = inventory_scenario_base(
            "https://opendata.muenchen.de/",
            {"name": "budget", "tags": [{"name": "budget"}]},
            0,
            {},
        )

        self.assertEqual(german["request"], "Hast du Daten zu Fahrrad?")
        self.assertEqual(german_umlaut["request"], "Hast du Daten zu Sitzplätze?")
        self.assertEqual(english["request"], "Do you have data about budget?")
        self.assertTrue(is_probably_german("Fahrrad"))
        self.assertFalse(is_probably_german("budget"))


if __name__ == "__main__":
    main()
