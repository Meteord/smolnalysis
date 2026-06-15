from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import TestCase, main


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import openui_adapter_demo


class OpenUIAdapterDemoTests(TestCase):
    def test_parse_and_make_user_message_match_training_format(self) -> None:
        content = 'Zeige den Wert.\n\nTool result:\n{"value": 12, "unit": "h"}'

        query, tool_result = openui_adapter_demo.parse_user_content(content)
        rebuilt = openui_adapter_demo.make_user_message(query, json.dumps(tool_result))

        self.assertEqual(query, "Zeige den Wert.")
        self.assertEqual(tool_result["value"], 12)
        self.assertNotIn("User query:", rebuilt)
        self.assertTrue(rebuilt.startswith("Zeige den Wert.\n\nTool result:\n"))

    def test_load_examples_prefers_diverse_training_shapes(self) -> None:
        rows = [
            _sample("scalar", "StatCard"),
            _sample("comparison", "ComparisonCard"),
            _sample("table", "TableCard"),
            _sample("scalar", "StatCard", query="another"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            examples = openui_adapter_demo.load_examples(path, limit=3)

        self.assertEqual([example.label.split(" / ")[1] for example in examples], ["scalar", "comparison", "table"])

    def test_preview_renders_trained_component_families(self) -> None:
        comparison = (
            '<ComparisonCard title="A" currentLabel="2024" currentValue={12} '
            'previousLabel="2023" previousValue={9} deltaValue={3} deltaDirection="up" unit="h" />'
        )
        table = (
            '<TableCard title="T" data={[{ office: "KVR", value: 7, unit: "h" }]} />'
        )
        progress = '<ProgressCard title="P" value={25} max={50} unit="%" />'

        self.assertIn("Delta up", openui_adapter_demo.render_component_preview(comparison))
        self.assertIn("<table>", openui_adapter_demo.render_component_preview(table))
        self.assertIn("width:50.0%", openui_adapter_demo.render_component_preview(progress))

    def test_preview_renders_openui_lang_bar_chart(self) -> None:
        openui_lang = (
            'root = Root([summary, chart])\n'
            'summary = InsightCard("Sonnenstunden nach Stadtbezirk 2020", "Ranking nach Sonnenstunden in h.")\n'
            'chart = BarChart("Sonnenstunden nach Stadtbezirk 2020", "label", "value", '
            '[{"label":"Giesing","value":1998},{"label":"Laim","value":1545}])\n'
        )

        preview = openui_adapter_demo.render_component_preview(openui_lang)

        self.assertIn("Sonnenstunden nach Stadtbezirk 2020", preview)
        self.assertIn("Giesing", preview)
        self.assertIn("width:100.0%", preview)

    def test_clean_output_preserves_openui_lang_root(self) -> None:
        raw = "notes\nroot = Root([summary])\nsummary = InsightCard(\"A\", \"B\")"

        self.assertEqual(
            openui_adapter_demo.clean_component_output(raw),
            'root = Root([summary])\nsummary = InsightCard("A", "B")',
        )


def _sample(data_shape: str, component: str, query: str = "q") -> dict:
    return {
        "messages": [
            {"role": "system", "content": openui_adapter_demo.SYSTEM_PROMPT},
            {"role": "user", "content": f'{query}\n\nTool result:\n{{"value": 1}}'},
            {"role": "assistant", "content": f"<{component} value={{1}} />"},
        ],
        "metadata": {"domain": "weather", "data_shape": data_shape, "component": component},
    }


if __name__ == "__main__":
    main()
