import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "train" / "openui_lang"
DATA_DIR = SCRIPT_DIR / "data"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(DATA_DIR))

from demo_adapters import (  # noqa: E402
    DemoResult,
    apply_question_override,
    clean_model_output,
    discover_adapters,
    prompt_messages_for_generation,
    parse_openui_assignments,
    render_openui_preview,
    select_sample,
    write_report,
)


class OpenUIDemoAdaptersTest(unittest.TestCase):
    def test_discovers_checkpoint_adapters_in_numeric_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["checkpoint-10", "checkpoint-2", "not-an-adapter"]:
                (root / name).mkdir()
            (root / "checkpoint-10" / "adapter_config.json").write_text("{}", encoding="utf-8")
            (root / "checkpoint-2" / "adapter_config.json").write_text("{}", encoding="utf-8")

            adapters = discover_adapters(root)

        self.assertEqual([adapter.name for adapter in adapters], ["checkpoint-2", "checkpoint-10"])

    def test_select_sample_reads_real_test_sample(self):
        sample_path, sample = select_sample(None, "test", 0)

        self.assertTrue(sample_path.name.endswith(".json"))
        self.assertIn("messages", sample)

    def test_question_override_updates_json_user_payload_without_mutating_sample(self):
        _, sample = select_sample(None, "test", 0)
        original_content = sample["messages"][1]["content"]

        updated = apply_question_override(sample, "Show average sunshine hours.")
        prompt_messages = prompt_messages_for_generation(updated)

        self.assertEqual(sample["messages"][1]["content"], original_content)
        self.assertIn("Show average sunshine hours.", prompt_messages[-1]["content"])
        self.assertNotIn("Show average sunshine hours.", original_content)

    def test_question_override_updates_dict_user_payload(self):
        sample = {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": {"task": "render_openui", "user_question": "old"}},
                {"role": "assistant", "content": "root = Card([])"},
            ]
        }

        updated = apply_question_override(sample, "new question")

        self.assertIn('"user_question": "new question"', updated["messages"][1]["content"])
        self.assertEqual(sample["messages"][1]["content"]["user_question"], "old")

    def test_clean_model_output_removes_fences_and_prefix(self):
        output = clean_model_output('notes first\n```openui-lang\nroot = Card([header])\nheader = CardHeader("A", "B")\n```')

        self.assertEqual(output, 'root = Card([header])\nheader = CardHeader("A", "B")')

    def test_parse_and_render_common_training_components(self):
        openui_lang = "\n".join(
            [
                "root = Card([header, intro, table, chart, followups])",
                'header = CardHeader("Dataset quality", "3 rows")',
                'intro = TextContent("Short summary")',
                "table = Table([col_a, col_b])",
                'col_a = Col("Column", ["A", "B"])',
                'col_b = Col("Value", [1, 2])',
                'chart = BarChart(["A", "B"], [series], "Column", "Value")',
                'series = Series("Count", [1, 2])',
                "followups = FollowUpBlock([followup])",
                'followup = FollowUpItem("Inspect missing values")',
            ]
        )

        components = parse_openui_assignments(openui_lang)
        html = render_openui_preview(openui_lang)

        self.assertEqual(components["root"]["type"], "Card")
        self.assertIn("Dataset quality", html)
        self.assertIn("<table>", html)
        self.assertIn("Inspect missing values", html)
        self.assertIn("bar-track", html)

    def test_write_report_includes_prompt_expected_output_and_rendered_result(self):
        sample = {
            "query_result": {"dataset_title": "Test dataset"},
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
                {"role": "assistant", "content": 'root = Card([header])\nheader = CardHeader("Expected", "UI")'},
            ],
        }
        result = DemoResult(
            name="checkpoint-1",
            adapter_path="/tmp/checkpoint-1",
            output='root = Card([header])\nheader = CardHeader("Actual", "UI")',
            rendered_html="<section>Actual</section>",
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "report.html"
            write_report(
                output_path=output_path,
                sample_path=Path("sample.json"),
                sample=sample,
                prompt_messages=sample["messages"][:2],
                results=[result],
            )
            html = output_path.read_text(encoding="utf-8")

        self.assertIn("OpenUI Adapter Demo", html)
        self.assertIn("checkpoint-1", html)
        self.assertIn("Expected", html)
        self.assertIn("Actual", html)


if __name__ == "__main__":
    unittest.main()
