import json
import sys
import tempfile
import unittest
from pathlib import Path
from random import Random


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "train" / "router" / "data"
sys.path.insert(0, str(DATA_DIR))
sys.path.insert(0, str(REPO_ROOT / "train" / "router"))

from dataset import ID_TO_LABEL, ROUTER_LABELS, RouterDataCollator, RouterTrainingDataset, _DebugTokenizer  # noqa: E402
from generate_router_data import collect_openui_samples, collect_smalltalk_samples, split_tool_result_prompt  # noqa: E402


class RouterDatasetTest(unittest.TestCase):
    def test_dataset_returns_tokenized_classification_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample_path = Path(tmp) / "sample.json"
            sample_path.write_text(
                json.dumps(
                    {
                        "label": "ckan_retrieval",
                        "messages": [{"role": "user", "content": "Find a bicycle dataset."}],
                    }
                ),
                encoding="utf-8",
            )

            dataset = RouterTrainingDataset(sample_path, _DebugTokenizer())
            item = dataset[0]

        self.assertGreater(item["input_ids"].numel(), 0)
        self.assertEqual(item["attention_mask"].shape, item["input_ids"].shape)
        self.assertEqual(ID_TO_LABEL[int(item["labels"].item())], "ckan_retrieval")

    def test_collator_pads_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, text in enumerate(["hi", "Find a dataset about mobility"], start=1):
                (root / f"{index}.json").write_text(
                    json.dumps({"label": ROUTER_LABELS[index - 1], "messages": [{"role": "user", "content": text}]}),
                    encoding="utf-8",
                )

            dataset = RouterTrainingDataset(root, _DebugTokenizer())
            batch = RouterDataCollator(_DebugTokenizer())([dataset[0], dataset[1]])

        self.assertEqual(batch["input_ids"].shape[0], 2)
        self.assertEqual(batch["attention_mask"].shape, batch["input_ids"].shape)
        self.assertEqual(batch["labels"].shape[0], 2)

    def test_router_generator_has_enough_general_negatives(self):
        samples = collect_smalltalk_samples(500, Random(42))

        self.assertEqual(len(samples), 500)
        self.assertEqual({sample["label"] for sample in samples}, {"general_agent"})
        self.assertIn("router_stage", samples[0]["metadata"])

    def test_tool_result_prompt_splitter_recognizes_openui_stage(self):
        content = 'Show weather\n\nTool result:\n{"value": 12}'

        self.assertEqual(split_tool_result_prompt(content), ("Show weather", '{"value": 12}'))

    def test_openui_router_samples_use_adapter_input_only(self):
        sample = collect_openui_samples(1)[0]

        self.assertEqual(sample["label"], "openui_translator")
        self.assertEqual([message["role"] for message in sample["messages"]], ["user"])
        self.assertIn("Tool result:", sample["messages"][0]["content"])
        self.assertFalse(sample["messages"][0]["content"].startswith("Create the UI"))
        self.assertFalse(sample["messages"][0]["content"].startswith("Render this"))


if __name__ == "__main__":
    unittest.main()
