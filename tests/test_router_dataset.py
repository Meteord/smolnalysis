import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "train" / "router" / "data"
sys.path.insert(0, str(DATA_DIR))

from dataset import ID_TO_LABEL, ROUTER_LABELS, RouterDataCollator, RouterTrainingDataset, _DebugTokenizer  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
