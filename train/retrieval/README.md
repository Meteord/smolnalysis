# Tool Result Retrieval Adapter Data

This folder contains a shortcut PoC dataset for training a small adapter to generate the structured tool result expected by the OpenUI-Lang adapter.

The dataset is derived from `train/openui_lang/data/openui_sft_{train,eval,test}.jsonl`.
Each source user message has this shape:

```text
<user question>

Tool result:
<structured JSON>
```

The derived task uses:

- input: the user question only
- output: the structured JSON only

The assistant label intentionally does not include `Tool result:`.

Regenerate:

```bash
python train/retrieval/build_tool_result_dataset.py
```

Generated files:

- `data/tool_result_train.jsonl`
- `data/tool_result_eval.jsonl`
- `data/tool_result_test.jsonl`
- `data/manifest.json`

Train a MiniCPM5 LoRA adapter with the same base model and LoRA defaults as the OpenUI-Lang adapter:

```bash
python train/retrieval/train_minicpm_lora.py \
  --train-limit 1000 \
  --eval-limit 100 \
  --num-train-epochs 1 \
  --output-dir train/retrieval/outputs/tool-result-minicpm5-lora
```

For a quick tokenization check without loading the model:

```bash
python train/retrieval/train_minicpm_lora.py --dry-run --train-limit 2 --eval-limit 2
```

The default base model is `openbmb/MiniCPM5-1B`.

After training, sanity-check generated tool results:

```bash
python train/retrieval/sanity_check_adapter.py \
  --adapter-path train/retrieval/outputs/tool-result-minicpm5-lora \
  --data-path train/retrieval/data/tool_result_test.jsonl \
  --max-samples 5
```

The sanity check reports valid JSON rate, exact-match rate, whether outputs avoid the source marker, and whether all expected top-level keys are present.
