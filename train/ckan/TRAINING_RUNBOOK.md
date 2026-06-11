# CKAN MiniCPM5 LoRA Training Runbook

This runbook trains the CKAN retrieval adapter from the curated dataset:

- Train: `train/ckan/data/generated/valid_train_1000_repaired.jsonl`
- Eval: `train/ckan/data/generated/valid_eval_golden_60_repaired.jsonl`
- Stats: `train/ckan/data/DATASET_STATS.md`
- Results: `train/ckan/EVALUATION_RESULTS.md`

The adapter target is `smolnalysis-ckan-retrieval-minicpm5-lora` on top of `openbmb/MiniCPM5-1B`.

## Local Syntax Check

The app environment intentionally does not install training dependencies. This check only verifies that the training scripts parse:

```powershell
uv run python -m py_compile train/ckan/train_minicpm_lora.py train/ckan/modal_train_ckan.py
```

## Modal Setup

Install Modal locally if needed:

```powershell
uv pip install modal
modal setup
```

The Modal app uses:

- App name: `smolnalysis-ckan-minicpm5-lora`
- Volume: `smolnalysis-ckan-training`
- GPU: `A100`
- Output path in the volume:
  - smoke: `/outputs/smolnalysis-ckan-retrieval-minicpm5-lora-smoke`
  - full: `/outputs/smolnalysis-ckan-retrieval-minicpm5-lora`

## Smoke Run

Always run the smoke job first:

```powershell
uv run modal run train/ckan/modal_train_ckan.py --smoke
```

The smoke job uses 24 train examples and 12 eval examples for one epoch.

## Full Run

Run the full 1,000-example training job only after the smoke job completes:

```powershell
uv run modal run train/ckan/modal_train_ckan.py --no-smoke
```

Equivalent explicit train commands:

```powershell
uv run modal run train/ckan/modal_train_ckan.py --mode train --smoke
uv run modal run train/ckan/modal_train_ckan.py --mode train --no-smoke
```

Default training settings:

- LoRA `r=16`
- LoRA alpha `32`
- LoRA dropout `0.05`
- target modules: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- max length `2048`
- epochs `2`
- learning rate `2e-4`
- batch size `1`
- gradient accumulation `8`
- bf16 enabled
- gradient checkpointing enabled

## Outputs

The training script writes:

- adapter files
- tokenizer files
- checkpoints
- `eval_metrics.json`

Use the Modal volume browser or Modal CLI to download the adapter after training.

## Adapter Evaluation

After training, run generation-based evaluation on the package-disjoint golden eval set:

```powershell
uv run modal run train/ckan/modal_train_ckan.py --mode evaluate --no-smoke
```

For a smoke adapter:

```powershell
uv run modal run train/ckan/modal_train_ckan.py --mode evaluate --smoke
```

Evaluation writes to the Modal volume:

- `/outputs/eval/eval_predictions.jsonl`
- `/outputs/eval/eval_summary.json`

Run the human-authored challenge eval:

```powershell
uv run modal run train/ckan/modal_train_ckan.py --mode evaluate --no-smoke --challenge
```

Challenge evaluation writes:

- `/outputs/eval-challenge/eval_predictions.jsonl`
- `/outputs/eval-challenge/eval_summary.json`

If challenge eval shows schema drift, run the challenge-mix training variant:

```powershell
uv run modal run train/ckan/modal_train_ckan.py --mode train --no-smoke --challenge
```

This trains on `valid_train_1000_repaired.jsonl` plus four repeats of the 30-example human-authored challenge set, writing the adapter to:

```text
/outputs/smolnalysis-ckan-retrieval-minicpm5-lora-challenge
```

Then evaluate that adapter with:

```powershell
uv run modal run train/ckan/modal_train_ckan.py --mode evaluate --no-smoke --challenge --adapter-variant challenge
```

Challenge-adapter evaluation writes to `/outputs/eval-challenge-adapter-challenge`.

Metrics include:

- JSON parse rate
- valid action rate
- exact action match rate
- issue counts
- predicted action counts

## Notes

- This first training script is intentionally simple SFT.
- It does not merge the adapter into the base model.
- It does not push to Hugging Face yet.
- The generation-based eval checks syntax and action selection, but it is still benchmarked against teacher labels.
