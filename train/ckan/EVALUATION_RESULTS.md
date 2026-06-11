# CKAN Retrieval Adapter Evaluation Results

## Summary

The first CKAN retrieval LoRA adapter trained successfully on Modal.

- Base model: `openbmb/MiniCPM5-1B`
- Adapter: `smolnalysis-ckan-retrieval-minicpm5-lora`
- Training data: `valid_train_1000_repaired.jsonl`
- Golden eval data: `valid_eval_golden_60_repaired.jsonl`
- Human challenge eval data: `challenge_eval_30.jsonl`
- Training method: LoRA SFT with TRL + PEFT

Status: usable experimental adapter with backend guardrails.

## Training Run

Full Modal training completed successfully.

```json
{
  "train_runtime": 638,
  "train_samples_per_second": 3.135,
  "train_steps_per_second": 0.392,
  "train_loss": 0.5888,
  "epoch": 2
}
```

Trainer eval metrics:

```json
{
  "eval_loss": 0.3616762161254883,
  "eval_runtime": 3.5108,
  "eval_samples_per_second": 17.09,
  "eval_steps_per_second": 17.09,
  "eval_entropy": 0.34011556940774124,
  "eval_num_tokens": 617868.0,
  "eval_mean_token_accuracy": 0.9116872588793437,
  "epoch": 2.0
}
```

## Golden Eval

Golden eval uses package-disjoint teacher-generated examples.

```json
{
  "total": 60,
  "json_parse_rate": 1.0,
  "valid_action_rate": 1.0,
  "exact_action_match_rate": 1.0,
  "exact_action_matches": 60,
  "issue_counts": {},
  "predicted_action_counts": {
    "package_search": 14,
    "package_show": 14,
    "reject_result": 5,
    "select_resource": 18,
    "finish": 9
  }
}
```

Interpretation:

- The adapter learned the supervised output format.
- It emits valid CKAN action JSON on the teacher-shaped held-out set.
- No syntax, schema, or action-selection failures appeared on this eval.

## Human Challenge Eval

Challenge eval uses hand-authored German and English prompts with mismatches, document-only candidates, enough-evidence finish cases, and less teacher-shaped wording.

Result after adding the strict inference prompt:

```json
{
  "total": 30,
  "json_parse_rate": 0.9666666666666667,
  "valid_action_rate": 0.9,
  "exact_action_match_rate": 0.4,
  "exact_action_matches": 12,
  "issue_counts": {
    "invalid_json": 2,
    "invalid_args": 2,
    "invalid_confidence": 2,
    "missing_candidates": 2,
    "missing_rationale": 2
  },
  "predicted_action_counts": {
    "package_search": 18,
    "package_show": 5,
    "finish": 5,
    "select_resource": 1
  }
}
```

Interpretation:

- The strict inference prompt substantially improved behavior on challenge examples.
- Valid output rate is acceptable for an experimental adapter.
- Exact action match is weak on hand-authored cases.
- The model over-predicts `package_search`.
- The model under-predicts `reject_result`; no `reject_result` actions appeared in the challenge predictions.
- Remaining invalid outputs are mostly malformed `finish` shapes or minor JSON/schema issues.

## Known Weaknesses

- Over-selects `package_search` under ambiguity.
- Under-selects `reject_result` for mismatched or document-only candidates.
- Can produce malformed `finish` arguments on harder prompts.
- Golden eval is very strong, but still teacher-shaped.
- Challenge eval better reflects expected backend risk.

## Backend Guardrails

When integrating this adapter as the `ckan_tool` role:

- Always parse and validate model output before executing tools.
- Reject unknown action names.
- Reject actions with invalid `args`.
- Verify `package_id` and `resource_id` against observed context.
- Retry once with the strict schema prompt after invalid output.
- Fall back to deterministic retrieval policy if retry fails.
- Never allow model output to construct arbitrary URLs or credentials.
- Treat `finish` as valid only when enough evidence exists.

## Recommendation

Use this adapter as an experimental CKAN retrieval policy behind validator and retry guardrails.

Do not train more immediately. First integrate the adapter in a guarded backend path and collect failure cases. A later improvement pass should add more human-authored challenge examples, especially for:

- `reject_result`
- document-only packages
- filtered result mismatch
- finish-vs-search decisions
- German vague requests
