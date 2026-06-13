# CKAN Retrieval Adapter Evaluation Results

## Summary

The current CKAN retrieval LoRA adapter trains MiniCPM5-1B for strict multi-turn CKAN tool calling.

- Base model: `openbmb/MiniCPM5-1B`
- Adapter: `smolnalysis-ckan-retrieval-minicpm5-lora`
- Training data: `generated/valid_examples_multitool_train_1600_repaired.jsonl`
- Golden eval data: `generated/valid_examples_multitool_eval_160.jsonl`
- Hand challenge eval data: `multitool_eval_golden.jsonl`
- Training method: LoRA SFT with TRL + PEFT, assistant-only loss, MiniCPM training chat template
- Output contract: one compact JSON action with `thought`, `action`, `args`, and numeric `confidence`

Status: ready for guarded backend integration as the `ckan_retrieval` policy.

## Full Compact Run

The successful run uses compact assistant JSON during dataset preparation. This avoids teaching the model long pretty-printed responses that are easy to truncate during generation.

```json
{
  "train_runtime": 1124,
  "train_samples_per_second": 2.847,
  "train_steps_per_second": 0.356,
  "train_loss": 0.293,
  "epoch": 2
}
```

Trainer eval metrics:

```json
{
  "eval_loss": 0.21680307388305664,
  "eval_runtime": 9.6415,
  "eval_samples_per_second": 16.595,
  "eval_steps_per_second": 16.595,
  "eval_entropy": 0.1920227211434394,
  "eval_mean_token_accuracy": 0.9298324260860682,
  "eval_num_tokens": 1253968.0,
  "epoch": 2.0
}
```

## Golden Eval

Golden eval uses 160 package-disjoint examples generated from the Munich CKAN inventory.

```json
{
  "total": 160,
  "json_parse_rate": 1.0,
  "valid_action_rate": 1.0,
  "exact_action_match_rate": 1.0,
  "exact_action_matches": 160,
  "issue_counts": {},
  "predicted_action_counts": {
    "tag_search": 23,
    "organization_list": 23,
    "select_resource": 24,
    "package_show": 23,
    "group_list": 23,
    "package_search": 32,
    "finish": 12
  }
}
```

## Hand Challenge Eval

The hand challenge set has one example per action, including `ask_clarification`.

```json
{
  "total": 8,
  "json_parse_rate": 1.0,
  "valid_action_rate": 1.0,
  "exact_action_match_rate": 1.0,
  "exact_action_matches": 8,
  "issue_counts": {},
  "predicted_action_counts": {
    "tag_search": 1,
    "group_list": 1,
    "organization_list": 1,
    "package_search": 1,
    "package_show": 1,
    "select_resource": 1,
    "finish": 1,
    "ask_clarification": 1
  }
}
```

## Notes

- The earlier TRL failure was caused by the MiniCPM tokenizer chat template not being training-compatible. The trainer now installs a training-only template with `{% generation %}` markers and restores the original tokenizer template before saving.
- The earlier weak outputs were mostly protocol shape problems: missing `confidence`, truncated JSON, and occasional prefixed-vs-bare CKAN resource ids.
- The evaluator now extracts the first balanced JSON object and validates resource ids against observed context.
- Runtime should still validate every action before execution and retry/repair only the JSON envelope, not the semantic tool choice.

## Recommendation

Integrate this adapter behind the existing CKAN agent validator next. Keep Python in charge of loop control, observed-id validation, retries, and CKAN tool execution. The next training pass should use real runtime traces from that guarded path, especially failed searches, ambiguous German requests, and incorrect resource selections.
