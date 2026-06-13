# Multi-Tool Training Strategy

## Goal

Train small LoRA-ready specialists around explicit tool contracts instead of expecting a base 1B model to improvise the full workflow.

First priority:

1. `ckan_retrieval`: choose the next CKAN tool call until a suitable package/resource is found.
2. `data_analysis`: choose/request deterministic Python analysis tools after retrieval.
3. `openui_translator`: turn structured retrieval/analysis payloads into valid OpenUI-Lang.

Python remains the executor and validator. Models decide the next bounded action or generate OpenUI-Lang under validation.

## Existing München Data

The current CKAN inventory collection is already enough for a first retrieval dataset:

- `train/ckan/data/dataset_inventory.jsonl`: 336 München packages.
- `train/ckan/data/harvested_inventory_scenarios.jsonl`: 1,321 scenario fragments.
- Existing generated train/eval files are legacy-contract data and should be regenerated before the next LoRA run.
- Useful real topics already appear in the inventory, including Fahrrad/Raddauerzählstellen, Verkehr, Bevölkerung, Bildung, Verwaltung, Geodaten, Oktoberfest, Parken, Personal, and Digitaler Zwilling München.

Use this portal data as observations, not as hardcoded runtime synonyms. German language and misspellings belong in training/eval examples.

## CKAN Retrieval Adapter

Role name: `ckan_retrieval`.

Allowed actions:

```text
tag_search
group_list
organization_list
package_search
package_show
select_resource
finish
ask_clarification
```

Train the model to follow this typical loop:

1. Start with `tag_search` when user wording may not match package titles, especially German nouns and noisy spelling.
2. Use `group_list` or `organization_list` after empty, broad, or confusing results.
3. Run `package_search` with user intent plus discovered catalog vocabulary.
4. Run `package_show` for promising observed packages.
5. Use `select_resource` only for observed resources, with concrete `match_evidence`.
6. Use `finish` only after selection evidence exists.
7. Use `ask_clarification` for genuinely ambiguous requests.

Target first balanced eval:

- 1 golden example per current action in `train/ckan/data/generated/multitool_eval_golden.jsonl`.
- Add hard German/noisy prompts, for example `Fahrräder`, `Radverkehr`, `bycycles`, `was weißt du über fahrräder`, and typo variants.
- Keep package/resource IDs disjoint between train and eval.

Target first training mix:

- 15-20% catalog discovery actions.
- 25-35% package search/refinement.
- 15-20% package inspection.
- 15-20% resource selection.
- 5-10% finish/clarification.

Failure cases should teach retries with real tools:

- Empty `tag_search` -> `group_list` or refined `package_search`.
- Broad `package_search` -> `organization_list` or refined `package_search`.
- Document-only package -> refined `package_search`.
- Weak topic match -> refined `package_search`, not `select_resource`.
- Service/tool error observation -> pick a different query/tool or ask clarification.

## Dataset Analysis Adapter

Role name: `data_analysis`.

Do not train it to invent statistics. Train it to choose and parameterize deterministic Python tools.

Initial actions can be:

```text
analyze_resource
profile_columns
build_chart_data
summarize_quality
ask_clarification
```

Python should emit a typed `AnalysisResult`:

- row/column counts
- schema
- missingness
- numeric/text/date columns
- sample rows
- chart-ready series
- deterministic observations

The model may add short labels or choose which analysis view is useful, but pandas owns the numbers.

## OpenUI Translator Adapter

Role name: `openui_translator`.

This adapter should be trained separately from retrieval. Input should follow the structured payload shape from commit `70eb38962ab42f95c3389042bff65ba8a518fdec`, especially:

```json
{
  "task": "render_openui",
  "user_question": "...",
  "query_result": {
    "dataset_title": "...",
    "package_id": "...",
    "resource_id": "...",
    "resource_format": "CSV",
    "row_count": 0,
    "sampled_rows": 0,
    "column_count": 0,
    "columns": []
  },
  "component_hints": {
    "recommended_components": [],
    "numeric_columns": [],
    "category_columns": [],
    "time_columns": [],
    "primary_chart": {}
  },
  "quality_score": 0.0,
  "cleaning_notes": {}
}
```

Output must be OpenUI-Lang only. Validate with the existing parser path before accepting examples or model outputs. On invalid output, keep one repair attempt, then deterministic fallback.

Training should include:

- retrieval result cards
- dataset metadata views
- schema/quality summaries
- chart views from deterministic `AnalysisResult`
- German labels and questions
- negative examples where markdown/prose is rejected during validation

## Regeneration Path

1. Regenerate inventory scenarios from `https://opendata.muenchen.de/`.
2. Split by package ID before teacher generation.
3. Generate teacher examples with the current action contract.
4. Validate with `ckan_dataset_tools.py`.
5. Evaluate on `multitool_eval_golden.jsonl` plus a German hard set.
6. Train `ckan_retrieval` LoRA.
7. Add deterministic analysis payloads.
8. Train `openui_translator` from validated OpenUI-Lang examples.

## Acceptance Targets

- CKAN adapter: 98% JSON parse rate, 95% valid action rate, 85% correct next-action rate on held-out multi-tool scenarios.
- Resource selection: no unobserved package/resource IDs.
- German hard set: correctly handles catalog discovery for Fahrrad/Radverkehr-style prompts.
- OpenUI adapter: parser-valid output above 95%, deterministic fallback for the rest.
