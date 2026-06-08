# OpenUI-Lang Translation Fine-Tuning Plan

## Summary

Train a small specialist LoRA adapter for OpenUI-Lang translation using `openbmb/MiniCPM5-1B` with supervised fine-tuning. The model should receive a user request, final analysis result, CKAN metadata, and desired UI intent, then output valid OpenUI-Lang that renders in the existing OpenUI chat library.

This is a 2-3 day first pass for Modal with a target budget around `$250`. The priority is valid, renderable UI output over creative prose. The training loop is lightweight: benchmark first, generate synthetic data, filter it with deterministic validators, train, evaluate failures, then generate a small hard-example batch.

References:

- MiniCPM5-1B model card: <https://huggingface.co/openbmb/MiniCPM5-1B>
- MiniCPM TRL LoRA skill: <https://github.com/OpenBMB/MiniCPM/blob/main/skills/minicpm5-finetune-trl/SKILL.md>
- Liquid AI Modal fine-tuning example: <https://docs.liquid.ai/examples/customize-models/home-assistant>
- Modal pricing: <https://modal.com/pricing>
- AgenticQwen paper: <https://arxiv.org/pdf/2604.21590>

## Model And Adapter

- Base model: `openbmb/MiniCPM5-1B`
- Adapter name: `smolnalysis-openui-translator-minicpm5-lora`
- Training method: LoRA SFT with TRL + PEFT
- Loss: assistant-only loss with the MiniCPM TRL chat-template patch
- Do not train this together with the CKAN retrieval adapter. Keep retrieval and translation as separate LoRA adapters.

Recommended LoRA defaults:

```yaml
r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
num_train_epochs: 2
learning_rate: 2e-4
max_length: 2048
bf16: true
assistant_only_loss: true
```

## Objective

The adapter should convert structured workflow results into compact OpenUI-Lang.

Inputs should include:

- User request.
- CKAN endpoint and selected dataset/resource metadata.
- Final analysis result.
- Desired UI intent, such as summary, chart, data quality, comparison, or no-data state.

The adapter should output OpenUI-Lang only. It should not explain the UI, wrap output in markdown, or produce generic prose.

## Training Output Contract

Assistant output must be OpenUI-Lang only:

- No markdown fences.
- No surrounding commentary.
- Must include a `root = ...` entry point.
- Must only reference variables that are defined.
- Must use supported OpenUI chat-library components.

Preferred components:

- `Card`
- `CardHeader`
- `TextContent`
- `ListBlock`
- `ListItem`
- `Table`
- `Col`
- `BarChart`
- `Series`
- `Callout`
- `FollowUpBlock`
- `FollowUpItem`

Example output shape:

```text
root = Card([header, summary, table, callout, followups])
header = CardHeader("Dataset summary", "CKAN analysis result")
summary = TextContent("The selected resource contains district-level mobility data.", "default")
col1 = Col("Metric", ["Rows", "Columns", "Missing values"], "string")
col2 = Col("Value", ["4,280", "9", "2.1%"], "string")
table = Table([col1, col2])
callout = Callout("info", "Next step", "A grouped bar chart would be useful for district comparison.")
followups = FollowUpBlock([f1, f2])
f1 = FollowUpItem("Show a bar chart")
f2 = FollowUpItem("Check data quality")
```

## Dataset Plan

Target size:

- Train: 1,000-2,000 examples.
- Eval: 150-250 examples.
- Smoke-test subset: 20-50 examples before full training.

Data should be synthetic teacher-generated, then deterministically filtered with the existing OpenUI parser.

Generate examples from structured analysis JSON:

- Summary result.
- Chart-ready result.
- Data-quality result.
- CKAN candidate comparison.
- Failed/no-data result.
- Mixed insight and follow-up suggestions.

Include edge cases:

- Empty observations.
- Long dataset names.
- Missing values.
- Unavailable chart data.
- German and English user prompts.
- Very small tables.
- Multiple valid UI variants for the same analysis payload.

Keep a small golden eval set from current app outputs and hand-authored examples. Do not include golden eval prompts in synthetic training generation.

## Synthetic Generation Prompt Shape

Each generated example should be a chat sample:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You translate smolnalysis workflow results into OpenUI-Lang. Output OpenUI-Lang only."
    },
    {
      "role": "user",
      "content": "User request, CKAN metadata, analysis JSON, and UI intent."
    },
    {
      "role": "assistant",
      "content": "root = Card([header, ...])\nheader = CardHeader(...)"
    }
  ]
}
```

Generate multiple valid UI variants per analysis payload so the adapter does not overfit to one rigid template.

## Validation And Filtering

Keep only examples that pass all checks:

- `openui_support.parse_openui_lang` accepts the output.
- Output includes a `root = ...` entry point.
- Every referenced variable is defined.
- Component names are supported by the existing OpenUI chat library.
- Component arguments match the expected simple shapes used in the current app.
- Chart examples include numeric series values.
- Table examples have matching column lengths.
- Output includes the user request or a faithful summary of it.
- Output includes dataset/resource context when present.
- Output includes the main analysis observations.

Reject examples that:

- Include markdown fences.
- Include prose before or after OpenUI-Lang.
- Invent unsupported components.
- Ignore the analysis result.
- Produce an invalid chart/table from unavailable data.

Parser validation is the first gate. A later implementation can add browser render validation for a sample batch.

## Evaluation

Report these metrics:

- Parse success rate.
- Undefined-reference rate.
- Component validity rate.
- Semantic coverage score.
- Chart/table appropriateness.
- Average output length.

Semantic coverage checks:

- Includes user request.
- Includes dataset/resource context when present.
- Includes main analysis observations.
- Includes chart/table when appropriate.
- Includes useful follow-up items.

Failure buckets:

- Invalid syntax.
- Missing `root`.
- Undefined references.
- Wrong component arguments.
- Too much prose instead of UI.
- Ignores analysis data.
- Uses chart/table when data does not support it.

Run a visual smoke test on a representative sample through the existing frontend renderer after parser metrics pass.

## Modal Training Strategy

Budget plan:

- Smoke tests: `<$20`.
- First full training pass for both adapters: `<$80`.
- Reserve the remaining budget for failed runs, extra synthetic data, and one improvement pass.

GPU defaults:

- Smoke runs: cheapest viable GPU available on Modal.
- Full runs: `A100 40GB`, `A100 80GB`, or `H100`, depending on availability.
- Current planning estimate: Modal lists H100 around `$0.001097/sec`, roughly `$3.95/hour`; `$250` should cover about 60 H100-hours before storage, CPU, and region multipliers.

Guardrails:

- Set hard timeouts per job.
- Never run full training before a 20-50 example smoke test passes.
- Push datasets and adapters to Hugging Face or a persistent Modal volume.
- Log metrics and sample predictions.
- Stop after the first useful adapter unless eval clearly justifies another run.

## 2-3 Day Execution Schedule

Day 1:

- Write golden benchmark prompts and expected OpenUI-Lang outputs.
- Generate structured analysis payloads.
- Generate initial synthetic translations.
- Filter with `openui_support.parse_openui_lang`.
- Run a tiny smoke fine-tune.

Day 2:

- Run the first full LoRA training job.
- Evaluate on the held-out benchmark.
- Inspect failure buckets.
- Generate a small hard-example batch from failures.

Day 3, if available:

- Continue training or retrain once with hard examples.
- Run parser and visual smoke tests.
- Save final adapter, eval report, and example predictions.
- Document how the backend should call the adapter later.

## Acceptance Criteria

- Adapter emits parseable OpenUI-Lang on at least 95% of held-out prompts.
- Output includes `root = ...` on at least 98% of held-out prompts.
- Undefined-reference rate is below 5%.
- It selects chart/table/list layouts appropriately for the analysis payload.
- It does not emit markdown fences or explanatory prose around the OpenUI-Lang.
- A representative browser smoke sample renders without obvious blank messages.

## Assumptions

- Synthetic teacher data is acceptable for the first pass.
- The current app parser is the first validation authority.
- The adapter should optimize for valid renderable OpenUI-Lang over creative wording.
- This adapter is for UI translation only, not CKAN retrieval or data analysis.
- LFM remains a future comparison candidate, not part of this sprint.
