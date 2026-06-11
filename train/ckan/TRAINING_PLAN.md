# CKAN Retrieval Fine-Tuning Plan

## Summary

Train a small specialist LoRA adapter for CKAN retrieval decisions using `openbmb/MiniCPM5-1B` with supervised fine-tuning. The model should receive a user request, connected CKAN endpoint, previous tool observations, and current retrieval state, then emit the next CKAN retrieval action as strict JSON.

This is a 2-3 day first pass for Modal with a target budget around `$250`. The goal is a usable specialist adapter, not a publishable RL system. The training loop borrows the lightweight part of the AgenticQwen-style data flywheel: benchmark first, generate synthetic data, filter it, train, evaluate failures, then generate a small hard-example batch.

References:

- MiniCPM5-1B model card: <https://huggingface.co/openbmb/MiniCPM5-1B>
- MiniCPM TRL LoRA skill: <https://github.com/OpenBMB/MiniCPM/blob/main/skills/minicpm5-finetune-trl/SKILL.md>
- Liquid AI Modal fine-tuning example: <https://docs.liquid.ai/examples/customize-models/home-assistant>
- Modal pricing: <https://modal.com/pricing>
- AgenticQwen paper: <https://arxiv.org/pdf/2604.21590>

Dataset generation scaffold:

- CKAN dataset tools: [ckan_dataset_tools.py](ckan_dataset_tools.py)
- CKAN scenario harvester: [harvest_ckan_scenarios.py](harvest_ckan_scenarios.py)
- Teacher generator: [generate_teacher_data.py](generate_teacher_data.py)
- Dataset generation guide: [DATASET_GENERATION.md](DATASET_GENERATION.md)

## Model And Adapter

- Base model: `openbmb/MiniCPM5-1B`
- Adapter name: `smolnalysis-ckan-retrieval-minicpm5-lora`
- Training method: LoRA SFT with TRL + PEFT
- Loss: assistant-only loss with the MiniCPM TRL chat-template patch
- Do not train this together with the OpenUI-Lang adapter. Keep retrieval and translation as separate LoRA adapters.

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

The adapter should learn when to:

- Search CKAN packages.
- Refine a weak query.
- Inspect package details.
- Reject unsuitable resources.
- Rerun retrieval when the request is ambiguous or comparative.
- Stop with selected dataset/resource candidates when there is enough evidence.

The model is not responsible for calling CKAN itself. The backend will later parse the model output and execute the selected action through real tools.

## Training Output Contract

Assistant messages must be strict JSON only. Do not emit markdown fences or prose outside JSON.

```json
{
  "thought": "short private-style reasoning summary, no chain-of-thought dump",
  "action": "package_search | package_show | select_resource | reject_result | finish",
  "args": {},
  "confidence": 0.0
}
```

Action arguments:

- `package_search`: `query`, `rows`, `start`, optional `filters`.
- `package_show`: `package_id`.
- `select_resource`: `package_id`, `resource_id`, `reason`.
- `reject_result`: `reason`, `next_query`.
- `finish`: `selected_candidates`, `rationale`.

Keep `thought` short. It should summarize the decision, not expose long chain-of-thought.

## Dataset Plan

Target size:

- Train: 800-1,500 examples.
- Eval: 100-200 examples.
- Smoke-test subset: 20-50 examples before full training.

Data should be synthetic teacher-generated, then deterministically filtered.

Use [ckan_dataset_tools.py](ckan_dataset_tools.py) as the first hard validation gate before any example enters the training set.

Prefer inventory-grounded scenarios from [harvest_ckan_scenarios.py](harvest_ckan_scenarios.py) before scaling teacher generation. The goal is broad CKAN behavior coverage across groups, organizations, tags, and resource formats, not memorization of concrete München dataset ids.

Generate examples around:

- Search by topic, city, district, time range, and resource type.
- Ambiguous user requests.
- Failed search requiring query rewrite.
- Multiple candidate datasets requiring ranking.
- Unsuitable resource formats requiring rejection.
- Multi-step trajectories where previous CKAN observations influence the next action.

Use `https://opendata.muenchen.de/`-style examples first. Add generic CKAN portal examples later if time remains.

Create the benchmark set before generating training examples. Blocklist benchmark prompts from the synthetic generator so eval is not contaminated.

## Synthetic Generation Prompt Shape

Each generated example should be a chat sample:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are the CKAN retrieval policy for smolnalysis. Emit strict JSON actions only."
    },
    {
      "role": "user",
      "content": "User request, connected endpoint, current state, and previous tool observations."
    },
    {
      "role": "assistant",
      "content": "{\"thought\":\"...\",\"action\":\"package_search\",\"args\":{\"query\":\"...\",\"rows\":5,\"start\":0},\"confidence\":0.74}"
    }
  ]
}
```

Prefer trajectory fragments over isolated one-shot answers. Include enough observation context that the correct next action is checkable.

## Validation And Filtering

Keep only examples that pass all checks:

- Assistant content parses as JSON.
- `action` is one of the allowed action names.
- `args` matches the selected action.
- `confidence` is numeric and between `0.0` and `1.0`.
- Query rewrites do not invent endpoint credentials, private URLs, or impossible CKAN API paths.
- `package_id` and `resource_id` choices are present in the synthetic observation context.
- `finish` is only used after enough candidate evidence exists.
- Example passes a simple simulated executor that applies the action to the provided state.

Reject examples with long hidden reasoning, markdown, invalid JSON, or hallucinated resources.

## Evaluation

Report these metrics:

- Exact JSON parse rate.
- Valid action rate.
- Correct next-action accuracy on held-out trajectories.
- Resource-selection accuracy.
- Rerun/refinement behavior on ambiguous prompts.
- Average output length.

Failure buckets:

- Invalid JSON.
- Wrong action.
- Hallucinated dataset/resource.
- Stops too early.
- Keeps searching after enough evidence.
- Unsafe endpoint or URL behavior.

Before looking at model-generated examples, run the full benchmark and save metrics plus raw predictions.

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

- Write benchmark prompts and expected actions.
- Generate initial synthetic trajectories.
- Implement validators and filter data.
- Run a tiny smoke fine-tune.

Day 2:

- Run the first full LoRA training job.
- Evaluate on the held-out benchmark.
- Inspect failure buckets.
- Generate a small hard-example batch from failures.

Day 3, if available:

- Continue training or retrain once with hard examples.
- Save final adapter, eval report, and example predictions.
- Document how the backend should call the adapter later.

## Acceptance Criteria

- Adapter emits parseable JSON on at least 95% of held-out prompts.
- Valid action rate is at least 90%.
- It can choose `package_search`, `package_show`, `select_resource`, `reject_result`, and `finish` in appropriate benchmark cases.
- It reruns or refines retrieval for ambiguous/comparative prompts.
- It does not invent credentials, private endpoints, or resources outside the supplied observation context.

## Assumptions

- Synthetic teacher data is acceptable for the first pass.
- Real CKAN API execution is handled by the backend, not the model.
- This adapter is for retrieval/tool policy only, not data analysis or OpenUI rendering.
- LFM remains a future comparison candidate, not part of this sprint.
