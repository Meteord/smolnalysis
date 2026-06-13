# CKAN Retrieval Dataset Generation

This folder contains the first practical scaffold for generating and filtering CKAN retrieval training data.

The goal is not to trust synthetic data blindly. The goal is to generate many candidate tool traces, then keep only examples that pass deterministic CKAN-specific checks.

## Recommended Stack

- Use a teacher LLM to generate candidate chat examples.
- Use `ckan_dataset_tools.py` to validate strict JSON actions and simulated CKAN state.
- Optionally use Distilabel for scalable generation and judging pipelines.
- Optionally use Argilla for manual review of a small golden set.

Useful references:

- Distilabel: <https://distilabel.argilla.io/latest/>
- Argilla: <https://docs.argilla.io/latest/>
- DataDreamer: <https://github.com/datadreamer-dev/DataDreamer>

## Dataset Format

Use JSONL with one chat example per line:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are the CKAN retrieval policy for smolnalysis. Emit strict JSON actions only."
    },
    {
      "role": "user",
      "content": "Request, endpoint, current retrieval state, and previous CKAN observations."
    },
    {
      "role": "assistant",
      "content": "{\"thought\":\"Need an initial search.\",\"action\":\"package_search\",\"args\":{\"query\":\"population Munich\",\"rows\":5,\"start\":0},\"confidence\":0.82}"
    }
  ],
  "metadata": {
    "ckan_context": {
      "observed_packages": [],
      "observed_resources": [],
      "has_enough_evidence": false
    }
  }
}
```

Assistant content must be strict JSON only:

```json
{
  "thought": "short decision summary, not long chain-of-thought",
  "action": "tag_search | group_list | organization_list | package_search | package_show | select_resource | finish | ask_clarification",
  "args": {},
  "confidence": 0.0
}
```

## Generate Seed Examples

```bash
uv run python train/ckan/ckan_dataset_tools.py seed --output train/ckan/data/seed_examples.jsonl
```

The seed file is intentionally tiny. It exists to check schemas and trainer plumbing before generating a larger synthetic set.

## Harvest CKAN-Grounded Scenarios

Use the harvester to pull real public CKAN package/resource metadata and turn it into teacher-generation scenarios:

```bash
uv run python train/ckan/harvest_ckan_scenarios.py \
  harvest \
  --endpoint https://opendata.muenchen.de/ \
  --query population \
  --query mobility \
  --query environment \
  --rows-per-query 3 \
  --max-scenarios 50 \
  --output train/ckan/data/harvested_scenarios.jsonl
```

PowerShell:

```powershell
uv run python train/ckan/harvest_ckan_scenarios.py `
  harvest `
  --endpoint https://opendata.muenchen.de/ `
  --query population `
  --query mobility `
  --query environment `
  --rows-per-query 3 `
  --max-scenarios 50 `
  --output train/ckan/data/harvested_scenarios.jsonl
```

The output scenarios include real package ids, resource ids, resource formats, and summarized metadata. They are still only scenario inputs; the teacher model generates the final assistant action JSON.

To cover the whole dataset portal instead of hand-picked search terms, use inventory mode:

```powershell
uv run python train/ckan/harvest_ckan_scenarios.py inventory `
  --endpoint https://opendata.muenchen.de/ `
  --rows-per-page 100 `
  --inventory-output train/ckan/data/dataset_inventory.jsonl `
  --scenarios-output train/ckan/data/harvested_inventory_scenarios.jsonl
```

Inventory mode pages through `package_search` with `rows` and `start`, summarizes the portal datasets, and creates scenarios from real groups, organizations, tags, and resource formats. The scenario requests stay natural on purpose, for example `Do you have data about Sitzplätze?`. Group and organization constraints are stored as retrieval context in `filters` and `state`, not leaked into the user prompt.

Before teacher generation, create a deterministic balanced sample:

```powershell
uv run python train/ckan/ckan_dataset_tools.py sample `
  --input train/ckan/data/harvested_inventory_scenarios.jsonl `
  --output train/ckan/data/scenarios_200.jsonl `
  --limit 200 `
  --key target_action
```

For train/eval work, prefer a package-disjoint split before sampling teacher data:

```powershell
uv run python train/ckan/ckan_dataset_tools.py split `
  --input train/ckan/data/harvested_inventory_scenarios.jsonl `
  --train-output train/ckan/data/scenarios_train.jsonl `
  --eval-output train/ckan/data/scenarios_eval_golden.jsonl `
  --train-size 1000 `
  --eval-size 60 `
  --key target_action
```

The split reserves whole package ids for eval, so train and eval do not share the same dataset package.

You can discover available CKAN groups and organizations first:

```powershell
uv run python train/ckan/harvest_ckan_scenarios.py list-groups `
  --endpoint https://opendata.muenchen.de/ `
  --limit 20

uv run python train/ckan/harvest_ckan_scenarios.py list-organizations `
  --endpoint https://opendata.muenchen.de/ `
  --limit 20
```

Then harvest scenarios with CKAN-native filters:

```powershell
uv run python train/ckan/harvest_ckan_scenarios.py `
  harvest `
  --endpoint https://opendata.muenchen.de/ `
  --query mobility `
  --group transport `
  --organization referat-fuer-stadtplanung-und-bauordnung `
  --rows-per-query 3 `
  --max-scenarios 50 `
  --output train/ckan/data/harvested_filtered_scenarios.jsonl
```

The harvester sends these filters through `package_search` as `fq=groups:<name>` and `fq=organization:<name>`, then records the selected filters in each scenario.

## Connect A Teacher Model

The teacher generator reads OpenAI-compatible settings from environment variables:

```bash
SMOLNALYSIS_TEACHER_BASE_URL=https://api.openai.com/v1
SMOLNALYSIS_TEACHER_API_KEY=replace-with-your-key
SMOLNALYSIS_TEACHER_MODEL=gpt-4.1-mini
SMOLNALYSIS_TEACHER_TIMEOUT_SECONDS=30
```

For a local OpenAI-compatible endpoint:

```bash
SMOLNALYSIS_TEACHER_BASE_URL=http://localhost:11434/v1
SMOLNALYSIS_TEACHER_API_KEY=local
SMOLNALYSIS_TEACHER_MODEL=qwen2.5:14b
```

The generator also loads `.env` by default, so local env files work without manually exporting every variable.

## Generate Teacher Examples

Start with the checked-in scenario file:

```bash
uv run python train/ckan/generate_teacher_data.py \
  --scenarios train/ckan/data/scenarios.example.jsonl \
  --output train/ckan/data/generated/raw_examples.jsonl \
  --valid-output train/ckan/data/generated/valid_examples.jsonl \
  --report train/ckan/data/generated/validation_report.jsonl \
  --validate \
  --limit 5
```

The raw output keeps everything the teacher produced. The valid output contains only examples accepted by `ckan_dataset_tools.py`.

After harvesting, point the generator at the harvested scenarios:

```powershell
uv run python train/ckan/generate_teacher_data.py `
  --scenarios train/ckan/data/scenarios_200.jsonl `
  --output train/ckan/data/generated/raw_examples.jsonl `
  --valid-output train/ckan/data/generated/valid_examples.jsonl `
  --report train/ckan/data/generated/validation_report.jsonl `
  --validate `
  --limit 200
```

## Validate Examples

```bash
uv run python train/ckan/ckan_dataset_tools.py validate \
  --input train/ckan/data/generated/raw_examples.jsonl \
  --valid-output train/ckan/data/generated/valid_examples.jsonl \
  --report train/ckan/data/generated/validation_report.jsonl
```

The validator checks:

- JSON parses.
- `action` is allowed.
- `args` matches the action.
- `confidence` is between `0.0` and `1.0`.
- Searches do not contain credentials or URLs.
- `package_show` only references observed packages when context is present.
- `select_resource` only references observed resources when context is present.
- `tag_search`, `group_list`, and `organization_list` are used to discover catalog vocabulary before or between package searches.
- `finish` only happens when `has_enough_evidence` is true.

## Distilabel Pipeline Shape

Use Distilabel when scaling from seed examples to hundreds or thousands of examples.

Recommended pipeline stages:

1. Harvest CKAN-grounded scenarios with `harvest_ckan_scenarios.py inventory`.
2. Add a small number of hand-authored edge-case scenarios.
3. Ask a teacher model for one strict JSON assistant action, or call `generate_teacher_data.py`.
4. Run `validate_training_example`.
5. Ask a judge model only for examples that pass deterministic validation.
6. Export valid examples to JSONL or Hugging Face datasets.

Keep the deterministic validator as the hard gate. LLM judges are helpful for quality, but they should not override schema/tool safety failures.

## Teacher Prompt Template

```text
You generate one supervised fine-tuning example for a CKAN retrieval policy.

The assistant must emit strict JSON only:
{
  "thought": "short decision summary, no long chain-of-thought",
  "action": "tag_search | group_list | organization_list | package_search | package_show | select_resource | finish | ask_clarification",
  "args": {},
  "confidence": 0.0
}

Generate the next action for:
- user request: {request}
- endpoint: {endpoint}
- observed packages: {observed_packages}
- observed resources: {observed_resources}
- enough evidence: {has_enough_evidence}

Rules:
- package_show must use an observed package.
- select_resource must use an observed resource.
- finish only if enough evidence is true.
- package_search queries must not contain URLs, credentials, or API keys.
- output strict JSON only.
```

## Suggested First Batch

Start with 200 raw teacher examples and filter them aggressively:

- 35 `tag_search`.
- 25 `group_list`.
- 25 `organization_list`.
- 45 initial or refined `package_search`.
- 35 `package_show`.
- 25 `select_resource`.
- 5 `finish`.
- 5 `ask_clarification`.

If fewer than 70% pass validation, improve the teacher prompt before generating more.

## Generated Data Policy

Generated datasets and model outputs can become large. Keep small seed/golden files in git, but keep bulk generated data out of git unless intentionally curated.

Avoiding overfit:

- Use real dataset metadata as observations, but keep user requests generic.
- Keep CKAN implementation details, group ids, and organization ids out of the user request.
- Put group and organization constraints into structured scenario context so the retrieval policy learns them as internal state.
- Mix groups, organizations, and unfiltered searches.
- Hold out a golden eval split by dataset id so train and eval do not share the same package.
- Prefer action diversity and format diversity over repeating the same popular dataset many times.
- Include document-only, no-tabular-resource, empty-result, and broad-result cases that force another concrete tool call instead of a legacy abstract rejection action.
