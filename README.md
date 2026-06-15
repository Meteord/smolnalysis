---
title: smolnalysis
emoji: "📊"
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: 6.18.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
short_description: Interactive open data analysis app for CKAN datasets.
---

# smolnalysis

Interactive open data chat for the build small hackathon.

The current app is intentionally simple: a Gradio Blocks chat frontend backed by `SmolnalysisMoE`. OpenUI-Lang returned by the model is rendered inside the chat with the lightweight OpenUI renderer bundle.

## Local Setup

```bash
uv venv
uv sync
npm install
npm run build:openui-renderer
uv run python app/app.py
```

Open [http://127.0.0.1:7860/](http://127.0.0.1:7860/).

## Chat Flow

- `hi`: returns `hi, there how can i help you?`.
- Any other input runs `ckan_retrieval`.
- Retrieval generation receives only the latest user message.
- The retrieval adapter output is passed with the user question to `openui_translator`.
- If the final output is OpenUI-Lang, the Gradio chat renders it inline.

## Frontend

The old fullscreen React chat frontend was removed. The only remaining frontend bundle is the OpenUI renderer:

- Source: `app/frontend/openui-renderer.jsx`
- Built asset: `app/static/openui-renderer.js`

Rebuild after renderer edits:

```bash
npm run build:openui-renderer
```

## Runtime Configuration

The shared MiniCPM base model defaults to `openbmb/MiniCPM5-1B`.

Common settings:

```text
SMOLNALYSIS_MINICPM_TRANSFORMERS_MODEL_ID=openbmb/MiniCPM5-1B
SMOLNALYSIS_MINICPM_MAX_NEW_TOKENS=512
SMOLNALYSIS_MINICPM_TEMPERATURE=0.7
```

Adapter paths are local and fixed in `app/backend/adapter_registry.py`:

- `ckan_retrieval`: `train/retrieval/outputs/tool-results-minicpm5-lora/checkpoint-260`
- `openui_translator`: `train/openui_lang/outputs/openui-translate-mini-lora/checkpoint-160`

## Useful Commands

```bash
uv run python app/app.py
uv run python -m unittest tests.test_smolnalysis_model_wrapper
uv run python -m unittest tests.test_openui_adapter_demo
npm run build:openui-renderer
```

## Planning

- Project vision and idea: [tasks/vision.md](tasks/vision.md)
- Task tracker: [tasks/task_list.md](tasks/task_list.md)
