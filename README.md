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
tags:
  - track:backyard
  - sponsor:openbmb
  - achievement:offgrid
  - achievement:welltuned
  - achievement:offbrand
  - achievement:fieldnotes
---

# 📊 smolnalysis

**Ask a question about open data. Get UI generated on the fly. All powered by small expert models.**

`smolnalysis` is an interactive open data agent built for the Build Small Hackathon. It combines a fine-tuned MiniCPM-1B language model with OpenUI-Lang code generation to create dynamic, data-driven interfaces directly from natural language questions about CKAN datasets.

The app runs in Gradio Server Mode with a custom OpenUI chat frontend. Gradio provides the Python API server and Space-friendly runtime, while OpenUI drives the browser chat experience.

## Submission Links

- Live app: [smolnalysis on Hugging Face Spaces](https://huggingface.co/spaces/build-small-hackathon/smolnalysis)
- Source code: [Meteord/smolnalysis on GitHub](https://github.com/Meteord/smolnalysis)
- Demo video: [In github](https://github.com/Meteord/smolnalysis/blob/main/demo.mov)
- Field notes (German): [Build Small Hackathon Blog Post](https://ki.muenchen.de/blog/2026-06-15-build-small-hackathon)
- Models:
  - Base model: [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B)
  - Fine-tuned LoRA adapters:
    - [MiniCPM5-1B Adapter für CKAN Retrieval](https://huggingface.co/build-small-hackathon/smolnalysis-translation-minicpm5-lora)
    - [MiniCPM5-1B Adapter für OpenUI-Lang Generation](https://huggingface.co/build-small-hackathon/smolnalysis-generation-minicpm5-lora)
- Contributors: `illuminate25` and `Meteord`

## How It Works

smolnalysis combines two key ideas:

### 1. OpenUI-Lang for Token-Efficient UI Generation

Instead of generating full HTML or JSON UI specifications, the app uses [OpenUI-Lang](https://www.openui.com/docs/openui-lang/specification-v05), a lightweight declarative language for component-based interfaces.

```text
root = Stack([header, cards, footer])
header = CardHeader("Weather in Munich", "Current Forecast")
cards = Stack([tempCard, windCard, humCard], "row", "m", "stretch", "start", true)
tempCard = Card([CardHeader("Temperature", "Partly Cloudy"), TextContent("14 C", "large-heavy")], "card")
windCard = Card([CardHeader("Wind", "From Northwest"), TextContent("18 km/h", "large-heavy")], "card")
humCard = Card([CardHeader("Humidity", "Moderate"), TextContent("62%", "large-heavy")], "card")
footer = Card([CardHeader("5-Day Forecast", ""), forecastChart], "sunk")
```

### 2. CKAN Integration with Specialist Adapters

The app connects to CKAN portals such as [opendata.muenchen.de](https://opendata.muenchen.de/), discovers relevant datasets, and uses role-specific adapters to:

- Parse natural language questions about datasets
- Produce validated CKAN retrieval actions
- Generate OpenUI-Lang from retrieved context

## Architecture

The system uses a role-based routing pattern:

1. Query routing selects the most suitable role for the latest user message.
2. MiniCPM-1B is used as the shared base model.
3. Task-specific LoRA adapters specialize behavior without full model retraining.
4. OpenUI-Lang output is rendered inline in the chat frontend.

## Models Used

| Component | Model | Parameters | Purpose |
|-----------|-------|------------|---------|
| Base LLM | [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) | 1B | Core language understanding and generation |
| CKAN adapter | LoRA adapter | ~8M | CKAN retrieval actions |
| OpenUI adapter | LoRA adapter | ~8M | OpenUI-Lang generation |
| Router | Lightweight classifier | Small | Role selection |

## Local Setup

```bash
uv venv
uv sync
npm install
npm run build:openui-chat
uv run python app/app.py
```

Open [http://127.0.0.1:7860/](http://127.0.0.1:7860/).

## Runtime Configuration

Common settings:

```text
SMOLNALYSIS_MINICPM_TRANSFORMERS_MODEL_ID=openbmb/MiniCPM5-1B
SMOLNALYSIS_MINICPM_MAX_NEW_TOKENS=512
SMOLNALYSIS_MINICPM_TEMPERATURE=0.7
```

Adapter defaults are configured in `app/backend/adapter_registry.py`.

## Useful Commands

```bash
uv run python app/app.py
uv run python -m unittest tests.test_smolnalysis_model_wrapper
uv run python -m unittest tests.test_openui_adapter_demo
npm run build:openui-chat
npm run build:openui-renderer
```

## Planning

- Project vision and idea: [tasks/vision.md](tasks/vision.md)
- Task tracker: [tasks/task_list.md](tasks/task_list.md)

## Acknowledgements

Special thanks to:

- [Hugging Face](https://huggingface.co/) for Gradio, Spaces, and the hackathon
- [OpenBMB](https://www.openbmb.cn/) for MiniCPM and sponsorship
- [Modal](https://modal.com) for providing training credits
- The Build Small Hackathon organizers and community
