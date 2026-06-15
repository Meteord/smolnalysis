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
  - sponsor:modal
  - achievement:offgrid
  - achievement:welltuned
  - achievement:offbrand
  - achievement:fieldnotes
---

# 📊 smolnalysis

**Ask a question about open data. Get UI generated on the fly. All powered by small expert models.**

`smolnalysis` is an interactive open data agent built for the Build Small Hackathon. It combines MiniCPM5-1B, specialist LoRA adapters, and a learned router to create dynamic, data-driven interfaces directly from natural language questions about CKAN-style datasets.

The app runs on `gr.Server`. Gradio provides the Python API server and Space-friendly runtime, while a custom lightweight HTML chat frontend calls `/api/chat`. OpenUI-Lang output is cleaned and rendered server-side before it is inserted into the chat.

## Submission Links

- Live app: [smolnalysis on Hugging Face Spaces](https://huggingface.co/spaces/build-small-hackathon/smolnalysis)
- Source code: [Meteord/smolnalysis on GitHub](https://github.com/Meteord/smolnalysis)
- Demo video: [In github](https://github.com/Meteord/smolnalysis/blob/main/demo.mov)
- Field notes (German): [Build Small Hackathon Blog Post](https://ki.muenchen.de/blog/2026-06-15-build-small-hackathon)
- Models (all below 2B and 4Bit quantized and we therefore want to participate in the Tiny Titan challenge)
  - Base model: [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B)
  - Fine-tuned LoRA adapters:
    - [MiniCPM5-1B Adapter für CKAN Retrieval](https://huggingface.co/build-small-hackathon/smolnalysis-translation-minicpm5-lora)
    - [MiniCPM5-1B Adapter für OpenUI-Lang Generation](https://huggingface.co/build-small-hackathon/smolnalysis-generation-minicpm5-lora)
- Contributors: `illuminate25` and `Meteord`
- [Social Media Post](https://www.linkedin.com/posts/michael-jaumann-a4736a263_think-big-war-gesternsebastian-berger-und-share-7472425559524950018-R1Qi/?utm_source=share&utm_medium=member_desktop&rcm=ACoAAECow1cBepLyGVVeRVQlppPim4o-EHvxmoM)
- Modal-Challenge: [Our Documentation, how we used modal (German)](https://ki.muenchen.de/blog/2026-06-15-build-small-hackathon#training)

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

### 3. Learned Adapter Routing

Incoming requests are routed by a small classifier trained on the same chat-template prompts used by the adapters. The router uses a frozen MiniCPM encoder and a lightweight MLP head, then selects one of:

- `general_agent`: base model, no adapter
- `ckan_retrieval`: initial data/retrieval prompt
- `openui_translator`: prompt that already contains a `Tool result`

## Architecture

The system uses a role-based routing pattern:

1. The router selects the most suitable role for the latest user message.
2. MiniCPM-1B is used as the shared base model.
3. Task-specific LoRA adapters specialize behavior without full model retraining.
4. CKAN-style requests run retrieval first, then pass `user question + Tool result` to the OpenUI adapter.
5. OpenUI-Lang output is rendered inline in the custom chat frontend.

## Models Used

| Component | Model | Parameters | Purpose |
|-----------|-------|------------|---------|
| Base LLM | [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) | 1B | Core language understanding and generation |
| CKAN adapter | LoRA adapter | ~11M | CKAN retrieval actions |
| OpenUI adapter | LoRA adapter | ~11M | OpenUI-Lang generation |
| Router | Frozen MiniCPM encoder + MLP head | <1M trainable head | Role selection |

## Local Setup

```bash
uv venv
uv sync
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
The router is enabled by default. If local router artifacts are missing, the runtime downloads them from `build-small-hackathon/smolnalysis-adapter-router`. Override with `SMOLNALYSIS_ROUTER_REPO_ID` only if you publish a different router repo.

## Useful Commands

```bash
uv run python app/app.py
uv run python -m unittest tests.test_smolnalysis_model_wrapper
uv run python -m unittest tests.test_openui_adapter_demo
npm run build:openui-renderer
HF_TOKEN=... python train/router/upload_router_to_hf.py --router-dir train/router/outputs/router-mlp --repo-id build-small-hackathon/smolnalysis-adapter-router
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
