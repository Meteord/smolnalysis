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

`smolnalysis` is an interactive open data agent built for th Build Small Hackathon. It combines a fine-tuned MiniCPM-1B language model with OpenUI-Lang code generation to create dynamic, data-driven interfaces directly from natural language questions about CKAN datasets.

The app is a Gradio Server Mode app with a custom fullscreen OpenUI chat frontend. Gradio provides the Python API server, queue-compatible endpoints, and Hugging Face Space-friendly runtime, while OpenUI owns the browser chat experience.

## Submission Links

- Live Space: [https://huggingface.co/spaces/build-small-hackathon/smolnalysis](https://huggingface.co/spaces/build-small-hackathon/smolnalysis)
- GitHub repo: [https://github.com/Meteord/smolnalysis](https://github.com/Meteord/smolnalysis)
- Field notes (Very long in detail blog post in german): [Build Small Hackathon Blog Post](https://ki.muenchen.de/blog/2026-06-15-build-small-hackathon.html)
- Fine-tuned model: [MiniCPM5-Finetune für CKAN Retrieval](https://huggingface.co/build-small-hackathon/smolnalysis-ckan-retrieval-minicpm5-lora)

## How It Works

smolnalysis combines two key innovations:

### 1. OpenUI-Lang for Token-Efficient UI Generation

Instead of generating full HTML or JSON UI specifications, the app uses [OpenUI-Lang](https://www.openui.com/docs/openui-lang/specification-v05)—a lightweight declarative language for describing component-based interfaces. This approach is significantly more token-efficient than direct HTML rendering:

```
root = Stack([header, cards, footer])
header = CardHeader("Weather in Munich", "Current Forecast")
cards = Stack([tempCard, windCard, humCard], "row", "m", "stretch", "start", true)
tempCard = Card([CardHeader("🌤️ Temperature", "Partly Cloudy"), TextContent("14 °C", "large-heavy")], "card")
windCard = Card([CardHeader("💨 Wind", "From Northwest"), TextContent("18 km/h", "large-heavy")], "card")
humCard = Card([CardHeader("💧 Humidity", "Moderate"), TextContent("62 %", "large-heavy")], "card")
footer = Card([CardHeader("5-Day Forecast", ""), forecastChart], "sunk")
```

### 2. CKAN Integration with Fine-Tuned Expert Models

The app connects to open data portals based on CKAN (like [opendata.muenchen.de](https://opendata.muenchen.de/)), automatically discovering datasets and retrieving relevant information. A fine-tuned MiniCPM-1B model with LoRA adapters learns to:
- Parse natural language questions about datasets
- Call the CKAN API intelligently
- Generate OpenUI-Lang directly from the query results

## Architecture

The system uses a router pattern to efficiently manage multiple specialized tasks:

1. **Query Router (MLP)**: A small multilayer perceptron that detects whether the user's question requires:
   - CKAN dataset retrieval (uses CKAN-specific LoRA adapter)
   - OpenUI-Lang generation (uses OpenUI-specific LoRA adapter)

2. **Base Model**: MiniCPM-1B provides the foundational language understanding across tasks.

3. **LoRA Adapters**: Task-specific adapters added during inference—no full model retraining required.

4. **Frontend**: OpenUI-Lang components render dynamically in a custom Gradio Server frontend.

## Models Used

| Component | Model | Parameters | Purpose |
|-----------|-------|------------|---------|
| Base LLM | [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) | 1B | Natural language understanding and code generation |
| CKAN Adapter | smolnalysis-ckan-retrieval-minicpm5-lora (LoRA) | ~8M | Fine-tuned for CKAN query generation |
| OpenUI Adapter | smolnalysis-openui-minicpm5-lora (LoRA) | ~8M | Fine-tuned for OpenUI-Lang generation |
| Router | Custom MLP | ~50K | Selects which expert adapter to use |

## Key Features

- ✨ **Dynamic UI Generation**: Real-time component creation from natural language
- 🔍 **CKAN Integration**: Direct access to open data portals and dataset discovery
- ⚡ **Small Model Efficiency**: Runs on modest hardware with LoRA fine-tuning (vs. full model training)
- 🔄 **Expert Router**: MLP-based routing ensures optimal adapter selection
- 💬 **Custom Frontend**: Full-screen OpenUI chat experience with streaming responses


## 🙌 Acknowledgements

Special thanks to:
- [Hugging Face](https://huggingface.co/) for Gradio, Spaces, and the awesome event
- [OpenBMB](https://www.openbmb.cn/) for MiniCPM and hackathon sponsorship
- [Modal](https://modal.com) for providing training credits
