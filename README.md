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
Interactive open data agent for the build small hackathon

`smolnalysis` is currently a Gradio Server Mode app with a custom fullscreen OpenUI chat frontend. Gradio provides the Python API server, queue-compatible endpoints, and Hugging Face Space-friendly runtime, while OpenUI owns the browser chat experience.

## Local setup

```bash
uv venv
uv sync
npm install
npm run build:openui-chat
uv run python app/app.py
```

Open the app at [http://127.0.0.1:7860/](http://127.0.0.1:7860/).

## Hugging Face Space

This repository is ready to run as a Hugging Face Gradio Space from the repository root. Create or sync the Space under the target organisation, for example:

```bash
huggingface-cli repo create YOUR_ORG/smolnalysis --type space --space_sdk gradio
```

The Space uses the root [app.py](app.py) launcher, [requirements.txt](requirements.txt), and the metadata above so it appears as a Gradio Space in the organisation namespace at `https://huggingface.co/spaces/build-small-hackathon/smolnalysis`.

### GitHub Actions Deployment

The Space can be updated automatically from GitHub with [.github/workflows/sync-space.yml](.github/workflows/sync-space.yml). The workflow builds the OpenUI frontend bundle, creates a clean `_space/` snapshot containing only runtime files, and syncs that snapshot to `build-small-hackathon/smolnalysis` with `huggingface/hub-sync`.

Required GitHub repository secret:

```text
HF_TOKEN=<a Hugging Face token with write access to build-small-hackathon/smolnalysis>
```

The sync action mirrors files over the Hub API rather than pushing Git history. This avoids the previous large-file history problem as long as generated training data and model artifacts are not copied into `_space/`.

### PEFT Adapter Deployment

The first CKAN specialist is a PEFT LoRA adapter, so the fastest Space integration path is the transformers backend:

```text
SMOLNALYSIS_MINICPM_BACKEND=transformers
SMOLNALYSIS_MINICPM_TRANSFORMERS_MODEL_ID=openbmb/MiniCPM5-1B
SMOLNALYSIS_MINICPM_CKAN_RETRIEVAL_ADAPTER_REPO_ID=build-small-hackathon/smolnalysis-ckan-retrieval-minicpm5-lora
SMOLNALYSIS_MINICPM_CKAN_RETRIEVAL_TEMPERATURE=0
SMOLNALYSIS_MINICPM_MAX_NEW_TOKENS=384
```

Upload the trained adapter from a machine with `HF_TOKEN` set:

```bash
uv run python train/ckan/upload_adapter_to_hf.py \
  --repo-id build-small-hackathon/smolnalysis-ckan-retrieval-minicpm5-lora
```

The upload script only pushes the deployable top-level adapter files and skips checkpoints and optimizer state. Keep the CKAN agent validator enabled in production; the LoRA proposes tool actions, while Python validates observed package/resource ids and executes tools.

### llama.cpp Deployment Target

The intended production path should use the `build-small-hackathon/CodeFlow` llama.cpp pattern: the Gradio Space runs `llama-cpp-python` directly, downloads a GGUF with `huggingface_hub`, and serves a custom frontend through `gr.Server`.

For `smolnalysis`, prefer an in-Space llama.cpp runtime because the target MiniCPM model is small. The desired hosted hardware is Hugging Face ZeroGPU, not Modal. Because ZeroGPU is Gradio-only and primarily designed around `@spaces.GPU` GPU sections, the first deployment task is to verify whether `llama-cpp-python` with CUDA offload works correctly under ZeroGPU. If it does not, keep the same self-contained Space and run a quantized MiniCPM GGUF on CPU.

The deployed model stack should be MiniCPM-only, not Gemma. Use `openbmb/MiniCPM5-1B` as the shared base model unless later benchmarks select another MiniCPM checkpoint. Deployment artifacts should be GGUF:

- MiniCPM base GGUF, preferably quantized for the selected Space hardware
- `ckan_retrieval` LoRA GGUF
- `openui_translator` LoRA GGUF
- optional future `data_analysis` LoRA GGUF

The app calls a local llama.cpp adapter when running inside the Space. It can share one base GGUF across roles, attach optional role-specific LoRA GGUF files, or use pre-merged role-specific GGUF models.

Runtime configuration:

```text
MODEL_REPO_ID=your-org/minicpm5-1b-gguf
MODEL_FILENAME=minicpm5-1b.Q4_K_M.gguf
SMOLNALYSIS_MINICPM_N_CTX=4096
SMOLNALYSIS_MINICPM_N_GPU_LAYERS=0
SMOLNALYSIS_MINICPM_MAX_NEW_TOKENS=850
```

Optional per-role overrides use the same GGUF base with role-specific LoRAs or pre-merged model files:

```text
SMOLNALYSIS_MINICPM_GENERAL_AGENT_MODEL_PATH=/models/general.gguf
SMOLNALYSIS_MINICPM_CKAN_RETRIEVAL_LORA_PATH=/models/ckan-retrieval-lora.gguf
SMOLNALYSIS_MINICPM_DATA_ANALYSIS_LORA_PATH=/models/data-analysis-lora.gguf
SMOLNALYSIS_MINICPM_OPENUI_TRANSLATOR_LORA_PATH=/models/openui-translator-lora.gguf
```

LoRAs can also live in Hugging Face repos and be downloaded by the Space at runtime:

```text
SMOLNALYSIS_MINICPM_CKAN_RETRIEVAL_LORA_REPO_ID=your-org/smolnalysis-loras
SMOLNALYSIS_MINICPM_CKAN_RETRIEVAL_LORA_FILENAME=ckan-retrieval-lora.gguf
SMOLNALYSIS_MINICPM_DATA_ANALYSIS_LORA_REPO_ID=your-org/smolnalysis-loras
SMOLNALYSIS_MINICPM_DATA_ANALYSIS_LORA_FILENAME=data-analysis-lora.gguf
SMOLNALYSIS_MINICPM_OPENUI_TRANSLATOR_LORA_REPO_ID=your-org/smolnalysis-loras
SMOLNALYSIS_MINICPM_OPENUI_TRANSLATOR_LORA_FILENAME=openui-translator-lora.gguf
```

Supported runtime roles are `general_agent`, `ckan_retrieval`, `data_analysis`, and `openui_translator`. The frontend can still send `adapter: "auto"`; the backend routes to the best role from the latest user message.

ZeroGPU deployment notes:

- Keep `sdk: gradio`.
- Select ZeroGPU hardware in the Space settings.
- The `spaces` package is installed, and startup exposes a small GPU-decorated probe so ZeroGPU detects the app correctly.
- Generation is wrapped in `@spaces.GPU(duration=...)`; keep `SMOLNALYSIS_MINICPM_N_GPU_LAYERS=0` for CPU-only llama.cpp if GPU offload is not compatible.
- Do not use Modal for the deployed path.

## Current MVP

The app includes:

- Gradio `Server` as the Python backend
- A custom React frontend served from `/`
- OpenUI's native fullscreen `FullScreen` chat component
- Public CKAN endpoint configuration with `https://opendata.muenchen.de/` as the default
- Server-side configuration for four OpenAI-compatible LLM roles
- A simple Python CKAN agent loop exposed to the fullscreen frontend through `/api/chat`
- LoRA-ready role boundaries for CKAN retrieval and OpenUI-Lang generation, with Python-owned tool execution and validation
- OpenUI's `openuiChatLibrary` for rendered assistant responses
- A demo city dataset used by the current deterministic analysis flow
- A public Gradio API endpoint at `/gradio_api/call/respond`

The current frontend does not use Gradio's built-in Blocks UI. It uses Gradio as the server/runtime and renders the full OpenUI chat application in the browser. `/api/chat` now routes dataset search messages through a CKAN specialist loop, streams validated OpenUI-Lang progress in the OpenAI-compatible SSE shape expected by the frontend, and records the model/tool trace. The MiniCPM role backends remain available for probing and later LoRA-backed specialist integration.

## LLM role configuration

The backend reads OpenAI-compatible provider settings from environment variables. API keys stay server-side and are never returned by status endpoints.

Use [example.env](example.env) as a starting point:

```bash
cp example.env .env
```

```bash
SMOLNALYSIS_LLM_BASE_URL=https://api.openai.com
SMOLNALYSIS_LLM_API_KEY=...
SMOLNALYSIS_LLM_TIMEOUT_SECONDS=8
SMOLNALYSIS_LLM_GENERAL_AGENT_MODEL=gpt-4.1-mini
SMOLNALYSIS_LLM_CKAN_TOOL_MODEL=gpt-4.1-mini
SMOLNALYSIS_LLM_DATA_ANALYSIS_MODEL=gpt-4.1-mini
SMOLNALYSIS_LLM_OPENUI_TRANSLATOR_MODEL=gpt-4.1-mini
```

Optional per-role overrides exist for future provider mixing:

```bash
SMOLNALYSIS_LLM_CKAN_TOOL_BASE_URL=https://provider.example
SMOLNALYSIS_LLM_CKAN_TOOL_API_KEY=...
```

The legacy `CKAN_TOOL` environment variable names are kept as aliases for compatibility. Internally, the runtime normalizes that role to `ckan_retrieval`.

## Hugging Face tracing

The Hugging Face model backends can emit OpenTelemetry spans for tokenizer/model loading, PEFT adapter loading, and generation. Tracing is off by default.

```bash
SMOLNALYSIS_HF_TRACING_ENABLED=true
SMOLNALYSIS_HF_TRACING_CONSOLE=true
# or send spans to an OTLP HTTP collector:
SMOLNALYSIS_HF_TRACING_OTLP_ENDPOINT=http://localhost:4318/v1/traces
SMOLNALYSIS_HF_TRACING_SERVICE_NAME=smolnalysis
```

Generation spans include the base model, active adapter, sampling settings, message count, and input/output token counts. Prompt and response text are not added to spans.

## Useful commands

```bash
# Run the app
uv run python app/app.py

# Rebuild the fullscreen OpenUI chat bundle
npm run build:openui-chat

# Rebuild the earlier embedded renderer prototype bundle
npm run build:openui-renderer

# Run CKAN connector tests
uv run python -m unittest tests.test_ckan_support

# Run LLM settings tests
uv run python -m unittest tests.test_llm_support

# Run CKAN agent workflow tests
uv run python -m unittest tests.test_agent_workflow
```

## Planning

- Project vision and idea: [tasks/vision.md](tasks/vision.md)
- Task tracker: [tasks/task_list.md](tasks/task_list.md)
