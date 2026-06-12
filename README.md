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

The Space uses the root [app.py](app.py) launcher, [requirements.txt](requirements.txt), and the metadata above so it appears as a Gradio Space in the organisation namespace at `https://huggingface.co/spaces/YOUR_ORG/smolnalysis`.

### llama.cpp Deployment Target

The intended production path should use the `build-small-hackathon/CodeFlow` llama.cpp pattern: the Gradio Space runs `llama-cpp-python` directly, downloads a GGUF with `huggingface_hub`, and serves a custom frontend through `gr.Server`.

For `smolnalysis`, prefer an in-Space llama.cpp runtime because the target MiniCPM model is small. The desired hosted hardware is Hugging Face ZeroGPU, not Modal. Because ZeroGPU is Gradio-only and primarily designed around `@spaces.GPU` GPU sections, the first deployment task is to verify whether `llama-cpp-python` with CUDA offload works correctly under ZeroGPU. If it does not, keep the same self-contained Space and run a quantized MiniCPM GGUF on CPU.

The deployed model stack should be MiniCPM-only, not Gemma. Use `openbmb/MiniCPM5-1B` as the shared base model unless later benchmarks select another MiniCPM checkpoint. Deployment artifacts should be GGUF:

- MiniCPM base GGUF, preferably quantized for the selected Space hardware
- `ckan_retrieval` LoRA GGUF
- `openui_translator` LoRA GGUF
- optional future `data_analysis` LoRA GGUF

The app should call a local llama.cpp adapter when running inside the Space. If per-request LoRA switching is not reliable enough for this workflow, use pre-merged role-specific GGUF models.

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

Supported runtime roles are `general_agent`, `ckan_retrieval`, `data_analysis`, and `openui_translator`. The frontend can still send `adapter: "auto"`; the backend routes to the best role from the latest user message.

ZeroGPU deployment notes:

- Keep `sdk: gradio`.
- Select ZeroGPU hardware in the Space settings.
- Add the `spaces` package only when GPU-decorated functions are introduced.
- Wrap generation in `@spaces.GPU(duration=...)` if llama.cpp GPU offload is compatible with ZeroGPU.
- Do not use Modal for the deployed path.

## Current MVP

The app includes:

- Gradio `Server` as the Python backend
- A custom React frontend served from `/`
- OpenUI's native fullscreen `FullScreen` chat component
- Public CKAN endpoint configuration with `https://opendata.muenchen.de/` as the default
- Server-side configuration for four OpenAI-compatible LLM roles
- A Gemma backend chat service exposed to the fullscreen frontend through `/api/chat`
- A ReAct-style LangGraph backend workflow with delayed, randomized stub CKAN, analysis, and OpenUI translation nodes for the Gradio `respond` API
- OpenUI's `openuiChatLibrary` for rendered assistant responses
- A demo city dataset used by the current mock analysis flow
- A public Gradio API endpoint at `/gradio_api/call/respond`

The current frontend does not use Gradio's built-in Blocks UI. It uses Gradio as the server/runtime and renders the full OpenUI chat application in the browser. CKAN support is connection-only for now, and LLM support is configuration-only. `/api/chat` now forwards browser chat messages to the backend Gemma service and streams the response in the OpenAI-compatible SSE shape expected by the frontend.

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

## Hugging Face tracing

The Gemma backend can emit OpenTelemetry spans for Hugging Face tokenizer/model loading, PEFT adapter loading, and generation. Tracing is off by default.

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

# Run LangGraph workflow tests
uv run python -m unittest tests.test_agent_workflow
```

## Planning

- Project vision and idea: [tasks/vision.md](tasks/vision.md)
- Task tracker: [tasks/task_list.md](tasks/task_list.md)
