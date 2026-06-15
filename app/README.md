# smolnalysis App

This folder contains the current MVP app for `smolnalysis`.

The app now uses Gradio Server Mode instead of a traditional Gradio Blocks interface. Gradio runs the Python backend and API endpoints; the page at `/` is a bundled React frontend using OpenUI's fullscreen chat component and `openuiChatLibrary`.

## Local Run

From the repository root:

```bash
uv venv
uv sync
npm install
npm run build:openui-chat
uv run python app/app.py
```

Then open [http://127.0.0.1:7860/](http://127.0.0.1:7860/).

If you change `app/frontend/openui-chat.jsx` or `app/frontend/openui-chat.css`, rebuild the bundled OpenUI chat frontend:

```bash
npm run build:openui-chat
```

## Hugging Face Space

The repository root is prepared as the Space root. Hugging Face runs the root `app.py` launcher, which loads this app module while keeping the existing local command `uv run python app/app.py` working.

The Space needs the root `README.md` metadata, root `requirements.txt`, and these app assets:

- `app/app.py`
- `app/openui_support.py`
- files in `app/examples/`
- built assets in `app/static/`

Longer term, the Space should keep this Gradio Server app as the public UI and run llama.cpp through `llama-cpp-python` in the Space, following the `build-small-hackathon/CodeFlow` pattern. The target hosted hardware is Hugging Face ZeroGPU, with CPU GGUF inference as the fallback if llama.cpp CUDA offload is not compatible with ZeroGPU. The model backend should become MiniCPM-only with GGUF base/LoRA artifacts, replacing the current lazy Gemma backend. Modal is not part of the desired deployed path.

## MVP Features

- Fullscreen OpenUI chat UI
- Gradio `Server` backend with custom FastAPI routes
- Public CKAN endpoint configuration and validation
- Server-side OpenAI-compatible LLM role configuration
- Simple CKAN agent loop behind `/api/chat`, with streaming tool progress
- Dataset-aware deterministic chat interaction using `examples/demo_cities.csv`
- Template-backed OpenUI-Lang backend contract
- OpenUI's native `FullScreen` chat component
- OpenUI's chat-optimized `openuiChatLibrary`
- Streaming `/api/chat` route that adapts Python responses to the OpenUI chat stream
- CKAN connection routes at `/api/ckan/default` and `/api/ckan/connect`
- LLM status routes at `/api/llms/status` and `/api/llms/validate`
- Public Gradio API function at `/gradio_api/call/respond`

## OpenUI Chat Architecture

The app keeps OpenUI support modular:

- `app.py` owns the Gradio `Server`, serves the frontend, and exposes `/api/chat` plus the Gradio `respond` API.
- MiniCPM backend modules own the optional model runtimes used for status/probe endpoints and LoRA-backed specialists.
- `ckan_agent.py` defines the explicit CKAN retrieval loop, role action contract, tool execution, progress events, and retrieval OpenUI fallback.
- `agent_workflow.py` is a small compatibility wrapper for the Gradio `respond` API and demo dataset fallback.
- `ckan_support.py` validates public CKAN endpoints through the Action API v3.
- `llm_support.py` parses server-side LLM role settings with `pydantic-settings`.
- `openui_support.py` defines OpenUI-Lang parsing, validation, rendering support, and legacy demo helpers.
- `app/frontend/openui-chat.jsx` mounts OpenUI's `FullScreen` chat component.
- `app/frontend/openui-chat.css` contains the app-specific frontend styling.
- `app/static/openui-chat.js` and `app/static/openui-chat.css` are the bundled browser assets loaded by `/`.
- The frontend chat contract streams assistant content through OpenAI-compatible SSE chunks.
- The Gradio `respond` API exposes the same deterministic OpenUI-Lang workflow.
- `backend/minicpm_llama_cpp.py` owns the MiniCPM llama.cpp runtime path for `general_agent`, `ckan_retrieval`, `data_analysis`, and `openui_translator`.

## LLM Backend Configuration

The backend exposes four configurable OpenAI-compatible LLM roles:

- `general_agent`: plans the overall workflow
- `ckan_retrieval`: proposes CKAN search, package inspection, and resource selection actions
- `data_analysis`: analyzes loaded resource data
- `openui_translator`: converts analysis results to OpenUI-Lang

Required environment variables:

Use the repository-level `example.env` as a starting point:

```bash
cp example.env .env
```

```bash
SMOLNALYSIS_LLM_BASE_URL=https://api.openai.com
SMOLNALYSIS_LLM_API_KEY=...
SMOLNALYSIS_LLM_GENERAL_AGENT_MODEL=...
SMOLNALYSIS_LLM_CKAN_TOOL_MODEL=...
SMOLNALYSIS_LLM_DATA_ANALYSIS_MODEL=...
SMOLNALYSIS_LLM_OPENUI_TRANSLATOR_MODEL=...
```

Optional:

```bash
SMOLNALYSIS_LLM_TIMEOUT_SECONDS=8
SMOLNALYSIS_LLM_<ROLE>_BASE_URL=...
SMOLNALYSIS_LLM_<ROLE>_API_KEY=...
```

The current `/api/chat` path runs the CKAN retrieval loop for dataset search prompts and streams OpenUI-Lang progress as OpenAI-compatible SSE chunks. The `ckan_tool` environment variable names remain supported as compatibility aliases, but runtime logic normalizes that role to `ckan_retrieval`.

## Adapter Router

The MiniCPM backends can route `adapter="auto"` through the trained router MLP at `train/router/outputs/router-mlp` instead of the keyword heuristic:

```bash
SMOLNALYSIS_ROUTER_ENABLED=true
SMOLNALYSIS_ROUTER_OUTPUT_DIR=train/router/outputs/router-mlp
```

The router uses the MiniCPM tokenizer and expects `router_mlp.pt` plus `config.json` in the output directory. If artifacts are missing, confidence is below `SMOLNALYSIS_ROUTER_MIN_CONFIDENCE`, or inference fails, the app falls back to the existing heuristic. The current training labels are `general_agent`, `ckan_retrieval`, and `openui_translator`; retrain with `data_analysis` if you want the router to choose that role directly.

## Hugging Face Tracing

Set `SMOLNALYSIS_HF_TRACING_ENABLED=true` to emit OpenTelemetry spans around the Hugging Face runtime path:

- `huggingface.tokenizer.load`
- `huggingface.model.load`
- `huggingface.adapter.load`
- `huggingface.model.generate`

For local debugging, set `SMOLNALYSIS_HF_TRACING_CONSOLE=true`. To export to a collector, set `SMOLNALYSIS_HF_TRACING_OTLP_ENDPOINT`, for example `http://localhost:4318/v1/traces`. Spans include model, adapter, generation parameters, and token counts, but not prompt or response text.

## CKAN Agent Loop

Dataset/catalog prompts run a small Python-owned loop:

- `ckan_retrieval` receives the user request, endpoint, message history, and compact tool observations.
- The role may emit only strict JSON actions: `package_search`, `package_show`, `select_resource`, `finish`, or `ask_clarification`.
- Python validates every action, executes CKAN Action API calls, records progress events, and stops after a bounded number of tool calls.
- `/api/chat` streams those events as OpenUI-Lang list items, then renders a retrieval result table and selected resource callout.
- Later, `openui_translator` can turn the selected resource plus analysis payload into richer OpenUI-Lang; the fallback template stays available when generated OpenUI is invalid.

## CKAN Endpoint Connection

The current CKAN slice is intentionally small:

- Default endpoint: `https://opendata.muenchen.de/`
- Authentication: public/anonymous only
- Validation: `/api/3/action/site_read` plus `/api/3/action/package_search?rows=0`
- UI state: the last successful endpoint is stored in browser `localStorage`
- Included in the MVP workflow: CKAN package search/package inspection, CSV-like resource selection, progress streaming, and deterministic retrieval result UI

For deployed safety, private and link-local endpoint addresses are blocked unless `SMOLNALYSIS_ALLOW_LOCAL_CKAN=true` is set.

The earlier embedded renderer prototype is still represented by:

- `app/frontend/openui-renderer.jsx`
- `app/static/openui-renderer.js`
- `OpenUIRenderer` / `openui_component()` in `openui_support.py`

That path rendered OpenUI inside a Gradio `Chatbot`. The current main app renders the full OpenUI chat frontend instead.

## OpenUI Chat Examples

The current server uses `app/examples/demo_cities.csv` and supports prompts like:

- `Summarize this dataset` -> summary list and sample table
- `Show a bar chart of population by city` -> rendered bar chart
- `Show a histogram of median_age` -> bucketed bar chart
- `List the columns and missing values` -> schema table
- `Return invalid OpenUI for fallback testing` -> legacy warning/debug response

## Tests

```bash
uv run python -m unittest tests.test_ckan_support
uv run python -m unittest tests.test_llm_support
uv run python -m unittest tests.test_agent_workflow
```
