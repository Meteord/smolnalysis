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

## Current MVP

The app includes:

- Gradio `Server` as the Python backend
- A custom React frontend served from `/`
- OpenUI's native fullscreen `FullScreen` chat component
- Public CKAN endpoint configuration with `https://opendata.muenchen.de/` as the default
- Server-side configuration for four OpenAI-compatible LLM roles
- A LangGraph backend workflow with delayed, randomized stub CKAN, analysis, and OpenUI translation nodes
- Mock backend responses that emit deterministic OpenUI-Lang
- OpenUI's `openuiChatLibrary` for rendered assistant responses
- A demo city dataset used by the current mock analysis flow
- A public Gradio API endpoint at `/gradio_api/call/respond`

The current frontend does not use Gradio's built-in Blocks UI. It uses Gradio as the server/runtime and renders the full OpenUI chat application in the browser. CKAN support is connection-only for now, and LLM support is configuration-only. `/api/chat` now runs a delayed LangGraph stub workflow: user request -> CKAN retrieval stub -> data analysis stub -> randomized OpenUI-Lang translation stub -> frontend render.

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
