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

For a later Hugging Face Space, use this folder as the Space root or copy this app structure into the Space root. The Space needs:

- `app.py`
- `requirements.txt`
- `openui_support.py`
- files in `examples/`
- built assets in `static/`

## MVP Features

- Fullscreen OpenUI chat UI
- Gradio `Server` backend with custom FastAPI routes
- Public CKAN endpoint configuration and validation
- Server-side OpenAI-compatible LLM role configuration
- Dataset-aware mocked chat interaction using `examples/demo_cities.csv`
- Mocked deterministic OpenUI-Lang backend contract
- OpenUI's native `FullScreen` chat component
- OpenUI's chat-optimized `openuiChatLibrary`
- Streaming `/api/chat` route that adapts Python responses to the OpenUI chat stream
- CKAN connection routes at `/api/ckan/default` and `/api/ckan/connect`
- LLM status routes at `/api/llms/status` and `/api/llms/validate`
- Public Gradio API function at `/gradio_api/call/respond`

## OpenUI Chat Architecture

The app keeps OpenUI support modular:

- `app.py` owns the Gradio `Server`, serves the frontend, and exposes `/api/chat` plus the Gradio `respond` API.
- `ckan_support.py` validates public CKAN endpoints through the Action API v3.
- `llm_support.py` parses server-side LLM role settings with `pydantic-settings`.
- `openui_support.py` defines deterministic mock OpenUI-Lang responses for the demo dataset.
- `app/frontend/openui-chat.jsx` mounts OpenUI's `FullScreen` chat component.
- `app/frontend/openui-chat.css` contains the app-specific frontend styling.
- `app/static/openui-chat.js` and `app/static/openui-chat.css` are the bundled browser assets loaded by `/`.
- The current chat contract emits line-oriented OpenUI-Lang with a `root = Card([...])` entry point for `openuiChatLibrary`.
- Later, an LLM can replace the mock generator while keeping the Gradio server and OpenUI chat contract.

## LLM Backend Configuration

The backend exposes four configurable OpenAI-compatible LLM roles:

- `general_agent`: plans the overall workflow
- `ckan_tool`: works with CKAN search/tool-calling
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

The current chat path still uses mocked responses. It renders a visible workflow trace for the future flow: user request -> CKAN search -> data analysis -> OpenUI-Lang translation -> frontend render.

## CKAN Endpoint Connection

The current CKAN slice is intentionally small:

- Default endpoint: `https://opendata.muenchen.de/`
- Authentication: public/anonymous only
- Validation: `/api/3/action/site_read` plus `/api/3/action/package_search?rows=0`
- UI state: the last successful endpoint is stored in browser `localStorage`
- Deferred: agentic CKAN search, resource loading, and dataset analysis

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
- `Return invalid OpenUI for fallback testing` -> mocked warning/debug response

## Tests

```bash
uv run python -m unittest tests.test_ckan_support
uv run python -m unittest tests.test_llm_support
```
