# smolnalysis App

This app uses `gr.Server` with a custom HTML chat frontend.

## Local Run

From the repository root:

```bash
uv venv
uv sync
npm install
npm run build:openui-renderer
uv run python app/app.py
```

Open [http://127.0.0.1:7860/](http://127.0.0.1:7860/).

## Runtime Shape

- `app.py` owns the Gradio server, `/` frontend route, and `/api/chat` backend route.
- Backend chat goes through `SmolnalysisMoE`.
- The frontend is a dynamic server-rendered page with chat state, training-set example prompts, loading states, and adapter trace display.
- Auto mode uses the learned router, not the old hardcoded CKAN-first heuristic.
- Router outputs select `general_agent`, `ckan_retrieval`, or `openui_translator`.
- CKAN-style retrieval output is passed with the user question to the `openui_translator` adapter.
- OpenUI-Lang responses are rendered server-side into the custom chat UI.

The old Blocks chat UI, fullscreen React chat frontend, and streaming route are removed.

## Frontend Assets

The app no longer needs a bundled chat frontend. Only the embedded OpenUI renderer remains:

- `app/frontend/openui-renderer.jsx`
- `app/static/openui-renderer.js`

Rebuild it after editing the renderer source:

```bash
npm run build:openui-renderer
```

## Useful Tests

```bash
uv run python -m unittest tests.test_smolnalysis_model_wrapper
uv run python -m unittest tests.test_openui_adapter_demo
```
