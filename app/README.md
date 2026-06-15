# smolnalysis App

This app uses `gr.Server` with a small custom HTML chat frontend.

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
- `hi` returns `hi, there how can i help you?`.
- Any other input runs the `ckan_retrieval` adapter with only the latest user message.
- The retrieval adapter output is passed with the user question to the `openui_translator` adapter.
- OpenUI-Lang responses are rendered server-side into the custom chat UI.

The old fullscreen React chat frontend and streaming route are still removed.

## Frontend Assets

Only the embedded OpenUI renderer remains:

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
