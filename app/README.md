# smolnalysis App

This app is now a fresh Gradio Blocks frontend.

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

- `app.py` owns the Gradio chat UI.
- Backend chat goes through `SmolnalysisMoE`.
- `hi` returns `hi, there how can i help you?`.
- Any other input runs the `ckan_retrieval` adapter with only the latest user message.
- The retrieval adapter output is passed with the user question to the `openui_translator` adapter.
- OpenUI-Lang responses are rendered inside the Gradio chat using `app/static/openui-renderer.js`.

The old fullscreen React chat frontend and `/api/chat` streaming route have been removed.

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
