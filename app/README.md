# smolnalysis Gradio App

This folder contains the first MVP Gradio app for `smolnalysis`.

## Local Run

From the repository root:

```bash
uv venv
uv sync
uv run gradio app/app.py
```

If you change `app/frontend/openui-renderer.jsx`, rebuild the bundled OpenUI renderer:

```bash
npm install
npm run build:openui-renderer
```

## Hugging Face Space

For a later Hugging Face Space, use this folder as the Space root. The Space needs:

- `app.py`
- `requirements.txt`
- optional files in `examples/`

## MVP Features

- CSV upload
- Dataset-aware chat interaction
- Single Gradio `Chatbot` with OpenUI-rendered assistant responses in the conversation history
- Mocked deterministic OpenUI-Lang backend contract
- OpenUI-Lang validation and fallback rendering
- OpenUI's native React `<Renderer />` hosted inside a Gradio 6 `gr.HTML` component

## OpenUI Chat Architecture

The Gradio app keeps OpenUI support modular:

- `app.py` owns the Gradio interface, dataset upload, and single chat history.
- `openui_support.py` defines the message schema, backend orchestration, OpenUI-Lang generation, validation, parsing, and rendering adapter.
- `app/frontend/openui-renderer.jsx` defines the OpenUI component library and mounts OpenUI's React `<Renderer />`.
- `app/static/openui-renderer.js` is the bundled browser asset loaded by Gradio.
- The backend emits line-oriented OpenUI-Lang with a `root = Root([...])` entry point.
- The parser validates supported components before rendering.
- Each assistant message is a Gradio HTML component that passes raw OpenUI-Lang into OpenUI's bundled React renderer.
- Later, an LLM can replace the mock generator without changing the renderer contract.

## OpenUI Chat Examples

Use `app/examples/demo_cities.csv` and try:

- `Summarize this dataset` -> metric cards and a sample table
- `Show a bar chart of population by city` -> rendered bar chart
- `Show a histogram of median_age` -> rendered histogram
- `List the columns and missing values` -> schema table
- `Return invalid OpenUI for fallback testing` -> friendly fallback instead of a crash
