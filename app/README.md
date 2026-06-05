# smolnalysis Gradio App

This folder contains the first MVP Gradio app for `smolnalysis`.

## Local Run

From the repository root:

```bash
uv venv
uv sync
uv run gradio app/app.py
```

## Hugging Face Space

For a later Hugging Face Space, use this folder as the Space root. The Space needs:

- `app.py`
- `requirements.txt`
- optional files in `examples/`

## MVP Features

- CSV upload and preview
- Baseline dataset summary
- Column profile with missing-value counts
- Automatic chart for the first numeric column
- Chat-style OpenUI flow with conversation history
- Deterministic OpenUI-Lang backend contract
- OpenUI-Lang validation and fallback rendering
- Basic OpenUI-style commands:
  - `/summary`
  - `/columns`
  - `/plot <column>`
  - `/plot histogram of <column>`
  - `/plot <column> against <other column>`

## OpenUI Chat Architecture

The Gradio app keeps OpenUI support modular:

- `app.py` owns the Gradio interface and session history.
- `openui_support.py` defines the message schema, backend orchestration, OpenUI-Lang generation, validation, parsing, and rendering adapter.
- The backend emits line-oriented OpenUI-Lang with a `root = Root([...])` entry point.
- The parser validates supported components before rendering.
- The frontend boots the OpenUI React `Renderer` in-browser through a Gradio 6 custom `gr.HTML` template component and a small custom library matching the backend contract.
- Matplotlib still renders the analytical chart output separately for reliability, while the OpenUI React runtime renders the structured component response.

## OpenUI Chat Examples

Use `app/examples/demo_cities.csv` and try:

- `Summarize this dataset` -> metric cards and a sample table
- `Show a bar chart of population by city` -> rendered bar chart
- `Show a histogram of median_age` -> rendered histogram
- `List the columns and missing values` -> schema table
- `Return invalid OpenUI for fallback testing` -> friendly fallback instead of a crash
