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
- Basic OpenUI-style commands:
  - `/summary`
  - `/columns`
  - `/plot <column>`
  - `/plot histogram of <column>`
  - `/plot <column> against <other column>`
