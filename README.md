# smolnalysis
Interactive open data agent for the build small hackathon

## Local setup

```bash
uv venv
uv sync
uv run gradio app/app.py
```

The MVP app includes:

- CSV upload and baseline analysis
- OpenUI-style slash commands
- OpenUI chat flow with deterministic OpenUI-Lang generation, validation, and Gradio rendering
- Raw OpenUI-Lang debug output for evaluation

## Planning

- Project vision and idea: [tasks/vision.md](tasks/vision.md)
- Task tracker: [tasks/task_list.md](tasks/task_list.md)
