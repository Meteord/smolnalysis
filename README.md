# smolnalysis
Interactive open data agent for the build small hackathon

## Local setup

```bash
uv venv
uv sync
uv run gradio app/app.py
```

The MVP app includes:

- CSV upload
- Chat-style dataset interaction
- Mock backend responses that emit deterministic OpenUI-Lang
- OpenUI-Lang validation plus OpenUI's native React renderer hosted in a Gradio 6 custom `gr.HTML` component
- Assistant responses rendered as OpenUI components directly inside a single Gradio chat

## Planning

- Project vision and idea: [tasks/vision.md](tasks/vision.md)
- Task tracker: [tasks/task_list.md](tasks/task_list.md)
