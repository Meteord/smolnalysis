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
- Mock backend responses that emit deterministic OpenUI-Lang
- OpenUI's `openuiChatLibrary` for rendered assistant responses
- A demo city dataset used by the current mock analysis flow
- A public Gradio API endpoint at `/gradio_api/call/respond`

The current frontend does not use Gradio's built-in Blocks UI. It uses Gradio as the server/runtime and renders the full OpenUI chat application in the browser. CKAN support is connection-only for now: users can configure and validate a public endpoint, but chat does not yet search or analyze CKAN datasets.

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
```

## Planning

- Project vision and idea: [tasks/vision.md](tasks/vision.md)
- Task tracker: [tasks/task_list.md](tasks/task_list.md)
