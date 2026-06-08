from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
from fastapi import Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from openui_support import generate_openui_chat_response


APP_DIR = Path(__file__).parent
STATIC_DIR = APP_DIR / "static"
DEMO_CSV = APP_DIR / "examples" / "demo_cities.csv"


def _demo_dataframe() -> pd.DataFrame:
    if DEMO_CSV.exists():
        return pd.read_csv(DEMO_CSV)

    return pd.DataFrame(
        [
            {"city": "Berlin", "population": 3677000, "median_age": 42.6},
            {"city": "Hamburg", "population": 1906000, "median_age": 42.1},
            {"city": "Munich", "population": 1512000, "median_age": 41.5},
            {"city": "Cologne", "population": 1086000, "median_age": 42.3},
        ]
    )


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content)


def _last_user_prompt(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return _message_text(message).strip()
    return ""


def build_openui_response(prompt: str) -> str:
    return generate_openui_chat_response(_demo_dataframe(), prompt or "Summarize this dataset")


def _openai_sse_chunk(delta: dict[str, Any], finish_reason: str | None = None) -> str:
    payload = {"choices": [{"delta": delta, "finish_reason": finish_reason}]}
    return f"data: {json.dumps(payload)}\n\n"


async def _stream_openui_response(openui_lang: str):
    yield _openai_sse_chunk({"role": "assistant"})
    await asyncio.sleep(0)
    yield _openai_sse_chunk({"content": openui_lang})
    await asyncio.sleep(0)
    yield _openai_sse_chunk({}, "stop")
    yield "data: [DONE]\n\n"


app = gr.Server(title="smolnalysis")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.api(name="respond")
def respond(prompt: str) -> str:
    return build_openui_response(prompt)


@app.post("/api/chat")
async def chat(request: Request) -> StreamingResponse:
    body = await request.json()
    messages = body.get("messages") or []
    prompt = _last_user_prompt(messages)
    openui_lang = build_openui_response(prompt)
    return StreamingResponse(_stream_openui_response(openui_lang), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def homepage() -> str:
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>smolnalysis</title>
    <link rel="stylesheet" href="/static/openui-chat.css" />
  </head>
  <body>
    <div id="root"></div>
    <script src="/static/openui-chat.js"></script>
  </body>
</html>
"""


if __name__ == "__main__":
    app.launch()
