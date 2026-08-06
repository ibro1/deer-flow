"""
claude_proxy.py
OpenAI-compatible /v1/chat/completions proxy backed by the Claude Agent SDK.

Auth: reuses the Claude Code credential store mounted at /root/.claude
(or set CLAUDE_CODE_OAUTH_TOKEN env var instead). No Anthropic API key needed.

Run:  uvicorn claude_proxy:app --host 0.0.0.0 --port 8082
"""

import json
import time
import uuid

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    query,
)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

DEFAULT_MODEL = "claude-sonnet-4-6"


def messages_to_prompt(messages: list[dict]) -> tuple[str | None, str]:
    """Flatten OpenAI-style chat history into (system_prompt, user_prompt)."""
    system_parts = []
    convo_parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        # content may be a list of parts (vision-style); keep text parts only
        if isinstance(content, list):
            content = "\n".join(
                p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
            )
        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            convo_parts.append(f"[Previous assistant reply]\n{content}")
        else:
            convo_parts.append(f"[User]\n{content}")
    system_prompt = "\n\n".join(system_parts) or None
    prompt = "\n\n".join(convo_parts)
    return system_prompt, prompt


async def run_claude(model: str, system_prompt: str | None, prompt: str) -> str:
    """Single-turn text completion through the Agent SDK (no tools)."""
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        max_turns=1,
        allowed_tools=[],  # DeerFlow supplies its own tools; Claude here is just the brain
    )
    chunks: list[str] = []
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    chunks.append(block.text)
    return "".join(chunks)


def openai_response(model: str, text: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def sse_chunks(model: str, text: str):
    """Fake streaming: send the full text as one delta chunk, then [DONE]."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    first = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
    }
    last = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(first)}\n\n"
    yield f"data: {json.dumps(last)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "owned_by": "anthropic"}
            for m in ("claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5")
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model") or DEFAULT_MODEL
    messages = body.get("messages", [])
    stream = bool(body.get("stream", False))

    system_prompt, prompt = messages_to_prompt(messages)

    try:
        text = await run_claude(model, system_prompt, prompt)
    except Exception as exc:  # surface SDK/auth errors to DeerFlow logs
        return JSONResponse(
            status_code=500,
            content={"error": {"message": f"claude-agent-sdk error: {exc}", "type": "proxy_error"}},
        )

    if stream:
        return StreamingResponse(sse_chunks(model, text), media_type="text/event-stream")
    return JSONResponse(content=openai_response(model, text))


@app.get("/health")
async def health():
    return {"status": "ok"}
