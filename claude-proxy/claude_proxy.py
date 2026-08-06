"""
claude_proxy.py  (v3)
OpenAI-compatible /v1/chat/completions proxy backed by the Claude Agent SDK.

v3 fixes/adds:
  - max_turns raised (thinking + answer no longer exhausts the session).
  - Function-calling emulation: OpenAI `tools` schemas are injected into the
    prompt; when Claude wants a tool, it emits a TOOL_CALL JSON block which is
    parsed and returned as OpenAI tool_calls (finish_reason="tool_calls").
    role:"tool" result messages in the history are fed back as tool results.
  - Vision (image_url parts) and thinking-budget mapping retained from v2.

Auth: CLAUDE_CODE_OAUTH_TOKEN env var.
Run:  uvicorn claude_proxy:app --host 0.0.0.0 --port 8082
"""

import base64
import json
import re
import time
import uuid

import httpx
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TURNS = 6
EFFORT_BUDGETS = {"low": 4_000, "medium": 12_000, "high": 32_000}
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

TOOL_CALL_RE = re.compile(r"```tool_call\s*\n(.*?)\n```", re.DOTALL)

TOOL_PROTOCOL = """
# Tool calling protocol

You have access to the tools listed below (JSON Schema). You cannot execute
them yourself. To call a tool, end your reply with exactly one fenced block:

```tool_call
{"name": "<tool_name>", "arguments": { ... }}
```

Rules:
- At most ONE tool_call block per reply, always at the very end.
- "arguments" must match the tool's parameter schema.
- If no tool is needed, reply normally with no tool_call block.
- After a tool result arrives (shown as [Tool result ...]), continue the task.

## Available tools
"""


# ── content conversion ──────────────────────────────────────────────────────

async def _image_part_to_block(part: dict) -> dict | None:
    url = (part.get("image_url") or {}).get("url", "")
    if not url:
        return None
    if url.startswith("data:"):
        try:
            header, data = url.split(",", 1)
            media_type = header.split(":", 1)[1].split(";", 1)[0]
        except (ValueError, IndexError):
            return None
    else:
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
            media_type = resp.headers.get("content-type", "image/png").split(";")[0]
            data = base64.b64encode(resp.content).decode()
        except Exception:
            return None
    if media_type not in ALLOWED_IMAGE_TYPES:
        return None
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


def _tools_to_system_suffix(tools: list[dict]) -> str:
    lines = [TOOL_PROTOCOL]
    for t in tools:
        fn = t.get("function", {}) if t.get("type") == "function" else t
        lines.append(
            f"- {fn.get('name')}: {fn.get('description', '')}\n"
            f"  parameters: {json.dumps(fn.get('parameters', {}), separators=(',', ':'))}"
        )
    return "\n".join(lines)


async def messages_to_blocks(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Flatten OpenAI chat history (incl. assistant tool_calls and tool results)
    into (system_prompt, content blocks for one user turn)."""
    system_parts: list[str] = []
    blocks: list[dict] = []
    transcript: list[str] = []

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        if role == "system":
            if isinstance(content, list):
                content = "\n".join(
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            system_parts.append(content)
            continue

        if role == "tool":
            transcript.append(
                f"[Tool result for call {m.get('tool_call_id', '?')}]\n{content}"
            )
            continue

        if role == "assistant":
            parts = []
            if content:
                parts.append(str(content))
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                parts.append(
                    "```tool_call\n"
                    + json.dumps({"name": fn.get("name"), "arguments": json.loads(fn.get("arguments") or "{}")})
                    + "\n```"
                )
            if parts:
                transcript.append("[Previous assistant reply]\n" + "\n".join(parts))
            continue

        # user
        if isinstance(content, list):
            text_bits = []
            for p in content:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "text":
                    text_bits.append(p.get("text", ""))
                elif p.get("type") == "image_url":
                    block = await _image_part_to_block(p)
                    if block:
                        blocks.append(block)
            if text_bits:
                transcript.append("[User]\n" + "\n".join(text_bits))
        else:
            transcript.append(f"[User]\n{content}")

    if transcript:
        blocks.insert(0, {"type": "text", "text": "\n\n".join(transcript)})
    system_prompt = "\n\n".join(system_parts) or None
    return system_prompt, blocks


def extract_thinking_budget(body: dict) -> int | None:
    if isinstance(body.get("max_thinking_tokens"), int):
        return body["max_thinking_tokens"]
    thinking = body.get("thinking")
    if isinstance(thinking, dict) and isinstance(thinking.get("budget_tokens"), int):
        return thinking["budget_tokens"]
    effort = body.get("reasoning_effort")
    if isinstance(effort, str) and effort.lower() in EFFORT_BUDGETS:
        return EFFORT_BUDGETS[effort.lower()]
    return None


# ── model call ──────────────────────────────────────────────────────────────

BUILTIN_TOOLS = [
    "Bash", "Read", "Write", "Edit", "MultiEdit", "NotebookEdit",
    "Glob", "Grep", "WebSearch", "WebFetch", "Task", "TodoWrite", "KillShell",
]


async def run_claude(model, system_prompt, blocks, thinking_budget) -> str:
    opts = dict(
        model=model,
        system_prompt=system_prompt,
        max_turns=MAX_TURNS,
        allowed_tools=[],
        disallowed_tools=BUILTIN_TOOLS,
    )
    if thinking_budget:
        opts["max_thinking_tokens"] = thinking_budget
    options = ClaudeAgentOptions(**opts)

    async def prompt_stream():
        yield {"type": "user", "message": {"role": "user", "content": blocks}}

    chunks: list[str] = []
    result_text: str | None = None
    async for message in query(prompt=prompt_stream(), options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    chunks.append(block.text)
        elif isinstance(message, ResultMessage):
            result_text = getattr(message, "result", None)
    return "".join(chunks) or (result_text or "")


# ── response shaping ────────────────────────────────────────────────────────

def parse_tool_call(text: str) -> tuple[str, list[dict] | None]:
    """Split Claude's reply into (plain_text, openai_tool_calls|None)."""
    match = TOOL_CALL_RE.search(text)
    if not match:
        return text, None
    try:
        payload = json.loads(match.group(1))
        name = payload["name"]
        arguments = payload.get("arguments", {})
    except (json.JSONDecodeError, KeyError):
        return text, None
    plain = TOOL_CALL_RE.sub("", text).strip()
    tool_calls = [
        {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }
    ]
    return plain, tool_calls


def openai_response(model: str, text: str, tool_calls: list[dict] | None) -> dict:
    message: dict = {"role": "assistant", "content": text or None}
    finish = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish = "tool_calls"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def sse_chunks(model: str, text: str, tool_calls: list[dict] | None):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def chunk(delta, finish=None):
        return {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    if tool_calls:
        delta_calls = [
            {
                "index": i,
                "id": tc["id"],
                "type": "function",
                "function": tc["function"],
            }
            for i, tc in enumerate(tool_calls)
        ]
        yield f"data: {json.dumps(chunk({'role': 'assistant', 'content': text or None, 'tool_calls': delta_calls}))}\n\n"
        yield f"data: {json.dumps(chunk({}, 'tool_calls'))}\n\n"
    else:
        yield f"data: {json.dumps(chunk({'role': 'assistant', 'content': text}))}\n\n"
        yield f"data: {json.dumps(chunk({}, 'stop'))}\n\n"
    yield "data: [DONE]\n\n"


# ── endpoints ───────────────────────────────────────────────────────────────

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
    tools = body.get("tools") or []

    system_prompt, blocks = await messages_to_blocks(messages)
    if not blocks:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "no usable content in messages", "type": "invalid_request"}},
        )
    if tools:
        system_prompt = (system_prompt or "") + "\n\n" + _tools_to_system_suffix(tools)
    thinking_budget = extract_thinking_budget(body)

    try:
        raw = await run_claude(model, system_prompt, blocks, thinking_budget)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": f"claude-agent-sdk error: {exc}", "type": "proxy_error"}},
        )

    text, tool_calls = parse_tool_call(raw) if tools else (raw, None)

    if stream:
        return StreamingResponse(sse_chunks(model, text, tool_calls), media_type="text/event-stream")
    return JSONResponse(content=openai_response(model, text, tool_calls))


@app.get("/health")
async def health():
    return {"status": "ok"}
