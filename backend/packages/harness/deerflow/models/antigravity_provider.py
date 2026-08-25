"""Custom Antigravity CLI (`agy`) provider — wraps Google's Antigravity CLI
headless print mode as a LangChain chat model.

Unlike Claude Code, Antigravity CLI does not document any way to reuse its
cached OAuth credentials against a raw HTTP API — headless mode only works by
shelling out to the `agy` binary itself on a machine where an interactive
`agy` login has already happened (`agy` caches credentials locally; there is
no token/API-key env var). So this provider is a subprocess wrapper around
`agy -p ... --output-format json`, not a direct-API client the way
``claude_provider.ClaudeChatModel`` is.

Important behavioural difference from other DeerFlow models: `agy -p` is a
full **agentic** Antigravity run, not a bare chat completion — there is no
documented way to get a raw completion without agy's own agent loop
(which may use its own tools). Keep it constrained:
  - don't set `add_dirs` unless you deliberately want agy to see a workspace
  - leave `dangerously_skip_permissions` off unless you understand the
    consequences (unattended headless runs otherwise stall/fail on any
    action needing approval, exactly as with headless Claude Code)

Each DeerFlow turn is stateless from agy's point of view: the full LangChain
message history is flattened into a single prompt per call. DeerFlow's own
checkpointer is the source of truth for conversation history — this mirrors
how ``ClaudeChatModel`` uses the Anthropic API statelessly rather than
reaching for Claude Code's own session/resume machinery.

Verified against a real `agy 1.1.17` install (2026-08-19):
  $ agy -p "say OK" --output-format json
  {"conversation_id": "...", "status": "SUCCESS", "response": "OK\\n",
   "duration_seconds": 3.74, "num_turns": 1,
   "usage": {"input_tokens": 15904, "output_tokens": 13,
             "thinking_tokens": 0, "cache_read_tokens": 0,
             "total_tokens": 15917}}

Config example:
    - name: antigravity-agy
      display_name: Antigravity (agy)
      use: deerflow.models.antigravity_provider:AntigravityChatModel
      model: gemini-3-pro           # see `agy models` on the host machine
      supports_thinking: false
      supports_vision: false
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import uuid
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

logger = logging.getLogger(__name__)

DEFAULT_SUBPROCESS_TIMEOUT_SECONDS = 300

# Linux caps a *single* argv element at MAX_ARG_STRLEN (32 pages = 128 KiB) and
# execve() fails with E2BIG / OSError errno 7. The prompt goes through `-p
# <prompt>` as one argument, so a long flattened history (the memory updater
# routinely produces one) crashes the call. Stay well under the cap to leave
# room for the rest of argv and the environment block.
DEFAULT_MAX_PROMPT_CHARS = 96_000

_ELISION_MARKER = "\n\n[... earlier conversation elided to fit the agy CLI argument limit ...]\n\n"


def _render_message(m: BaseMessage) -> str:
    if isinstance(m, SystemMessage):
        role = "System"
    elif isinstance(m, HumanMessage):
        role = "Human"
    elif isinstance(m, ToolMessage):
        role = "Tool"
    elif isinstance(m, AIMessage):
        role = "Assistant"
    else:
        role = m.type.capitalize() if getattr(m, "type", None) else "Message"
    content = m.content if isinstance(m.content, str) else json.dumps(m.content)
    return f"[{role}]\n{content}"


def _flatten_messages(
    messages: list[BaseMessage],
    max_chars: int = DEFAULT_MAX_PROMPT_CHARS,
) -> str:
    """Flatten a LangChain message history into a single `agy -p` prompt.

    `agy` headless mode takes one prompt string per invocation — there is no
    multi-message chat-completion endpoint to call into. This mirrors how
    ``ClaudeChatModel`` forwards the full message list statelessly on every
    call rather than relying on the underlying CLI's own session concept.

    Because the prompt is passed as a single argv element, it must stay under
    the kernel's per-argument limit. When the full history doesn't fit we keep
    the system messages (instructions) and as many of the *most recent* turns
    as will fit, dropping from the middle — the same shape a context window
    would trim, rather than truncating the tail and losing the actual question.
    """
    rendered = [_render_message(m) for m in messages]
    full = "\n\n".join(rendered)
    if len(full) <= max_chars:
        return full

    system_parts = [r for m, r in zip(messages, rendered) if isinstance(m, SystemMessage)]
    other_parts = [r for m, r in zip(messages, rendered) if not isinstance(m, SystemMessage)]

    head = "\n\n".join(system_parts)
    budget = max_chars - len(head) - len(_ELISION_MARKER)
    if budget <= 0:
        # System prompt alone exceeds the limit — keep its tail so the most
        # recent/specific instructions survive.
        logger.warning(
            "agy prompt: system messages alone exceed %d chars; truncating them.",
            max_chars,
        )
        return head[-max_chars:]

    kept: list[str] = []
    used = 0
    for part in reversed(other_parts):
        cost = len(part) + 2  # separator
        if used + cost > budget:
            break
        kept.append(part)
        used += cost
    kept.reverse()

    logger.warning(
        "agy prompt exceeded %d chars (%d); elided %d of %d non-system messages "
        "to stay under the CLI argument limit.",
        max_chars,
        len(full),
        len(other_parts) - len(kept),
        len(other_parts),
    )

    tail = "\n\n".join(kept)
    if head and tail:
        return head + _ELISION_MARKER + tail
    return (head or tail)[-max_chars:]


# ---------------------------------------------------------------------------
# Tool-calling emulation
#
# `agy` has no native function-calling API — its headless envelope carries only
# text. But `--json-schema` forces the reply into a shape we choose, and the
# result comes back pre-parsed in the envelope's `structured_output` field.
# Combined with `--mode plan` (which stops agy from executing its own tools and
# makes it *propose* instead), that's enough to emulate tool calls: we describe
# DeerFlow's tools in the prompt, constrain the answer to the envelope below,
# and translate `type: "tool_call"` back into LangChain tool_calls.
#
# Verified against agy 1.1.20:
#   plan mode + this schema -> structured_output:
#     {"type": "tool_call", "tool_name": "get_weather",
#      "tool_arguments": {"location": "Paris"}}
# ---------------------------------------------------------------------------

_TOOL_ENVELOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["text", "tool_call"]},
        "text": {"type": "string"},
        "tool_name": {"type": "string"},
        "tool_arguments": {"type": "object"},
    },
    "required": ["type"],
}

_TOOL_PROMPT_PREAMBLE = """\
You are the reasoning layer of another agent framework. You have NO tools of \
your own and must NOT execute, read, write, or run anything yourself.

The CALLER owns the tools listed below and will execute them for you.

Respond ONLY with JSON matching the required schema:
- To use a tool: set type="tool_call", tool_name to one of the tool names \
below, and tool_arguments to an object matching that tool's parameters.
- To answer directly: set type="text" and put your reply in the text field.

Available tools:
{tool_descriptions}
"""


def _describe_tools(tools: list[dict[str, Any]]) -> str:
    """Render OpenAI-format tool schemas into prompt text agy can follow."""
    lines: list[str] = []
    for tool in tools:
        fn = tool.get("function", tool)
        name = fn.get("name", "")
        if not name:
            continue
        description = (fn.get("description") or "").strip()
        params = fn.get("parameters") or {}
        lines.append(
            f"- {name}: {description}\n  parameters (JSON Schema): {json.dumps(params)}"
        )
    return "\n".join(lines) if lines else "- (none)"


def _extract_structured_fallback(response: str) -> dict[str, Any] | None:
    """Recover the envelope from `response` when `structured_output` is absent.

    agy normally supplies a parsed `structured_output`, but its `response`
    field has been observed carrying several concatenated JSON objects (the
    schema-shaped one plus a chattier variant with toolAction/toolSummary).
    Scan lines from the end and return the first that looks like our envelope,
    so a renamed/missing field in a future agy release degrades gracefully
    instead of breaking the turn outright.
    """
    for line in reversed([ln.strip() for ln in (response or "").splitlines() if ln.strip()]):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "type" in candidate:
            return candidate
    return None


def _parse_tool_envelope(structured: dict[str, Any]) -> AIMessage:
    """Translate agy's structured_output envelope into a LangChain AIMessage."""
    if structured.get("type") == "tool_call":
        name = structured.get("tool_name") or ""
        args = structured.get("tool_arguments")
        if not isinstance(args, dict):
            args = {}
        if name:
            return AIMessage(
                content=structured.get("text") or "",
                tool_calls=[
                    {
                        "name": name,
                        "args": args,
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "tool_call",
                    }
                ],
            )
        logger.warning("agy returned type=tool_call with no tool_name: %s", structured)
    return AIMessage(content=structured.get("text") or "")


class AntigravityChatModel(BaseChatModel):
    """LangChain chat model backed by the `agy` CLI's headless print mode."""

    model: str = ""
    agy_binary: str = "agy"
    effort: str | None = None  # low | medium | high
    agent: str | None = None
    mode: str | None = None  # accept-edits | plan
    dangerously_skip_permissions: bool = False
    add_dirs: list[str] = Field(default_factory=list)
    print_timeout: str = "5m"
    subprocess_timeout_seconds: int = DEFAULT_SUBPROCESS_TIMEOUT_SECONDS
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS

    # `agy` accepts no sampling params (temperature/max_tokens/etc.) over the
    # CLI, so silently ignore any such keys a shared config profile might
    # still carry, rather than erroring at construction time.
    model_config = {"arbitrary_types_allowed": True, "extra": "ignore"}

    @property
    def _llm_type(self) -> str:
        return "antigravity-agy"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any | BaseTool],
        **kwargs: Any,
    ) -> Runnable[Any, BaseMessage]:
        """Emulated tool calling via `--json-schema` + `--mode plan`.

        `agy` has no native function-calling API, so instead of a `tools`
        parameter we constrain the reply to a small envelope schema and
        describe the caller's tools in the prompt. Plan mode is what stops
        agy from running its *own* tools and makes it propose instead —
        without it, a tool-shaped request makes agy try to execute something
        and die on the permission prompt headless mode can't answer.

        Caveats vs a native implementation:
        - one tool call per turn (no parallel calls)
        - prompt-constrained JSON is less reliable than a real API contract
        - agy prepends its own large system context to every call, so each
          turn carries meaningful token overhead
        """
        formatted = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=formatted, **kwargs)

    def _build_command(
        self,
        prompt: str,
        json_schema: dict[str, Any] | None = None,
        force_plan_mode: bool = False,
    ) -> list[str]:
        binary = shutil.which(self.agy_binary) or self.agy_binary
        cmd = [binary, "-p", prompt, "--output-format", "json"]
        if self.model:
            cmd += ["--model", self.model]
        if self.effort:
            cmd += ["--effort", self.effort]
        if self.agent:
            cmd += ["--agent", self.agent]
        # Plan mode is mandatory for tool-calling turns: it's what keeps agy
        # proposing instead of executing its own tools (which headless mode
        # can't approve, producing status=CANCELED).
        mode = "plan" if force_plan_mode else self.mode
        if mode:
            cmd += ["--mode", mode]
        if json_schema is not None:
            cmd += ["--json-schema", json.dumps(json_schema)]
        if self.dangerously_skip_permissions:
            cmd += ["--dangerously-skip-permissions"]
        for d in self.add_dirs:
            cmd += ["--add-dir", d]
        if self.print_timeout:
            cmd += ["--print-timeout", self.print_timeout]
        return cmd

    def _run_agy(
        self,
        prompt: str,
        json_schema: dict[str, Any] | None = None,
        force_plan_mode: bool = False,
    ) -> dict[str, Any]:
        cmd = self._build_command(prompt, json_schema=json_schema, force_plan_mode=force_plan_mode)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.subprocess_timeout_seconds,
                # CRITICAL: without this agy inherits the gateway's stdin. When
                # its cached credentials are missing/expired it prints an OAuth
                # URL and blocks on "paste the authorization code here", hanging
                # the call until the timeout. With stdin closed it instead exits
                # non-zero and emits a clean JSON error envelope.
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"agy timed out after {self.subprocess_timeout_seconds}s"
            ) from exc
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"agy binary not found (looked for '{self.agy_binary}'). "
                "Install Antigravity CLI and complete an interactive `agy` "
                "login on this machine first — headless mode has no "
                "token/API-key env var, only cached local credentials."
            ) from exc
        except OSError as exc:
            # errno 7 (E2BIG) — the prompt still exceeded the kernel's argv
            # limit. _flatten_messages caps this, so reaching here means
            # max_prompt_chars is configured too high for this host.
            if exc.errno == 7:
                raise RuntimeError(
                    f"agy prompt too large for the CLI argument limit "
                    f"({len(prompt)} chars). Lower `max_prompt_chars` for this "
                    f"model in config.yaml (currently {self.max_prompt_chars})."
                ) from exc
            raise

        # Parse stdout FIRST: agy emits a structured JSON error envelope even on
        # non-zero exit (e.g. {"status": "ERROR", "error": "authentication
        # failed or timed out"}). Surfacing that beats dumping raw stderr, which
        # for auth failures is a wall of OAuth URL.
        data: dict[str, Any] | None = None
        if proc.stdout.strip():
            try:
                parsed = json.loads(proc.stdout)
                if isinstance(parsed, dict):
                    data = parsed
            except json.JSONDecodeError:
                data = None

        if data is not None and data.get("status") != "SUCCESS":
            agy_error = str(data.get("error") or "").strip()
            if "auth" in agy_error.lower():
                raise RuntimeError(
                    f"agy authentication failed ({agy_error}). Headless runs use "
                    "credentials cached by a prior interactive login as THIS "
                    "user on THIS host — run `agy` interactively in the same "
                    "container and complete sign-in, or set GEMINI_API_KEY. The "
                    "credential directory (~/.gemini) must persist across "
                    "container rebuilds or you will have to re-authenticate."
                )
            raise RuntimeError(f"agy run did not succeed: {agy_error or data}")

        if proc.returncode != 0:
            raise RuntimeError(
                f"agy exited {proc.returncode}: "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )

        if data is None:
            raise RuntimeError(
                f"agy returned non-JSON stdout: {proc.stdout[:500]!r}"
            )

        return data

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools: list[dict[str, Any]] = kwargs.get("tools") or []

        conversation = _flatten_messages(messages, max_chars=self.max_prompt_chars)

        if tools:
            preamble = _TOOL_PROMPT_PREAMBLE.format(
                tool_descriptions=_describe_tools(tools)
            )
            # Budget the conversation so preamble + history still fits argv.
            room = max(1000, self.max_prompt_chars - len(preamble) - 100)
            conversation = _flatten_messages(messages, max_chars=room)
            prompt = f"{preamble}\n\n{conversation}"
            data = self._run_agy(
                prompt,
                json_schema=_TOOL_ENVELOPE_SCHEMA,
                force_plan_mode=True,
            )
        else:
            prompt = conversation
            data = self._run_agy(prompt)

        usage = data.get("usage") or {}
        usage_metadata = {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get(
                "total_tokens",
                usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            ),
        }
        response_metadata = {
            "conversation_id": data.get("conversation_id"),
            "duration_seconds": data.get("duration_seconds"),
            "num_turns": data.get("num_turns"),
            "model_name": self.model,
        }

        if tools:
            # `structured_output` is agy's pre-parsed, schema-conforming object.
            # Prefer it over `response`, which carries duplicated blobs and
            # extra keys (toolAction/toolSummary) that aren't in our schema.
            structured = data.get("structured_output")
            if not isinstance(structured, dict):
                structured = _extract_structured_fallback(data.get("response", ""))
            if structured is None:
                raise RuntimeError(
                    "agy returned no structured_output for a tool-calling turn; "
                    f"raw response: {str(data.get('response'))[:400]!r}"
                )
            message = _parse_tool_envelope(structured)
            message.usage_metadata = usage_metadata
            message.response_metadata = response_metadata
        else:
            message = AIMessage(
                content=data.get("response", ""),
                usage_metadata=usage_metadata,
                response_metadata=response_metadata,
            )

        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # subprocess.run is blocking and an agy run can legitimately take
        # minutes; offload to a worker thread so the event loop isn't
        # blocked for the duration.
        return await asyncio.to_thread(
            self._generate, messages, stop, run_manager, **kwargs
        )
