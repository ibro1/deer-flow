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

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        """Not supported — `agy -p` returns prose, not structured tool calls.

        DeerFlow's lead agent binds its tool set to the model and drives a
        tool-calling loop; `agy` headless mode exposes no tool-call channel
        (its `--output-format json` envelope carries only a text `response`),
        so this model cannot serve as the lead agent's model. It works fine
        for text-only roles — title generation, memory updates, suggestions,
        input polishing, or plain chat with tools disabled.

        Note that `agy` still runs its *own* internal agent loop and may use
        its own tools during a run; those are invisible to DeerFlow and are
        not reported back as LangChain tool calls.
        """
        raise NotImplementedError(
            "AntigravityChatModel does not support tool calling: `agy -p` returns "
            "plain text, not structured tool calls. Select it only for text-only "
            "roles (titles, memory, suggestions, tool-free chat), or use a "
            "tool-capable model for the lead agent."
        )

    def _build_command(self, prompt: str) -> list[str]:
        binary = shutil.which(self.agy_binary) or self.agy_binary
        cmd = [binary, "-p", prompt, "--output-format", "json"]
        if self.model:
            cmd += ["--model", self.model]
        if self.effort:
            cmd += ["--effort", self.effort]
        if self.agent:
            cmd += ["--agent", self.agent]
        if self.mode:
            cmd += ["--mode", self.mode]
        if self.dangerously_skip_permissions:
            cmd += ["--dangerously-skip-permissions"]
        for d in self.add_dirs:
            cmd += ["--add-dir", d]
        if self.print_timeout:
            cmd += ["--print-timeout", self.print_timeout]
        return cmd

    def _run_agy(self, prompt: str) -> dict[str, Any]:
        cmd = self._build_command(prompt)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.subprocess_timeout_seconds,
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

        if proc.returncode != 0:
            raise RuntimeError(
                f"agy exited {proc.returncode}: "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )

        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"agy returned non-JSON stdout: {proc.stdout[:500]!r}"
            ) from exc

        if data.get("status") != "SUCCESS":
            raise RuntimeError(f"agy run did not succeed: {data}")

        return data

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = _flatten_messages(messages, max_chars=self.max_prompt_chars)
        data = self._run_agy(prompt)
        text = data.get("response", "")

        usage = data.get("usage") or {}
        message = AIMessage(
            content=text,
            usage_metadata={
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get(
                    "total_tokens",
                    usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                ),
            },
            response_metadata={
                "conversation_id": data.get("conversation_id"),
                "duration_seconds": data.get("duration_seconds"),
                "num_turns": data.get("num_turns"),
                "model_name": self.model,
            },
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
