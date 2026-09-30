"""Deliver steers — user messages sent while the run streams — into the run.

Before each model call the run's queued steers become ordinary user messages at
the end of the conversation, so the model reads them at its next step and the
thread state records them in order. When the model has just answered without
calling a tool, the turn would end; a steer that arrived during that answer
instead sends the loop back to the model, so the agent replies to it within
the same turn. Steers left when the agent finishes belong to no later run and
are dropped; the client, which did not see them in the thread, sends them as a
new turn.

The messages are the user's own words, not middleware-made content, so they
carry no provenance stamp; ``steer_id`` lets the client match a delivered steer
to the one it sent.
"""

from __future__ import annotations

from typing import Any, override

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import hook_config
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from deerflow.agents.thread_state import ThreadState
from deerflow.runtime.steer import SteerInbox, get_steer_inbox

#: ``additional_kwargs`` key carrying the client's id for a delivered steer.
STEER_ID_KEY = "steer_id"


def _ids(runtime: Runtime) -> tuple[str, str] | None:
    context = getattr(runtime, "context", None)
    thread_id = context.get("thread_id") if context else None
    run_id = context.get("run_id") if context else None
    if not thread_id or not run_id:
        return None
    return str(thread_id), str(run_id)


class SteerMiddleware(AgentMiddleware[ThreadState]):
    """Turn the run's queued steers into user messages at each step."""

    state_schema = ThreadState

    def __init__(self, inbox: SteerInbox | None = None) -> None:
        super().__init__()
        self._inbox = inbox or get_steer_inbox()

    def _take(self, runtime: Runtime) -> list[HumanMessage]:
        ids = _ids(runtime)
        if ids is None:
            return []
        return [
            HumanMessage(content=steer.text, additional_kwargs={STEER_ID_KEY: steer.steer_id})
            for steer in self._inbox.drain(*ids)
        ]

    @override
    def before_model(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        """Add steers that arrived since the last step."""
        messages = self._take(runtime)
        return {"messages": messages} if messages else None

    @override
    async def abefore_model(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        """Async version of before_model."""
        return self.before_model(state, runtime)

    @hook_config(can_jump_to=["model"])
    @override
    def after_model(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        """Keep the turn going when a steer arrived during a final answer."""
        last = (state.get("messages") or [None])[-1]
        if not isinstance(last, AIMessage) or last.tool_calls:
            # Tool calls follow; the next before_model delivers any steer.
            return None
        messages = self._take(runtime)
        if not messages:
            return None
        return {"messages": messages, "jump_to": "model"}

    @hook_config(can_jump_to=["model"])
    @override
    async def aafter_model(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        """Async version of after_model."""
        return self.after_model(state, runtime)

    @override
    def after_agent(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        """Drop steers the finished run never reached; the client resends them."""
        ids = _ids(runtime)
        if ids is not None:
            self._inbox.discard(*ids)
        return None

    @override
    async def aafter_agent(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        """Async version of after_agent."""
        return self.after_agent(state, runtime)
