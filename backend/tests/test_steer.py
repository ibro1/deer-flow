"""Steering: user messages sent while a run streams reach the running agent."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from _agent_e2e_helpers import FakeToolCallingModel
from _router_auth_helpers import make_authed_test_app
from fastapi.testclient import TestClient
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool

from app.gateway.routers import thread_runs
from deerflow.agents.middlewares.steer_middleware import STEER_ID_KEY, SteerMiddleware
from deerflow.agents.thread_state import ThreadState
from deerflow.runtime import RunManager, RunStatus
from deerflow.runtime.steer import MAX_STEERS_PER_RUN, SteerInbox, SteerMessage, get_steer_inbox

THREAD = "thread-steer"
RUN = "run-steer"


class RecordingModel(FakeToolCallingModel):
    """The fake model, recording each call's messages, with an optional hook per call."""

    calls: list[list[BaseMessage]] = []
    on_call: dict[int, Any] = {}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        self.calls.append(list(messages))
        hook = self.on_call.get(len(self.calls))
        if hook is not None:
            hook()
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _model(responses: list[AIMessage], on_call: dict[int, Any] | None = None) -> RecordingModel:
    model = RecordingModel(responses=responses)
    model.calls = []
    model.on_call = on_call or {}
    return model


def _texts(messages: list[BaseMessage]) -> list[str]:
    return [m.content for m in messages if isinstance(m, HumanMessage)]


class TestSteerInbox:
    def test_drains_one_runs_steers_in_order_and_keeps_runs_apart(self):
        inbox = SteerInbox()
        inbox.put(THREAD, RUN, SteerMessage("a", "first"))
        inbox.put(THREAD, RUN, SteerMessage("b", "second"))
        inbox.put(THREAD, "other-run", SteerMessage("c", "elsewhere"))
        assert [m.text for m in inbox.drain(THREAD, RUN)] == ["first", "second"]
        assert inbox.drain(THREAD, RUN) == []
        assert [m.text for m in inbox.drain(THREAD, "other-run")] == ["elsewhere"]

    def test_caps_steers_per_run_and_discards_a_finished_run(self):
        inbox = SteerInbox()
        for index in range(MAX_STEERS_PER_RUN):
            assert inbox.put(THREAD, RUN, SteerMessage(str(index), "x"))
        assert not inbox.put(THREAD, RUN, SteerMessage("over", "x"))
        inbox.discard(THREAD, RUN)
        assert inbox.drain(THREAD, RUN) == []


class TestSteerInTheAgentLoop:
    def test_a_steer_sent_during_a_tool_reaches_the_next_model_call(self):
        inbox = SteerInbox()

        @tool
        def slow_task() -> str:
            """A tool the user interrupts with a steer while it runs."""
            inbox.put(THREAD, RUN, SteerMessage("s1", "also check the logs"))
            return "task output"

        model = _model([
            AIMessage(content="", tool_calls=[{"name": "slow_task", "args": {}, "id": "call-1", "type": "tool_call"}]),
            AIMessage(content="done, and the logs are clean"),
        ])
        graph = create_agent(model=model, tools=[slow_task], middleware=[SteerMiddleware(inbox)], state_schema=ThreadState)
        state = graph.invoke({"messages": [HumanMessage("run the task")]}, context={"thread_id": THREAD, "run_id": RUN})

        assert _texts(model.calls[1]) == ["run the task", "also check the logs"]
        steered = [m for m in state["messages"] if isinstance(m, HumanMessage) and m.additional_kwargs.get(STEER_ID_KEY) == "s1"]
        assert len(steered) == 1
        assert state["messages"][-1].content == "done, and the logs are clean"

    def test_a_steer_sent_during_the_final_answer_keeps_the_turn_going(self):
        inbox = SteerInbox()
        model = _model(
            [AIMessage(content="here is the answer"), AIMessage(content="answering your follow-up too")],
            on_call={1: lambda: inbox.put(THREAD, RUN, SteerMessage("s2", "and one more thing"))},
        )
        graph = create_agent(model=model, tools=[], middleware=[SteerMiddleware(inbox)], state_schema=ThreadState)
        state = graph.invoke({"messages": [HumanMessage("question")]}, context={"thread_id": THREAD, "run_id": RUN})

        assert len(model.calls) == 2
        assert _texts(model.calls[1]) == ["question", "and one more thing"]
        assert [m.content for m in state["messages"]][-3:] == ["here is the answer", "and one more thing", "answering your follow-up too"]

    def test_a_run_without_steers_is_unchanged_and_other_runs_steers_stay_out(self):
        inbox = SteerInbox()
        inbox.put(THREAD, "an-earlier-run", SteerMessage("old", "stale"))
        model = _model([AIMessage(content="plain answer")])
        graph = create_agent(model=model, tools=[], middleware=[SteerMiddleware(inbox)], state_schema=ThreadState)
        state = graph.invoke({"messages": [HumanMessage("hi")]}, context={"thread_id": THREAD, "run_id": RUN})
        assert len(model.calls) == 1
        assert [m.content for m in state["messages"]] == ["hi", "plain answer"]


class TestSteerEndpoint:
    def _client(self, mgr: RunManager) -> TestClient:
        app = make_authed_test_app()
        app.include_router(thread_runs.router)
        app.state.run_manager = mgr
        return TestClient(app, raise_server_exceptions=False)

    def test_accepts_a_steer_for_the_live_run_and_queues_it(self, monkeypatch):
        mgr = RunManager()
        monkeypatch.setattr(mgr, "local_running_run", lambda thread_id: asyncio.sleep(0, SimpleNamespace(run_id=RUN)))
        response = self._client(mgr).post(f"/api/threads/{THREAD}/steer", json={"text": "look at X", "steer_id": "abc-1"})
        assert response.status_code == 202
        assert response.json() == {"accepted": True, "run_id": RUN}
        assert [(m.steer_id, m.text) for m in get_steer_inbox().drain(THREAD, RUN)] == [("abc-1", "look at X")]

    def test_refuses_when_no_run_is_streaming(self):
        response = self._client(RunManager()).post(f"/api/threads/{THREAD}/steer", json={"text": "hello", "steer_id": "abc-2"})
        assert response.status_code == 409

    def test_rejects_an_empty_or_malformed_steer(self):
        client = self._client(RunManager())
        assert client.post(f"/api/threads/{THREAD}/steer", json={"text": "   ", "steer_id": "abc"}).status_code == 422
        assert client.post(f"/api/threads/{THREAD}/steer", json={"text": "hi", "steer_id": "bad id!"}).status_code == 422

    def test_refuses_a_thread_the_caller_does_not_own(self):
        app = make_authed_test_app(owner_check_passes=False)
        app.include_router(thread_runs.router)
        app.state.run_manager = RunManager()
        response = TestClient(app, raise_server_exceptions=False).post(f"/api/threads/{THREAD}/steer", json={"text": "hi", "steer_id": "abc"})
        assert response.status_code in (403, 404)


class TestLocalRunningRun:
    def test_finds_only_a_run_executing_here(self):
        async def scenario():
            mgr = RunManager()
            record = await mgr.create(THREAD)
            assert await mgr.local_running_run(THREAD) is None  # pending
            await mgr.set_status(record.run_id, RunStatus.running)
            assert await mgr.local_running_run(THREAD) is None  # no task on this worker
            record.task = asyncio.create_task(asyncio.sleep(10))
            found = await mgr.local_running_run(THREAD)
            assert found is not None and found.run_id == record.run_id
            record.task.cancel()
            await asyncio.sleep(0)
            assert await mgr.local_running_run(THREAD) is None

        asyncio.run(scenario())
