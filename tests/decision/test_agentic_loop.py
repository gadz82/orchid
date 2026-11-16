"""Tests for the decision-backed agentic loop bound.

The ``agentic_loop`` point is advisory: it can stop the loop early after a round
and the stop reason is recorded, but it can never extend the loop past
``max_tool_rounds``. Without a binding — or after any invalid/uncertain answer,
timeout, or provider failure — the loop keeps today's max-round behaviour.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from orchid_ai.agents.agentic_loop import AgenticLoop
from orchid_ai.agents.generic_agent import GenericAgent
from orchid_ai.agents.mcp_dispatcher import MCPCapabilities
from orchid_ai.config.schema import OrchidAgentConfig
from orchid_ai.config.schema_decision import OrchidDecisionPointConfig
from orchid_ai.core.decision import (
    OrchidDecisionAnswer,
    OrchidDecisionError,
    OrchidDecisionQuestion,
    OrchidDecisionResult,
    OrchidDecisionState,
)
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.decision.registry import get_decision_point
from orchid_ai.decision.stub import StubDecisionModel
from orchid_ai.graph.decision_bindings import DecisionBinding
from orchid_ai.rag.backends.null import NullVectorReader

SENTINEL = "SENTINEL-USER-TEXT-DO-NOT-LOG"
TOOL_NAME = "lookup"


# ── Helpers ─────────────────────────────────────────────────


def _auth() -> OrchidAuthContext:
    return OrchidAuthContext(access_token="test-token")


def _binding(provider: Any, *, threshold: float = 0.75) -> DecisionBinding:
    return DecisionBinding(
        spec=get_decision_point("agentic_loop"),
        config=OrchidDecisionPointConfig(model="scripted", confidence_threshold=threshold),
        model=provider,
    )


def _tool_call(tc_id: str = "call_1") -> dict[str, Any]:
    return {"name": TOOL_NAME, "args": {"query": tc_id}, "id": tc_id}


def _tool_call_message(tc_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[_tool_call(tc_id)])


class _ScriptedChatModel:
    """Offline chat model returning one scripted response per call."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def bind_tools(self, tool_defs: list[dict[str, Any]]) -> _ScriptedChatModel:
        return self

    async def ainvoke(self, messages: list, **kwargs: Any) -> AIMessage:
        index = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[index]


class _StubTool:
    def __init__(self, name: str = TOOL_NAME) -> None:
        self.name = name
        self.requires_approval = False
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, args: dict[str, Any]) -> str:
        self.calls.append(dict(args))
        return f"result_{self.name}"


class _ScriptedProvider:
    """Decision provider returning a fixed answer or raising a fixed error."""

    def __init__(
        self,
        answer: Any = None,
        *,
        error: BaseException | None = None,
        delay: float = 0.0,
    ) -> None:
        self._answer = answer
        self._error = error
        self._delay = delay
        self.questions: list[OrchidDecisionQuestion] = []
        self.states: list[OrchidDecisionState] = []
        self.timeouts: list[float | None] = []

    async def decide(
        self,
        state: OrchidDecisionState,
        questions: list[OrchidDecisionQuestion],
        *,
        timeout: float | None = None,
    ) -> OrchidDecisionResult:
        self.states.append(state)
        self.questions.extend(questions)
        self.timeouts.append(timeout)
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        answers: dict[str, OrchidDecisionAnswer] = {}
        if isinstance(self._answer, OrchidDecisionAnswer):
            answers[self._answer.name] = self._answer
        elif isinstance(self._answer, dict):
            for name, spec in self._answer.items():
                answers[name] = OrchidDecisionAnswer(name=name, **spec)
        return OrchidDecisionResult(answers=answers, provider="scripted", model="scripted")


def _loop(
    model: _ScriptedChatModel,
    *,
    binding: Any | None = None,
    max_tool_rounds: int = 3,
    tool: _StubTool | None = None,
    decision_timeout: float | None = 2.0,
) -> tuple[AgenticLoop, _StubTool]:
    stub_tool = tool or _StubTool()
    loop = AgenticLoop(
        agent_name="a",
        chat_model=model,
        tool_map={stub_tool.name: stub_tool},
        all_tool_defs=[
            {
                "type": "function",
                "function": {
                    "name": stub_tool.name,
                    "description": "lookup tool",
                    "parameters": {"type": "object"},
                },
            }
        ],
        agentic_loop_binding=binding,
        max_tool_rounds=max_tool_rounds,
        decision_timeout=decision_timeout,
    )
    return loop, stub_tool


def _messages(query: str = "q") -> list[dict[str, Any]]:
    return [{"role": "user", "content": query}]


def _decision_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == "orchid.decision"]


# ── Accepted continue / stop ────────────────────────────────


async def test_continue_keeps_the_loop_running(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ScriptedProvider({"continue": {"kind": "boolean", "value": True, "confidence": 0.95}})
    model = _ScriptedChatModel([_tool_call_message(), _tool_call_message("call_2"), AIMessage(content="done")])
    loop, tool = _loop(model, binding=_binding(provider), max_tool_rounds=3)

    with caplog.at_level(logging.INFO):
        final_text, results = await loop.run(_messages())

    assert final_text == "done"
    assert results == {TOOL_NAME: "result_lookup"}
    assert model.calls == 3
    assert len(tool.calls) == 2
    assert len(provider.questions) == 2
    assert loop.stop_reason == "final_answer"
    lines = _decision_lines(caplog)
    assert len(lines) == 2
    assert all("outcome=accepted" in line for line in lines)


async def test_stop_ends_the_loop_and_records_the_reason(caplog: pytest.LogCaptureFixture) -> None:
    provider = StubDecisionModel(answers={"continue": {"value": False, "confidence": 0.95}})
    model = _ScriptedChatModel([_tool_call_message(), AIMessage(content="never reached")])
    loop, tool = _loop(model, binding=_binding(provider), max_tool_rounds=3)

    with caplog.at_level(logging.INFO):
        final_text, results = await loop.run(_messages())

    assert final_text is None
    assert results == {TOOL_NAME: "result_lookup"}
    assert model.calls == 1
    assert len(tool.calls) == 1
    assert loop.stop_reason == "decision_stop"
    lines = _decision_lines(caplog)
    assert len(lines) == 1
    assert "outcome=accepted" in lines[0]
    assert "stopped by the agentic_loop decision point" in caplog.text


# ── Cap enforcement ─────────────────────────────────────────


async def test_continue_cannot_exceed_the_round_cap(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ScriptedProvider({"continue": {"kind": "boolean", "value": True, "confidence": 0.95}})
    model = _ScriptedChatModel([_tool_call_message(), _tool_call_message("call_2"), _tool_call_message("call_3")])
    loop, tool = _loop(model, binding=_binding(provider), max_tool_rounds=2)

    with caplog.at_level(logging.INFO):
        final_text, results = await loop.run(_messages())

    assert final_text is None
    assert results == {TOOL_NAME: "result_lookup"}
    assert model.calls == 2
    assert len(tool.calls) == 2
    # The provider is consulted after round 1 only; the cap is never negotiated.
    assert len(provider.questions) == 1
    assert loop.stop_reason == "max_rounds"
    assert "Hit max tool rounds (2)" in caplog.text


# ── Disabled point / fallbacks ──────────────────────────────


async def test_disabled_point_keeps_the_current_behaviour(caplog: pytest.LogCaptureFixture) -> None:
    model = _ScriptedChatModel([_tool_call_message(), AIMessage(content="done")])
    loop, _tool = _loop(model, binding=None, max_tool_rounds=3)

    with caplog.at_level(logging.INFO):
        final_text, results = await loop.run(_messages())

    assert final_text == "done"
    assert results == {TOOL_NAME: "result_lookup"}
    assert model.calls == 2
    assert loop.stop_reason == "final_answer"
    assert _decision_lines(caplog) == []


async def test_low_confidence_keeps_running_to_the_answer(caplog: pytest.LogCaptureFixture) -> None:
    provider = StubDecisionModel(mode="low_confidence")
    model = _ScriptedChatModel([_tool_call_message(), AIMessage(content="done")])
    loop, _tool = _loop(model, binding=_binding(provider, threshold=0.75))

    with caplog.at_level(logging.INFO):
        final_text, _results = await loop.run(_messages())

    assert final_text == "done"
    assert model.calls == 2
    assert loop.stop_reason == "final_answer"
    line = _decision_lines(caplog)[0]
    assert "outcome=fallback" in line
    assert "fallback_reason=low_confidence" in line


async def test_refusal_keeps_running_to_the_answer(caplog: pytest.LogCaptureFixture) -> None:
    provider = StubDecisionModel(answers={"continue": {"refusal": "not enough context"}})
    model = _ScriptedChatModel([_tool_call_message(), AIMessage(content="done")])
    loop, _tool = _loop(model, binding=_binding(provider))

    with caplog.at_level(logging.INFO):
        final_text, _results = await loop.run(_messages())

    assert final_text == "done"
    assert "fallback_reason=refusal" in _decision_lines(caplog)[0]


async def test_missing_answer_keeps_running(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ScriptedProvider({})
    model = _ScriptedChatModel([_tool_call_message(), AIMessage(content="done")])
    loop, _tool = _loop(model, binding=_binding(provider))

    with caplog.at_level(logging.INFO):
        final_text, _results = await loop.run(_messages())

    assert final_text == "done"
    assert "fallback_reason=invalid_response" in _decision_lines(caplog)[0]


async def test_non_boolean_answer_keeps_running(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ScriptedProvider(OrchidDecisionAnswer(name="continue", kind="choice", value="yes", confidence=0.95))
    model = _ScriptedChatModel([_tool_call_message(), AIMessage(content="done")])
    loop, _tool = _loop(model, binding=_binding(provider))

    with caplog.at_level(logging.INFO):
        final_text, _results = await loop.run(_messages())

    assert final_text == "done"
    assert "fallback_reason=invalid_response" in _decision_lines(caplog)[0]


async def test_timeout_keeps_running(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ScriptedProvider(error=TimeoutError())
    model = _ScriptedChatModel([_tool_call_message(), AIMessage(content="done")])
    loop, _tool = _loop(model, binding=_binding(provider))

    with caplog.at_level(logging.INFO):
        final_text, _results = await loop.run(_messages())

    assert final_text == "done"
    assert "fallback_reason=timeout" in _decision_lines(caplog)[0]


async def test_stalled_decision_provider_cannot_stall_the_run(caplog: pytest.LogCaptureFixture) -> None:
    # The provider would stop the loop, but it stalls past the wall-clock bound:
    # the timeout must win and keep the historical max-round behaviour.
    provider = _ScriptedProvider(
        {"continue": {"kind": "boolean", "value": False, "confidence": 0.95}},
        delay=0.05,
    )
    model = _ScriptedChatModel([_tool_call_message(), AIMessage(content="done")])
    loop, _tool = _loop(model, binding=_binding(provider), max_tool_rounds=3, decision_timeout=0.01)

    with caplog.at_level(logging.INFO):
        final_text, _results = await loop.run(_messages())

    assert final_text == "done"
    assert model.calls == 2
    assert loop.stop_reason == "final_answer"
    assert provider.timeouts == [0.01]
    line = _decision_lines(caplog)[0]
    assert "outcome=fallback" in line
    assert "fallback_reason=timeout" in line


async def test_provider_error_keeps_running(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ScriptedProvider(
        error=OrchidDecisionError("provider unavailable", code="provider_unavailable", provider="err")
    )
    model = _ScriptedChatModel([_tool_call_message(), AIMessage(content="done")])
    loop, _tool = _loop(model, binding=_binding(provider))

    with caplog.at_level(logging.INFO):
        final_text, _results = await loop.run(_messages())

    assert final_text == "done"
    line = _decision_lines(caplog)[0]
    assert "fallback_reason=provider_unavailable" in line
    assert "provider=err" in line


async def test_unexpected_exception_keeps_running(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ScriptedProvider(error=RuntimeError(SENTINEL))
    model = _ScriptedChatModel([_tool_call_message(), AIMessage(content="done")])
    loop, _tool = _loop(model, binding=_binding(provider))

    with caplog.at_level(logging.INFO):
        final_text, _results = await loop.run(_messages())

    assert final_text == "done"
    assert SENTINEL not in caplog.text
    assert "fallback_reason=provider_error" in _decision_lines(caplog)[0]


# ── Decision state and observability ────────────────────────


async def test_decision_state_is_bounded_and_payload_free(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ScriptedProvider({"continue": {"kind": "boolean", "value": True, "confidence": 0.95}})
    model = _ScriptedChatModel([_tool_call_message(), AIMessage(content="done")])
    loop, _tool = _loop(model, binding=_binding(provider), max_tool_rounds=3)

    with caplog.at_level(logging.INFO):
        await loop.run(_messages(SENTINEL + "x" * 5000))

    state = provider.states[0]
    assert state.text.startswith(SENTINEL)
    assert len(state.text) == 4000
    assert dict(state.payload) == {
        "agent": "a",
        "round": 1,
        "max_rounds": 3,
        "tools": [TOOL_NAME],
    }
    # The structured line never carries the user's text or tool arguments.
    line = _decision_lines(caplog)[0]
    assert SENTINEL not in line
    assert "query" not in line


async def test_stop_reason_is_none_before_the_first_run() -> None:
    model = _ScriptedChatModel([AIMessage(content="done")])
    loop, _tool = _loop(model)

    assert loop.stop_reason is None


def test_decision_timeout_must_be_positive() -> None:
    model = _ScriptedChatModel([AIMessage(content="done")])
    with pytest.raises(ValueError, match="decision timeout must be positive"):
        AgenticLoop(
            agent_name="a",
            chat_model=model,
            tool_map={},
            all_tool_defs=[],
            decision_timeout=0,
        )


# ── GenericAgent wiring ─────────────────────────────────────


class _FakeDispatcher:
    async def render_capabilities(self, auth: OrchidAuthContext, *, agent_name: str = "") -> MCPCapabilities:
        return MCPCapabilities(
            raw_tools=[{"name": TOOL_NAME, "description": "lookup tool", "schema": {"type": "object"}}],
            tool_client_map={TOOL_NAME: (object(), object())},
        )


class _StubLangChainTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.requires_approval = False

    async def ainvoke(self, args: dict[str, Any]) -> str:
        return "ok"


async def test_generic_agent_forwards_the_binding_to_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _CaptureLoop:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self._events: list = []

        async def run(self, messages: list) -> tuple[str | None, dict[str, Any]]:
            return "done", {}

    monkeypatch.setattr("orchid_ai.agents.agentic_loop.AgenticLoop", _CaptureLoop)
    monkeypatch.setattr(
        "orchid_ai.agents.tools.build_langchain_tools",
        lambda **kwargs: [_StubLangChainTool(TOOL_NAME)],
    )

    loop_binding = _binding(StubDecisionModel(answers={"continue": {"value": True}}))
    agent = GenericAgent(
        config=OrchidAgentConfig(description="d", prompt="p", name="a"),
        reader=NullVectorReader(),
        chat_model=_ScriptedChatModel([AIMessage(content="done")]),
        agentic_loop_binding=loop_binding,
    )
    agent._mcp_dispatcher = _FakeDispatcher()

    final_text, results, events = await agent._agentic_tool_loop("q", _auth(), None, [])

    assert final_text == "done"
    assert results == {}
    assert events == []
    assert captured["agentic_loop_binding"] is loop_binding
    assert captured["agent_name"] == "a"
