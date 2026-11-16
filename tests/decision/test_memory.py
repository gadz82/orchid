"""Tests for decision-backed conversation memory and summarisation.

The ``memory`` point is advisory: a confident answer may run the existing
summary path (``summarize``), leave memory untouched (``keep``), or write one
summary delta through the existing storage API (``persist``). Every uncertain
or failed outcome keeps the configured sliding-window heuristic and never adds
a second persistence write.

No test reaches the network: the decision suites use ``StubDecisionModel`` or a
local fake provider, and ``tests/decision/conftest.py`` makes constructing a
real ``httpx.AsyncClient`` fail loudly.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

import orchid_ai.graph.graph as graph_module
from orchid_ai.agents.generic_agent import GenericAgent
from orchid_ai.agents.memory_decision import (
    MAX_PAYLOAD_CHARS,
    MAX_STATE_TURNS,
    MAX_SUMMARY_CHARS,
    MAX_TURN_CHARS,
    MEMORY_ACTIONS,
    build_memory_decision_state,
    decide_memory_action,
)
from orchid_ai.config.schema import (
    OrchidAgentConfig,
    OrchidAgentsConfig,
    OrchidMemoryConfig,
    OrchidSupervisorConfig,
)
from orchid_ai.config.schema_decision import OrchidDecisionPointConfig
from orchid_ai.core.decision import (
    OrchidDecisionAnswer,
    OrchidDecisionError,
    OrchidDecisionQuestion,
    OrchidDecisionResult,
    OrchidDecisionState,
)
from orchid_ai.core.memory import NullConversationMemory
from orchid_ai.decision.registry import get_decision_point
from orchid_ai.decision.stub import StubDecisionModel
from orchid_ai.graph.decision_bindings import DecisionBinding
from orchid_ai.graph.graph import build_graph
from orchid_ai.graph.supervisor import (
    OrchidRoutingDecision,
    _extract_and_compress_history,
    create_supervisor_node,
)
from orchid_ai.rag.backends.null import NullVectorReader
from orchid_ai.runtime import OrchidRuntime

SENTINEL = "SENTINEL-DO-NOT-LEAK"


# ── Helpers ─────────────────────────────────────────────────


def _binding(provider: Any, *, threshold: float = 0.75) -> DecisionBinding:
    return DecisionBinding(
        spec=get_decision_point("memory"),
        config=OrchidDecisionPointConfig(model="stub", confidence_threshold=threshold, options={}),
        model=provider,
    )


def _decision_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == "orchid.decision"]


def _history(pairs: int = 2) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for index in range(pairs):
        history.append({"role": "user", "content": f"user-{index}"})
        history.append({"role": "assistant", "content": f"assistant-{index}"})
    return history


class _FakeChatModel:
    """Deterministic chat model; records compression/summarise calls."""

    def __init__(self, reply: str = "compressed summary") -> None:
        self.reply = reply
        self.calls: list[list[dict[str, Any]]] = []
        self.structured_schemas: list[Any] = []
        self.structured_reply: Any = None

    async def ainvoke(self, messages: list[dict[str, Any]], **_kwargs: Any) -> Any:
        self.calls.append(messages)
        return SimpleNamespace(content=self.reply)

    def with_structured_output(self, schema: Any, **_kwargs: Any) -> Any:
        self.structured_schemas.append(schema)
        reply = self.structured_reply

        async def _ainvoke(_messages: Any, **_extra: Any) -> Any:
            return reply

        return SimpleNamespace(ainvoke=_ainvoke)


class _FakeMemory:
    """Minimal conversation-memory stand-in that records persistence writes."""

    def __init__(self, summary: str | None = "existing summary") -> None:
        self.summary = summary
        self.reads: list[str] = []
        self.updates: list[tuple[str, list[dict[str, str]], str | None]] = []

    async def get_running_summary(self, chat_id: str) -> str | None:
        self.reads.append(chat_id)
        return self.summary

    async def update_running_summary(
        self,
        chat_id: str,
        new_messages: list[dict[str, str]],
        existing_summary: str | None,
    ) -> str:
        self.updates.append((chat_id, list(new_messages), existing_summary))
        return "updated summary"

    async def get_relevant_history(self, *_args: Any, **_kwargs: Any) -> list[dict[str, str]]:
        return []

    async def store_conversation_turn(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _RecordingProvider:
    """Return one fixed choice and capture the bounded state it received."""

    def __init__(self, action: str = "keep", *, confidence: float = 0.95) -> None:
        self.action = action
        self.confidence = confidence
        self.states: list[OrchidDecisionState] = []
        self.questions: list[OrchidDecisionQuestion] = []

    async def decide(self, state, questions, *, timeout=None):
        self.states.append(state)
        self.questions.extend(questions)
        return OrchidDecisionResult(
            answers={
                "action": OrchidDecisionAnswer(
                    name="action",
                    kind="choice",
                    value=self.action,
                    confidence=self.confidence,
                )
            },
            provider="recorder",
            model="recorder",
        )


class _ErrorProvider:
    def __init__(self, code: str) -> None:
        self.code = code

    async def decide(self, state, questions, *, timeout=None):
        raise OrchidDecisionError("provider failed", code=self.code, provider="error-provider")


class _RaisingProvider:
    async def decide(self, state, questions, *, timeout=None):
        raise RuntimeError(SENTINEL)


class _TimeoutProvider:
    async def decide(self, state, questions, *, timeout=None):
        raise TimeoutError


class _EmptyProvider:
    async def decide(self, state, questions, *, timeout=None):
        return OrchidDecisionResult(answers={}, provider="empty", model="empty")


class _BooleanProvider:
    async def decide(self, state, questions, *, timeout=None):
        return OrchidDecisionResult(
            answers={
                "action": OrchidDecisionAnswer(
                    name="action",
                    kind="boolean",
                    value=True,
                    confidence=0.95,
                )
            },
            provider="boolean",
            model="boolean",
        )


class _RefusalProvider:
    async def decide(self, state, questions, *, timeout=None):
        return OrchidDecisionResult(
            answers={
                "action": OrchidDecisionAnswer(
                    name="action",
                    kind="choice",
                    value=None,
                    confidence=0.9,
                    refusal="not enough context",
                )
            },
            provider="refusal",
            model="refusal",
        )


def _agent(
    *,
    binding: Any | None = None,
    memory: Any | None = None,
    recent_turns: int = 1,
) -> tuple[GenericAgent, _FakeChatModel]:
    chat_model = _FakeChatModel()
    agent = GenericAgent(
        config=OrchidAgentConfig(description="test", prompt="you are a test agent", name="a"),
        reader=NullVectorReader(),
        chat_model=chat_model,
        summary_config={"memory": memory, "recent_turns": recent_turns, "structured_output": False},
        memory_binding=binding,
    )
    return agent, chat_model


def _agent_state(pairs: int = 2, *, chat_id: str = "chat-1") -> dict[str, Any]:
    messages: list[Any] = []
    for index in range(pairs):
        messages.append(HumanMessage(content=f"user-{index}"))
        messages.append(AIMessage(content=f"assistant-{index}"))
    messages.append(HumanMessage(content="current question"))
    return {"messages": messages, "chat_id": chat_id}


# ── Helper: mapping and observability ───────────────────────


async def test_confident_summarize_is_accepted(caplog: pytest.LogCaptureFixture) -> None:
    provider = StubDecisionModel(answers={"action": {"choice": "summarize", "confidence": 0.95}})

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_memory_action(_binding(provider), recent_turns=_history())

    assert decision.accepted is True
    assert decision.action == "summarize"
    lines = _decision_lines(caplog)
    assert len(lines) == 1
    for fragment in (
        "decision point=memory",
        "path=decision",
        "provider=stub",
        "outcome=accepted",
        "fallback_reason=null",
    ):
        assert fragment in lines[0]


async def test_confident_keep_is_accepted(caplog: pytest.LogCaptureFixture) -> None:
    provider = StubDecisionModel(answers={"action": {"choice": "keep", "confidence": 0.9}})

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_memory_action(_binding(provider), recent_turns=_history())

    assert decision.action == "keep"
    assert decision.accepted is True
    assert "outcome=accepted" in _decision_lines(caplog)[0]


async def test_confident_persist_requires_storage(caplog: pytest.LogCaptureFixture) -> None:
    provider = _RecordingProvider("persist")

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        accepted = await decide_memory_action(
            _binding(provider),
            recent_turns=_history(),
            running_summary="summary",
            persist_available=True,
        )
        unavailable = await decide_memory_action(
            _binding(provider),
            recent_turns=_history(),
            running_summary="summary",
            persist_available=False,
        )

    assert accepted.action == "persist"
    assert accepted.accepted is True
    assert dict(provider.states[0].payload)["persist_available"] is True
    assert unavailable.action == "keep"
    assert unavailable.accepted is False
    assert unavailable.fallback_reason == "persist_unavailable"
    assert dict(provider.states[1].payload)["persist_available"] is False
    lines = _decision_lines(caplog)
    assert len(lines) == 2
    assert "outcome=accepted" in lines[0]
    assert "outcome=fallback" in lines[1]
    assert "fallback_reason=persist_unavailable" in lines[1]


async def test_low_confidence_keeps_the_heuristic(caplog: pytest.LogCaptureFixture) -> None:
    provider = StubDecisionModel(mode="low_confidence", answers={"action": {"choice": "summarize"}})

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_memory_action(
            _binding(provider, threshold=0.8),
            recent_turns=_history(),
        )

    assert decision.action == "keep"
    assert decision.source == "fallback"
    assert decision.fallback_reason == "low_confidence"
    lines = _decision_lines(caplog)
    assert len(lines) == 1
    assert "path=llm_fallback" in lines[0]
    assert "outcome=fallback" in lines[0]


async def test_provider_error_keeps_the_heuristic(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_memory_action(_binding(_ErrorProvider("provider_unavailable")), recent_turns=_history())

    assert decision.action == "keep"
    assert decision.source == "fallback"
    assert decision.fallback_reason == "provider_unavailable"
    assert "fallback_reason=provider_unavailable" in _decision_lines(caplog)[0]


async def test_unexpected_provider_error_logs_only_the_type(caplog: pytest.LogCaptureFixture) -> None:
    with (
        caplog.at_level(logging.WARNING, logger="orchid_ai.agents.memory_decision"),
        caplog.at_level(logging.INFO, logger="orchid.decision"),
    ):
        decision = await decide_memory_action(_binding(_RaisingProvider()), recent_turns=_history())

    assert decision.fallback_reason == "provider_error"
    assert SENTINEL not in caplog.text
    assert "RuntimeError" in caplog.text
    assert len(_decision_lines(caplog)) == 1


async def test_builtin_timeout_keeps_the_heuristic(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_memory_action(_binding(_TimeoutProvider()), recent_turns=_history())

    assert decision.fallback_reason == "timeout"
    assert "fallback_reason=timeout" in _decision_lines(caplog)[0]


async def test_refusal_keeps_the_heuristic(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_memory_action(_binding(_RefusalProvider()), recent_turns=_history())

    assert decision.action == "keep"
    assert decision.fallback_reason == "refusal"
    assert "fallback_reason=refusal" in _decision_lines(caplog)[0]


async def test_missing_answer_is_an_invalid_response(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_memory_action(_binding(_EmptyProvider()), recent_turns=_history())

    assert decision.fallback_reason == "invalid_response"
    assert len(_decision_lines(caplog)) == 1


async def test_non_choice_answer_is_an_invalid_response(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_memory_action(_binding(_BooleanProvider()), recent_turns=_history())

    assert decision.action == "keep"
    assert decision.fallback_reason == "invalid_response"
    assert "fallback_reason=invalid_response" in _decision_lines(caplog)[0]


async def test_summarize_unavailable_keeps_the_heuristic(caplog: pytest.LogCaptureFixture) -> None:
    provider = _RecordingProvider("summarize")

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_memory_action(
            _binding(provider),
            recent_turns=_history(),
            summarize_available=False,
        )

    assert decision.action == "keep"
    assert decision.fallback_reason == "summarize_unavailable"
    assert dict(provider.states[0].payload)["summarize_available"] is False
    assert "fallback_reason=summarize_unavailable" in _decision_lines(caplog)[0]


async def test_unknown_action_is_rejected(caplog: pytest.LogCaptureFixture) -> None:
    provider = _RecordingProvider("telepathy")

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_memory_action(_binding(provider), recent_turns=_history())

    assert decision.fallback_reason == "unknown_action"
    assert provider.questions[0].options == MEMORY_ACTIONS


async def test_no_provider_is_silent_and_keeps_the_heuristic(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_memory_action(None, recent_turns=_history())

    assert decision.action == "keep"
    assert decision.source == "disabled"
    assert decision.fallback_reason is None
    assert _decision_lines(caplog) == []


def test_bounded_state_excludes_tool_output_and_secrets() -> None:
    turns: list[dict[str, Any]] = []
    for _ in range(10):
        turns.append({"role": "user", "content": "u" * 1000})
        turns.append({"role": "assistant", "content": "a" * 1000})
    turns.append({"role": "tool", "content": f"TOOL-OUTPUT-{SENTINEL}"})
    turns.append({"role": "system", "content": f"SYSTEM-{SENTINEL}"})

    state = build_memory_decision_state(
        turns,
        "s" * 5000,
        persist_available=True,
    )
    payload = dict(state.payload)
    serialised = json.dumps(payload)

    assert state.text == ""
    assert len(serialised) <= MAX_PAYLOAD_CHARS
    assert payload["turn_count"] <= MAX_STATE_TURNS
    assert payload["has_more_turns"] is True
    assert all(len(turn["excerpt"]) <= MAX_TURN_CHARS for turn in payload["turns"])
    assert len(payload["summary_excerpt"]) <= MAX_SUMMARY_CHARS
    assert all(turn["role"] in ("user", "assistant") for turn in payload["turns"])
    assert SENTINEL not in serialised


async def test_decision_line_never_contains_raw_text(caplog: pytest.LogCaptureFixture) -> None:
    provider = _RecordingProvider("summarize")
    turns = [{"role": "user", "content": SENTINEL}]

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        await decide_memory_action(_binding(provider), recent_turns=turns)

    assert SENTINEL not in caplog.text
    # The bounded state carries the excerpt so the provider can decide; the
    # structured log line carries counts only.
    assert SENTINEL in json.dumps(dict(provider.states[0].payload))


async def test_question_offers_the_three_actions() -> None:
    provider = _RecordingProvider("keep")

    await decide_memory_action(_binding(provider), recent_turns=_history())

    question = provider.questions[0]
    assert question.name == "action"
    assert question.kind == "choice"
    assert set(question.options) == {"keep", "persist", "summarize"}


async def test_stub_decision_never_touches_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """The local provider path must not construct any HTTP client."""
    import httpx

    def _fail(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("memory decisions must not create an HTTP client")

    monkeypatch.setattr(httpx, "AsyncClient", _fail)
    decision = await decide_memory_action(
        _binding(StubDecisionModel(answers={"action": {"choice": "keep"}})),
        recent_turns=_history(),
    )
    assert decision.accepted is True


# ── GenericAgent: summary / persistence mapping ─────────────


async def test_agent_summarize_runs_the_existing_summary_path(caplog: pytest.LogCaptureFixture) -> None:
    provider = _RecordingProvider("summarize")
    memory = _FakeMemory()
    agent, chat = _agent(binding=_binding(provider), memory=memory)

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        await agent._step_summarise("current question", {}, [], _agent_state())

    # One compression call plus the final summarise call.
    assert len(chat.calls) == 2
    assert len(memory.updates) == 1
    chat_id, delta, existing = memory.updates[0]
    assert chat_id == "chat-1"
    assert delta == [
        {"role": "user", "content": "user-1"},
        {"role": "assistant", "content": "assistant-1"},
    ]
    assert existing == "existing summary"
    assert "outcome=accepted" in _decision_lines(caplog)[0]


async def test_agent_keep_leaves_memory_untouched(caplog: pytest.LogCaptureFixture) -> None:
    provider = _RecordingProvider("keep")
    memory = _FakeMemory()
    agent, chat = _agent(binding=_binding(provider), memory=memory)

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        await agent._step_summarise("current question", {}, [], _agent_state())

    # Only the final summarise call — no compression, no persistence.
    assert len(chat.calls) == 1
    assert memory.updates == []
    assert "outcome=accepted" in _decision_lines(caplog)[0]


async def test_agent_persist_writes_one_delta(caplog: pytest.LogCaptureFixture) -> None:
    provider = _RecordingProvider("persist")
    memory = _FakeMemory()
    agent, chat = _agent(binding=_binding(provider), memory=memory)

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        await agent._step_summarise("current question", {}, [], _agent_state())

    assert len(chat.calls) == 1  # no compression
    assert len(memory.updates) == 1
    _chat_id, delta, existing = memory.updates[0]
    # The full extracted history is persisted: an accepted ``keep`` can skip
    # compression for several turns, so a recent-window delta could lose them.
    assert delta == _history()
    assert existing == "existing summary"
    assert "outcome=accepted" in _decision_lines(caplog)[0]


async def test_agent_persist_without_a_running_summary_creates_one() -> None:
    provider = _RecordingProvider("persist")
    memory = _FakeMemory(summary=None)
    agent, _chat = _agent(binding=_binding(provider), memory=memory)

    await agent._step_summarise("current question", {}, [], _agent_state())

    assert len(memory.updates) == 1
    assert memory.updates[0][1] == _history()
    assert memory.updates[0][2] is None


async def test_agent_low_confidence_keeps_the_heuristic_and_adds_no_decision_write() -> None:
    provider = _RecordingProvider("summarize", confidence=0.5)
    memory = _FakeMemory()
    agent, chat = _agent(binding=_binding(provider, threshold=0.8), memory=memory)

    await agent._step_summarise("current question", {}, [], _agent_state())

    # Fallback means no *additional* decision-driven write: the legacy
    # compression path runs and performs its normal one summary write.
    assert len(chat.calls) == 2
    assert len(memory.updates) == 1
    assert memory.updates[0][1] == [
        {"role": "user", "content": "user-1"},
        {"role": "assistant", "content": "assistant-1"},
    ]


async def test_agent_provider_error_keeps_the_heuristic_and_adds_no_decision_write() -> None:
    memory = _FakeMemory()
    agent, chat = _agent(binding=_binding(_ErrorProvider("provider_unavailable")), memory=memory)

    await agent._step_summarise("current question", {}, [], _agent_state())

    # The single write is the legacy heuristic's, not a decision-driven one.
    assert len(chat.calls) == 2
    assert len(memory.updates) == 1


async def test_agent_without_a_provider_keeps_the_default_behaviour(caplog: pytest.LogCaptureFixture) -> None:
    memory = _FakeMemory()
    agent, chat = _agent(binding=None, memory=memory)

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        await agent._step_summarise("current question", {}, [], _agent_state())

    assert len(chat.calls) == 2
    assert len(memory.updates) == 1
    assert _decision_lines(caplog) == []


async def test_agent_persist_without_storage_falls_back_to_the_heuristic() -> None:
    provider = _RecordingProvider("persist")
    agent, chat = _agent(binding=_binding(provider), memory=None)

    await agent._step_summarise("current question", {}, [], _agent_state())

    # No storage means no persistence: the legacy compression path runs once.
    assert len(chat.calls) == 2


async def test_agent_persistence_is_not_duplicated_per_call_site(caplog: pytest.LogCaptureFixture) -> None:
    """One decision path writes once; turn-level dedup across seams is out of scope."""
    for action in ("summarize", "persist"):
        provider = _RecordingProvider(action)
        memory = _FakeMemory()
        agent, _chat = _agent(binding=_binding(provider), memory=memory)

        with caplog.at_level(logging.INFO, logger="orchid.decision"):
            await agent._step_summarise("current question", {}, [], _agent_state())
            lines = _decision_lines(caplog)

        assert len(memory.updates) == 1, action
        assert len(lines) == 1, action
        caplog.clear()


async def test_persistence_is_per_call_site_not_per_turn() -> None:
    """Supervisor and agent seams each write once in the same turn (2 total).

    The point governs each call site, not the whole turn: ``synthesizer.py``
    and ``sequential_advancer.py`` still run the legacy path, and parallel
    agents each consult the point independently.
    """
    provider = _RecordingProvider("persist")
    memory = _FakeMemory()

    await _supervisor_history(_agent_state(), binding=_binding(provider), memory=memory)
    agent, _chat = _agent(binding=_binding(provider), memory=memory)
    await agent._step_summarise("current question", {}, [], _agent_state())

    assert len(memory.updates) == 2


# ── Supervisor: summary / persistence mapping ───────────────


def _supervisor_config() -> OrchidSupervisorConfig:
    return OrchidSupervisorConfig(
        history_summary_recent_turns=1,
        memory=OrchidMemoryConfig(strategy="running_summary"),
    )


async def _supervisor_history(
    state: dict[str, Any], *, binding: Any, memory: Any
) -> tuple[list[dict[str, str]], _FakeChatModel]:
    chat = _FakeChatModel()
    history = await _extract_and_compress_history(
        state,
        _supervisor_config(),
        chat,
        memory,
        memory_binding=binding,
    )
    return history, chat


async def test_supervisor_summarize_compresses_and_persists_once() -> None:
    provider = _RecordingProvider("summarize")
    memory = _FakeMemory()

    history, chat = await _supervisor_history(_agent_state(), binding=_binding(provider), memory=memory)

    # The supervisor strips the summary artefact from the prompt history; the
    # compressed window is visible in the persisted delta (2 recent messages).
    assert history == _history()[2:]
    assert len(chat.calls) == 1
    assert len(memory.updates) == 1
    assert memory.updates[0][1] == _history()[2:]


async def test_supervisor_keep_leaves_history_untouched() -> None:
    provider = _RecordingProvider("keep")
    memory = _FakeMemory()

    history, chat = await _supervisor_history(_agent_state(), binding=_binding(provider), memory=memory)

    assert history == _history()
    assert chat.calls == []
    assert memory.updates == []


async def test_supervisor_persist_writes_one_delta_without_compressing() -> None:
    provider = _RecordingProvider("persist")
    memory = _FakeMemory()

    history, chat = await _supervisor_history(_agent_state(), binding=_binding(provider), memory=memory)

    assert history == _history()
    assert chat.calls == []
    assert len(memory.updates) == 1
    assert memory.updates[0][1] == _history()


async def test_supervisor_low_confidence_keeps_the_heuristic() -> None:
    provider = _RecordingProvider("summarize", confidence=0.5)
    memory = _FakeMemory()

    history, chat = await _supervisor_history(
        _agent_state(),
        binding=_binding(provider, threshold=0.8),
        memory=memory,
    )

    assert history == _history()[2:]
    assert len(chat.calls) == 1
    # Fallback runs the legacy heuristic; its single write is not
    # decision-driven, so "no decision write" does not mean "zero writes".
    assert len(memory.updates) == 1


async def test_supervisor_without_a_binding_keeps_the_default_behaviour() -> None:
    memory = _FakeMemory()

    history, chat = await _supervisor_history(_agent_state(), binding=None, memory=memory)

    assert history == _history()[2:]
    assert len(chat.calls) == 1
    assert len(memory.updates) == 1


async def test_supervisor_seeds_the_first_running_summary() -> None:
    """H1 regression: the base path persisted even when no summary existed.

    That write is what seeds the first running summary for single-agent turns;
    the decision work must not gate it on ``running_summary is not None``.
    """
    memory = _FakeMemory(summary=None)

    history, chat = await _supervisor_history(_agent_state(), binding=None, memory=memory)

    assert history == _history()[2:]
    assert len(chat.calls) == 1
    assert len(memory.updates) == 1
    assert memory.updates[0][2] is None


async def test_supervisor_null_memory_is_not_persistable(caplog: pytest.LogCaptureFixture) -> None:
    """M5: a strategy without chat_storage resolves to NullConversationMemory."""
    provider = _RecordingProvider("persist")
    memory = NullConversationMemory()

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        history, chat = await _supervisor_history(_agent_state(), binding=_binding(provider), memory=memory)

    assert dict(provider.states[0].payload)["persist_available"] is False
    assert "fallback_reason=persist_unavailable" in _decision_lines(caplog)[0]
    # Fallback runs the legacy compression path; the null memory writes nothing.
    assert history == _history()[2:]
    assert len(chat.calls) == 1


async def test_supervisor_summary_disabled_never_calls_the_provider() -> None:
    provider = _RecordingProvider("summarize")
    memory = _FakeMemory()
    sup = OrchidSupervisorConfig(
        history_summary_enabled=False,
        history_summary_recent_turns=1,
        memory=OrchidMemoryConfig(strategy="running_summary"),
    )
    chat = _FakeChatModel()

    history = await _extract_and_compress_history(
        _agent_state(),
        sup,
        chat,
        memory,
        memory_binding=_binding(provider),
    )

    assert provider.states == []
    assert chat.calls == []
    assert memory.updates == []
    assert history == _history()


async def test_supervisor_node_threads_the_memory_binding() -> None:
    provider = _RecordingProvider("persist")
    memory = _FakeMemory()
    chat = _FakeChatModel()
    chat.structured_reply = OrchidRoutingDecision(reasoning="ok", execution="parallel", agents=["a"])
    node = create_supervisor_node(
        "test/model",
        {"a": "test agent"},
        chat_model=chat,
        supervisor_config=_supervisor_config(),
        memory=memory,
        memory_binding=_binding(provider),
    )

    result = await node(_agent_state())

    assert result["active_agents"] == ["a"]
    assert len(memory.updates) == 1
    assert chat.structured_schemas == [OrchidRoutingDecision]


# ── Graph wiring ────────────────────────────────────────────


def _agent_entry() -> dict[str, Any]:
    return {
        "description": "test agent",
        "prompt": "you are a test agent",
        "llm": {"model": "test/model"},
        "rag": {"namespace": "kb", "retrieval": {"strategy": "simple"}},
    }


def _graph_config(*, memory: dict[str, Any] | None = None, with_provider: bool = True) -> OrchidAgentsConfig:
    data: dict[str, Any] = {"version": "1", "agents": {"a": _agent_entry()}}
    if with_provider:
        data["decision_models"] = {"stub": {"protocol": "stub"}}
    if memory is not None:
        data["decisions"] = {"points": {"memory": memory}}
    return OrchidAgentsConfig.model_validate(data)


def _graph_runtime(
    *,
    with_provider: bool = True,
    chat_model: Any | None = None,
    decision_model: Any | None = None,
) -> OrchidRuntime:
    chat = chat_model or MagicMock(spec=BaseChatModel)
    chat.bind_tools = MagicMock(return_value=chat)
    providers: dict[str, Any] = {}
    if with_provider:
        providers["stub"] = decision_model or StubDecisionModel(
            answers={"action": {"choice": "keep", "confidence": 0.95}}
        )
    return OrchidRuntime(default_model="test/model", chat_model=chat, decision_models=providers)


def test_graph_passes_the_fleet_binding_to_agents() -> None:
    config = _graph_config(memory={"model": "stub"})
    agents: dict[str, Any] = {}

    build_graph(config=config, runtime=_graph_runtime(), agents_out=agents)

    binding = agents["a"]._memory_binding
    assert binding is not None
    assert binding.point == "memory"
    assert binding.provider == "stub"


def test_graph_disabled_memory_point_leaves_agents_unbound(caplog: pytest.LogCaptureFixture) -> None:
    config = _graph_config(memory={"model": "stub", "enabled": False})
    agents: dict[str, Any] = {}

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        build_graph(config=config, runtime=_graph_runtime(), agents_out=agents)

    assert agents["a"]._memory_binding is None
    assert _decision_lines(caplog) == []


def test_graph_unconfigured_memory_point_leaves_agents_unbound() -> None:
    config = _graph_config()
    agents: dict[str, Any] = {}

    build_graph(config=config, runtime=_graph_runtime(), agents_out=agents)

    assert agents["a"]._memory_binding is None


def test_graph_missing_memory_provider_leaves_agents_unbound(caplog: pytest.LogCaptureFixture) -> None:
    config = _graph_config(memory={"model": "stub"})
    agents: dict[str, Any] = {}

    with caplog.at_level(logging.WARNING, logger="orchid_ai.graph.decision_bindings"):
        build_graph(config=config, runtime=_graph_runtime(with_provider=False), agents_out=agents)

    assert agents["a"]._memory_binding is None
    assert "missing from the runtime" in caplog.text


def test_graph_child_agent_receives_the_fleet_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    parent = _agent_entry()
    parent["children"] = {"kid": _agent_entry()}
    config = OrchidAgentsConfig.model_validate(
        {
            "version": "1",
            "agents": {"parent": parent},
            "decision_models": {"stub": {"protocol": "stub"}},
            "decisions": {"points": {"memory": {"model": "stub"}}},
        }
    )

    created: dict[str, Any] = {}
    real_instantiate = graph_module._instantiate_agent

    def _capture(name, agent_config, *args, **kwargs):
        agent = real_instantiate(name, agent_config, *args, **kwargs)
        created[name] = agent
        return agent

    monkeypatch.setattr(graph_module, "_instantiate_agent", _capture)
    build_graph(config=config, runtime=_graph_runtime())

    binding = created["kid"]._memory_binding
    assert binding is not None
    assert binding.point == "memory"


async def test_nested_child_and_sub_supervisor_are_governed(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """M6: the subgraph gets the root supervisor config, memory and summary cfg."""
    parent = _agent_entry()
    parent["children"] = {"kid": _agent_entry()}
    config = OrchidAgentsConfig.model_validate(
        {
            "version": "1",
            "agents": {"parent": parent},
            "decision_models": {"stub": {"protocol": "stub"}},
            "decisions": {"points": {"memory": {"model": "stub"}}},
            "supervisor": {"history_summary_recent_turns": 1},
        }
    )
    provider = _RecordingProvider("summarize")
    chat = _FakeChatModel()

    supervisor_calls: list[tuple[dict[str, Any], dict[str, Any]]] = []
    real_supervisor = graph_module.create_supervisor_node

    def _capture_supervisor(model, descriptions, **kwargs):
        supervisor_calls.append((dict(descriptions), kwargs))
        return real_supervisor(model, descriptions, **kwargs)

    created: dict[str, Any] = {}
    real_instantiate = graph_module._instantiate_agent

    def _capture_agent(name, agent_config, *args, **kwargs):
        agent = real_instantiate(name, agent_config, *args, **kwargs)
        created[name] = agent
        return agent

    monkeypatch.setattr(graph_module, "create_supervisor_node", _capture_supervisor)
    monkeypatch.setattr(graph_module, "_instantiate_agent", _capture_agent)
    build_graph(
        config=config,
        runtime=_graph_runtime(chat_model=chat, decision_model=provider),
    )

    root = next(kwargs for descriptions, kwargs in supervisor_calls if descriptions == {"parent": "test agent"})
    sub = next(kwargs for descriptions, kwargs in supervisor_calls if descriptions == {"kid": "test agent"})
    assert sub["supervisor_config"] is config.supervisor
    assert sub["memory"] is root["memory"]
    assert sub["memory_binding"] is root["memory_binding"]

    child = created["kid"]
    assert child._summary_config is not None
    assert child._summary_config["recent_turns"] == 1
    assert child._memory_binding is not None

    # The binding now governs the child seam: a confident ``summarize`` runs
    # the existing compression path (compression call + final summarise call).
    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        await child._step_summarise("current question", {}, [], _agent_state())

    assert len(chat.calls) == 2
    assert "outcome=accepted" in _decision_lines(caplog)[0]
