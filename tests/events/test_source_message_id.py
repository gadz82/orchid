"""``ChatBinding.source_message_id`` auto-population (Phase 4.5 §LS5).

Covers the in-chat live progress anchoring contract:

- ``emit_signal(chat_id="self")`` from inside a chat turn auto-populates
  ``ChatBinding.source_message_id`` from ``OrchidAgent._current_message_id``.
- An explicit ``chat_binding_source_message_id=`` kwarg overrides the
  auto-population.
- Cross-chat ``emit_signal(chat_id="C-other")`` defaults the field to
  ``None`` regardless of ``_current_message_id`` — anchoring under
  another chat's message id would mis-render.
- An ``emit_signal(chat_id="self")`` outside a chat turn raises
  ``RuntimeError`` (the existing §15.1 contract).
- ``_create_agent_node`` (graph wrapper) sets ``_current_message_id``
  from the latest ``HumanMessage`` and restores the prior value
  after the agent runs.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from orchid_ai.core.agent import OrchidAgent
from orchid_ai.core.events.dispatcher import OrchidSignalDispatcher
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.events.producers.internal import DispatcherSignalEmitter
from orchid_ai.events.queues.inmemory import (
    InMemorySignalQueue,
    InMemorySignalStore,
)
from orchid_ai.graph.graph import _create_agent_node, _latest_human_message_id


# ── Fixture agent ───────────────────────────────────────────


class _Agent(OrchidAgent):
    @property
    def name(self) -> str:
        return "test-agent"

    @property
    def description(self) -> str:
        return "x"

    async def run(self, state):  # pragma: no cover — overridden by tests below
        return state


@pytest.fixture
def wired_agent():
    queue = InMemorySignalQueue()
    store = InMemorySignalStore()
    dispatcher = OrchidSignalDispatcher(store=store, queue=queue)
    emitter = DispatcherSignalEmitter(dispatcher)
    agent = _Agent(reader=None)  # type: ignore[arg-type]
    agent._signal_emitter = emitter
    agent._current_auth = OrchidAuthContext(access_token="t", tenant_key="t-1", user_id="u-7")
    agent._current_correlation_id = "corr-abc"
    agent._current_chat_id = "C-7"
    agent._current_message_id = "m-42"
    return {"agent": agent, "store": store}


# ── emit_signal auto-population ─────────────────────────────


async def test_self_chat_auto_populates_source_message_id(wired_agent) -> None:
    """``chat_id='self'`` lifts ``_current_message_id`` into the binding."""
    agent, store = wired_agent["agent"], wired_agent["store"]
    result = await agent.emit_signal("research.requested", {}, chat_id="self")
    sig = await store.get(result.signal_id)
    assert sig is not None
    assert sig.chat_binding == {
        "chat_id": "C-7",
        "mode": "append_final_message",
        "on_failure": "post_error",
        "source_message_id": "m-42",
    }


async def test_explicit_kwarg_overrides_auto_population(wired_agent) -> None:
    """``chat_binding_source_message_id=`` wins over ``_current_message_id``."""
    agent, store = wired_agent["agent"], wired_agent["store"]
    result = await agent.emit_signal(
        "research.requested",
        {},
        chat_id="self",
        chat_binding_source_message_id="m-explicit",
    )
    sig = await store.get(result.signal_id)
    assert sig is not None
    assert sig.chat_binding["source_message_id"] == "m-explicit"


async def test_cross_chat_defaults_source_message_id_to_none(wired_agent) -> None:
    """``chat_id='C-other'`` defaults ``source_message_id`` to ``None``.

    The ``_current_message_id`` is set on the agent (the user is in
    C-7) but the signal targets a different chat — anchoring under
    a foreign message id would mis-render, so we deliberately drop it.
    """
    agent, store = wired_agent["agent"], wired_agent["store"]
    result = await agent.emit_signal("demo.event", {}, chat_id="C-OTHER")
    sig = await store.get(result.signal_id)
    assert sig is not None
    assert sig.chat_binding == {
        "chat_id": "C-OTHER",
        "mode": "append_final_message",
        "on_failure": "post_error",
        "source_message_id": None,
    }


async def test_cross_chat_explicit_override_is_honoured(wired_agent) -> None:
    """Cross-chat caller MAY pass an explicit override (rare)."""
    agent, store = wired_agent["agent"], wired_agent["store"]
    result = await agent.emit_signal(
        "demo.event",
        {},
        chat_id="C-OTHER",
        chat_binding_source_message_id="m-from-caller",
    )
    sig = await store.get(result.signal_id)
    assert sig is not None
    assert sig.chat_binding["source_message_id"] == "m-from-caller"


async def test_self_outside_chat_run_still_raises() -> None:
    """``chat_id='self'`` outside a chat run is still a hard error."""
    queue = InMemorySignalQueue()
    store = InMemorySignalStore()
    dispatcher = OrchidSignalDispatcher(store=store, queue=queue)
    agent = _Agent(reader=None)  # type: ignore[arg-type]
    agent._signal_emitter = DispatcherSignalEmitter(dispatcher)
    agent._current_auth = OrchidAuthContext(access_token="t", tenant_key="t-1", user_id="u-7")
    # _current_chat_id is None, _current_message_id is None.
    with pytest.raises(RuntimeError, match="chat_id='self'"):
        await agent.emit_signal("demo.event", {}, chat_id="self")


# ── Graph plumbing ──────────────────────────────────────────


def test_latest_human_message_id_returns_most_recent_human_id() -> None:
    state = {
        "messages": [
            HumanMessage(content="first", id="m-1"),
            AIMessage(content="reply"),
            HumanMessage(content="second", id="m-2"),
        ]
    }
    assert _latest_human_message_id(state) == "m-2"  # type: ignore[arg-type]


def test_latest_human_message_id_returns_none_when_no_human_message() -> None:
    state = {"messages": [AIMessage(content="ai-only")]}
    assert _latest_human_message_id(state) is None  # type: ignore[arg-type]


def test_latest_human_message_id_returns_none_when_id_missing() -> None:
    state = {"messages": [HumanMessage(content="no id")]}
    assert _latest_human_message_id(state) is None  # type: ignore[arg-type]


def test_latest_human_message_id_handles_empty_messages() -> None:
    assert _latest_human_message_id({"messages": []}) is None  # type: ignore[arg-type]
    assert _latest_human_message_id({}) is None  # type: ignore[arg-type]


async def test_graph_wrapper_pins_message_id_during_run_and_restores() -> None:
    """``_create_agent_node`` sets + restores per-call agent context."""
    seen: dict[str, str | None] = {}

    class _Capturing(OrchidAgent):
        @property
        def name(self) -> str:
            return "cap"

        @property
        def description(self) -> str:
            return "x"

        async def run(self, state):
            seen["chat_id"] = self._current_chat_id
            seen["message_id"] = self._current_message_id
            return {"messages": [AIMessage(content="ok")]}

    agent = _Capturing(reader=None)  # type: ignore[arg-type]
    # Pre-existing values that must be restored.
    agent._current_chat_id = "PRIOR-CHAT"
    agent._current_message_id = "PRIOR-MSG"

    node = _create_agent_node(agent=agent)
    state = {
        "messages": [HumanMessage(content="hi", id="m-99")],
        "chat_id": "C-RUN",
        "auth_context": None,
    }
    await node(state)  # type: ignore[arg-type]

    # Inside ``run`` the agent saw the per-call context.
    assert seen == {"chat_id": "C-RUN", "message_id": "m-99"}
    # Prior values restored after the run.
    assert agent._current_chat_id == "PRIOR-CHAT"
    assert agent._current_message_id == "PRIOR-MSG"


async def test_graph_wrapper_restores_even_on_agent_exception() -> None:
    """The ``finally`` branch must restore context after a crash."""

    class _Crashing(OrchidAgent):
        @property
        def name(self) -> str:
            return "crash"

        @property
        def description(self) -> str:
            return "x"

        async def run(self, state):
            raise RuntimeError("boom")

    agent = _Crashing(reader=None)  # type: ignore[arg-type]
    agent._current_chat_id = "PRIOR"
    agent._current_message_id = "PRIOR-MSG"

    node = _create_agent_node(agent=agent)
    state = {
        "messages": [HumanMessage(content="hi", id="m-7")],
        "chat_id": "C-RUN",
        "auth_context": None,
    }
    # Wrapper catches the exception (existing behaviour) and returns
    # a fallback message — but must still have restored agent state.
    await node(state)  # type: ignore[arg-type]
    assert agent._current_chat_id == "PRIOR"
    assert agent._current_message_id == "PRIOR-MSG"
