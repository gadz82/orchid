"""Phase-3 ``OrchidAgent.emit_signal`` tests (§15.1)."""

from __future__ import annotations

import pytest

from orchid_ai.core.agent import OrchidAgent
from orchid_ai.core.events.dispatcher import OrchidSignalDispatcher
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.events.producers.internal import DispatcherSignalEmitter
from orchid_ai.events.queues.inmemory import (
    InMemorySignalQueue,
    InMemorySignalStore,
)

# ── Fixture agent ───────────────────────────────────────────


class _Agent(OrchidAgent):
    @property
    def name(self) -> str:
        return "test-agent"

    @property
    def description(self) -> str:
        return "x"

    async def run(self, state):  # pragma: no cover — irrelevant here
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
    return {
        "agent": agent,
        "queue": queue,
        "store": store,
        "dispatcher": dispatcher,
    }


# ── Tests ───────────────────────────────────────────────────


async def test_emit_signal_without_emitter_raises() -> None:
    agent = _Agent(reader=None)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="events disabled"):
        await agent.emit_signal("demo.event", {})


async def test_emit_signal_basic_path(wired_agent) -> None:
    agent = wired_agent["agent"]
    store = wired_agent["store"]

    result = await agent.emit_signal("demo.event", {"k": "v"}, dedupe_key="abc")
    assert result.deduplicated is False
    assert result.signal_id is not None

    fetched = await store.get(result.signal_id)
    assert fetched is not None
    assert fetched.type == "demo.event"
    assert fetched.payload == {"k": "v"}
    assert fetched.source == "internal:agent:test-agent"
    assert fetched.tenant_key == "t-1"
    assert fetched.user_id == "u-7"
    assert fetched.correlation_id == "corr-abc"
    assert fetched.dedupe_key == "abc"
    # Default identity = act_as_user with current user.
    assert fetched.identity_claim == {
        "mode": "act_as_user",
        "user_id": "u-7",
    }
    # No chat binding by default.
    assert fetched.chat_binding is None


async def test_emit_signal_chat_id_self_binds_to_current_chat(
    wired_agent,
) -> None:
    agent = wired_agent["agent"]
    store = wired_agent["store"]
    result = await agent.emit_signal(
        "research.requested",
        {"question": "..."},
        chat_id="self",
    )
    sig = await store.get(result.signal_id)
    assert sig is not None
    # ``_current_message_id`` is None on the wired_agent fixture, so
    # ``source_message_id`` defaults to None even with chat_id="self".
    assert sig.chat_binding == {
        "chat_id": "C-7",
        "mode": "append_final_message",
        "on_failure": "post_error",
        "source_message_id": None,
    }


async def test_emit_signal_chat_id_self_outside_chat_run_raises() -> None:
    queue = InMemorySignalQueue()
    store = InMemorySignalStore()
    dispatcher = OrchidSignalDispatcher(store=store, queue=queue)
    agent = _Agent(reader=None)  # type: ignore[arg-type]
    agent._signal_emitter = DispatcherSignalEmitter(dispatcher)
    agent._current_auth = OrchidAuthContext(access_token="t", tenant_key="t-1", user_id="u-7")
    # Note: ``_current_chat_id`` is None — outside a chat run.
    with pytest.raises(RuntimeError, match="chat_id='self'"):
        await agent.emit_signal("demo.event", {}, chat_id="self")


async def test_emit_signal_explicit_chat_id_carries_through(wired_agent) -> None:
    agent = wired_agent["agent"]
    store = wired_agent["store"]
    result = await agent.emit_signal(
        "demo.event",
        {},
        chat_id="C-OTHER",
        chat_binding_mode="append_with_metadata",
        chat_binding_on_failure="silent",
    )
    sig = await store.get(result.signal_id)
    assert sig is not None
    # Cross-chat emit defaults ``source_message_id`` to None — anchoring
    # under another chat's message id would be meaningless.
    assert sig.chat_binding == {
        "chat_id": "C-OTHER",
        "mode": "append_with_metadata",
        "on_failure": "silent",
        "source_message_id": None,
    }


async def test_emit_signal_identity_override(wired_agent) -> None:
    agent = wired_agent["agent"]
    store = wired_agent["store"]
    result = await agent.emit_signal(
        "demo.event",
        {},
        identity={"mode": "service_account", "name": "ops-bot"},
    )
    sig = await store.get(result.signal_id)
    assert sig is not None
    assert sig.identity_claim == {
        "mode": "service_account",
        "name": "ops-bot",
    }


async def test_emit_signal_dedupe_returns_existing(wired_agent) -> None:
    agent = wired_agent["agent"]
    first = await agent.emit_signal("demo.event", {}, dedupe_key="dup")
    second = await agent.emit_signal("demo.event", {}, dedupe_key="dup")
    assert second.deduplicated is True
    assert second.signal_id == first.signal_id
