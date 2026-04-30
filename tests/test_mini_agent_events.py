"""Tests for the mini-agent lifecycle event surface (Phase B / B8).

Covers:
  - The wire-format helpers (``make_event_message`` / ``extract_event``).
  - The decomposer hook emits ``mini_agent.decomposed`` when forking.
  - The mini node emits ``mini_agent.started`` and ``mini_agent.finished``
    (both success and failure paths).
  - The aggregator emits ``mini_agent.aggregated`` ahead of the
    user-visible AIMessage in both synthesis and short-circuit paths.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from orchid_ai.agents.mini_agent_aggregator import aggregator_node_factory
from orchid_ai.agents.mini_agent_decomposer import (
    MiniAgentDecomposition,
    MiniAgentSubTask,
    maybe_decompose,
)
from orchid_ai.agents.mini_agent_node import mini_agent_node_factory
from orchid_ai.config.schema import (
    OrchidAgentConfig,
    OrchidLLMConfig,
    OrchidMiniAgentConfig,
    OrchidRAGConfig,
)
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.observability import (
    MINI_AGENT_EVENT_KEY,
    extract_event,
    is_event_message,
    make_event_message,
)


# ── Wire-format helpers ───────────────────────────────────────


class TestEventHelpers:
    def test_make_event_message_shape(self):
        msg = make_event_message(
            "mini_agent.started",
            {"parent": "support", "mini_id": "mini_0", "description": "lookup user"},
        )
        assert isinstance(msg, SystemMessage)
        assert msg.content == ""
        assert msg.additional_kwargs[MINI_AGENT_EVENT_KEY] == "mini_agent.started"
        assert msg.additional_kwargs["data"] == {
            "parent": "support",
            "mini_id": "mini_0",
            "description": "lookup user",
        }

    def test_extract_event_round_trip(self):
        original = {"parent": "support", "count": 2, "sub_tasks": []}
        msg = make_event_message("mini_agent.decomposed", original)
        result = extract_event(msg)
        assert result is not None
        name, data = result
        assert name == "mini_agent.decomposed"
        assert data == original

    def test_extract_event_returns_none_for_non_events(self):
        # AIMessage / HumanMessage / SystemMessage without metadata → None.
        assert extract_event(AIMessage(content="hello")) is None
        assert extract_event(HumanMessage(content="hi")) is None
        assert extract_event(SystemMessage(content="just a system message")) is None
        assert extract_event(None) is None
        assert extract_event({}) is None
        # SystemMessage with unrelated additional_kwargs → None.
        msg = SystemMessage(content="x", additional_kwargs={"unrelated": True})
        assert extract_event(msg) is None

    def test_extract_event_handles_malformed_data(self):
        # Event with non-dict ``data`` payload still parses (data → {}).
        msg = SystemMessage(
            content="",
            additional_kwargs={MINI_AGENT_EVENT_KEY: "mini_agent.started", "data": None},
        )
        result = extract_event(msg)
        assert result is not None
        assert result[0] == "mini_agent.started"
        assert result[1] == {}

    def test_is_event_message_convenience(self):
        assert is_event_message(make_event_message("mini_agent.aggregated", {}))
        assert not is_event_message(AIMessage(content="hi"))


# ── Decomposer hook emits mini_agent.decomposed ───────────────


def _agent_config(
    name: str = "support",
    *,
    tool_allowlist_mode: str = "strict",
) -> OrchidAgentConfig:
    return OrchidAgentConfig(
        name=name,
        description=f"{name} agent",
        prompt="be helpful",
        rag=OrchidRAGConfig(enabled=False),
        llm=OrchidLLMConfig(model="gemini/test"),
        mini_agent=OrchidMiniAgentConfig(enabled=True, tool_allowlist_mode=tool_allowlist_mode),
    )


def _structured_chat(decomposition: MiniAgentDecomposition):
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=decomposition)
    chat = MagicMock()
    chat.with_structured_output = MagicMock(return_value=structured)
    return chat


@pytest.mark.asyncio
async def test_maybe_decompose_emits_decomposed_event_when_forking():
    decomposition = MiniAgentDecomposition(
        should_fork=True,
        sub_tasks=[
            MiniAgentSubTask(
                id="mini_0",
                description="lookup user",
                instruction="find the user",
                allowed_tools=[],
                rationale="r",
            ),
            MiniAgentSubTask(
                id="mini_1",
                description="lookup orders",
                instruction="list orders",
                allowed_tools=[],
                rationale="r",
            ),
        ],
    )
    chat = _structured_chat(decomposition)

    # ``parent_full`` allowlist mode lets empty ``allowed_tools`` through
    # — the test cares about the event emission, not the allowlist
    # enforcement (covered by ``test_mini_agent_decomposer.py``).
    update = await maybe_decompose(
        agent_config=_agent_config(tool_allowlist_mode="parent_full"),
        chat_model=chat,
        mcp_clients=[],
        auth=OrchidAuthContext(access_token="t", tenant_key="x", user_id="u"),
        state={"messages": [HumanMessage(content="please do A and B")]},
    )

    assert update is not None
    # Decision recorded for the fork router.
    assert "support" in update["mini_agent_decisions"]
    # Lifecycle event piggy-backed in messages.
    msgs = update["messages"]
    assert len(msgs) == 1
    name, data = extract_event(msgs[0])
    assert name == "mini_agent.decomposed"
    assert data["parent"] == "support"
    assert data["count"] == 2
    assert {st["id"] for st in data["sub_tasks"]} == {"mini_0", "mini_1"}
    # Sub-task entries carry the description but NOT the full instruction.
    for st in data["sub_tasks"]:
        assert "description" in st
        assert "instruction" not in st


@pytest.mark.asyncio
async def test_maybe_decompose_no_event_when_should_fork_false():
    chat = _structured_chat(MiniAgentDecomposition(should_fork=False, reasoning="single task"))
    update = await maybe_decompose(
        agent_config=_agent_config(),
        chat_model=chat,
        mcp_clients=[],
        auth=OrchidAuthContext(access_token="t", tenant_key="x", user_id="u"),
        state={"messages": [HumanMessage(content="hi")]},
    )
    assert update is None  # caller falls through to the agent's normal flow


# ── Mini node emits started + finished ───────────────────────


def _send_payload(*, parent: str, mini_id: str, sub_task: dict, tool_subset: list[str]) -> dict:
    return {
        "auth_context": OrchidAuthContext(access_token="t", tenant_key="x", user_id="u"),
        "messages": [HumanMessage(content="x")],
        "_active_mini_parent": parent,
        "_active_mini_id": mini_id,
        "_active_mini_subtask": sub_task,
        "_active_mini_tool_subset": tool_subset,
    }


def _patch_loop_and_caps(monkeypatch, run_impl):
    """Stub the agentic loop + capability rendering for fast unit tests."""
    caps = MagicMock()
    caps.raw_tools = []
    caps.tool_client_map = {}
    caps.tool_annotations = {}
    monkeypatch.setattr(
        "orchid_ai.agents.mcp_dispatcher.MCPDispatcher.render_capabilities",
        AsyncMock(return_value=caps),
    )
    monkeypatch.setattr("orchid_ai.agents.tools.build_langchain_tools", lambda **_kw: [])

    class _StubLoop:
        def __init__(self, **kwargs):
            self._kwargs = kwargs

        async def run(self, _messages):
            return await run_impl(self._kwargs)

    monkeypatch.setattr("orchid_ai.agents.agentic_loop.AgenticLoop", _StubLoop)


@pytest.mark.asyncio
async def test_mini_node_emits_started_and_finished_on_success(monkeypatch):
    async def _run_impl(_kwargs):
        return ("done", {"lookup_user": "u-1"})

    _patch_loop_and_caps(monkeypatch, _run_impl)

    cfg = _agent_config()
    node = mini_agent_node_factory(parent_config=cfg, chat_model=MagicMock(), mcp_clients=[])

    update = await node(
        _send_payload(
            parent=cfg.name,
            mini_id="mini_0",
            sub_task={
                "id": "mini_0",
                "description": "lookup user X",
                "instruction": "x",
                "allowed_tools": [],
                "rationale": "r",
            },
            tool_subset=[],
        ),
    )

    msgs = update["messages"]
    assert len(msgs) == 2
    started = extract_event(msgs[0])
    finished = extract_event(msgs[1])
    assert started == (
        "mini_agent.started",
        {
            "parent": "support",
            "mini_id": "mini_0",
            "description": "lookup user X",
        },
    )
    assert finished is not None
    assert finished[0] == "mini_agent.finished"
    assert finished[1]["parent"] == "support"
    assert finished[1]["mini_id"] == "mini_0"
    assert finished[1]["status"] == "ok"
    # No error key on success.
    assert "error" not in finished[1]


@pytest.mark.asyncio
async def test_mini_node_emits_finished_with_error_on_failure(monkeypatch):
    async def _run_impl(_kwargs):
        raise RuntimeError("kaboom")

    _patch_loop_and_caps(monkeypatch, _run_impl)

    cfg = _agent_config()
    node = mini_agent_node_factory(parent_config=cfg, chat_model=MagicMock(), mcp_clients=[])

    update = await node(
        _send_payload(
            parent=cfg.name,
            mini_id="mini_boom",
            sub_task={
                "id": "mini_boom",
                "description": "boom",
                "instruction": "x",
                "allowed_tools": [],
                "rationale": "r",
            },
            tool_subset=[],
        ),
    )

    msgs = update["messages"]
    assert len(msgs) == 2
    assert extract_event(msgs[0])[0] == "mini_agent.started"
    name, data = extract_event(msgs[1])
    assert name == "mini_agent.finished"
    assert data["status"] == "failed"
    assert "kaboom" in data["error"]


# ── Aggregator emits aggregated event ─────────────────────────


def _outcome(*, mini_id, status, summary=None, description=None):
    return {
        "mini_id": mini_id,
        "sub_task_description": description or mini_id,
        "status": status,
        "summary": summary,
        "error": None,
        "duration_ms": 5,
        "tool_results": {},
    }


@pytest.mark.asyncio
async def test_aggregator_emits_aggregated_event_before_aimessage():
    cfg = _agent_config()
    chat = MagicMock()
    chat.ainvoke = AsyncMock(return_value=MagicMock(content="synthesis"))
    node = aggregator_node_factory(parent_config=cfg, chat_model=chat)

    state = {
        "messages": [HumanMessage(content="x")],
        "mini_agent_outcomes": {
            "support#mini_0": _outcome(mini_id="mini_0", status="ok", summary="A", description="alpha"),
            "support#mini_1": _outcome(mini_id="mini_1", status="failed", description="beta"),
        },
    }

    result = await node(state)
    msgs = result["messages"]
    assert len(msgs) == 2

    # Event first.
    name, data = extract_event(msgs[0])
    assert name == "mini_agent.aggregated"
    assert data["parent"] == "support"
    statuses = {o["status"] for o in data["outcomes"]}
    assert statuses == {"ok", "failed"}

    # AIMessage second.
    assert isinstance(msgs[1], AIMessage)
    assert msgs[1].content == "synthesis"


@pytest.mark.asyncio
async def test_aggregator_emits_aggregated_event_on_short_circuit():
    cfg = _agent_config()
    chat = MagicMock()
    chat.ainvoke = AsyncMock()
    node = aggregator_node_factory(parent_config=cfg, chat_model=chat)

    state = {
        "messages": [HumanMessage(content="x")],
        "mini_agent_outcomes": {
            "support#mini_0": _outcome(mini_id="mini_0", status="failed", description="alpha"),
            "support#mini_1": _outcome(mini_id="mini_1", status="failed", description="beta"),
        },
    }
    result = await node(state)
    chat.ainvoke.assert_not_called()
    msgs = result["messages"]
    assert extract_event(msgs[0])[0] == "mini_agent.aggregated"
    assert "Sorry" in msgs[1].content
