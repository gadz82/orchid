"""HITL through a mini-agent.

The critical contract: when a mini calls ``interrupt()`` (because
its tool has ``requires_approval=True``), LangGraph raises
``GraphInterrupt`` from inside the agentic loop.  This MUST escape
the mini node's broad ``except Exception`` block so LangGraph's
runtime can pause the graph; otherwise the suspension is silently
converted into a ``status="failed"`` outcome and the frontend never
gets to ask the user.

This test does NOT spin up a full compiled graph — that would require
fully-working ``LangGraph + checkpointer + Command(resume=...)``
plumbing which the existing test suite does not exercise either.
Instead we drive the mini node directly with a stubbed loop that
raises a ``GraphInterrupt`` and assert it propagates unchanged —
which is exactly what LangGraph's runtime needs to suspend the run.

The "resume → aggregator runs after both minis complete" half is
structurally guaranteed by LangGraph's join semantics on ``Send``
fan-out (the aggregator only fires once every mini has recorded an
outcome) and is exercised end-to-end in the streaming integration test.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.errors import GraphInterrupt

from orchid_ai.agents.mini_agent_node import mini_agent_node_factory
from orchid_ai.config.schema import (
    OrchidAgentConfig,
    OrchidLLMConfig,
    OrchidMiniAgentConfig,
    OrchidRAGConfig,
)
from orchid_ai.core.state import OrchidAuthContext


def _parent_config() -> OrchidAgentConfig:
    return OrchidAgentConfig(
        name="support",
        description="support",
        prompt="be helpful",
        rag=OrchidRAGConfig(enabled=False),
        llm=OrchidLLMConfig(model="gemini/test"),
        mini_agent=OrchidMiniAgentConfig(enabled=True),
    )


def _patch_loop(monkeypatch, run_impl):
    class _StubLoop:
        def __init__(self, **kwargs):
            self._kwargs = kwargs

        async def run(self, _messages):
            return await run_impl(self._kwargs)

    monkeypatch.setattr("orchid_ai.agents.agentic_loop.AgenticLoop", _StubLoop)


def _patch_capability_rendering(monkeypatch):
    caps = MagicMock()
    caps.raw_tools = []
    caps.tool_client_map = {}
    caps.tool_annotations = {}
    monkeypatch.setattr(
        "orchid_ai.agents.mcp_dispatcher.MCPDispatcher.render_capabilities",
        AsyncMock(return_value=caps),
    )


def _patch_build_langchain_tools(monkeypatch):
    monkeypatch.setattr(
        "orchid_ai.agents.tools.build_langchain_tools",
        lambda **_kw: [],
    )


def _payload(parent: str, mini_id: str) -> dict:
    return {
        "auth_context": OrchidAuthContext(access_token="t", tenant_key="x", user_id="u"),
        "messages": [],
        "_active_mini_parent": parent,
        "_active_mini_id": mini_id,
        "_active_mini_subtask": {
            "id": mini_id,
            "description": f"sub-task {mini_id}",
            "instruction": "x",
            "allowed_tools": [],
            "rationale": "r",
        },
        "_active_mini_tool_subset": [],
    }


# ── T14 — interrupt() propagates out of the mini node ─────────


@pytest.mark.asyncio
async def test_graph_interrupt_propagates_through_mini_node(monkeypatch):
    """An ``interrupt()`` call inside the agentic loop must NOT be
    converted into a ``status="failed"`` outcome.  It must surface as
    a ``GraphInterrupt`` so LangGraph's runtime can pause the graph.
    """
    _patch_capability_rendering(monkeypatch)
    _patch_build_langchain_tools(monkeypatch)

    async def _run_impl(_kwargs):
        # Mimic exactly what ``langgraph.types.interrupt`` does on
        # first call: raise ``GraphInterrupt`` with the approval payload.
        raise GraphInterrupt(
            (
                {
                    "type": "tool_approval",
                    "tool": "delete_record",
                    "args": {"id": "u-1"},
                    "agent": "support.mini_0",
                },
            )
        )

    _patch_loop(monkeypatch, _run_impl)

    cfg = _parent_config()
    node = mini_agent_node_factory(parent_config=cfg, chat_model=MagicMock(), mcp_clients=[])

    with pytest.raises(GraphInterrupt) as excinfo:
        await node(_payload("support", "mini_0"))

    # The interrupt's payload survives the round-trip — LangGraph
    # would attach this to the suspended state for the frontend.
    interrupt_payload = excinfo.value.args[0]
    assert isinstance(interrupt_payload, tuple)
    approval = interrupt_payload[0]
    assert approval["type"] == "tool_approval"
    assert approval["tool"] == "delete_record"


@pytest.mark.asyncio
async def test_non_interrupt_exception_still_recorded_as_failed(monkeypatch):
    """Sanity check the special case is NARROW — a regular
    ``RuntimeError`` from inside the loop still gets converted to a
    ``failed`` outcome (the critical behaviour for non-HITL errors).
    """
    _patch_capability_rendering(monkeypatch)
    _patch_build_langchain_tools(monkeypatch)

    async def _run_impl(_kwargs):
        raise RuntimeError("normal failure")

    _patch_loop(monkeypatch, _run_impl)

    cfg = _parent_config()
    node = mini_agent_node_factory(parent_config=cfg, chat_model=MagicMock(), mcp_clients=[])

    update = await node(_payload("support", "mini_0"))
    outcome = update["mini_agent_outcomes"]["support#mini_0"]
    assert outcome["status"] == "failed"
    assert "normal failure" in (outcome["error"] or "")
