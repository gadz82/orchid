"""Tests for the graph-builder wiring and the fork router.

Covers:
  - T7  — ``should_fork=False`` → no extra ``_mini``/``_aggregator``
          nodes invoked; supervisor sees the parent's normal flow.
  - T8  — ``should_fork=True`` with N sub-tasks → ``_make_fork_router``
          returns N ``Send`` payloads pointed at ``{name}_mini``.
  - T20 — non-opt-in agents have ZERO ``_mini``/``_aggregator``
          nodes in the compiled graph.

Plus fork-router fallthrough cases (no decision; empty sub_tasks).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langgraph.types import Send

from orchid_ai.config.schema import (
    OrchidAgentConfig,
    OrchidAgentsConfig,
    OrchidLLMConfig,
    OrchidMiniAgentConfig,
    OrchidRAGConfig,
)
from orchid_ai.graph.graph import _make_fork_router, build_graph
from orchid_ai.runtime import OrchidRuntime

# ── Fork router (pure function — no graph) ─────────────────────


class TestForkRouter:
    def test_no_decision_returns_supervisor(self):
        router = _make_fork_router("support")
        assert router({}) == "supervisor"

    def test_should_fork_false_returns_supervisor(self):
        router = _make_fork_router("support")
        state = {
            "mini_agent_decisions": {
                "support": {"should_fork": False, "sub_tasks": []},
            },
        }
        assert router(state) == "supervisor"

    def test_should_fork_true_no_subtasks_returns_supervisor(self):
        """Defensive: should_fork=True but sub_tasks empty → fallback."""
        router = _make_fork_router("support")
        state = {
            "mini_agent_decisions": {
                "support": {"should_fork": True, "sub_tasks": []},
            },
        }
        assert router(state) == "supervisor"

    def test_other_parents_decision_ignored(self):
        """Decisions for other parents must NOT trigger this router."""
        router = _make_fork_router("support")
        state = {
            "mini_agent_decisions": {
                "other": {
                    "should_fork": True,
                    "sub_tasks": [
                        {"id": "mini_0", "description": "x", "instruction": "x", "allowed_tools": []},
                    ],
                },
            },
        }
        assert router(state) == "supervisor"

    def test_should_fork_true_returns_sends(self):
        """T8 — N sub-tasks → N ``Send`` instances, all targeting the parent's _mini node."""
        router = _make_fork_router("support")
        state = {
            "auth_context": MagicMock(),
            "messages": [],
            "mini_agent_decisions": {
                "support": {
                    "should_fork": True,
                    "sub_tasks": [
                        {
                            "id": "mini_0",
                            "description": "lookup user",
                            "instruction": "find user",
                            "allowed_tools": ["lookup_user"],
                            "rationale": "r",
                            "resolved_tool_subset": ["lookup_user"],
                        },
                        {
                            "id": "mini_1",
                            "description": "lookup orders",
                            "instruction": "list orders",
                            "allowed_tools": ["lookup_order"],
                            "rationale": "r",
                            "resolved_tool_subset": ["lookup_order"],
                        },
                        {
                            "id": "mini_2",
                            "description": "lookup invoices",
                            "instruction": "list invoices",
                            "allowed_tools": ["lookup_invoice"],
                            "rationale": "r",
                            "resolved_tool_subset": ["lookup_invoice"],
                        },
                    ],
                },
            },
        }

        sends = router(state)
        assert isinstance(sends, list)
        assert len(sends) == 3
        for send in sends:
            assert isinstance(send, Send)
            assert send.node == "support_mini"

        # Each Send carries the per-mini sentinel keys.
        ids = {s.arg.get("_active_mini_id") for s in sends}
        assert ids == {"mini_0", "mini_1", "mini_2"}
        parents = {s.arg.get("_active_mini_parent") for s in sends}
        assert parents == {"support"}
        # Tool subsets propagated.
        for send in sends:
            payload = send.arg
            sub_task = payload["_active_mini_subtask"]
            assert payload["_active_mini_tool_subset"] == sub_task["resolved_tool_subset"]


# ── Graph topology — opt-in vs opt-out ─────────────────────────


def _build_minimal_runtime() -> OrchidRuntime:
    chat = MagicMock(spec=BaseChatModel)
    chat.bind_tools = MagicMock(return_value=chat)
    return OrchidRuntime(default_model="gemini/test", chat_model=chat)


@pytest.mark.asyncio
async def test_non_opt_in_agent_has_no_extra_nodes():
    """T20 — agents without ``mini_agent.enabled`` keep today's wiring."""
    config = OrchidAgentsConfig(
        agents={
            "plain": OrchidAgentConfig(
                description="plain",
                prompt="prompt",
                rag=OrchidRAGConfig(enabled=False),
                llm=OrchidLLMConfig(model="gemini/test"),
                # mini_agent stays at default (enabled=False).
            ),
        },
    )
    compiled = build_graph(config=config, runtime=_build_minimal_runtime())
    nodes = set(compiled.get_graph().nodes)
    assert "plain_agent" in nodes
    assert "plain_mini" not in nodes
    assert "plain_aggregator" not in nodes


@pytest.mark.asyncio
async def test_opt_in_agent_has_three_nodes():
    """An agent with ``mini_agent.enabled=true`` has _agent + _mini + _aggregator."""
    config = OrchidAgentsConfig(
        agents={
            "support": OrchidAgentConfig(
                description="support",
                prompt="prompt",
                rag=OrchidRAGConfig(enabled=False),
                llm=OrchidLLMConfig(model="gemini/test"),
                mini_agent=OrchidMiniAgentConfig(enabled=True),
            ),
        },
    )
    compiled = build_graph(config=config, runtime=_build_minimal_runtime())
    nodes = set(compiled.get_graph().nodes)
    assert "support_agent" in nodes
    assert "support_mini" in nodes
    assert "support_aggregator" in nodes


@pytest.mark.asyncio
async def test_mixed_topology():
    """Two agents — one opted in, one not — coexist in the same graph."""
    config = OrchidAgentsConfig(
        agents={
            "plain": OrchidAgentConfig(
                description="plain",
                prompt="prompt",
                rag=OrchidRAGConfig(enabled=False),
                llm=OrchidLLMConfig(model="gemini/test"),
            ),
            "support": OrchidAgentConfig(
                description="support",
                prompt="prompt",
                rag=OrchidRAGConfig(enabled=False),
                llm=OrchidLLMConfig(model="gemini/test"),
                mini_agent=OrchidMiniAgentConfig(enabled=True),
            ),
        },
    )
    compiled = build_graph(config=config, runtime=_build_minimal_runtime())
    nodes = set(compiled.get_graph().nodes)

    # Plain agent: only _agent.
    assert "plain_agent" in nodes
    assert "plain_mini" not in nodes
    assert "plain_aggregator" not in nodes

    # Support agent: full triple.
    assert "support_agent" in nodes
    assert "support_mini" in nodes
    assert "support_aggregator" in nodes
