"""Tests for ``mini_agent_node_factory``.

Covers:
  - T10 — tool subset enforcement: ``bind_tools`` only sees ``allowed_tools``.
  - T13 — timeout: a mini exceeding ``timeout_seconds`` produces
          ``status="timeout"``; cancellation propagates.
  - T17 — mini-of-a-mini guard: the mini node bypasses
          ``GenericAgent.run()`` entirely so the decomposer LLM is
          never invoked from within a mini's execution.

Plus failure-path coverage (generic exception → ``status="failed"``)
and the "missing auth_context" defensive branch.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchid_ai.agents.mini_agent_node import (
    MiniAgentRuntimeError,
    mini_agent_node_factory,
)
from orchid_ai.config.schema import (
    OrchidAgentConfig,
    OrchidLLMConfig,
    OrchidMiniAgentConfig,
    OrchidRAGConfig,
)
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.core.run_config import with_auth


# ── Helpers ────────────────────────────────────────────────────


def _parent_config(
    *,
    name: str = "support",
    timeout: int = 60,
) -> OrchidAgentConfig:
    return OrchidAgentConfig(
        name=name,
        description=f"{name} agent",
        prompt="be helpful",
        rag=OrchidRAGConfig(enabled=False),
        llm=OrchidLLMConfig(model="gemini/test"),
        mini_agent=OrchidMiniAgentConfig(enabled=True, timeout_seconds=timeout),
    )


def _send_payload(state: dict, *, parent: str, mini_id: str, sub_task: dict, tool_subset: list[str]) -> dict:
    return {
        **state,
        "_active_mini_parent": parent,
        "_active_mini_id": mini_id,
        "_active_mini_subtask": sub_task,
        "_active_mini_tool_subset": tool_subset,
    }


def _auth() -> OrchidAuthContext:
    return OrchidAuthContext(access_token="t", tenant_key="x", user_id="u")


def _patch_loop(monkeypatch, run_impl):
    """Patch ``AgenticLoop`` so the mini node uses our stubbed loop.

    ``run_impl`` is an awaitable callable returning ``(final_text, tool_results)``.
    Returns a list collecting every ``__init__`` kwargs dict for assertions.
    """
    init_calls: list[dict[str, Any]] = []

    class _StubLoop:
        def __init__(self, **kwargs):
            init_calls.append(kwargs)
            self._kwargs = kwargs

        async def run(self, _messages):
            return await run_impl(self._kwargs)

    monkeypatch.setattr("orchid_ai.agents.agentic_loop.AgenticLoop", _StubLoop)
    return init_calls


def _patch_capability_rendering(monkeypatch, raw_tools=None):
    """Stub ``MCPDispatcher.render_capabilities`` to avoid network."""
    raw = raw_tools or []
    caps = MagicMock()
    caps.raw_tools = raw
    caps.tool_client_map = {t["name"]: (MagicMock(), MagicMock()) for t in raw}
    caps.tool_annotations = {}

    async def _fake(*args, **kwargs):
        return caps

    monkeypatch.setattr(
        "orchid_ai.agents.mcp_dispatcher.MCPDispatcher.render_capabilities",
        AsyncMock(side_effect=_fake),
    )
    return caps


def _patch_build_langchain_tools(monkeypatch, tool_names: list[str]):
    """Stub ``build_langchain_tools`` to return simple stand-ins."""

    def _fake(*, builtin_names, builtin_tool_defs, mcp_tool_defs, mcp_tool_client_map, **kwargs):
        names = [td["function"]["name"] for td in (builtin_tool_defs + mcp_tool_defs)]
        results = []
        for name in names:
            t = MagicMock()
            t.name = name
            t.requires_approval = False
            t.ainvoke = AsyncMock(return_value=f"result_{name}")
            results.append(t)
        return results

    monkeypatch.setattr("orchid_ai.agents.tools.build_langchain_tools", _fake)


# ── T10: tool subset enforcement ──────────────────────────────


@pytest.mark.asyncio
async def test_tool_subset_filters_inventory(monkeypatch):
    """Loop receives only the tools listed in ``_active_mini_tool_subset``."""
    _patch_capability_rendering(
        monkeypatch,
        raw_tools=[
            {"name": "lookup_user", "description": "find user", "schema": {}, "annotations": None},
            {"name": "lookup_order", "description": "find order", "schema": {}, "annotations": None},
            {"name": "delete_record", "description": "delete", "schema": {}, "annotations": None},
        ],
    )
    _patch_build_langchain_tools(monkeypatch, ["lookup_user", "lookup_order", "delete_record"])

    async def _run_impl(kwargs):
        return ("done", {"lookup_user": "u-1"})

    init_calls = _patch_loop(monkeypatch, _run_impl)

    cfg = _parent_config()
    chat_model = MagicMock()
    node = mini_agent_node_factory(parent_config=cfg, chat_model=chat_model, mcp_clients=[])

    state = {"auth_context": _auth(), "messages": []}
    payload = _send_payload(
        state,
        parent=cfg.name,
        mini_id="mini_0",
        sub_task={
            "id": "mini_0",
            "description": "fetch user",
            "instruction": "find the user",
            "allowed_tools": ["lookup_user"],
            "rationale": "r",
        },
        tool_subset=["lookup_user"],
    )

    update = await node(payload, config=with_auth(payload.get("auth_context")))

    # Loop saw exactly one tool def — the subset.
    assert len(init_calls) == 1
    tool_defs = init_calls[0]["all_tool_defs"]
    assert {td["function"]["name"] for td in tool_defs} == {"lookup_user"}
    # tool_map filtered identically.
    assert set(init_calls[0]["tool_map"].keys()) == {"lookup_user"}
    # is_mini flag flipped.
    assert init_calls[0]["is_mini"] is True
    # tool_subset propagated.
    assert init_calls[0]["tool_subset"] == ["lookup_user"]

    # Outcome shape.
    outcome_slot = update["mini_agent_outcomes"][f"{cfg.name}#mini_0"]
    assert outcome_slot["status"] == "ok"
    assert outcome_slot["summary"] == "done"
    assert outcome_slot["tool_results"] == {"lookup_user": "u-1"}


# ── T13: timeout handling ──────────────────────────────────────


@pytest.mark.asyncio
async def test_timeout_sets_status_timeout(monkeypatch):
    """A mini exceeding ``timeout_seconds`` records ``status="timeout"``."""
    _patch_capability_rendering(monkeypatch)
    _patch_build_langchain_tools(monkeypatch, [])

    async def _run_impl(kwargs):
        await asyncio.sleep(5)  # well past the 1s timeout below
        return ("never", {})

    _patch_loop(monkeypatch, _run_impl)

    cfg = _parent_config(timeout=5)
    # OrchidMiniAgentConfig validator forbids timeout < 5 — work around
    # by reaching past Pydantic and assigning directly to the instance.
    object.__setattr__(cfg.mini_agent, "timeout_seconds", 1)
    node = mini_agent_node_factory(parent_config=cfg, chat_model=MagicMock(), mcp_clients=[])

    payload = _send_payload(
        {"auth_context": _auth(), "messages": []},
        parent=cfg.name,
        mini_id="mini_slow",
        sub_task={"id": "mini_slow", "description": "slow", "instruction": "x", "allowed_tools": [], "rationale": "r"},
        tool_subset=[],
    )

    update = await node(payload, config=with_auth(payload.get("auth_context")))
    outcome = update["mini_agent_outcomes"][f"{cfg.name}#mini_slow"]
    assert outcome["status"] == "timeout"
    assert "timed out" in (outcome["error"] or "")
    assert outcome["summary"] is None


# ── Failure path: arbitrary exception → status=failed ─────────


@pytest.mark.asyncio
async def test_unhandled_exception_sets_status_failed(monkeypatch):
    _patch_capability_rendering(monkeypatch)
    _patch_build_langchain_tools(monkeypatch, [])

    async def _run_impl(_kwargs):
        raise RuntimeError("kaboom")

    _patch_loop(monkeypatch, _run_impl)

    cfg = _parent_config()
    node = mini_agent_node_factory(parent_config=cfg, chat_model=MagicMock(), mcp_clients=[])

    payload = _send_payload(
        {"auth_context": _auth(), "messages": []},
        parent=cfg.name,
        mini_id="mini_boom",
        sub_task={"id": "mini_boom", "description": "boom", "instruction": "x", "allowed_tools": [], "rationale": "r"},
        tool_subset=[],
    )
    update = await node(payload, config=with_auth(payload.get("auth_context")))
    outcome = update["mini_agent_outcomes"][f"{cfg.name}#mini_boom"]
    assert outcome["status"] == "failed"
    assert "kaboom" in (outcome["error"] or "")


# ── T17: mini-of-a-mini guard ─────────────────────────────────


@pytest.mark.asyncio
async def test_mini_node_does_not_invoke_decomposer(monkeypatch):
    """Defensive guard against mini-of-a-mini.

    The mini node calls ``AgenticLoop`` directly and never goes
    through ``GenericAgent.run()``, so even if the parent has
    ``mini_agent.enabled=True`` no decomposer LLM call fires inside
    a mini's execution.  We assert this by patching the decomposer
    module and confirming it's untouched.
    """
    _patch_capability_rendering(monkeypatch)
    _patch_build_langchain_tools(monkeypatch, [])

    decomposer_calls = {"count": 0}

    class _SpyDecomposer:
        def __init__(self, **kwargs):
            decomposer_calls["count"] += 1

        async def decompose(self, **kwargs):
            decomposer_calls["count"] += 1
            return MagicMock()

    monkeypatch.setattr(
        "orchid_ai.agents.mini_agent_decomposer.MiniAgentDecomposer",
        _SpyDecomposer,
    )

    async def _run_impl(_kwargs):
        return ("nested-task-output", {})

    _patch_loop(monkeypatch, _run_impl)

    cfg = _parent_config()
    node = mini_agent_node_factory(parent_config=cfg, chat_model=MagicMock(), mcp_clients=[])

    payload = _send_payload(
        {"auth_context": _auth(), "messages": []},
        parent=cfg.name,
        mini_id="mini_nested",
        sub_task={"id": "mini_nested", "description": "x", "instruction": "x", "allowed_tools": [], "rationale": "r"},
        tool_subset=[],
    )
    update = await node(payload, config=with_auth(payload.get("auth_context")))

    assert update["mini_agent_outcomes"][f"{cfg.name}#mini_nested"]["status"] == "ok"
    # Decomposer was NEVER instantiated or invoked from inside the mini.
    assert decomposer_calls["count"] == 0


# ── Defensive: missing auth context ───────────────────────────


@pytest.mark.asyncio
async def test_missing_auth_marks_failed(monkeypatch):
    """The mini node refuses to run without ``auth_context`` and records
    a failed outcome rather than crashing."""
    _patch_capability_rendering(monkeypatch)
    _patch_build_langchain_tools(monkeypatch, [])

    async def _run_impl(_kwargs):
        return ("never", {})

    _patch_loop(monkeypatch, _run_impl)

    cfg = _parent_config()
    node = mini_agent_node_factory(parent_config=cfg, chat_model=MagicMock(), mcp_clients=[])

    payload = _send_payload(
        {"messages": []},  # no auth_context
        parent=cfg.name,
        mini_id="mini_0",
        sub_task={"id": "mini_0", "description": "x", "instruction": "x", "allowed_tools": [], "rationale": "r"},
        tool_subset=[],
    )
    update = await node(payload, config=with_auth(payload.get("auth_context")))
    outcome = update["mini_agent_outcomes"][f"{cfg.name}#mini_0"]
    assert outcome["status"] == "failed"
    assert "auth_context" in (outcome["error"] or "")


# ── Cross-parent mis-routing is rejected ──────────────────────


@pytest.mark.asyncio
async def test_cross_parent_send_rejected(monkeypatch):
    """If a Send accidentally targets the wrong parent's node, refuse
    to run rather than corrupt the outcomes slot."""
    _patch_capability_rendering(monkeypatch)
    _patch_build_langchain_tools(monkeypatch, [])

    async def _run_impl(_kwargs):
        return ("never", {})

    _patch_loop(monkeypatch, _run_impl)

    cfg = _parent_config(name="support")
    node = mini_agent_node_factory(parent_config=cfg, chat_model=MagicMock(), mcp_clients=[])

    payload = _send_payload(
        {"auth_context": _auth(), "messages": []},
        parent="other_agent",  # wrong!
        mini_id="mini_0",
        sub_task={"id": "mini_0", "description": "x", "instruction": "x", "allowed_tools": [], "rationale": "r"},
        tool_subset=[],
    )
    with pytest.raises(MiniAgentRuntimeError):
        await node(payload)
