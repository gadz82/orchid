"""Regression tests for the H4 concurrency fix.

Two activations of the same :class:`OrchidAgent` instance, fired
concurrently via :func:`asyncio.gather`, must see independent
``_current_auth`` / ``_current_chat_id`` / ``_current_message_id``
bindings.  Before the ContextVar migration the four fields lived on
shared instance attributes and races corrupted them under
LangGraph ``Send()`` fan-out.

Also covers the ``SkillExecutor`` depth save/restore bug: the prior
implementation set ``self._skill_depth = 0`` in ``finally``, so a
two-level nested skill chain unwound to depth ``0`` mid-chain and
the depth guard could be bypassed.
"""

from __future__ import annotations

import asyncio

import pytest

from orchid_ai.agents.skill_executor import (
    MAX_AGENT_SKILL_DEPTH,
    SkillExecutor,
    _skill_depth_var,
)
from orchid_ai.core.agent import OrchidAgent, OrchidAgentRunContext
from orchid_ai.core.state import OrchidAuthContext


class _ProbeAgent(OrchidAgent):
    """Agent whose ``run`` reads the current run context after a sleep.

    The sleep forces the two ``gather``'d tasks to overlap inside
    ``run`` — without ContextVar isolation, whichever task set its
    context last would win for both.
    """

    @property
    def name(self) -> str:
        return "probe"

    @property
    def description(self) -> str:
        return "x"

    async def run(self, state):  # type: ignore[override]
        await asyncio.sleep(0.01)
        return {
            "auth": self._current_auth,
            "chat_id": self._current_chat_id,
            "message_id": self._current_message_id,
            "correlation_id": self._current_correlation_id,
        }


async def _activate(agent: _ProbeAgent, ctx: OrchidAgentRunContext) -> dict:
    token = agent.set_run_context(ctx)
    try:
        return await agent.run({})  # type: ignore[arg-type]
    finally:
        agent.reset_run_context(token)


class TestConcurrentRunContext:
    async def test_two_concurrent_activations_do_not_cross_contaminate(self) -> None:
        agent = _ProbeAgent(reader=None)  # type: ignore[arg-type]
        ctx_a = OrchidAgentRunContext(
            auth=OrchidAuthContext(access_token="ta", tenant_key="t-a", user_id="u-a"),
            chat_id="C-A",
            message_id="m-a",
            correlation_id="corr-a",
        )
        ctx_b = OrchidAgentRunContext(
            auth=OrchidAuthContext(access_token="tb", tenant_key="t-b", user_id="u-b"),
            chat_id="C-B",
            message_id="m-b",
            correlation_id="corr-b",
        )
        result_a, result_b = await asyncio.gather(
            _activate(agent, ctx_a),
            _activate(agent, ctx_b),
        )
        assert result_a["chat_id"] == "C-A"
        assert result_a["message_id"] == "m-a"
        assert result_a["correlation_id"] == "corr-a"
        assert result_a["auth"].tenant_key == "t-a"
        assert result_b["chat_id"] == "C-B"
        assert result_b["message_id"] == "m-b"
        assert result_b["correlation_id"] == "corr-b"
        assert result_b["auth"].tenant_key == "t-b"

    async def test_reset_restores_previous_binding(self) -> None:
        agent = _ProbeAgent(reader=None)  # type: ignore[arg-type]
        outer = OrchidAgentRunContext(chat_id="outer")
        outer_token = agent.set_run_context(outer)
        try:
            inner = OrchidAgentRunContext(chat_id="inner")
            inner_token = agent.set_run_context(inner)
            try:
                assert agent._current_chat_id == "inner"
            finally:
                agent.reset_run_context(inner_token)
            assert agent._current_chat_id == "outer"
        finally:
            agent.reset_run_context(outer_token)

    async def test_default_run_context_is_empty(self) -> None:
        agent = _ProbeAgent(reader=None)  # type: ignore[arg-type]
        assert agent._current_auth is None
        assert agent._current_chat_id is None
        assert agent._current_message_id is None
        assert agent._current_correlation_id is None

    async def test_setter_proxies_to_contextvar(self) -> None:
        """Legacy fixture pattern (``agent._current_auth = X``) keeps working."""
        agent = _ProbeAgent(reader=None)  # type: ignore[arg-type]
        agent._current_chat_id = "C-fixture"
        assert agent._current_chat_id == "C-fixture"

        async def reader() -> str | None:
            return agent._current_chat_id

        assert await reader() == "C-fixture"


class TestSkillExecutorDepth:
    """Ensures the depth counter is task-local and restores on unwind."""

    def _make_executor(self, agent_name: str, peers: dict | None = None) -> SkillExecutor:
        return SkillExecutor(
            agent_name=agent_name,
            mcp_dispatcher=None,
            builtin_tool_caller=lambda *a, **k: None,  # type: ignore[arg-type]
            agent_peers=peers or {},
        )

    async def test_depth_starts_at_zero(self) -> None:
        executor = self._make_executor("a")
        assert executor._skill_depth == 0

    async def test_depth_restores_to_prev_not_zero_after_nested_call(self) -> None:
        """Reproduces the pre-fix bug: two-level nesting must unwind to depth 1, not 0."""
        # Simulate two-level nesting: depth was 1 before the call;
        # after the call (and its finally) it must be back to 1.
        outer_token = _skill_depth_var.set(1)
        try:
            inner_token = _skill_depth_var.set(_skill_depth_var.get() + 1)
            try:
                assert _skill_depth_var.get() == 2
            finally:
                _skill_depth_var.reset(inner_token)
            assert _skill_depth_var.get() == 1
        finally:
            _skill_depth_var.reset(outer_token)

    async def test_concurrent_skill_chains_do_not_share_depth(self) -> None:
        """Two parallel skill chains must each have their own counter."""

        async def chain(depth_setter: int) -> int:
            token = _skill_depth_var.set(depth_setter)
            try:
                await asyncio.sleep(0.01)
                return _skill_depth_var.get()
            finally:
                _skill_depth_var.reset(token)

        a, b = await asyncio.gather(chain(2), chain(3))
        assert a == 2
        assert b == 3

    async def test_max_depth_guard_uses_current_task_depth(self) -> None:
        """The guard inside ``_run_agent_step`` must read the task-local value."""

        class _StubPeer:
            async def run(self, state):
                return {"messages": [], "mcp_context": {}}

        executor = self._make_executor("a", peers={"peer": _StubPeer()})

        # Force depth to the max via the ContextVar — the next call
        # should raise without poking instance attributes.
        token = _skill_depth_var.set(MAX_AGENT_SKILL_DEPTH)
        try:
            with pytest.raises(RecursionError, match="Agent skill depth exceeded"):
                await executor._run_agent_step(
                    "peer",
                    instruction="do thing",
                    query="q",
                    auth=OrchidAuthContext(access_token="t", tenant_key="t", user_id="u"),
                    previous_results={},
                )
        finally:
            _skill_depth_var.reset(token)
