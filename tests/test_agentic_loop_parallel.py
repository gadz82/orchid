"""Parallel tool-call dispatch within one agentic round.

All tests drive ``AgenticLoop._dispatch_tool_calls`` directly so we
can observe ``asyncio.gather`` invocations, ``ToolMessage`` ordering,
and HITL serialisation without mocking the LLM.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import ToolMessage

from orchid_ai.agents.agentic_loop import AgenticLoop, _is_parallel_safe


class _StubTool:
    """Minimal LangChain ``BaseTool`` stand-in.

    We avoid pydantic's strict ``BaseTool`` here — the loop only needs
    ``name``, ``requires_approval``, and ``ainvoke``.  ``ainvoke`` may
    sleep, raise, or return a string — each test plugs in its own.
    """

    def __init__(
        self,
        *,
        name: str,
        sleep: float = 0.0,
        result: str | None = None,
        raise_exc: BaseException | None = None,
        requires_approval: bool = False,
    ) -> None:
        self.name = name
        self.requires_approval = requires_approval
        self._sleep = sleep
        self._result = result if result is not None else f"result_{name}"
        self._raise = raise_exc
        self.calls: list[dict[str, Any]] = []
        self.invocation_times: list[float] = []

    async def ainvoke(self, args: dict[str, Any]) -> str:
        self.invocation_times.append(time.perf_counter())
        self.calls.append(dict(args))
        if self._sleep:
            await asyncio.sleep(self._sleep)
        if self._raise is not None:
            raise self._raise
        return self._result


def _make_loop(
    *,
    tool_map: dict[str, _StubTool],
    parallel_safety: dict[str, bool] | None,
) -> AgenticLoop:
    """Construct an ``AgenticLoop`` with a no-op chat model."""
    chat_model = AsyncMock()
    chat_model.bind_tools = lambda _defs: chat_model  # noqa: E731 — terse stub
    return AgenticLoop(
        agent_name="test_agent",
        chat_model=chat_model,
        tool_map=tool_map,
        all_tool_defs=[],
        parallel_safety=parallel_safety,
    )


def _tc(name: str, args: dict[str, Any] | None = None, tc_id: str | None = None) -> dict[str, Any]:
    """Shape an LLM ``tool_call`` payload."""
    return {"name": name, "args": args or {}, "id": tc_id or f"call_{name}"}


# ── Helper-level tests ──


class TestIsParallelSafeHelper:
    """Pure lookup against the resolved safety map."""

    def test_none_map_means_serial(self):
        assert _is_parallel_safe("anything", None) is False

    def test_missing_key_returns_false(self):
        assert _is_parallel_safe("absent", {"present": True}) is False

    def test_explicit_true(self):
        assert _is_parallel_safe("present", {"present": True}) is True

    def test_explicit_false(self):
        assert _is_parallel_safe("present", {"present": False}) is False


# ── test 1 — both safe → gather ──


@pytest.mark.asyncio
async def test_two_parallel_safe_tools_run_concurrently():
    """T1: two parallel-safe tools → gathered in one shot, in original order."""
    tool_a = _StubTool(name="lookup_a", sleep=0.10, result="A")
    tool_b = _StubTool(name="lookup_b", sleep=0.10, result="B")
    loop = _make_loop(
        tool_map={"lookup_a": tool_a, "lookup_b": tool_b},
        parallel_safety={"lookup_a": True, "lookup_b": True},
    )

    messages: list = []
    started = time.perf_counter()
    await loop._dispatch_tool_calls(
        [_tc("lookup_a"), _tc("lookup_b")],
        messages,
        round_num=0,
    )
    elapsed = time.perf_counter() - started

    # Both tools executed exactly once.
    assert len(tool_a.calls) == 1
    assert len(tool_b.calls) == 1
    # Concurrency proof: total wall-clock < sum of individual sleeps.
    # Two 100ms sleeps run serially would take ≥ 0.2s; parallel ≈ 0.1s.
    assert elapsed < 0.18, f"Expected concurrent execution, got {elapsed:.3f}s"
    # ToolMessages appended in original order.
    assert len(messages) == 2
    assert isinstance(messages[0], ToolMessage)
    assert messages[0].content == "A"
    assert messages[0].tool_call_id == "call_lookup_a"
    assert messages[1].content == "B"
    assert messages[1].tool_call_id == "call_lookup_b"


# ── test 2 — mixed safe + unsafe ──


@pytest.mark.asyncio
async def test_mixed_safe_and_unsafe_partitions_correctly():
    """T2: safe ones gather, unsafe runs sequentially after."""
    safe_a = _StubTool(name="safe_a", sleep=0.05, result="A")
    safe_b = _StubTool(name="safe_b", sleep=0.05, result="B")
    unsafe = _StubTool(name="unsafe", sleep=0.05, result="C")
    loop = _make_loop(
        tool_map={"safe_a": safe_a, "safe_b": safe_b, "unsafe": unsafe},
        parallel_safety={"safe_a": True, "safe_b": True, "unsafe": False},
    )

    messages: list = []
    await loop._dispatch_tool_calls(
        [_tc("safe_a"), _tc("unsafe"), _tc("safe_b")],
        messages,
        round_num=0,
    )

    # The unsafe tool started AFTER both safe tools' invocations
    # (because the parallel batch runs first, then the sequential tail).
    safe_started_max = max(safe_a.invocation_times[0], safe_b.invocation_times[0])
    assert unsafe.invocation_times[0] > safe_started_max

    # Original order preserved when appending ToolMessages.
    assert [m.content for m in messages] == ["A", "C", "B"]
    assert [m.tool_call_id for m in messages] == ["call_safe_a", "call_unsafe", "call_safe_b"]


# ── test 3 — one safe call raises in the gather ──


@pytest.mark.asyncio
async def test_safe_call_raises_others_complete():
    """T3: one parallel leg raises → loop continues, error becomes [Tool error] msg."""
    boom = _StubTool(name="boom", raise_exc=RuntimeError("kaboom"))
    ok = _StubTool(name="ok", result="fine")
    loop = _make_loop(
        tool_map={"boom": boom, "ok": ok},
        parallel_safety={"boom": True, "ok": True},
    )

    messages: list = []
    await loop._dispatch_tool_calls(
        [_tc("boom"), _tc("ok")],
        messages,
        round_num=0,
    )

    assert len(messages) == 2
    # Error rendered into a ToolMessage in original order.
    assert messages[0].tool_call_id == "call_boom"
    assert messages[0].content.startswith("[Tool error]")
    assert "kaboom" in messages[0].content
    # Successful leg landed normally.
    assert messages[1].tool_call_id == "call_ok"
    assert messages[1].content == "fine"


# ── test 4 — approval serialises whole round ──


@pytest.mark.asyncio
async def test_requires_approval_round_runs_serially(monkeypatch):
    """T4: any approval-tool in the round → no gather; interrupt fires."""
    approval = _StubTool(name="delete_record", result="deleted", requires_approval=True)
    safe_a = _StubTool(name="lookup_a", sleep=0.05, result="A")
    safe_b = _StubTool(name="lookup_b", sleep=0.05, result="B")

    # Patch ``asyncio.gather`` so we can prove it is NEVER invoked when
    # a round contains an approval tool.  Counter increments on every
    # call site reachable from this test's event loop.
    gather_calls = {"count": 0}
    real_gather = asyncio.gather

    def _counted_gather(*args: Any, **kwargs: Any) -> Any:
        gather_calls["count"] += 1
        return real_gather(*args, **kwargs)

    monkeypatch.setattr("orchid_ai.agents.agentic_loop.asyncio.gather", _counted_gather)

    # Auto-approve the interrupt (the loop reads ``decision["approved"]``).
    fake_interrupt_calls: list[dict[str, Any]] = []

    def _fake_interrupt(payload: dict[str, Any]) -> dict[str, Any]:
        fake_interrupt_calls.append(payload)
        return {"approved": True}

    import langgraph.types

    monkeypatch.setattr(langgraph.types, "interrupt", _fake_interrupt)

    loop = _make_loop(
        tool_map={"delete_record": approval, "lookup_a": safe_a, "lookup_b": safe_b},
        parallel_safety={"lookup_a": True, "lookup_b": True, "delete_record": False},
    )

    messages: list = []
    await loop._dispatch_tool_calls(
        [_tc("lookup_a"), _tc("delete_record"), _tc("lookup_b")],
        messages,
        round_num=0,
    )

    # No gather started this round — approval serialises the whole round.
    assert gather_calls["count"] == 0
    # Interrupt fired exactly once (for the approval tool).
    assert len(fake_interrupt_calls) == 1
    assert fake_interrupt_calls[0]["tool"] == "delete_record"
    # All three tools ran, results in original order.
    assert [m.content for m in messages] == ["A", "deleted", "B"]


# ── test 5 — opt-out preserves byte-identical behaviour ──


@pytest.mark.asyncio
async def test_parallel_tools_disabled_no_gather(monkeypatch):
    """T5: ``parallel_safety=None`` (parallel_tools=false) → never calls gather."""
    tool_a = _StubTool(name="a", result="A")
    tool_b = _StubTool(name="b", result="B")

    gather_calls = {"count": 0}
    real_gather = asyncio.gather

    def _counted_gather(*args: Any, **kwargs: Any) -> Any:
        gather_calls["count"] += 1
        return real_gather(*args, **kwargs)

    monkeypatch.setattr("orchid_ai.agents.agentic_loop.asyncio.gather", _counted_gather)

    loop = _make_loop(
        tool_map={"a": tool_a, "b": tool_b},
        parallel_safety=None,  # opted out
    )

    messages: list = []
    await loop._dispatch_tool_calls(
        [_tc("a"), _tc("b")],
        messages,
        round_num=0,
    )

    assert gather_calls["count"] == 0
    assert [m.content for m in messages] == ["A", "B"]


# ── test 6 — annotation precedence covered at agent level ──


@pytest.mark.asyncio
async def test_resolve_parallel_safety_precedence(monkeypatch):
    """T6: explicit YAML ``parallel_safe`` overrides MCP ``readOnlyHint``.

    ``GenericAgent._resolve_parallel_safety`` is the precedence
    resolver — explicit YAML > MCP annotation > False.  We exercise it
    against a hand-constructed ``OrchidAgentConfig`` + faked
    ``MCPCapabilities``.
    """
    from orchid_ai.agents.generic_agent import GenericAgent
    from orchid_ai.agents.mcp_dispatcher import MCPCapabilities
    from orchid_ai.config.schema import (
        OrchidAgentConfig,
        OrchidAgentsConfig,
        OrchidBuiltinToolConfig,
        OrchidMCPServerConfig,
        OrchidRAGConfig,
        OrchidToolConfig,
    )
    from orchid_ai.mcp.inventory import MCPToolAnnotations

    config = OrchidAgentsConfig(
        tools={
            "format_date": OrchidBuiltinToolConfig(
                handler="myapp.tools.format_date",
                parallel_safe=True,
            ),
            "delete_user": OrchidBuiltinToolConfig(
                handler="myapp.tools.delete_user",
                parallel_safe=None,  # built-in default → False
            ),
        },
        agents={
            "support": OrchidAgentConfig(
                description="support",
                prompt="prompt",
                rag=OrchidRAGConfig(enabled=False),
                parallel_tools=True,
                tools=["format_date", "delete_user"],
                mcp_servers=[
                    OrchidMCPServerConfig(
                        name="kb",
                        url="http://kb.example.com",
                        tools=[
                            # Annotation says read-only; YAML overrides to False.
                            OrchidToolConfig(name="search_kb", parallel_safe=False),
                            # No YAML override; annotation hints read-only=True.
                            OrchidToolConfig(name="lookup_record"),
                            # No YAML override; no annotation → False.
                            OrchidToolConfig(name="unknown_tool"),
                        ],
                    ),
                ],
            ),
        },
    )
    agent_cfg = config.agents["support"]

    # Build a bare-bones GenericAgent (we only call the resolver method).
    from orchid_ai.rag.backends.null import NullVectorReader

    agent = GenericAgent.__new__(GenericAgent)
    agent._config = agent_cfg

    caps = MCPCapabilities()
    caps.tool_annotations = {
        "search_kb": MCPToolAnnotations(read_only_hint=True),
        "lookup_record": MCPToolAnnotations(read_only_hint=True),
        "unknown_tool": MCPToolAnnotations(read_only_hint=False),
    }

    # ``tool_map`` keys drive iteration order; every tool the loop knows about.
    tool_map = {
        "format_date": object(),
        "delete_user": object(),
        "search_kb": object(),
        "lookup_record": object(),
        "unknown_tool": object(),
    }
    builtin_tool_names = {"format_date", "delete_user"}

    safety = agent._resolve_parallel_safety(
        tool_map=tool_map,
        builtin_tool_names=builtin_tool_names,
        caps=caps,
    )

    # Built-in: explicit True → True.
    assert safety["format_date"] is True
    # Built-in: no override → False (no annotation fallback for built-ins).
    assert safety["delete_user"] is False
    # MCP: YAML override beats the read-only annotation.
    assert safety["search_kb"] is False
    # MCP: no override → annotation read_only_hint=True wins.
    assert safety["lookup_record"] is True
    # MCP: no override + read_only_hint=False → False.
    assert safety["unknown_tool"] is False

    # Sanity: opting out of parallel_tools returns None regardless.
    agent_cfg.parallel_tools = False
    assert (
        agent._resolve_parallel_safety(
            tool_map=tool_map,
            builtin_tool_names=builtin_tool_names,
            caps=caps,
        )
        is None
    )
    _ = NullVectorReader  # silence unused import (kept for fixture parity)


@pytest.mark.asyncio
async def test_resolve_parallel_safety_approval_overrides_annotation():
    """``requires_approval=True`` always wins regardless of annotation."""
    from orchid_ai.agents.generic_agent import GenericAgent
    from orchid_ai.agents.mcp_dispatcher import MCPCapabilities
    from orchid_ai.config.schema import (
        OrchidAgentConfig,
        OrchidAgentsConfig,
        OrchidMCPServerConfig,
        OrchidRAGConfig,
        OrchidToolConfig,
    )
    from orchid_ai.mcp.inventory import MCPToolAnnotations

    config = OrchidAgentsConfig(
        agents={
            "support": OrchidAgentConfig(
                description="support",
                prompt="prompt",
                rag=OrchidRAGConfig(enabled=False),
                parallel_tools=True,
                mcp_servers=[
                    OrchidMCPServerConfig(
                        name="kb",
                        url="http://kb.example.com",
                        tools=[
                            OrchidToolConfig(
                                name="risky_lookup",
                                parallel_safe=True,
                                requires_approval=True,
                            ),
                        ],
                    ),
                ],
            ),
        },
    )
    agent_cfg = config.agents["support"]

    agent = GenericAgent.__new__(GenericAgent)
    agent._config = agent_cfg

    caps = MCPCapabilities()
    caps.tool_annotations = {
        "risky_lookup": MCPToolAnnotations(read_only_hint=True),
    }

    safety = agent._resolve_parallel_safety(
        tool_map={"risky_lookup": object()},
        builtin_tool_names=set(),
        caps=caps,
    )
    assert safety == {"risky_lookup": False}


# ── Bonus: ToolMessage ordering preserved when both buckets coexist ──


@pytest.mark.asyncio
async def test_tool_message_order_matches_tool_calls():
    """Defensive: even when bucket counts differ, output order = input order."""
    safe = _StubTool(name="safe", sleep=0.02, result="S")
    seq = _StubTool(name="seq", sleep=0.02, result="Q")
    loop = _make_loop(
        tool_map={"safe": safe, "seq": seq},
        parallel_safety={"safe": True, "seq": False},
    )

    messages: list = []
    # Order: seq, safe, seq, safe.
    calls = [
        _tc("seq", tc_id="t0"),
        _tc("safe", tc_id="t1"),
        _tc("seq", {"x": 1}, tc_id="t2"),
        _tc("safe", {"y": 2}, tc_id="t3"),
    ]
    await loop._dispatch_tool_calls(calls, messages, round_num=0)

    assert [m.tool_call_id for m in messages] == ["t0", "t1", "t2", "t3"]
    assert [m.content for m in messages] == ["Q", "S", "Q", "S"]
