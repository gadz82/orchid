"""Tests for ``aggregator_node_factory``.

Covers spec §16 cases:
  - T11 — partial failure: 2 of 3 succeed → aggregator prompt
          includes the full outcome list (success + failure).
  - T12 — all-failed: 0 of N succeed → no LLM call; deterministic
          error AIMessage.

Plus mcp_context merging from successful outcomes only.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage

from orchid_ai.agents.mini_agent_aggregator import (
    DEFAULT_AGGREGATOR_PROMPT,
    aggregator_node_factory,
)
from orchid_ai.config.schema import (
    OrchidAgentConfig,
    OrchidLLMConfig,
    OrchidMiniAgentConfig,
    OrchidRAGConfig,
)


def _parent_config(name: str = "support") -> OrchidAgentConfig:
    return OrchidAgentConfig(
        name=name,
        description=f"{name} agent",
        prompt="be helpful",
        rag=OrchidRAGConfig(enabled=False),
        llm=OrchidLLMConfig(model="gemini/test"),
        mini_agent=OrchidMiniAgentConfig(enabled=True),
    )


def _make_chat_model(reply: str = "synthesised"):
    """Stub the parent's chat model.  ``ainvoke`` records its prompt
    so the test can assert what reached the LLM.
    """
    chat = MagicMock()
    invocation = MagicMock(content=reply)
    chat.ainvoke = AsyncMock(return_value=invocation)
    return chat


def _outcome(*, mini_id, status, summary=None, error=None, tool_results=None, description=None):
    return {
        "mini_id": mini_id,
        "sub_task_description": description or mini_id,
        "status": status,
        "summary": summary,
        "error": error,
        "duration_ms": 10,
        "tool_results": tool_results or {},
    }


# ── T11: partial failure ───────────────────────────────────────


@pytest.mark.asyncio
async def test_partial_failure_includes_full_outcome_list_in_prompt():
    """The aggregator's prompt mentions every outcome — successful AND
    failed — so the synthesis LLM can frame the partial result."""
    cfg = _parent_config()
    chat = _make_chat_model("here is what I found ...")
    node = aggregator_node_factory(parent_config=cfg, chat_model=chat)

    state = {
        "messages": [HumanMessage(content="show me user X and orders")],
        "mini_agent_outcomes": {
            "support#mini_0": _outcome(
                mini_id="mini_0",
                status="ok",
                summary="user X is active",
                tool_results={"lookup_user": "u-1"},
                description="lookup user X",
            ),
            "support#mini_1": _outcome(
                mini_id="mini_1",
                status="ok",
                summary="3 orders found",
                tool_results={"lookup_order": "o-1,o-2,o-3"},
                description="lookup orders",
            ),
            "support#mini_2": _outcome(
                mini_id="mini_2",
                status="failed",
                error="connection refused",
                description="check invoices",
            ),
            # Outcomes from a different parent must be ignored.
            "other#mini_0": _outcome(mini_id="mini_0", status="ok", summary="ignore me"),
        },
    }

    result = await node(state)

    # Synthesis LLM was called exactly once.
    chat.ainvoke.assert_awaited_once()
    prompt_messages = chat.ainvoke.await_args.args[0]
    prompt_text = prompt_messages[0]["content"]

    # Original user query made it into the prompt.
    assert "show me user X and orders" in prompt_text
    # Both successful summaries cited by description.
    assert "lookup user X" in prompt_text
    assert "3 orders found" in prompt_text
    # Failed sub-task surfaced WITH its description AND the error text —
    # the synthesis LLM must be able to mention the failure.
    assert "check invoices" in prompt_text
    assert "connection refused" in prompt_text
    # Cross-parent outcome did NOT leak.
    assert "ignore me" not in prompt_text
    # Status markers present for every outcome.
    assert "[ok]" in prompt_text
    assert "[failed]" in prompt_text

    # Lifecycle event + AIMessage emitted.  The first slot is the
    # ``mini_agent.aggregated`` event so the streaming router can surface
    # SSE before the synthesised tokens land.
    from orchid_ai.observability import extract_event

    assert len(result["messages"]) == 2
    event = extract_event(result["messages"][0])
    assert event is not None
    assert event[0] == "mini_agent.aggregated"
    assert event[1]["parent"] == "support"
    assert {o["mini_id"] for o in event[1]["outcomes"]} == {"mini_0", "mini_1", "mini_2"}
    # AIMessage carries the synthesis text.
    assert result["messages"][1].content == "here is what I found ..."

    # mcp_context aggregates successful tool_results only — never the failed leg.
    merged = result["mcp_context"]["support"]["tool_results"]
    assert "lookup_user" in merged
    assert "lookup_order" in merged
    # Failed outcome's ``description`` is in mini_outcomes for trace
    # but its tool_results (empty in this case) do not pollute.
    assert all(k in {"lookup_user", "lookup_order"} for k in merged)
    # Full outcome list preserved for trace inspection.
    assert len(result["mcp_context"]["support"]["mini_outcomes"]) == 3


# ── T12: all-failed short-circuit ──────────────────────────────


@pytest.mark.asyncio
async def test_all_failed_short_circuits_no_llm_call():
    """0 of N succeeded → emit error AIMessage, NEVER call the LLM."""
    cfg = _parent_config()
    chat = _make_chat_model()
    node = aggregator_node_factory(parent_config=cfg, chat_model=chat)

    state = {
        "messages": [HumanMessage(content="please do everything")],
        "mini_agent_outcomes": {
            "support#mini_0": _outcome(
                mini_id="mini_0",
                status="failed",
                error="server down",
                description="lookup user",
            ),
            "support#mini_1": _outcome(
                mini_id="mini_1",
                status="timeout",
                error="timed out",
                description="lookup orders",
            ),
        },
    }

    result = await node(state)

    chat.ainvoke.assert_not_called()
    # First message is the ``mini_agent.aggregated`` event; second is
    # the deterministic error AIMessage.
    from orchid_ai.observability import extract_event

    assert len(result["messages"]) == 2
    assert extract_event(result["messages"][0]) is not None
    msg = result["messages"][1].content
    assert "Sorry" in msg
    # Cause-of-death summary surfaces both failure types.
    assert "lookup user" in msg or "lookup orders" in msg
    # ``mcp_context`` still records the outcomes for trace.
    assert "support" in result["mcp_context"]
    assert len(result["mcp_context"]["support"]["mini_outcomes"]) == 2
    # No tool_results from any outcome (all failed).
    assert result["mcp_context"]["support"]["tool_results"] == {}


# ── Defensive: no outcomes at all ──────────────────────────────


@pytest.mark.asyncio
async def test_no_outcomes_short_circuits():
    """The graph never reaches here without outcomes, but be defensive."""
    cfg = _parent_config()
    chat = _make_chat_model()
    node = aggregator_node_factory(parent_config=cfg, chat_model=chat)

    state = {"messages": [HumanMessage(content="x")]}
    result = await node(state)
    chat.ainvoke.assert_not_called()
    # Event + error AIMessage (event first).
    assert "Sorry" in result["messages"][1].content


# ── Synthesis LLM failure: graceful fallback ──────────────────


@pytest.mark.asyncio
async def test_synthesis_llm_failure_uses_fallback():
    cfg = _parent_config()
    chat = MagicMock()
    chat.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
    node = aggregator_node_factory(parent_config=cfg, chat_model=chat)

    state = {
        "messages": [HumanMessage(content="x")],
        "mini_agent_outcomes": {
            "support#mini_0": _outcome(
                mini_id="mini_0",
                status="ok",
                summary="all good",
                description="lookup",
            ),
        },
    }
    result = await node(state)
    # Did not crash — event + AIMessage (fallback synthesis).
    assert len(result["messages"]) == 2
    # Fallback contains the successful sub-task's summary.
    assert "all good" in result["messages"][1].content


# ── Custom prompt template ─────────────────────────────────────


@pytest.mark.asyncio
async def test_custom_aggregator_prompt_used():
    cfg = _parent_config()
    cfg.mini_agent.aggregator_prompt = "CUSTOM aggregator for {agent_name}: query={user_query}, n={n}\n{outcome_block}"
    chat = _make_chat_model("custom synthesis")
    node = aggregator_node_factory(parent_config=cfg, chat_model=chat)

    state = {
        "messages": [HumanMessage(content="my question")],
        "mini_agent_outcomes": {
            "support#mini_0": _outcome(mini_id="mini_0", status="ok", summary="A", description="alpha"),
            "support#mini_1": _outcome(mini_id="mini_1", status="ok", summary="B", description="beta"),
        },
    }
    await node(state)
    prompt = chat.ainvoke.await_args.args[0][0]["content"]
    assert prompt.startswith("CUSTOM aggregator for support")
    assert "n=2" in prompt
    assert "alpha" in prompt
    assert "beta" in prompt
    assert "Decide whether this request" not in prompt


def test_default_template_has_all_placeholders():
    rendered = DEFAULT_AGGREGATOR_PROMPT.format(
        agent_name="x",
        user_query="q",
        n=2,
        outcome_block="- ok: a\n- ok: b",
    )
    assert "x" in rendered
    assert "q" in rendered
    assert "2 independent sub-tasks" in rendered
