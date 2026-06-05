"""
Graph-specific state — extends the core OrchidAgentState with LangGraph annotations.

The core/ layer defines the *shape* (TypedDict with stdlib types).
This module re-defines the state with:
  - ``Annotated[list, add_messages]`` for automatic message merging
  - ``merge_dicts`` reducer for ``mcp_context`` / ``rag_context``
    so parallel sub-agents can each contribute data without overwriting.
  - ``execution_mode`` + ``pending_agents``
    (Supervisor decides parallel vs sequential per request).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ── Reducers ─────────────────────────────────────────────────


def merge_dicts(existing: dict[str, Any] | None, new: dict[str, Any] | None) -> dict[str, Any]:
    """Shallow merge — each sub-agent writes a unique key (e.g. "knowledge-base", "messaging")."""
    base = existing or {}
    update = new or {}
    return {**base, **update}


def replace_list(existing: list[str] | None, new: list[str] | None) -> list[str]:
    """Replace reducer — last write wins.  Handles parallel fan-in safely."""
    return new if new is not None else (existing or [])


# ── Graph State ──────────────────────────────────────────────


class GraphState(TypedDict, total=False):
    """
    LangGraph state schema with reducers for parallel sub-agent execution.

    Reducers:
      messages    → ``add_messages``  (append, deduplicate by id)
      mcp_context → ``merge_dicts``   (shallow merge per-agent keys)
      rag_context → ``merge_dicts``   (shallow merge per-agent keys)

    Execution control:
      execution_mode  → "parallel" | "sequential" — how the Supervisor dispatches
      pending_agents  → agents still waiting in a sequential pipeline
    """

    messages: Annotated[list[BaseMessage], add_messages]
    # auth is execution context (RunnableConfig), never checkpointed state —
    # see orchid_ai.core.run_config.
    chat_id: str  # chat session identifier for RAG scoping
    active_agents: Annotated[list[str], replace_list]
    mcp_context: Annotated[dict[str, Any], merge_dicts]
    rag_context: Annotated[dict[str, Any], merge_dicts]
    final_response: str | None

    # ── execution control ────────────────────────────────────
    execution_mode: Literal["parallel", "sequential"]
    pending_agents: Annotated[list[str], replace_list]

    # ── orchestrator skill instructions ──────────────────────
    skill_instructions: Annotated[dict[str, Any], merge_dicts]

    # ── guardrail routing hint ───────────────────────────────
    _has_output_guardrails: bool  # sentinel for route_to_agents()

    # ── MCP per-server OAuth status (per-request) ────────────
    mcp_auth_status: Annotated[dict[str, Any], merge_dicts]  # {server_name: authorized_bool}

    # ── mini-agents (self-clones) ────────────────────────────
    # Per-parent decomposer output, keyed by parent agent name
    # (e.g. ``"support"``).  Stored as plain dicts (``MiniAgentDecomposition.model_dump()``)
    # to keep state checkpointer-safe.  The fork router reads this
    # to decide between ``"supervisor"`` and a ``Send`` fan-out.
    mini_agent_decisions: Annotated[dict[str, Any], merge_dicts]

    # Per-mini outcome, keyed ``f"{parent_name}#{mini_id}"``
    # (e.g. ``"support#mini_0"``).  Stored as plain dicts
    # (``MiniAgentOutcome.model_dump()``).  The aggregator filters
    # by the parent's prefix and synthesises the parent's final answer.
    mini_agent_outcomes: Annotated[dict[str, Any], merge_dicts]

    # Per-Send sentinels — set by the fork router on each ``Send``
    # payload so the mini node can identify its sub-task without
    # threading kwargs through LangGraph.  Each Send branch sees its
    # own values; nothing else reads these keys.
    _active_mini_parent: str
    _active_mini_id: str
    _active_mini_subtask: dict[str, Any]
    _active_mini_tool_subset: list[str]
