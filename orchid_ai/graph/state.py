"""
Graph-specific state — extends the core OrchidAgentState with LangGraph annotations.

The core/ layer defines the *shape* (TypedDict with stdlib types).
This module re-defines the state with:
  - ``Annotated[list, add_messages]`` for automatic message merging
  - ``merge_dicts`` reducer for ``mcp_context`` / ``rag_context``
    so parallel sub-agents can each contribute data without overwriting.
  - ``execution_mode`` + ``pending_agents`` for ADR-013
    (Supervisor decides parallel vs sequential per request).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from ..core.state import OrchidAuthContext


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

    Execution control (ADR-013):
      execution_mode  → "parallel" | "sequential" — how the Supervisor dispatches
      pending_agents  → agents still waiting in a sequential pipeline
    """

    messages: Annotated[list[BaseMessage], add_messages]
    auth_context: OrchidAuthContext  # ADR-014: tenant key = auth_context.tenant_key (installation_id)
    chat_id: str  # chat session identifier for RAG scoping
    active_agents: Annotated[list[str], replace_list]
    mcp_context: Annotated[dict[str, Any], merge_dicts]
    rag_context: Annotated[dict[str, Any], merge_dicts]
    final_response: str | None

    # ── ADR-013: execution control ────────────────────────────
    execution_mode: Literal["parallel", "sequential"]
    pending_agents: Annotated[list[str], replace_list]

    # ── ADR-017: orchestrator skill instructions ──────────────
    skill_instructions: Annotated[dict[str, Any], merge_dicts]

    # ── ADR-018: guardrail routing hint ──────────────────────
    _has_output_guardrails: bool  # sentinel for route_to_agents()

    # ── MCP per-server OAuth status (per-request) ────────────
    mcp_auth_status: Annotated[dict[str, Any], merge_dicts]  # {server_name: authorized_bool}
