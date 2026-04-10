"""
Hierarchical RAG scoping — 5-level partition scheme.

Scope hierarchy (most specific → broadest):
  chat_agent  → only this agent in this chat
  chat_shared → all agents in this chat
  user        → all chats for this user
  tenant      → all users in this tenant
  __shared__  → all tenants (root common data)

At query time, an agent sees ALL levels it's entitled to
(its own agent-private data + chat shared + user common + tenant + shared).
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

SHARED_TENANT = "__shared__"


@dataclass(frozen=True)
class RAGScope:
    """Full position in the RAG hierarchy."""

    tenant_id: str
    user_id: str = ""
    chat_id: str = ""
    agent_id: str = ""


def build_qdrant_filter(scope: RAGScope) -> Filter:
    """
    Build a Qdrant ``Filter`` with ``should`` (OR) clauses covering
    every scope level visible to the caller.

    The result is a single filter that, when passed to ``client.search()``,
    returns documents from all accessible levels ranked by relevance.
    """
    clauses: list[Filter] = []

    # 1. Root common — tenant_id = "__shared__"
    clauses.append(
        Filter(must=[
            FieldCondition(key="tenant_id", match=MatchValue(value=SHARED_TENANT)),
        ])
    )

    # 2. Tenant-level — tenant_id = T AND scope = "tenant"
    clauses.append(
        Filter(must=[
            FieldCondition(key="tenant_id", match=MatchValue(value=scope.tenant_id)),
            FieldCondition(key="scope", match=MatchValue(value="tenant")),
        ])
    )

    # 3. User-common — requires user_id
    if scope.user_id:
        clauses.append(
            Filter(must=[
                FieldCondition(key="tenant_id", match=MatchValue(value=scope.tenant_id)),
                FieldCondition(key="user_id", match=MatchValue(value=scope.user_id)),
                FieldCondition(key="scope", match=MatchValue(value="user")),
            ])
        )

    # 4. Chat-shared — requires user_id + chat_id
    if scope.user_id and scope.chat_id:
        clauses.append(
            Filter(must=[
                FieldCondition(key="tenant_id", match=MatchValue(value=scope.tenant_id)),
                FieldCondition(key="user_id", match=MatchValue(value=scope.user_id)),
                FieldCondition(key="chat_id", match=MatchValue(value=scope.chat_id)),
                FieldCondition(key="scope", match=MatchValue(value="chat_shared")),
            ])
        )

    # 5. Agent-private — requires user_id + chat_id + agent_id
    if scope.user_id and scope.chat_id and scope.agent_id:
        clauses.append(
            Filter(must=[
                FieldCondition(key="tenant_id", match=MatchValue(value=scope.tenant_id)),
                FieldCondition(key="user_id", match=MatchValue(value=scope.user_id)),
                FieldCondition(key="chat_id", match=MatchValue(value=scope.chat_id)),
                FieldCondition(key="agent_id", match=MatchValue(value=scope.agent_id)),
                FieldCondition(key="scope", match=MatchValue(value="chat_agent")),
            ])
        )

    return Filter(should=clauses)
