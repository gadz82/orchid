"""
Hierarchical RAG scoping — 5-level partition scheme.

``RAGScope`` lives in ``core/scopes.py`` (zero external deps) and is
re-exported here for backward compatibility.  The Qdrant-specific
``build_qdrant_filter()`` stays here because it depends on ``qdrant_client``.
"""

from __future__ import annotations

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

# Re-export from core for backward compatibility
from ..core.scopes import SHARED_TENANT, RAGScope

__all__ = ["SHARED_TENANT", "RAGScope", "build_qdrant_filter"]


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
        Filter(
            must=[
                FieldCondition(key="tenant_id", match=MatchValue(value=SHARED_TENANT)),
            ]
        )
    )

    # 2. Tenant-level — tenant_id = T AND scope = "tenant"
    clauses.append(
        Filter(
            must=[
                FieldCondition(key="tenant_id", match=MatchValue(value=scope.tenant_id)),
                FieldCondition(key="scope", match=MatchValue(value="tenant")),
            ]
        )
    )

    # 3. User-common — requires user_id
    if scope.user_id:
        clauses.append(
            Filter(
                must=[
                    FieldCondition(key="tenant_id", match=MatchValue(value=scope.tenant_id)),
                    FieldCondition(key="user_id", match=MatchValue(value=scope.user_id)),
                    FieldCondition(key="scope", match=MatchValue(value="user")),
                ]
            )
        )

    # 4. Chat-shared — requires user_id + chat_id
    if scope.user_id and scope.chat_id:
        clauses.append(
            Filter(
                must=[
                    FieldCondition(key="tenant_id", match=MatchValue(value=scope.tenant_id)),
                    FieldCondition(key="user_id", match=MatchValue(value=scope.user_id)),
                    FieldCondition(key="chat_id", match=MatchValue(value=scope.chat_id)),
                    FieldCondition(key="scope", match=MatchValue(value="chat_shared")),
                ]
            )
        )

    # 5. Agent-private — requires user_id + chat_id + agent_id
    if scope.user_id and scope.chat_id and scope.agent_id:
        clauses.append(
            Filter(
                must=[
                    FieldCondition(key="tenant_id", match=MatchValue(value=scope.tenant_id)),
                    FieldCondition(key="user_id", match=MatchValue(value=scope.user_id)),
                    FieldCondition(key="chat_id", match=MatchValue(value=scope.chat_id)),
                    FieldCondition(key="agent_id", match=MatchValue(value=scope.agent_id)),
                    FieldCondition(key="scope", match=MatchValue(value="chat_agent")),
                ]
            )
        )

    return Filter(should=clauses)
