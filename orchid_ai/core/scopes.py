"""
RAG scope — pure dataclass for hierarchical partition keys.

Lives in ``core/`` because it has zero external dependencies and is
used by ``OrchidVectorReader.retrieve()`` and ``OrchidAgent.fetch_rag_context()``.
The Qdrant-specific ``build_qdrant_filter()`` stays in ``rag/scopes.py``
and imports ``OrchidRAGScope`` from here.
"""

from __future__ import annotations

from dataclasses import dataclass

SHARED_TENANT = "__shared__"

_SCOPE_KEY_SEP = "\x1f"


@dataclass(frozen=True)
class OrchidRAGScope:
    """Full position in the RAG hierarchy.

    Scope hierarchy (most specific to broadest):
      chat_agent  → only this agent in this chat
      chat_shared → all agents in this chat
      user        → all chats for this user
      tenant      → all users in this tenant
      __shared__  → all tenants (root common data)
    """

    tenant_id: str
    user_id: str = ""
    chat_id: str = ""
    agent_id: str = ""


def resolve_scope_level(scope: OrchidRAGScope) -> str:
    """Derive the visibility level from which scope fields are populated.

    The level is written as the ``scope`` metadata field on indexed
    chunks and is matched by the retrieval scope filter.  The
    precedence mirrors the hierarchy: the most specific populated field
    wins.

    Returns one of ``"chat_agent"``, ``"chat_shared"``, ``"user"``,
    or ``"tenant"``.
    """
    if scope.agent_id:
        return "chat_agent"
    if scope.chat_id:
        return "chat_shared"
    if scope.user_id:
        return "user"
    return "tenant"


def scope_key(scope: OrchidRAGScope) -> str:
    """Return a canonical string identifying a scope for deduplication.

    Used by the ingestion manifest so that re-indexing the same source
    into a *different* scope is not treated as unchanged.  The key
    encodes every identifying field; the level is intentionally omitted
    because it is already derivable from the fields themselves.
    """
    return _SCOPE_KEY_SEP.join(
        (
            scope.tenant_id,
            scope.user_id,
            scope.chat_id,
            scope.agent_id,
        )
    )
