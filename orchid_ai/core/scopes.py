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
