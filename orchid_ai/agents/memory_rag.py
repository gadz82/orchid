"""
RAG-augmented conversation memory.

Extends the in-memory running summary with Qdrant-backed semantic
retrieval of past conversation turns.  Uses three strategies in
combination:

1. Running summary (inherited) for broad conversation context.
2. Recent verbatim turns for immediate context.
3. RAG-retrieved turns for semantically relevant context from any
   point in the conversation history.

Qdrant interactions go through ``OrchidVectorReader`` /
``OrchidVectorWriter`` ABCs — no concrete backend imports.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from ..core.repository import OrchidDocument
from ..core.scopes import OrchidRAGScope
from .memory import OrchidInMemoryConversationMemory

logger = logging.getLogger(__name__)


class OrchidRAGConversationMemory(OrchidInMemoryConversationMemory):
    """Runs running-summary memory AND Qdrant semantic retrieval.

    Inherits running-summary logic from
    :class:`OrchidInMemoryConversationMemory` and adds semantic
    retrieval of past turns stored in Qdrant under the reserved
    ``__memory__`` namespace.
    """

    MEMORY_NAMESPACE = "__memory__"

    def __init__(
        self,
        chat_storage: Any,
        chat_model: Any,
        reader: Any,
        writer: Any,
        *,
        structured_output: bool = True,
    ):
        super().__init__(
            chat_storage=chat_storage,
            chat_model=chat_model,
            structured_output=structured_output,
        )
        self._reader = reader
        self._writer = writer

    async def store_conversation_turn(
        self,
        chat_id: str,
        tenant_id: str,
        user_id: str,
        turn: dict[str, str],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Embed and store a conversation turn in Qdrant.

        The turn content is stored as a single document under the
        ``__memory__`` namespace.  Scope metadata (tenant_id, user_id,
        chat_id) is embedded so retrieval can filter by
        ``OrchidRAGScope``.
        """
        content = turn.get("content", "")
        if not content.strip():
            return

        content_hash = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]
        doc_id = f"mem-{chat_id}-{int(time.time() * 1000)}-{content_hash}"

        doc_metadata: dict[str, Any] = {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "chat_id": chat_id,
            "scope": "chat_shared",
            "source": "conversation_memory",
            "turn_role": turn.get("role", ""),
            **(metadata or {}),
        }

        doc = OrchidDocument(id=doc_id, page_content=content, metadata=doc_metadata)

        try:
            await self._writer.upsert(documents=[doc], namespace=self.MEMORY_NAMESPACE)
        except Exception as exc:
            logger.warning("[RAGMemory] Failed to store turn for chat %s: %s", chat_id, exc)

    async def get_relevant_history(
        self,
        query: str,
        chat_id: str,
        k: int = 5,
        *,
        tenant_id: str = "default",
        user_id: str = "",
        similarity_threshold: float = 0.5,
    ) -> list[dict[str, str]]:
        """Retrieve the k most relevant past turns via semantic search.

        Uses ``OrchidRAGScope`` for hierarchical tenant isolation.
        Results below ``similarity_threshold`` are discarded.
        """
        scope = OrchidRAGScope(
            tenant_id=tenant_id,
            user_id=user_id,
            chat_id=chat_id,
            agent_id="",
        )
        try:
            results = await self._reader.retrieve(
                query=query,
                namespace=self.MEMORY_NAMESPACE,
                k=k,
                scope=scope,
            )
        except Exception:
            logger.warning(
                "[RAGMemory] Retrieval failed for chat %s (namespace=%s), returning empty",
                chat_id,
                self.MEMORY_NAMESPACE,
            )
            return []

        relevant: list[dict[str, str]] = []
        for r in results:
            if r.score < similarity_threshold:
                continue
            relevant.append(
                {
                    "role": "assistant",
                    "content": r.document.page_content,
                    "score": r.score,
                }
            )
        return relevant

    async def get_relevant_history_merged(
        self,
        query: str,
        chat_id: str,
        recent_verbatim: list[dict[str, str]],
        *,
        tenant_id: str = "default",
        user_id: str = "",
        k: int = 5,
        similarity_threshold: float = 0.5,
    ) -> list[dict[str, str]]:
        """Retrieve RAG-relevant turns and merge with the verbatim window.

        Deduplication: RAG-retrieved turns whose content substantially
        overlaps with a verbatim turn are dropped (content-hash check).
        The result order is: RAG turns first, then verbatim.
        """
        rag_turns = await self.get_relevant_history(
            query=query,
            chat_id=chat_id,
            k=k,
            tenant_id=tenant_id,
            user_id=user_id,
            similarity_threshold=similarity_threshold,
        )
        if not rag_turns:
            return list(recent_verbatim)

        # Build a set of content hashes for the verbatim window
        verbatim_hashes: set[str] = set()
        for m in recent_verbatim:
            verbatim_hashes.add(hashlib.sha256(m.get("content", "").encode("utf-8", errors="replace")).hexdigest())

        # Filter RAG turns that overlap with verbatim
        deduped: list[dict[str, str]] = []
        for t in rag_turns:
            content = t.get("content", "")
            ch = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
            if ch not in verbatim_hashes:
                t_clean = {"role": t.get("role", "assistant"), "content": content}
                deduped.append(t_clean)

        # RAG turns first, then verbatim
        merged = deduped + list(recent_verbatim)
        return merged
