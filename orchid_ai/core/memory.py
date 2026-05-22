"""
Conversation memory strategies.

Provides the ABC for incremental conversation summarization
and a no-op implementation for when memory is disabled.

This module lives in ``core/`` so the graph layer can depend on it
without importing concrete backends.  It has ZERO external dependencies
beyond stdlib.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class OrchidConversationMemory(ABC):
    """ABC for conversation memory strategies.

    Implementations provide different strategies for storing and
    retrieving conversation context beyond the current session's
    LangGraph state.  The base class lives in ``core/`` so the graph
    layer can depend on it without importing concrete backends.
    """

    @abstractmethod
    async def get_running_summary(self, chat_id: str) -> str | None:
        """Return the current running summary for a chat, or None if no summary exists."""
        ...

    @abstractmethod
    async def update_running_summary(
        self,
        chat_id: str,
        new_messages: list[dict[str, str]],
        existing_summary: str | None,
    ) -> str:
        """Incrementally update the running summary with new messages.

        If existing_summary is None, creates a new summary from scratch.
        If existing_summary is provided, extends it with new information.
        Returns the updated summary text.
        """
        ...

    @abstractmethod
    async def get_relevant_history(
        self,
        query: str,
        chat_id: str,
        k: int = 5,
    ) -> list[dict[str, str]]:
        """Retrieve the k most relevant past turns for the given query.

        Phase 1 implementations return an empty list.
        Phase 3 (RAG-augmented) will implement semantic retrieval.
        """
        ...

    @abstractmethod
    async def store_conversation_turn(
        self,
        chat_id: str,
        tenant_id: str,
        user_id: str,
        turn: dict[str, str],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a conversation turn for future retrieval.

        Phase 1 implementations are no-ops.
        Phase 3 (RAG-augmented) will embed and store in Qdrant.
        """
        ...


class NullConversationMemory(OrchidConversationMemory):
    """No-op implementation used when memory is disabled (strategy: "none")."""

    async def get_running_summary(self, chat_id: str) -> str | None:
        return None

    async def update_running_summary(
        self,
        chat_id: str,
        new_messages: list[dict[str, str]],
        existing_summary: str | None,
    ) -> str:
        return existing_summary or ""

    async def get_relevant_history(
        self,
        query: str,
        chat_id: str,
        k: int = 5,
    ) -> list[dict[str, str]]:
        return []

    async def store_conversation_turn(
        self,
        chat_id: str,
        tenant_id: str,
        user_id: str,
        turn: dict[str, str],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        pass
