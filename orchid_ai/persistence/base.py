"""
Abstract chat storage interface.

Any concrete backend (PostgreSQL, SQLite, MySQL, etc.) must implement
the `OrchidChatStorage` ABC. The API layer depends only on this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import OrchidChatMessage, OrchidChatSession


class OrchidChatStorage(ABC):
    """Abstract base class for chat persistence backends."""

    # ── Lifecycle ────────────────────────────────────────────

    @abstractmethod
    async def init_db(self) -> None:
        """Initialise the database connection and run pending migrations."""

    @abstractmethod
    async def close(self) -> None:
        """Release database connections / pools."""

    # ── Sessions ─────────────────────────────────────────────

    @abstractmethod
    async def create_chat(
        self,
        tenant_id: str,
        user_id: str,
        title: str = "",
    ) -> OrchidChatSession:
        """Create a new chat session and return it."""

    @abstractmethod
    async def list_chats(
        self,
        tenant_id: str,
        user_id: str,
    ) -> list[OrchidChatSession]:
        """List all chats for a user, most recently updated first."""

    @abstractmethod
    async def get_chat(self, chat_id: str) -> OrchidChatSession | None:
        """Get a single chat session by ID, or None."""

    @abstractmethod
    async def delete_chat(self, chat_id: str) -> None:
        """Delete a chat session and all its messages."""

    @abstractmethod
    async def update_title(self, chat_id: str, title: str) -> None:
        """Update a chat's title."""

    @abstractmethod
    async def mark_shared(self, chat_id: str) -> None:
        """Mark a chat as shared."""

    # ── Messages ─────────────────────────────────────────────

    @abstractmethod
    async def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        agents_used: list[str] | None = None,
        metadata: dict | None = None,
    ) -> OrchidChatMessage:
        """Add a message to a chat and touch the session's updated_at."""

    @abstractmethod
    async def get_messages(
        self,
        chat_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[OrchidChatMessage]:
        """Get messages for a chat, oldest first."""
