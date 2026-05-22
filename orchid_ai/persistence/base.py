"""
Abstract chat storage interface.

Any concrete backend (PostgreSQL, SQLite, MySQL, etc.) must implement
the `OrchidChatStorage` ABC. The API layer depends only on this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .models import OrchidChatMessage, OrchidChatSession

if TYPE_CHECKING:
    from ..core.state import OrchidAuthContext


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

    # ── Events / chat-binding helpers (§25) ──────────────────
    #
    # Concrete defaults so existing backends keep working without
    # modification.  The chat-binding runtime (``GraphJobRunner
    # ._resolve_chat_binding``) calls these to enforce per-call
    # authorization.

    async def get_chat_metadata(self, chat_id: str) -> "OrchidChatSession | None":
        """Return the chat-session metadata or ``None`` when missing.

        Default: alias for :meth:`get_chat`.  Backends with separate
        cheap-metadata storage (e.g. a session cache without messages)
        may override for performance, but the default is correct.
        """
        return await self.get_chat(chat_id)

    async def can_write(
        self,
        chat: OrchidChatSession,
        auth: "OrchidAuthContext",
    ) -> bool:
        """Authorize ``auth`` to append messages to ``chat``.

        Default rule: same tenant AND same user.  Concrete backends
        may override for richer semantics (e.g. team-shared chats,
        admin override) but the default is the safe baseline that
        rejects cross-user writes — exactly what the §25 chat-binding
        contract requires.

        Returns ``True`` iff:

        - ``chat.tenant_id == auth.tenant_key`` AND
        - ``chat.user_id == auth.user_id``  OR
        - ``"admin"`` is in ``auth.roles`` (admins write any chat in
          their tenant).
        """
        if chat.tenant_id != auth.tenant_key:
            return False
        if chat.user_id == auth.user_id:
            return True
        if "admin" in getattr(auth, "roles", frozenset()):
            return True
        return False

    # ── Conversation summaries (running-summary memory) ──────

    async def get_conversation_summary(self, chat_id: str) -> str | None:
        """Return the current running summary for a chat, or None.

        Default: no-op for backward compat.  Concrete backends that
        support running-summary persistence override this.
        """
        return None

    async def save_conversation_summary(self, chat_id: str, summary: str, turn_number: int) -> None:
        """Persist a running summary for a chat.

        Default: no-op for backward compat.  Concrete backends that
        support running-summary persistence override this.
        """
        pass
