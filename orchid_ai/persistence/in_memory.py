"""Built-in in-memory persistence backends — the framework default.

These implementations are **process-local, non-durable** and depend only
on the standard library.  They exist so ``orchid-ai`` works out of the box
with zero storage packages installed: the bootstrap defaults resolve to
the classes in this module when no storage backend is configured.

For durable storage install a storage plugin (e.g.
``orchid-storage-sqlite``) and point the relevant config key at its
dotted class path, or pass the ``"memory"`` sentinel to the factories in
:mod:`orchid_ai.persistence` / :mod:`orchid_ai.config.storage_factory` to
select these classes explicitly.

Construction follows the house contract: keyword-only ``dsn`` plus the
optional ``extra_migrations_package`` hook (both accepted and ignored —
there is no database and no migration system here).
"""

from __future__ import annotations

import asyncio
import copy
import time
import uuid
from typing import Any

from ..core.ingestion_manifest import OrchidIngestionManifest
from ..core.mcp import (
    OrchidMCPClientRegistration,
    OrchidMCPClientRegistrationStore,
    OrchidMCPTokenRecord,
    OrchidMCPTokenStore,
)
from ..core.mcp_gateway_state import (
    OrchidMCPGatewayAuthCode,
    OrchidMCPGatewayAuthCodeStore,
    OrchidMCPGatewayClient,
    OrchidMCPGatewayClientStore,
    OrchidMCPGatewayToken,
    OrchidMCPGatewayTokenStore,
)
from .base import OrchidChatStorage
from .models import OrchidChatMessage, OrchidChatSession, utcnow

__all__ = [
    "OrchidInMemoryChatStorage",
    "OrchidInMemoryIngestionManifest",
    "OrchidInMemoryMCPClientRegistrationStore",
    "OrchidInMemoryMCPGatewayStateStore",
    "OrchidInMemoryMCPTokenStore",
]


class OrchidInMemoryChatStorage(OrchidChatStorage):
    """Process-local chat sessions + messages (framework default).

    Chats live in dictionaries for the lifetime of the process — they are
    **not** durable across restarts.  Suitable for tests, embedded runs,
    and zero-configuration quick starts; install a storage backend for
    persistence.
    """

    def __init__(self, *, dsn: str = "", extra_migrations_package: str | None = None) -> None:
        # ``dsn`` / ``extra_migrations_package`` exist for factory-contract
        # compatibility; neither applies to an in-memory store.
        self._sessions: dict[str, OrchidChatSession] = {}
        self._messages: dict[str, list[OrchidChatMessage]] = {}
        self._summaries: dict[str, tuple[str, int]] = {}

    # ── Lifecycle ────────────────────────────────────────────

    async def init_db(self) -> None:
        """No-op — nothing to open or migrate."""

    async def close(self) -> None:
        """No-op — nothing to release."""

    # ── Sessions ─────────────────────────────────────────────

    async def create_chat(
        self,
        tenant_id: str,
        user_id: str,
        title: str = "",
    ) -> OrchidChatSession:
        now = utcnow()
        chat = OrchidChatSession(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            user_id=user_id,
            title=title or "New chat",
            created_at=now,
            updated_at=now,
        )
        self._sessions[chat.id] = copy.deepcopy(chat)
        self._messages[chat.id] = []
        return chat

    async def list_chats(
        self,
        tenant_id: str,
        user_id: str,
    ) -> list[OrchidChatSession]:
        chats = [
            session
            for session in self._sessions.values()
            if session.tenant_id == tenant_id and session.user_id == user_id
        ]
        chats.sort(key=lambda session: session.updated_at, reverse=True)
        return [copy.deepcopy(session) for session in chats]

    async def get_chat(self, chat_id: str) -> OrchidChatSession | None:
        session = self._sessions.get(chat_id)
        return copy.deepcopy(session) if session is not None else None

    async def delete_chat(self, chat_id: str) -> None:
        self._sessions.pop(chat_id, None)
        self._messages.pop(chat_id, None)
        self._summaries.pop(chat_id, None)

    async def update_title(self, chat_id: str, title: str) -> None:
        session = self._sessions.get(chat_id)
        if session is None:
            return
        session.title = title
        session.updated_at = utcnow()

    async def mark_shared(self, chat_id: str) -> None:
        session = self._sessions.get(chat_id)
        if session is None:
            return
        session.is_shared = True
        session.updated_at = utcnow()

    # ── Messages ─────────────────────────────────────────────

    async def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        agents_used: list[str] | None = None,
        metadata: dict | None = None,
    ) -> OrchidChatMessage:
        now = utcnow()
        msg = OrchidChatMessage(
            id=str(uuid.uuid4()),
            chat_id=chat_id,
            role=role,
            content=content,
            agents_used=list(agents_used or []),
            created_at=now,
            metadata=dict(metadata or {}),
        )
        self._messages.setdefault(chat_id, []).append(copy.deepcopy(msg))
        session = self._sessions.get(chat_id)
        if session is not None:
            session.updated_at = now
        return msg

    async def get_messages(
        self,
        chat_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[OrchidChatMessage]:
        messages = self._messages.get(chat_id, [])
        window = messages[offset : offset + limit]
        return [copy.deepcopy(msg) for msg in window]

    # ── Conversation summaries (running-summary memory) ──────

    async def get_conversation_summary(self, chat_id: str) -> str | None:
        entry = self._summaries.get(chat_id)
        return entry[0] if entry is not None else None

    async def save_conversation_summary(self, chat_id: str, summary: str, turn_number: int) -> None:
        self._summaries[chat_id] = (summary, turn_number)


class OrchidInMemoryMCPTokenStore(OrchidMCPTokenStore):
    """Process-local per-user MCP OAuth token store."""

    def __init__(self, *, dsn: str = "", extra_migrations_package: str | None = None) -> None:
        self._tokens: dict[tuple[str, str, str], OrchidMCPTokenRecord] = {}

    async def init_db(self) -> None:
        """No-op — nothing to open or migrate."""

    async def close(self) -> None:
        """No-op — nothing to release."""

    async def get_token(
        self,
        tenant_id: str,
        user_id: str,
        server_name: str,
    ) -> OrchidMCPTokenRecord | None:
        record = self._tokens.get((tenant_id, user_id, server_name))
        return copy.deepcopy(record) if record is not None else None

    async def save_token(self, record: OrchidMCPTokenRecord) -> None:
        stored = copy.deepcopy(record)
        stored.updated_at = time.time()
        self._tokens[(record.tenant_id, record.user_id, record.server_name)] = stored

    async def delete_token(
        self,
        tenant_id: str,
        user_id: str,
        server_name: str,
    ) -> bool:
        return self._tokens.pop((tenant_id, user_id, server_name), None) is not None

    async def list_tokens(
        self,
        tenant_id: str,
        user_id: str,
    ) -> list[OrchidMCPTokenRecord]:
        return [
            copy.deepcopy(record)
            for (token_tenant, token_user, _server), record in self._tokens.items()
            if token_tenant == tenant_id and token_user == user_id
        ]

    async def cleanup_expired(self, *, before: float | None = None) -> int:
        cutoff = time.time() if before is None else before
        doomed = [key for key, record in self._tokens.items() if record.expires_at > 0 and record.expires_at < cutoff]
        for key in doomed:
            del self._tokens[key]
        return len(doomed)


class OrchidInMemoryMCPClientRegistrationStore(OrchidMCPClientRegistrationStore):
    """Process-local per-server MCP DCR registration store."""

    def __init__(self, *, dsn: str = "", extra_migrations_package: str | None = None) -> None:
        self._registrations: dict[str, OrchidMCPClientRegistration] = {}

    async def init_db(self) -> None:
        """No-op — nothing to open or migrate."""

    async def close(self) -> None:
        """No-op — nothing to release."""

    async def get(self, server_name: str) -> OrchidMCPClientRegistration | None:
        record = self._registrations.get(server_name)
        return copy.deepcopy(record) if record is not None else None

    async def save(self, record: OrchidMCPClientRegistration) -> None:
        stored = copy.deepcopy(record)
        stored.updated_at = time.time()
        self._registrations[record.server_name] = stored

    async def delete(self, server_name: str) -> bool:
        return self._registrations.pop(server_name, None) is not None


class OrchidInMemoryMCPGatewayStateStore(
    OrchidMCPGatewayClientStore,
    OrchidMCPGatewayAuthCodeStore,
    OrchidMCPGatewayTokenStore,
):
    """Process-local inbound MCP-gateway state (all three gateway ABCs).

    ``consume`` pops the auth code under an ``asyncio.Lock`` so one-shot
    semantics hold even when two coroutines race the same code.
    """

    def __init__(self, *, dsn: str = "", extra_migrations_package: str | None = None) -> None:
        self._clients: dict[str, OrchidMCPGatewayClient] = {}
        self._auth_codes: dict[str, OrchidMCPGatewayAuthCode] = {}
        self._tokens: dict[str, OrchidMCPGatewayToken] = {}
        self._lock = asyncio.Lock()

    async def init_db(self) -> None:
        """No-op — nothing to open or migrate."""

    async def close(self) -> None:
        """No-op — nothing to release."""

    # ── Clients ──────────────────────────────────────────────

    async def register(self, record: OrchidMCPGatewayClient) -> None:
        self._clients[record.client_id] = copy.deepcopy(record)

    async def get(self, client_id: str) -> OrchidMCPGatewayClient | None:
        record = self._clients.get(client_id)
        return copy.deepcopy(record) if record is not None else None

    # ── Auth codes ───────────────────────────────────────────

    async def put(self, record: OrchidMCPGatewayAuthCode) -> None:
        self._auth_codes[record.code] = copy.deepcopy(record)

    async def get_by_upstream_state(
        self,
        upstream_state: str,
    ) -> OrchidMCPGatewayAuthCode | None:
        for record in self._auth_codes.values():
            if record.upstream_state == upstream_state:
                return copy.deepcopy(record)
        return None

    async def update(
        self,
        code: str,
        *,
        identity: dict[str, Any] | None = None,
        idp_access_token: str | None = None,
        idp_refresh_token: str | None = None,
        idp_expires_at: float | None = None,
    ) -> None:
        async with self._lock:
            record = self._auth_codes.get(code)
            if record is None:
                return
            updated = copy.deepcopy(record)
            if identity is not None:
                updated.identity = identity
            if idp_access_token is not None:
                updated.idp_access_token = idp_access_token
            if idp_refresh_token is not None:
                updated.idp_refresh_token = idp_refresh_token
            if idp_expires_at is not None:
                updated.idp_expires_at = float(idp_expires_at)
            self._auth_codes[code] = updated

    async def consume(self, code: str) -> OrchidMCPGatewayAuthCode | None:
        async with self._lock:
            record = self._auth_codes.pop(code, None)
            return copy.deepcopy(record) if record is not None else None

    # ── Tokens ───────────────────────────────────────────────

    async def issue(self, record: OrchidMCPGatewayToken) -> None:
        self._tokens[record.access_token] = copy.deepcopy(record)

    async def get_by_access_token(
        self,
        access_token: str,
    ) -> OrchidMCPGatewayToken | None:
        return self._live_token(self._tokens.get(access_token))

    async def get_by_refresh_token(
        self,
        refresh_token: str,
    ) -> OrchidMCPGatewayToken | None:
        for record in self._tokens.values():
            if record.refresh_token == refresh_token:
                return self._live_token(record)
        return None

    async def revoke(self, access_token: str) -> bool:
        return self._tokens.pop(access_token, None) is not None

    @staticmethod
    def _live_token(record: OrchidMCPGatewayToken | None) -> OrchidMCPGatewayToken | None:
        if record is None:
            return None
        if record.expires_at > 0 and time.time() >= record.expires_at:
            return None
        return copy.deepcopy(record)


class OrchidInMemoryIngestionManifest(OrchidIngestionManifest):
    """Process-local ingestion manifest keyed by ``(source_id, namespace, scope)``."""

    def __init__(self, *, dsn: str = "", extra_migrations_package: str | None = None) -> None:
        self._entries: dict[tuple[str, str, str], tuple[str, list[str]]] = {}

    async def init_db(self) -> None:
        """No-op — nothing to open or migrate."""

    async def close(self) -> None:
        """No-op — nothing to release."""

    async def should_skip(
        self,
        source_id: str,
        content_hash: str,
        namespace: str,
        scope: str = "",
    ) -> bool:
        entry = self._entries.get((source_id, namespace, scope))
        return entry is not None and entry[0] == content_hash

    async def record(
        self,
        source_id: str,
        content_hash: str,
        namespace: str,
        document_ids: list[str],
        scope: str = "",
    ) -> None:
        self._entries[(source_id, namespace, scope)] = (content_hash, list(document_ids))

    async def remove(self, source_id: str, namespace: str, scope: str = "") -> None:
        self._entries.pop((source_id, namespace, scope), None)

    async def list_known(self, namespace: str, scope: str = "") -> set[str]:
        return {
            source_id
            for (source_id, entry_namespace, entry_scope) in self._entries
            if entry_namespace == namespace and entry_scope == scope
        }

    async def get_document_ids(self, source_id: str, namespace: str, scope: str = "") -> list[str]:
        entry = self._entries.get((source_id, namespace, scope))
        return list(entry[1]) if entry is not None else []
