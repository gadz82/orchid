"""
PostgreSQL chat storage — built-in ChatStorage implementation.

This is the **default** storage backend shipped with the library.
Uses asyncpg for async connection pooling. Migrations are discovered
from ``src.persistence.migrations``.

Configuration:
    CHAT_STORAGE_CLASS=src.persistence.postgres.PostgresChatStorage
    CHAT_DB_DSN=postgresql://user:pass@host:5432/dbname
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any

import asyncpg

from .base import ChatStorage
from .migrations.runner import MigrationRunner
from .models import ChatMessage, ChatSession

logger = logging.getLogger(__name__)

MIGRATIONS_PACKAGE = "orchid_ai.persistence.migrations"


class PostgresMigrationRunner(MigrationRunner):
    """PostgreSQL-specific migration tracking."""

    dialect = "postgres"
    migrations_package = MIGRATIONS_PACKAGE

    async def ensure_migrations_table(self, conn: Any) -> None:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS _migrations (
                version TEXT PRIMARY KEY,
                description TEXT NOT NULL DEFAULT '',
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

    async def get_applied_versions(self, conn: Any) -> set[str]:
        rows = await conn.fetch("SELECT version FROM _migrations")
        return {r["version"] for r in rows}

    async def record_version(self, conn: Any, version: str, description: str) -> None:
        await conn.execute(
            "INSERT INTO _migrations (version, description) VALUES ($1, $2)",
            version,
            description,
        )

    async def remove_version(self, conn: Any, version: str) -> None:
        await conn.execute("DELETE FROM _migrations WHERE version = $1", version)


class PostgresChatStorage(ChatStorage):
    """
    Async PostgreSQL storage for chat sessions and messages.

    Constructor accepts a single ``dsn`` keyword argument.
    """

    def __init__(self, *, dsn: str):
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None
        self._migrator = PostgresMigrationRunner()

    # ── Lifecycle ────────────────────────────────────────────

    async def init_db(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
        async with self._pool.acquire() as conn:
            await self._migrator.run_up(conn)
        safe_dsn = self._dsn.split("@")[-1] if "@" in self._dsn else self._dsn
        logger.info("[ChatStorage:pg] Initialised — %s", safe_dsn)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    # ── Sessions ─────────────────────────────────────────────

    async def create_chat(
        self,
        tenant_id: str,
        user_id: str,
        title: str = "",
    ) -> ChatSession:
        now = datetime.utcnow()
        chat = ChatSession(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            user_id=user_id,
            title=title or "New chat",
            created_at=now,
            updated_at=now,
        )
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO chat_sessions (id, tenant_id, user_id, title, created_at, updated_at) "
                "VALUES ($1, $2, $3, $4, $5, $6)",
                chat.id,
                chat.tenant_id,
                chat.user_id,
                chat.title,
                now,
                now,
            )
        return chat

    async def list_chats(
        self,
        tenant_id: str,
        user_id: str,
    ) -> list[ChatSession]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM chat_sessions WHERE tenant_id = $1 AND user_id = $2 ORDER BY updated_at DESC",
                tenant_id,
                user_id,
            )
            return [_row_to_session(r) for r in rows]

    async def get_chat(self, chat_id: str) -> ChatSession | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM chat_sessions WHERE id = $1",
                chat_id,
            )
            return _row_to_session(row) if row else None

    async def delete_chat(self, chat_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM chat_sessions WHERE id = $1", chat_id)

    async def update_title(self, chat_id: str, title: str) -> None:
        now = datetime.utcnow()
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE chat_sessions SET title = $1, updated_at = $2 WHERE id = $3",
                title,
                now,
                chat_id,
            )

    async def mark_shared(self, chat_id: str) -> None:
        now = datetime.utcnow()
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE chat_sessions SET is_shared = TRUE, updated_at = $1 WHERE id = $2",
                now,
                chat_id,
            )

    # ── Messages ─────────────────────────────────────────────

    async def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        agents_used: list[str] | None = None,
        metadata: dict | None = None,
    ) -> ChatMessage:
        now = datetime.utcnow()
        msg = ChatMessage(
            id=str(uuid.uuid4()),
            chat_id=chat_id,
            role=role,
            content=content,
            agents_used=agents_used or [],
            created_at=now,
            metadata=metadata or {},
        )
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO chat_messages (id, chat_id, role, content, agents_used, created_at, metadata) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7)",
                    msg.id,
                    msg.chat_id,
                    msg.role,
                    msg.content,
                    json.dumps(msg.agents_used),
                    now,
                    json.dumps(msg.metadata),
                )
                await conn.execute(
                    "UPDATE chat_sessions SET updated_at = $1 WHERE id = $2",
                    now,
                    chat_id,
                )
        return msg

    async def get_messages(
        self,
        chat_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ChatMessage]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM chat_messages WHERE chat_id = $1 ORDER BY created_at ASC LIMIT $2 OFFSET $3",
                chat_id,
                limit,
                offset,
            )
            return [_row_to_message(r) for r in rows]


# ── Row mappers ──────────────────────────────────────────────


def _row_to_session(row: asyncpg.Record) -> ChatSession:
    return ChatSession(
        id=row["id"],
        tenant_id=row["tenant_id"],
        user_id=row["user_id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        is_shared=bool(row["is_shared"]),
    )


def _row_to_message(row: asyncpg.Record) -> ChatMessage:
    agents_used = row["agents_used"]
    if isinstance(agents_used, str):
        agents_used = json.loads(agents_used)
    meta = row["metadata"]
    if isinstance(meta, str):
        meta = json.loads(meta)
    return ChatMessage(
        id=row["id"],
        chat_id=row["chat_id"],
        role=row["role"],
        content=row["content"],
        agents_used=agents_used,
        created_at=row["created_at"],
        metadata=meta,
    )
