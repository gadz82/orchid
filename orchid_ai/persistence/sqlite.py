"""
SQLite chat storage — built-in lightweight OrchidChatStorage implementation.

Default storage backend for orchid. Data is stored in a single file
(default: ``~/.orchid/chats.db``) or ``:memory:`` for tests.

Configuration:
    CHAT_STORAGE_CLASS=orchid.persistence.sqlite.OrchidSQLiteChatStorage
    CHAT_DB_DSN=~/.orchid/chats.db
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any

import aiosqlite

from .base import OrchidChatStorage
from .migrations.runner import OrchidMigrationRunner
from .models import OrchidChatMessage, OrchidChatSession, utcnow

logger = logging.getLogger(__name__)

MIGRATIONS_PACKAGE = "orchid_ai.persistence.migrations"


class OrchidSQLiteMigrationRunner(OrchidMigrationRunner):
    """SQLite-specific migration tracking."""

    dialect = "sqlite"
    migrations_package = MIGRATIONS_PACKAGE

    async def ensure_migrations_table(self, conn: Any) -> None:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS _migrations (
                version TEXT PRIMARY KEY,
                description TEXT NOT NULL DEFAULT '',
                applied_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        await conn.commit()

    async def get_applied_versions(self, conn: Any) -> set[str]:
        cursor = await conn.execute("SELECT version FROM _migrations")
        rows = await cursor.fetchall()
        return {r[0] for r in rows}

    async def record_version(self, conn: Any, version: str, description: str) -> None:
        await conn.execute(
            "INSERT INTO _migrations (version, description) VALUES (?, ?)",
            (version, description),
        )
        await conn.commit()

    async def remove_version(self, conn: Any, version: str) -> None:
        await conn.execute("DELETE FROM _migrations WHERE version = ?", (version,))
        await conn.commit()


class OrchidSQLiteChatStorage(OrchidChatStorage):
    """
    Async SQLite storage for chat sessions and messages.

    Constructor accepts the file path via ``dsn`` and an optional
    ``extra_migrations_package`` (dotted import path) so integrators
    can append their own migrations after the framework's — see
    :class:`orchid_ai.persistence.migrations.runner.OrchidMigrationRunner`.

    Use ``:memory:`` for in-memory databases (tests).  The default path
    ``~/.orchid/chats.db`` is resolved at init time.
    """

    def __init__(self, *, dsn: str, extra_migrations_package: str | None = None):
        self._db_path = os.path.expanduser(dsn)
        self._conn: aiosqlite.Connection | None = None
        self._migrator = OrchidSQLiteMigrationRunner(
            extra_migrations_package=extra_migrations_package,
        )

    # ── Lifecycle ────────────────────────────────────────────

    async def init_db(self) -> None:
        if self._db_path != ":memory:":
            os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)

        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._migrator.run_up(self._conn)
        logger.info("[OrchidChatStorage:sqlite] Initialised — %s", self._db_path)

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()

    # ── Sessions ─────────────────────────────────────────────

    async def create_chat(
        self,
        tenant_id: str,
        user_id: str,
        title: str = "",
    ) -> OrchidChatSession:
        now = utcnow()
        now_iso = now.isoformat()
        chat = OrchidChatSession(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            user_id=user_id,
            title=title or "New chat",
            created_at=now,
            updated_at=now,
        )
        await self._conn.execute(
            "INSERT INTO chat_sessions (id, tenant_id, user_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (chat.id, chat.tenant_id, chat.user_id, chat.title, now_iso, now_iso),
        )
        await self._conn.commit()
        return chat

    async def list_chats(
        self,
        tenant_id: str,
        user_id: str,
    ) -> list[OrchidChatSession]:
        cursor = await self._conn.execute(
            "SELECT * FROM chat_sessions WHERE tenant_id = ? AND user_id = ? ORDER BY updated_at DESC",
            (tenant_id, user_id),
        )
        rows = await cursor.fetchall()
        return [_row_to_session(r) for r in rows]

    async def get_chat(self, chat_id: str) -> OrchidChatSession | None:
        cursor = await self._conn.execute(
            "SELECT * FROM chat_sessions WHERE id = ?",
            (chat_id,),
        )
        row = await cursor.fetchone()
        return _row_to_session(row) if row else None

    async def delete_chat(self, chat_id: str) -> None:
        await self._conn.execute("DELETE FROM chat_sessions WHERE id = ?", (chat_id,))
        await self._conn.commit()

    async def update_title(self, chat_id: str, title: str) -> None:
        now_iso = utcnow().isoformat()
        await self._conn.execute(
            "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, now_iso, chat_id),
        )
        await self._conn.commit()

    async def mark_shared(self, chat_id: str) -> None:
        now_iso = utcnow().isoformat()
        await self._conn.execute(
            "UPDATE chat_sessions SET is_shared = 1, updated_at = ? WHERE id = ?",
            (now_iso, chat_id),
        )
        await self._conn.commit()

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
        now_iso = now.isoformat()
        msg = OrchidChatMessage(
            id=str(uuid.uuid4()),
            chat_id=chat_id,
            role=role,
            content=content,
            agents_used=agents_used or [],
            created_at=now,
            metadata=metadata or {},
        )
        await self._conn.execute(
            "INSERT INTO chat_messages (id, chat_id, role, content, agents_used, created_at, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                msg.id,
                msg.chat_id,
                msg.role,
                msg.content,
                json.dumps(msg.agents_used),
                now_iso,
                json.dumps(msg.metadata),
            ),
        )
        await self._conn.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
            (now_iso, chat_id),
        )
        await self._conn.commit()
        return msg

    async def get_messages(
        self,
        chat_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[OrchidChatMessage]:
        cursor = await self._conn.execute(
            "SELECT * FROM chat_messages WHERE chat_id = ? ORDER BY created_at ASC LIMIT ? OFFSET ?",
            (chat_id, limit, offset),
        )
        rows = await cursor.fetchall()
        return [_row_to_message(r) for r in rows]


# ── Row mappers ──────────────────────────────────────────────


def _parse_dt(val: str | datetime) -> datetime:
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(val)
    except (ValueError, TypeError):
        return utcnow()


def _row_to_session(row: aiosqlite.Row) -> OrchidChatSession:
    return OrchidChatSession(
        id=row["id"],
        tenant_id=row["tenant_id"],
        user_id=row["user_id"],
        title=row["title"],
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
        is_shared=bool(row["is_shared"]),
    )


def _row_to_message(row: aiosqlite.Row) -> OrchidChatMessage:
    agents_used = row["agents_used"]
    if isinstance(agents_used, str):
        agents_used = json.loads(agents_used)
    meta = row["metadata"]
    if isinstance(meta, str):
        meta = json.loads(meta)
    return OrchidChatMessage(
        id=row["id"],
        chat_id=row["chat_id"],
        role=row["role"],
        content=row["content"],
        agents_used=agents_used,
        created_at=_parse_dt(row["created_at"]),
        metadata=meta,
    )
