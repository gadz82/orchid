"""
SQLite MCP token storage — built-in lightweight OrchidMCPTokenStore implementation.

Shares the same database and migration system as ``OrchidSQLiteChatStorage``.
The ``mcp_oauth_tokens`` table is created by the unified
``v001_initial_schema`` in the shared ``orchid_ai.persistence.migrations``
package.

Configuration:
    MCP_TOKEN_STORE_CLASS=orchid_ai.persistence.mcp_token_sqlite.OrchidSQLiteMCPTokenStore
    MCP_TOKEN_STORE_DSN=~/.orchid/chats.db
"""

from __future__ import annotations

import logging
import os
import time

import aiosqlite

from ..core.mcp import OrchidMCPTokenRecord, OrchidMCPTokenStore
from .sqlite import OrchidSQLiteMigrationRunner

logger = logging.getLogger(__name__)


class OrchidSQLiteMCPTokenStore(OrchidMCPTokenStore):
    """Async SQLite storage for per-server OAuth tokens.

    Constructor accepts the file path via ``dsn`` and an optional
    ``extra_migrations_package`` (dotted import path) so integrators
    can append their own migrations after the framework's — see
    :class:`orchid_ai.persistence.migrations.runner.OrchidMigrationRunner`.

    Use ``:memory:`` for in-memory databases (tests).
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
        await self._migrator.run_up(self._conn)
        logger.info("[OrchidMCPTokenStore:sqlite] Initialised — %s", self._db_path)

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()

    # ── CRUD ─────────────────────────────────────────────────

    async def get_token(
        self,
        tenant_id: str,
        user_id: str,
        server_name: str,
    ) -> OrchidMCPTokenRecord | None:
        cursor = await self._conn.execute(
            "SELECT * FROM mcp_oauth_tokens WHERE server_name = ? AND tenant_id = ? AND user_id = ?",
            (server_name, tenant_id, user_id),
        )
        row = await cursor.fetchone()
        return _row_to_record(row) if row else None

    async def save_token(self, record: OrchidMCPTokenRecord) -> None:
        now = time.time()
        await self._conn.execute(
            "INSERT OR REPLACE INTO mcp_oauth_tokens "
            "(server_name, tenant_id, user_id, access_token, refresh_token, "
            " expires_at, scopes, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.server_name,
                record.tenant_id,
                record.user_id,
                record.access_token,
                record.refresh_token,
                record.expires_at,
                record.scopes,
                record.created_at,
                now,
            ),
        )
        await self._conn.commit()

    async def delete_token(
        self,
        tenant_id: str,
        user_id: str,
        server_name: str,
    ) -> bool:
        cursor = await self._conn.execute(
            "DELETE FROM mcp_oauth_tokens WHERE server_name = ? AND tenant_id = ? AND user_id = ?",
            (server_name, tenant_id, user_id),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    async def list_tokens(
        self,
        tenant_id: str,
        user_id: str,
    ) -> list[OrchidMCPTokenRecord]:
        cursor = await self._conn.execute(
            "SELECT * FROM mcp_oauth_tokens WHERE tenant_id = ? AND user_id = ?",
            (tenant_id, user_id),
        )
        rows = await cursor.fetchall()
        return [_row_to_record(r) for r in rows]

    async def cleanup_expired(self, *, before: float | None = None) -> int:
        """Single ``DELETE`` purging rows whose ``expires_at`` is in the past.

        Safe to run while the gateway is serving traffic — callers
        check ``record.is_expired`` before use, so an expired row is
        unreachable to any live request before this method runs.
        """
        cutoff = before if before is not None else time.time()
        cursor = await self._conn.execute(
            "DELETE FROM mcp_oauth_tokens WHERE expires_at > 0 AND expires_at < ?",
            (cutoff,),
        )
        await self._conn.commit()
        return cursor.rowcount


# ── Row mapper ──────────────────────────────────────────────


def _row_to_record(row: aiosqlite.Row) -> OrchidMCPTokenRecord:
    return OrchidMCPTokenRecord(
        server_name=row["server_name"],
        tenant_id=row["tenant_id"],
        user_id=row["user_id"],
        access_token=row["access_token"],
        refresh_token=row["refresh_token"],
        expires_at=float(row["expires_at"]),
        scopes=row["scopes"],
        created_at=float(row["created_at"]) if isinstance(row["created_at"], (int, float)) else time.time(),
        updated_at=float(row["updated_at"]) if isinstance(row["updated_at"], (int, float)) else time.time(),
    )
