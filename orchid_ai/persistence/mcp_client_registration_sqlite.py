"""SQLite-backed client registration store (RFC 7591 DCR).

Mirrors :class:`orchid_ai.persistence.mcp_token_sqlite.OrchidSQLiteMCPTokenStore`
but for per-server metadata rather than per-user tokens — one row per
MCP server, keyed by ``server_name`` alone.  The underlying table is
created by the shared v001 initial schema.

Configuration::

    MCP_CLIENT_REGISTRATION_STORE_CLASS=orchid_ai.persistence.mcp_client_registration_sqlite.OrchidSQLiteMCPClientRegistrationStore
    MCP_CLIENT_REGISTRATION_STORE_DSN=~/.orchid/chats.db
"""

from __future__ import annotations

import logging
import os
import time

import aiosqlite

from ..core.mcp import OrchidMCPClientRegistration, OrchidMCPClientRegistrationStore
from .sqlite import OrchidSQLiteMigrationRunner

logger = logging.getLogger(__name__)


class OrchidSQLiteMCPClientRegistrationStore(OrchidMCPClientRegistrationStore):
    """Async SQLite storage for :class:`OrchidMCPClientRegistration`.

    Constructor accepts the file path via ``dsn`` and an optional
    ``extra_migrations_package`` so integrators can stack their own
    schema alongside the framework's.  ``:memory:`` is supported for
    tests.
    """

    def __init__(self, *, dsn: str, extra_migrations_package: str | None = None) -> None:
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
        logger.info(
            "[OrchidMCPClientRegistrationStore:sqlite] Initialised — %s",
            self._db_path,
        )

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    # ── CRUD ─────────────────────────────────────────────────

    async def get(self, server_name: str) -> OrchidMCPClientRegistration | None:
        assert self._conn is not None, "init_db() must be called first"
        cursor = await self._conn.execute(
            "SELECT * FROM mcp_client_registrations WHERE server_name = ?",
            (server_name,),
        )
        row = await cursor.fetchone()
        return _row_to_record(row) if row else None

    async def save(self, record: OrchidMCPClientRegistration) -> None:
        assert self._conn is not None, "init_db() must be called first"
        now = time.time()
        await self._conn.execute(
            "INSERT OR REPLACE INTO mcp_client_registrations "
            "(server_name, authorization_endpoint, token_endpoint, registration_endpoint, "
            " issuer, scopes_supported, token_endpoint_auth_methods_supported, "
            " client_id, client_secret, client_id_issued_at, client_secret_expires_at, "
            " created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.server_name,
                record.authorization_endpoint,
                record.token_endpoint,
                record.registration_endpoint,
                record.issuer,
                record.scopes_supported,
                record.token_endpoint_auth_methods_supported,
                record.client_id,
                record.client_secret,
                record.client_id_issued_at,
                record.client_secret_expires_at,
                record.created_at,
                now,
            ),
        )
        await self._conn.commit()

    async def delete(self, server_name: str) -> bool:
        assert self._conn is not None, "init_db() must be called first"
        cursor = await self._conn.execute(
            "DELETE FROM mcp_client_registrations WHERE server_name = ?",
            (server_name,),
        )
        await self._conn.commit()
        return cursor.rowcount > 0


# ── Row mapper ──────────────────────────────────────────────


def _row_to_record(row: aiosqlite.Row) -> OrchidMCPClientRegistration:
    return OrchidMCPClientRegistration(
        server_name=row["server_name"],
        authorization_endpoint=row["authorization_endpoint"],
        token_endpoint=row["token_endpoint"],
        registration_endpoint=row["registration_endpoint"],
        issuer=row["issuer"],
        scopes_supported=row["scopes_supported"],
        token_endpoint_auth_methods_supported=row["token_endpoint_auth_methods_supported"],
        client_id=row["client_id"],
        client_secret=row["client_secret"],
        client_id_issued_at=float(row["client_id_issued_at"]),
        client_secret_expires_at=float(row["client_secret_expires_at"]),
        created_at=float(row["created_at"]) if isinstance(row["created_at"], (int, float)) else time.time(),
        updated_at=float(row["updated_at"]) if isinstance(row["updated_at"], (int, float)) else time.time(),
    )
