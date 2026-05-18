"""PostgreSQL-backed client registration store (RFC 7591 DCR).

Mirrors :class:`orchid_ai.persistence.mcp_token_postgres.OrchidPostgresMCPTokenStore`
but for per-server registration metadata — one row per MCP server,
keyed by ``server_name`` alone.  The underlying table is created by
the shared v001 initial schema.

Optional dependency — install via ``pip install orchid-ai[postgres]``.

Configuration::

    MCP_CLIENT_REGISTRATION_STORE_CLASS=orchid_ai.persistence.mcp_client_registration_postgres.OrchidPostgresMCPClientRegistrationStore
    MCP_CLIENT_REGISTRATION_STORE_DSN=postgresql://user:pass@host:5432/db
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..core.mcp import OrchidMCPClientRegistration, OrchidMCPClientRegistrationStore
from .postgres import OrchidPostgresMigrationRunner

logger = logging.getLogger(__name__)


class OrchidPostgresMCPClientRegistrationStore(OrchidMCPClientRegistrationStore):
    """Async PostgreSQL storage for :class:`OrchidMCPClientRegistration`.

    Pool: ``min_size=2, max_size=10``, matching the token-store sibling
    to keep operator-visible connection usage symmetric across the two
    stores that typically share a DSN.
    """

    def __init__(self, *, dsn: str, extra_migrations_package: str | None = None) -> None:
        self._dsn = dsn
        self._pool: Any = None
        self._migrator = OrchidPostgresMigrationRunner(
            extra_migrations_package=extra_migrations_package,
        )

    # ── Lifecycle ────────────────────────────────────────────

    async def init_db(self) -> None:
        import asyncpg

        self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
        async with self._pool.acquire() as conn:
            await self._migrator.run_up(conn)
        logger.info(
            "[OrchidMCPClientRegistrationStore:postgres] Initialised — %s",
            self._dsn.split("@")[-1] if "@" in self._dsn else "***",
        )

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    # ── CRUD ─────────────────────────────────────────────────

    async def get(self, server_name: str) -> OrchidMCPClientRegistration | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM mcp_client_registrations WHERE server_name = $1",
                server_name,
            )
        return _row_to_record(row) if row else None

    async def save(self, record: OrchidMCPClientRegistration) -> None:
        now = time.time()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO mcp_client_registrations
                    (server_name, authorization_endpoint, token_endpoint, registration_endpoint,
                     issuer, scopes_supported, token_endpoint_auth_methods_supported,
                     client_id, client_secret, client_id_issued_at, client_secret_expires_at,
                     created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, to_timestamp($12), to_timestamp($13))
                ON CONFLICT (server_name) DO UPDATE SET
                    authorization_endpoint = EXCLUDED.authorization_endpoint,
                    token_endpoint = EXCLUDED.token_endpoint,
                    registration_endpoint = EXCLUDED.registration_endpoint,
                    issuer = EXCLUDED.issuer,
                    scopes_supported = EXCLUDED.scopes_supported,
                    token_endpoint_auth_methods_supported = EXCLUDED.token_endpoint_auth_methods_supported,
                    client_id = EXCLUDED.client_id,
                    client_secret = EXCLUDED.client_secret,
                    client_id_issued_at = EXCLUDED.client_id_issued_at,
                    client_secret_expires_at = EXCLUDED.client_secret_expires_at,
                    updated_at = to_timestamp($13)
                """,
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
            )

    async def delete(self, server_name: str) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM mcp_client_registrations WHERE server_name = $1",
                server_name,
            )
        return result.endswith("1") if isinstance(result, str) else False


# ── Row mapper ──────────────────────────────────────────────


def _row_to_record(row: Any) -> OrchidMCPClientRegistration:
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
        created_at=row["created_at"].timestamp()
        if hasattr(row["created_at"], "timestamp")
        else float(row["created_at"]),
        updated_at=row["updated_at"].timestamp()
        if hasattr(row["updated_at"], "timestamp")
        else float(row["updated_at"]),
    )
