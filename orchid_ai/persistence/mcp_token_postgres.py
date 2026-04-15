"""
PostgreSQL MCP token storage — MCPTokenStore implementation using asyncpg.

Shares the same database and migration system as ``PostgresChatStorage``.
The ``mcp_oauth_tokens`` table is created by ``v002_mcp_tokens_schema``
in the shared ``orchid_ai.persistence.migrations`` package.

Optional dependency — install via ``pip install orchid-ai[postgres]``.

Configuration:
    MCP_TOKEN_STORE_CLASS=orchid_ai.persistence.mcp_token_postgres.PostgresMCPTokenStore
    MCP_TOKEN_STORE_DSN=postgresql://user:pass@host:5432/db
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..core.mcp import MCPTokenRecord, MCPTokenStore
from .postgres import PostgresMigrationRunner

logger = logging.getLogger(__name__)


class PostgresMCPTokenStore(MCPTokenStore):
    """Async PostgreSQL storage for per-server OAuth tokens.

    Uses connection pooling via asyncpg (min_size=2, max_size=10).
    Constructor accepts a single ``dsn`` keyword argument.
    """

    def __init__(self, *, dsn: str):
        self._dsn = dsn
        self._pool: Any = None
        self._migrator = PostgresMigrationRunner()

    # ── Lifecycle ────────────────────────────────────────────

    async def init_db(self) -> None:
        import asyncpg

        self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
        async with self._pool.acquire() as conn:
            await self._migrator.run_up(conn)
        logger.info(
            "[MCPTokenStore:postgres] Initialised — %s", self._dsn.split("@")[-1] if "@" in self._dsn else "***"
        )

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    # ── CRUD ─────────────────────────────────────────────────

    async def get_token(
        self,
        tenant_id: str,
        user_id: str,
        server_name: str,
    ) -> MCPTokenRecord | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM mcp_oauth_tokens WHERE server_name = $1 AND tenant_id = $2 AND user_id = $3",
                server_name,
                tenant_id,
                user_id,
            )
        return _row_to_record(row) if row else None

    async def save_token(self, record: MCPTokenRecord) -> None:
        now = time.time()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO mcp_oauth_tokens
                    (server_name, tenant_id, user_id, access_token, refresh_token,
                     expires_at, scopes, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, to_timestamp($8), to_timestamp($9))
                ON CONFLICT (server_name, tenant_id, user_id) DO UPDATE SET
                    access_token = EXCLUDED.access_token,
                    refresh_token = EXCLUDED.refresh_token,
                    expires_at = EXCLUDED.expires_at,
                    scopes = EXCLUDED.scopes,
                    updated_at = to_timestamp($9)
                """,
                record.server_name,
                record.tenant_id,
                record.user_id,
                record.access_token,
                record.refresh_token,
                record.expires_at,
                record.scopes,
                record.created_at,
                now,
            )

    async def delete_token(
        self,
        tenant_id: str,
        user_id: str,
        server_name: str,
    ) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM mcp_oauth_tokens WHERE server_name = $1 AND tenant_id = $2 AND user_id = $3",
                server_name,
                tenant_id,
                user_id,
            )
        return result.endswith("1") if isinstance(result, str) else False

    async def list_tokens(
        self,
        tenant_id: str,
        user_id: str,
    ) -> list[MCPTokenRecord]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM mcp_oauth_tokens WHERE tenant_id = $1 AND user_id = $2",
                tenant_id,
                user_id,
            )
        return [_row_to_record(r) for r in rows]


# ── Row mapper ──────────────────────────────────────────────


def _row_to_record(row: Any) -> MCPTokenRecord:
    return MCPTokenRecord(
        server_name=row["server_name"],
        tenant_id=row["tenant_id"],
        user_id=row["user_id"],
        access_token=row["access_token"],
        refresh_token=row["refresh_token"],
        expires_at=float(row["expires_at"]),
        scopes=row["scopes"],
        created_at=row["created_at"].timestamp()
        if hasattr(row["created_at"], "timestamp")
        else float(row["created_at"]),
        updated_at=row["updated_at"].timestamp()
        if hasattr(row["updated_at"], "timestamp")
        else float(row["updated_at"]),
    )
