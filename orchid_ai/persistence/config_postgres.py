"""
PostgreSQL config storage — built-in OrchidConfigStorage implementation.

Uses asyncpg for async connection pooling. The ``agent_configs`` table
DDL lives in the shared ``_schema_ddl.py`` (same place as all other
framework tables) and is created when :meth:`init_db` runs.

Configuration::

    CONFIG_STORAGE_CLASS=orchid_ai.persistence.config_postgres.OrchidPostgresConfigStorage
    CONFIG_DB_DSN=postgresql://user:pass@host:5432/dbname
"""

from __future__ import annotations

import json
import logging
from typing import Any

import asyncpg

from orchid_ai.config.storage import OrchidConfigStorage
from orchid_ai.config.schema_agent import OrchidAgentConfig, _deep_merge
from orchid_ai.persistence.migrations.runner import OrchidMigrationRunner

logger = logging.getLogger(__name__)

MIGRATIONS_PACKAGE = "orchid_ai.persistence.migrations"


class OrchidPostgresMigrationRunner(OrchidMigrationRunner):
    """PostgreSQL-specific migration tracking for config storage."""

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


class OrchidPostgresConfigStorage(OrchidConfigStorage):
    """Async PostgreSQL storage for agent configurations.

    Uses the same migration runner as ``OrchidPostgresChatStorage`` — the
    shared ``_schema_ddl.py`` is applied once per database regardless of
    which storage class initiates it. ``CREATE TABLE IF NOT EXISTS``
    makes this safe to call multiple times.
    """

    def __init__(self, *, dsn: str) -> None:
        """Keyword-only constructor.

        Parameters
        ----------
        dsn : str
            PostgreSQL connection string
            (e.g. ``"postgresql://user:pass@host:5432/db"``).
        """
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None
        self._migrator = OrchidPostgresMigrationRunner()

    async def init_db(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
        async with self._pool.acquire() as conn:
            await self._migrator.run_up(conn)
        safe_dsn = self._dsn.split("@")[-1] if "@" in self._dsn else "***"
        logger.info("[OrchidConfigStorage:postgres] Initialised — %s", safe_dsn)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def list_configs(self) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT name, config, created_at, updated_at FROM agent_configs ORDER BY updated_at DESC"
            )
            return [_row_to_config(r) for r in rows]

    async def get_config(self, name: str) -> dict | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT name, config, created_at, updated_at FROM agent_configs WHERE name = $1",
                name,
            )
            return _row_to_config(row) if row else None

    async def upsert_config(self, name: str, config: dict) -> dict:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO agent_configs (name, config, created_at, updated_at)
                VALUES ($1, $2, NOW(), NOW())
                ON CONFLICT (name) DO UPDATE SET
                    config = EXCLUDED.config,
                    updated_at = NOW()
                RETURNING name, config, created_at, updated_at
                """,
                name,
                json.dumps(config),
            )
            return _row_to_config(row)

    async def patch_config(self, name: str, patch: dict) -> dict | None:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT name, config, created_at, updated_at FROM agent_configs WHERE name = $1 FOR UPDATE",
                    name,
                )
                if row is None:
                    return None
                existing = _row_to_config(row)
                merged = _deep_merge(existing["config"], patch)
                OrchidAgentConfig.model_validate(merged)
                updated = await conn.fetchrow(
                    """
                    UPDATE agent_configs
                    SET config = $2, updated_at = NOW()
                    WHERE name = $1
                    RETURNING name, config, created_at, updated_at
                    """,
                    name,
                    json.dumps(merged),
                )
                return _row_to_config(updated)

    async def delete_config(self, name: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM agent_configs WHERE name = $1", name)


def _row_to_config(row: asyncpg.Record) -> dict:
    config = row["config"]
    if isinstance(config, str):
        config = json.loads(config)
    return {
        "name": row["name"],
        "config": config,
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
    }
