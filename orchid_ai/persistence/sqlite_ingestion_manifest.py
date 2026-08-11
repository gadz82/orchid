"""SQLite ingestion manifest — built-in lightweight OrchidIngestionManifest implementation.

Shares the same database and migration system as ``OrchidSQLiteChatStorage``.
The ``ingestion_manifest`` table is created by the framework migration
``v002_ingestion_manifest``.

Configuration:
    INDEX_MANIFEST_CLASS=orchid_ai.persistence.sqlite_ingestion_manifest.OrchidSQLiteIngestionManifest
    INDEX_MANIFEST_DSN=~/.orchid/index_manifest.db
"""

from __future__ import annotations

import json
import logging
import os

import aiosqlite

from ..core.ingestion_manifest import OrchidIngestionManifest
from .sqlite import OrchidSQLiteMigrationRunner

logger = logging.getLogger(__name__)


class OrchidSQLiteIngestionManifest(OrchidIngestionManifest):
    """Async SQLite storage for ingestion manifests.

    Tracks content hashes and vector document IDs per ``(source_id,
    namespace)`` so callers can skip unchanged files and prune removed
    sources.

    Use ``:memory:`` for in-memory databases (tests).
    """

    def __init__(self, *, dsn: str, extra_migrations_package: str | None = None):
        self._db_path = os.path.expanduser(dsn)
        self._conn: aiosqlite.Connection | None = None
        self._migrator = OrchidSQLiteMigrationRunner(
            extra_migrations_package=extra_migrations_package,
        )

    async def init_db(self) -> None:
        if self._db_path != ":memory:":
            os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)

        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._migrator.run_up(self._conn)
        logger.info("[OrchidIngestionManifest:sqlite] Initialised — %s", self._db_path)

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def should_skip(self, source_id: str, content_hash: str, namespace: str) -> bool:
        row = await self._get_row(source_id, namespace)
        if row is None:
            return False
        return row["content_hash"] == content_hash

    async def record(
        self,
        source_id: str,
        content_hash: str,
        namespace: str,
        document_ids: list[str],
    ) -> None:
        self._ensure_conn()
        await self._conn.execute(
            "INSERT OR REPLACE INTO ingestion_manifest "
            "(source_id, namespace, content_hash, document_ids, indexed_at) "
            "VALUES (?, ?, ?, ?, strftime('%s', 'now') * 1000)",
            (source_id, namespace, content_hash, json.dumps(document_ids)),
        )
        await self._conn.commit()

    async def remove(self, source_id: str, namespace: str) -> None:
        self._ensure_conn()
        await self._conn.execute(
            "DELETE FROM ingestion_manifest WHERE source_id = ? AND namespace = ?",
            (source_id, namespace),
        )
        await self._conn.commit()

    async def list_known(self, namespace: str) -> set[str]:
        self._ensure_conn()
        cursor = await self._conn.execute(
            "SELECT source_id FROM ingestion_manifest WHERE namespace = ?",
            (namespace,),
        )
        rows = await cursor.fetchall()
        return {row["source_id"] for row in rows}

    async def get_document_ids(self, source_id: str, namespace: str) -> list[str]:
        row = await self._get_row(source_id, namespace)
        if row is None:
            return []
        try:
            return json.loads(row["document_ids"])
        except json.JSONDecodeError:
            return []

    async def _get_row(self, source_id: str, namespace: str) -> aiosqlite.Row | None:
        self._ensure_conn()
        cursor = await self._conn.execute(
            "SELECT * FROM ingestion_manifest WHERE source_id = ? AND namespace = ?",
            (source_id, namespace),
        )
        return await cursor.fetchone()

    def _ensure_conn(self) -> None:
        if self._conn is None:
            raise RuntimeError("OrchidSQLiteIngestionManifest is not initialised. Call init_db() first.")
