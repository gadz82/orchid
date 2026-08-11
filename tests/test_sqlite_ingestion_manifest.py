"""Tests for OrchidSQLiteIngestionManifest — in-memory SQLite manifest persistence."""

from __future__ import annotations

import pytest

from orchid_ai.persistence.sqlite_ingestion_manifest import OrchidSQLiteIngestionManifest


@pytest.fixture
async def manifest():
    """Create an in-memory manifest store for each test."""
    m = OrchidSQLiteIngestionManifest(dsn=":memory:")
    await m.init_db()
    yield m
    await m.close()


@pytest.mark.asyncio
class TestSQLiteIngestionManifest:
    async def test_should_skip_returns_false_for_unknown_source(self, manifest: OrchidSQLiteIngestionManifest):
        assert await manifest.should_skip("src-1", "hash-1", "ns-1") is False

    async def test_record_then_skip_unchanged(self, manifest: OrchidSQLiteIngestionManifest):
        await manifest.record("src-1", "hash-1", "ns-1", ["doc-1", "doc-2"])

        assert await manifest.should_skip("src-1", "hash-1", "ns-1") is True
        assert await manifest.should_skip("src-1", "hash-2", "ns-1") is False

    async def test_record_upserts_hash_and_document_ids(self, manifest: OrchidSQLiteIngestionManifest):
        await manifest.record("src-1", "hash-1", "ns-1", ["doc-1"])
        assert await manifest.should_skip("src-1", "hash-1", "ns-1") is True

        await manifest.record("src-1", "hash-2", "ns-1", ["doc-2", "doc-3"])
        assert await manifest.should_skip("src-1", "hash-1", "ns-1") is False
        assert await manifest.should_skip("src-1", "hash-2", "ns-1") is True

    async def test_list_known(self, manifest: OrchidSQLiteIngestionManifest):
        await manifest.record("src-1", "hash-1", "ns-1", ["doc-1"])
        await manifest.record("src-2", "hash-2", "ns-1", ["doc-2"])
        await manifest.record("src-3", "hash-3", "ns-2", ["doc-3"])

        known = await manifest.list_known("ns-1")
        assert known == {"src-1", "src-2"}

    async def test_get_document_ids(self, manifest: OrchidSQLiteIngestionManifest):
        await manifest.record("src-1", "hash-1", "ns-1", ["doc-1", "doc-2"])
        assert await manifest.get_document_ids("src-1", "ns-1") == ["doc-1", "doc-2"]
        assert await manifest.get_document_ids("unknown", "ns-1") == []

    async def test_remove(self, manifest: OrchidSQLiteIngestionManifest):
        await manifest.record("src-1", "hash-1", "ns-1", ["doc-1"])
        assert await manifest.should_skip("src-1", "hash-1", "ns-1") is True

        await manifest.remove("src-1", "ns-1")
        assert await manifest.should_skip("src-1", "hash-1", "ns-1") is False
        assert await manifest.list_known("ns-1") == set()

    async def test_namespaces_are_isolated(self, manifest: OrchidSQLiteIngestionManifest):
        await manifest.record("src-1", "hash-1", "ns-1", ["doc-1"])
        await manifest.record("src-1", "hash-2", "ns-2", ["doc-2"])

        assert await manifest.should_skip("src-1", "hash-1", "ns-1") is True
        assert await manifest.should_skip("src-1", "hash-2", "ns-2") is True
        assert await manifest.should_skip("src-1", "hash-1", "ns-2") is False

    async def test_close_then_operation_raises(self, manifest: OrchidSQLiteIngestionManifest):
        await manifest.close()
        with pytest.raises(RuntimeError, match="not initialised"):
            await manifest.should_skip("src-1", "hash-1", "ns-1")
