"""Tests for the built-in in-memory ingestion manifest (framework default)."""

from __future__ import annotations

from orchid_ai.core.ingestion_manifest import OrchidIngestionManifest
from orchid_ai.persistence.in_memory import OrchidInMemoryIngestionManifest


class TestInMemoryIngestionManifest:
    def test_is_an_ingestion_manifest(self):
        assert isinstance(OrchidInMemoryIngestionManifest(), OrchidIngestionManifest)

    def test_constructor_accepts_factory_kwargs(self):
        manifest = OrchidInMemoryIngestionManifest(dsn="ignored", extra_migrations_package="a.b")
        assert isinstance(manifest, OrchidInMemoryIngestionManifest)

    async def test_should_skip_only_on_matching_hash(self):
        manifest = OrchidInMemoryIngestionManifest()
        assert await manifest.should_skip("docs/a.md", "hash-1", "ns") is False

        await manifest.record("docs/a.md", "hash-1", "ns", ["doc-1", "doc-2"])
        assert await manifest.should_skip("docs/a.md", "hash-1", "ns") is True
        assert await manifest.should_skip("docs/a.md", "hash-2", "ns") is False

    async def test_record_is_an_upsert(self):
        manifest = OrchidInMemoryIngestionManifest()
        await manifest.record("docs/a.md", "hash-1", "ns", ["doc-1"])
        await manifest.record("docs/a.md", "hash-2", "ns", ["doc-2"])

        assert await manifest.should_skip("docs/a.md", "hash-2", "ns") is True
        assert await manifest.get_document_ids("docs/a.md", "ns") == ["doc-2"]

    async def test_scope_partitions_entries(self):
        manifest = OrchidInMemoryIngestionManifest()
        await manifest.record("docs/a.md", "hash-1", "ns", ["doc-1"], scope="tenant:t1")
        await manifest.record("docs/a.md", "hash-2", "ns", ["doc-2"], scope="tenant:t2")

        assert await manifest.should_skip("docs/a.md", "hash-1", "ns", "tenant:t1") is True
        assert await manifest.should_skip("docs/a.md", "hash-1", "ns", "tenant:t2") is False
        assert await manifest.get_document_ids("docs/a.md", "ns", "tenant:t2") == ["doc-2"]

    async def test_remove(self):
        manifest = OrchidInMemoryIngestionManifest()
        await manifest.record("docs/a.md", "hash-1", "ns", ["doc-1"])

        await manifest.remove("docs/a.md", "ns")
        assert await manifest.should_skip("docs/a.md", "hash-1", "ns") is False
        assert await manifest.get_document_ids("docs/a.md", "ns") == []

        await manifest.remove("docs/a.md", "ns")  # idempotent

    async def test_list_known(self):
        manifest = OrchidInMemoryIngestionManifest()
        await manifest.record("docs/a.md", "h1", "ns-a", ["d1"])
        await manifest.record("docs/b.md", "h2", "ns-a", ["d2"])
        await manifest.record("docs/c.md", "h3", "ns-b", ["d3"])

        assert await manifest.list_known("ns-a") == {"docs/a.md", "docs/b.md"}
        assert await manifest.list_known("ns-b") == {"docs/c.md"}
        assert await manifest.list_known("missing") == set()

    async def test_get_document_ids_returns_a_copy(self):
        manifest = OrchidInMemoryIngestionManifest()
        await manifest.record("docs/a.md", "h1", "ns", ["d1"])

        ids = await manifest.get_document_ids("docs/a.md", "ns")
        ids.append("mutated")
        assert await manifest.get_document_ids("docs/a.md", "ns") == ["d1"]

    async def test_lifecycle_noops(self):
        manifest = OrchidInMemoryIngestionManifest()
        await manifest.init_db()
        await manifest.close()
