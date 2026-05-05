"""Tests for Qdrant payload-index auto-creation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchid_ai.core.scopes import OrchidRAGScope
from orchid_ai.rag.backends.qdrant import QdrantRepository


def _mock_client() -> MagicMock:
    """Mock AsyncQdrantClient with the methods QdrantRepository touches."""
    client = MagicMock()
    client.collection_exists = AsyncMock(return_value=False)
    client.create_collection = AsyncMock()
    client.create_payload_index = AsyncMock()
    client.query_points = AsyncMock(return_value=MagicMock(points=[]))
    return client


def _repo(client) -> QdrantRepository:
    embeddings = MagicMock()
    embeddings.aembed_query = AsyncMock(return_value=[0.0])
    return QdrantRepository(
        url="http://x",
        embeddings=embeddings,
        embedding_dimension=1,
        client=client,
    )


def _scope() -> OrchidRAGScope:
    return OrchidRAGScope(tenant_id="t1", user_id="u1", chat_id="c1", agent_id="a1")


class TestEnsurePayloadIndexes:
    @pytest.mark.asyncio
    async def test_explicit_indexes_create_once(self):
        client = _mock_client()
        repo = _repo(client)
        await repo.ensure_payload_indexes("kb", {"status": "keyword", "view_count": "integer"})
        assert client.create_payload_index.await_count == 2

    @pytest.mark.asyncio
    async def test_idempotent_repeat_calls(self):
        client = _mock_client()
        repo = _repo(client)
        await repo.ensure_payload_indexes("kb", {"status": "keyword"})
        await repo.ensure_payload_indexes("kb", {"status": "keyword"})
        # Cached — only the first call hit Qdrant.
        assert client.create_payload_index.await_count == 1

    @pytest.mark.asyncio
    async def test_per_namespace_isolation(self):
        client = _mock_client()
        repo = _repo(client)
        await repo.ensure_payload_indexes("ns_a", {"status": "keyword"})
        await repo.ensure_payload_indexes("ns_b", {"status": "keyword"})
        # Same field name but different namespaces — both create calls fire.
        assert client.create_payload_index.await_count == 2

    @pytest.mark.asyncio
    async def test_qdrant_errors_swallowed(self, caplog):
        client = _mock_client()
        client.create_payload_index = AsyncMock(side_effect=RuntimeError("schema clash"))
        repo = _repo(client)
        with caplog.at_level("WARNING"):
            await repo.ensure_payload_indexes("kb", {"status": "keyword"})
        assert any("payload index" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_empty_indexes_short_circuits(self):
        client = _mock_client()
        repo = _repo(client)
        await repo.ensure_payload_indexes("kb", {})
        client.create_payload_index.assert_not_awaited()


class TestRetrieveAutoIndexes:
    @pytest.mark.asyncio
    async def test_retrieve_creates_inferred_indexes(self):
        """When ``metadata_filters`` is non-empty, retrieve infers and
        creates the matching payload indexes."""
        client = _mock_client()
        repo = _repo(client)
        await repo.retrieve(
            query="q",
            namespace="kb",
            scope=_scope(),
            metadata_filters={"status": "published", "view_count": {"gte": 100}},
        )
        # Two fields → two index creates.
        assert client.create_payload_index.await_count == 2
        called_fields = {call.kwargs["field_name"] for call in client.create_payload_index.await_args_list}
        assert called_fields == {"status", "view_count"}

    @pytest.mark.asyncio
    async def test_retrieve_without_filters_skips_indexes(self):
        client = _mock_client()
        repo = _repo(client)
        await repo.retrieve(query="q", namespace="kb", scope=_scope())
        client.create_payload_index.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_retrieve_caches_inferred_indexes(self):
        client = _mock_client()
        repo = _repo(client)
        filters = {"status": "published"}
        await repo.retrieve(query="q1", namespace="kb", scope=_scope(), metadata_filters=filters)
        await repo.retrieve(query="q2", namespace="kb", scope=_scope(), metadata_filters=filters)
        # Cached — second call hits the existing payload index.
        assert client.create_payload_index.await_count == 1
