"""Unit tests for ``QdrantDocStore`` against a mocked ``AsyncQdrantClient``."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from orchid_ai.rag.backends.qdrant_doc_store import QdrantDocStore, _doc_id_to_uuid


def _mock_client() -> AsyncMock:
    """Build a mocked ``AsyncQdrantClient`` with the methods QdrantDocStore uses."""
    client = AsyncMock()
    client.collection_exists = AsyncMock(return_value=False)
    client.create_collection = AsyncMock()
    client.upsert = AsyncMock()
    client.retrieve = AsyncMock(return_value=[])
    client.scroll = AsyncMock(return_value=([], None))
    client.delete = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_put_creates_collection_on_first_use():
    client = _mock_client()
    store = QdrantDocStore(url="http://x", client=client)
    await store.put("d1", "hello", {"k": "v"})
    client.collection_exists.assert_awaited_once()
    client.create_collection.assert_awaited_once()
    client.upsert.assert_awaited_once()


@pytest.mark.asyncio
async def test_put_idempotent_uuid():
    """Same doc_id always produces the same point UUID."""
    client = _mock_client()
    store = QdrantDocStore(url="http://x", client=client)
    await store.put("d1", "first", {"v": 1})
    await store.put("d1", "second", {"v": 2})
    assert client.upsert.await_count == 2
    # Both upserts target the same point id.
    first = client.upsert.await_args_list[0]
    second = client.upsert.await_args_list[1]
    assert first.kwargs["points"][0].id == second.kwargs["points"][0].id == _doc_id_to_uuid("d1")


@pytest.mark.asyncio
async def test_get_returns_none_when_missing():
    client = _mock_client()
    store = QdrantDocStore(url="http://x", client=client)
    record = await store.get("missing")
    assert record is None


@pytest.mark.asyncio
async def test_get_returns_payload_minus_doc_id_and_content():
    client = _mock_client()
    client.retrieve = AsyncMock(
        return_value=[
            SimpleNamespace(
                id="x",
                payload={"doc_id": "d1", "content": "hello world", "tag": "x", "n": 7},
            )
        ]
    )
    store = QdrantDocStore(url="http://x", client=client)
    record = await store.get("d1")
    assert record == ("hello world", {"tag": "x", "n": 7})


@pytest.mark.asyncio
async def test_get_many_filters_to_known_doc_ids():
    """Records returned by Qdrant whose payload lacks ``doc_id`` are dropped."""
    client = _mock_client()
    client.retrieve = AsyncMock(
        return_value=[
            SimpleNamespace(payload={"doc_id": "a", "content": "alpha", "k": 1}),
            SimpleNamespace(payload={"content": "orphan with no doc_id"}),
            SimpleNamespace(payload={"doc_id": "b", "content": "beta", "k": 2}),
        ]
    )
    store = QdrantDocStore(url="http://x", client=client)
    out = await store.get_many(["a", "b", "missing"])
    assert set(out) == {"a", "b"}
    assert out["a"] == ("alpha", {"k": 1})
    assert out["b"] == ("beta", {"k": 2})


@pytest.mark.asyncio
async def test_get_many_empty_list():
    client = _mock_client()
    store = QdrantDocStore(url="http://x", client=client)
    out = await store.get_many([])
    assert out == {}
    client.retrieve.assert_not_awaited()


@pytest.mark.asyncio
async def test_existing_collection_skips_create():
    client = _mock_client()
    client.collection_exists = AsyncMock(return_value=True)
    store = QdrantDocStore(url="http://x", client=client)
    await store.put("d1", "hello", {})
    client.create_collection.assert_not_awaited()


def test_is_null_marker_is_false():
    assert QdrantDocStore.is_null is False
