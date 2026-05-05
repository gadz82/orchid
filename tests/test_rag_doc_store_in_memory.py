"""Tests for ``InMemoryDocStore``."""

from __future__ import annotations

import pytest

from orchid_ai.rag.backends.in_memory_doc_store import InMemoryDocStore


@pytest.mark.asyncio
async def test_round_trip():
    store = InMemoryDocStore()
    await store.put("d1", "hello world", {"tag": "x"})
    record = await store.get("d1")
    assert record == ("hello world", {"tag": "x"})


@pytest.mark.asyncio
async def test_missing_returns_none():
    store = InMemoryDocStore()
    assert await store.get("missing") is None


@pytest.mark.asyncio
async def test_get_many_returns_only_present():
    store = InMemoryDocStore()
    await store.put("a", "alpha", {"x": 1})
    await store.put("b", "beta", {"x": 2})
    out = await store.get_many(["a", "b", "c"])
    assert set(out) == {"a", "b"}
    assert out["a"] == ("alpha", {"x": 1})
    assert out["b"] == ("beta", {"x": 2})


@pytest.mark.asyncio
async def test_put_overwrites():
    store = InMemoryDocStore()
    await store.put("d1", "v1", {"v": 1})
    await store.put("d1", "v2", {"v": 2})
    assert await store.get("d1") == ("v2", {"v": 2})


@pytest.mark.asyncio
async def test_metadata_is_copied():
    """Mutating the metadata returned from get() should not affect the store."""
    store = InMemoryDocStore()
    await store.put("d1", "hi", {"k": "v"})
    record = await store.get("d1")
    assert record is not None
    record[1]["k"] = "tampered"
    fresh = await store.get("d1")
    assert fresh == ("hi", {"k": "v"})


def test_is_null_marker_is_false():
    """InMemoryDocStore is a real backend — strategies must NOT treat
    it as the no-op fallback."""
    assert InMemoryDocStore.is_null is False
