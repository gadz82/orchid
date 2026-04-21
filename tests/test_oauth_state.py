"""Tests for ``orchid_ai.mcp.oauth_state``."""

from __future__ import annotations

import time

import pytest

from orchid_ai.mcp.oauth_state import (
    OrchidInMemoryOAuthStateStore,
    OrchidOAuthPendingState,
    OrchidOAuthStateStore,
    build_oauth_state_store,
    register_oauth_state_store,
)


def _payload(state_id: str = "srv", *, created_at: float | None = None) -> OrchidOAuthPendingState:
    return OrchidOAuthPendingState(
        server_name=state_id,
        tenant_id="t",
        user_id="u",
        code_verifier="v",
        token_endpoint="https://idp/token",
        created_at=created_at if created_at is not None else time.time(),
    )


class TestInMemoryStore:
    @pytest.mark.asyncio
    async def test_put_pop_roundtrip(self):
        store = OrchidInMemoryOAuthStateStore()
        payload = _payload()

        await store.put("abc", payload)
        fetched = await store.pop("abc")

        assert fetched is payload

    @pytest.mark.asyncio
    async def test_pop_consumes(self):
        store = OrchidInMemoryOAuthStateStore()
        await store.put("abc", _payload())

        first = await store.pop("abc")
        second = await store.pop("abc")

        assert first is not None
        assert second is None

    @pytest.mark.asyncio
    async def test_pop_missing_returns_none(self):
        store = OrchidInMemoryOAuthStateStore()
        assert await store.pop("never-put") is None

    @pytest.mark.asyncio
    async def test_expired_entries_are_evicted(self):
        store = OrchidInMemoryOAuthStateStore(ttl_seconds=0.01)
        await store.put("old", _payload(created_at=time.time() - 10))
        await store.put("new", _payload())

        # Access triggers lazy sweep
        assert await store.pop("old") is None
        assert await store.pop("new") is not None

    @pytest.mark.asyncio
    async def test_close_is_noop(self):
        store = OrchidInMemoryOAuthStateStore()
        await store.close()
        await store.close()  # idempotent


class TestFactory:
    @pytest.mark.asyncio
    async def test_memory_returns_in_memory_store(self):
        store = await build_oauth_state_store("memory")
        assert isinstance(store, OrchidInMemoryOAuthStateStore)

    @pytest.mark.asyncio
    async def test_default_type_is_memory(self):
        store = await build_oauth_state_store()
        assert isinstance(store, OrchidInMemoryOAuthStateStore)

    @pytest.mark.asyncio
    async def test_registered_type_wins(self):
        class DummyStore(OrchidOAuthStateStore):
            async def put(self, state: str, payload: OrchidOAuthPendingState) -> None:
                return None

            async def pop(self, state: str) -> OrchidOAuthPendingState | None:
                return None

        async def dummy_factory(*, dsn: str, ttl_seconds: float) -> OrchidOAuthStateStore:
            return DummyStore()

        register_oauth_state_store("dummy", dummy_factory)
        store = await build_oauth_state_store("dummy")
        assert isinstance(store, DummyStore)

    @pytest.mark.asyncio
    async def test_custom_class_path(self):
        store = await build_oauth_state_store(
            "orchid_ai.mcp.oauth_state.OrchidInMemoryOAuthStateStore",
            ttl_seconds=42.0,
        )
        assert isinstance(store, OrchidInMemoryOAuthStateStore)

    @pytest.mark.asyncio
    async def test_non_subclass_raises(self):
        with pytest.raises(TypeError):
            await build_oauth_state_store("orchid_ai.utils.import_class")
