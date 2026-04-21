"""Tests for OrchidSQLiteMCPTokenStore — in-memory SQLite token persistence."""

from __future__ import annotations

import time

import pytest

from orchid_ai.core.mcp import OrchidMCPTokenRecord
from orchid_ai.persistence.mcp_token_sqlite import OrchidSQLiteMCPTokenStore


@pytest.fixture
async def store():
    """Create an in-memory token store for each test."""
    s = OrchidSQLiteMCPTokenStore(dsn=":memory:")
    await s.init_db()
    yield s
    await s.close()


def _make_record(
    server_name: str = "ext-crm",
    tenant_id: str = "tenant1",
    user_id: str = "user1",
    access_token: str = "access-token-123",
    **kwargs,
) -> OrchidMCPTokenRecord:
    return OrchidMCPTokenRecord(
        server_name=server_name,
        tenant_id=tenant_id,
        user_id=user_id,
        access_token=access_token,
        **kwargs,
    )


@pytest.mark.asyncio
class TestSQLiteMCPTokenStore:
    async def test_get_token_returns_none_when_empty(self, store: OrchidSQLiteMCPTokenStore):
        result = await store.get_token("t", "u", "s")
        assert result is None

    async def test_save_and_get_token(self, store: OrchidSQLiteMCPTokenStore):
        record = _make_record()
        await store.save_token(record)

        loaded = await store.get_token("tenant1", "user1", "ext-crm")
        assert loaded is not None
        assert loaded.server_name == "ext-crm"
        assert loaded.access_token == "access-token-123"
        assert loaded.tenant_id == "tenant1"
        assert loaded.user_id == "user1"

    async def test_save_token_upserts(self, store: OrchidSQLiteMCPTokenStore):
        await store.save_token(_make_record(access_token="old-token"))
        await store.save_token(_make_record(access_token="new-token"))

        loaded = await store.get_token("tenant1", "user1", "ext-crm")
        assert loaded is not None
        assert loaded.access_token == "new-token"

    async def test_delete_token(self, store: OrchidSQLiteMCPTokenStore):
        await store.save_token(_make_record())
        deleted = await store.delete_token("tenant1", "user1", "ext-crm")
        assert deleted is True

        loaded = await store.get_token("tenant1", "user1", "ext-crm")
        assert loaded is None

    async def test_delete_nonexistent_returns_false(self, store: OrchidSQLiteMCPTokenStore):
        deleted = await store.delete_token("t", "u", "nonexistent")
        assert deleted is False

    async def test_list_tokens(self, store: OrchidSQLiteMCPTokenStore):
        await store.save_token(_make_record(server_name="server-a"))
        await store.save_token(_make_record(server_name="server-b"))
        await store.save_token(_make_record(server_name="server-c", tenant_id="other-tenant"))

        tokens = await store.list_tokens("tenant1", "user1")
        assert len(tokens) == 2
        names = {t.server_name for t in tokens}
        assert names == {"server-a", "server-b"}

    async def test_init_db_idempotent(self, store: OrchidSQLiteMCPTokenStore):
        """Calling init_db twice should not fail."""
        await store.init_db()  # already called in fixture
        # Should not raise

    async def test_preserves_refresh_token_and_scopes(self, store: OrchidSQLiteMCPTokenStore):
        record = _make_record(
            refresh_token="refresh-abc",
            scopes="openid crm.read",
            expires_at=time.time() + 3600,
        )
        await store.save_token(record)

        loaded = await store.get_token("tenant1", "user1", "ext-crm")
        assert loaded is not None
        assert loaded.refresh_token == "refresh-abc"
        assert loaded.scopes == "openid crm.read"
        assert loaded.expires_at > 0

    async def test_different_users_isolated(self, store: OrchidSQLiteMCPTokenStore):
        await store.save_token(_make_record(user_id="alice", access_token="alice-token"))
        await store.save_token(_make_record(user_id="bob", access_token="bob-token"))

        alice = await store.get_token("tenant1", "alice", "ext-crm")
        bob = await store.get_token("tenant1", "bob", "ext-crm")
        assert alice is not None and alice.access_token == "alice-token"
        assert bob is not None and bob.access_token == "bob-token"
