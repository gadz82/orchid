"""SQLite round-trip tests for :class:`OrchidSQLiteMCPClientRegistrationStore`.

Keep these lightweight — storage is a thin SQL wrapper and the schema
is locked by the v002 migration.  Anything more elaborate belongs in
the discovery tests where the store is one of several dependencies.
"""

from __future__ import annotations

import pytest

from orchid_ai.core.mcp import OrchidMCPClientRegistration
from orchid_ai.persistence.mcp_client_registration_sqlite import (
    OrchidSQLiteMCPClientRegistrationStore,
)


@pytest.mark.asyncio
async def test_round_trip_save_get():
    store = OrchidSQLiteMCPClientRegistrationStore(dsn=":memory:")
    await store.init_db()
    try:
        record = OrchidMCPClientRegistration(
            server_name="crm",
            authorization_endpoint="https://idp.example.com/oauth2/authorize",
            token_endpoint="https://idp.example.com/oauth2/token",
            registration_endpoint="https://idp.example.com/oauth2/register",
            issuer="https://idp.example.com",
            scopes_supported="openid mcp.read",
            token_endpoint_auth_methods_supported="client_secret_post",
            client_id="dyn-abc",
            client_secret="s3kr3t",
            client_id_issued_at=1700000000.0,
            client_secret_expires_at=0.0,
        )
        await store.save(record)
        loaded = await store.get("crm")
        assert loaded is not None
        assert loaded.client_id == "dyn-abc"
        assert loaded.client_secret == "s3kr3t"
        assert loaded.authorization_endpoint == record.authorization_endpoint
        assert loaded.token_endpoint_auth_methods_supported == "client_secret_post"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_save_is_upsert():
    store = OrchidSQLiteMCPClientRegistrationStore(dsn=":memory:")
    await store.init_db()
    try:
        base = OrchidMCPClientRegistration(
            server_name="crm",
            authorization_endpoint="https://idp/oauth2/authorize",
            token_endpoint="https://idp/oauth2/token",
            client_id="old-id",
        )
        await store.save(base)
        updated = OrchidMCPClientRegistration(
            server_name="crm",
            authorization_endpoint="https://idp/oauth2/authorize",
            token_endpoint="https://idp/oauth2/token",
            client_id="new-id",
            client_secret="new-secret",
        )
        await store.save(updated)

        loaded = await store.get("crm")
        assert loaded is not None
        assert loaded.client_id == "new-id"
        assert loaded.client_secret == "new-secret"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_get_returns_none_for_missing():
    store = OrchidSQLiteMCPClientRegistrationStore(dsn=":memory:")
    await store.init_db()
    try:
        assert await store.get("unknown") is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_delete_returns_true_only_on_hit():
    store = OrchidSQLiteMCPClientRegistrationStore(dsn=":memory:")
    await store.init_db()
    try:
        record = OrchidMCPClientRegistration(
            server_name="crm",
            authorization_endpoint="https://idp/oauth2/authorize",
            token_endpoint="https://idp/oauth2/token",
        )
        await store.save(record)
        assert await store.delete("crm") is True
        assert await store.delete("crm") is False  # second delete — nothing to remove
        assert await store.get("crm") is None
    finally:
        await store.close()
