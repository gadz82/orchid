"""Tests for the built-in in-memory MCP stores (framework default)."""

from __future__ import annotations

import time

from orchid_ai.core.mcp import (
    OrchidMCPClientRegistration,
    OrchidMCPClientRegistrationStore,
    OrchidMCPTokenRecord,
    OrchidMCPTokenStore,
)
from orchid_ai.core.mcp_gateway_state import (
    OrchidMCPGatewayAuthCode,
    OrchidMCPGatewayAuthCodeStore,
    OrchidMCPGatewayClient,
    OrchidMCPGatewayClientStore,
    OrchidMCPGatewayToken,
    OrchidMCPGatewayTokenStore,
)
from orchid_ai.persistence.in_memory import (
    OrchidInMemoryMCPClientRegistrationStore,
    OrchidInMemoryMCPGatewayStateStore,
    OrchidInMemoryMCPTokenStore,
)


def _token(**overrides) -> OrchidMCPTokenRecord:
    values = {
        "server_name": "srv",
        "tenant_id": "t1",
        "user_id": "u1",
        "access_token": "access",
        "refresh_token": "refresh",
        "expires_at": time.time() + 3600,
        "scopes": "read write",
    }
    values.update(overrides)
    return OrchidMCPTokenRecord(**values)


def _registration(**overrides) -> OrchidMCPClientRegistration:
    values = {
        "server_name": "srv",
        "authorization_endpoint": "https://auth.example/authorize",
        "token_endpoint": "https://auth.example/token",
        "registration_endpoint": "https://auth.example/register",
        "issuer": "https://auth.example",
        "client_id": "client-1",
        "client_secret": "secret",
    }
    values.update(overrides)
    return OrchidMCPClientRegistration(**values)


class TestInMemoryMCPTokenStore:
    def test_is_a_token_store(self):
        assert isinstance(OrchidInMemoryMCPTokenStore(), OrchidMCPTokenStore)

    def test_constructor_accepts_factory_kwargs(self):
        store = OrchidInMemoryMCPTokenStore(dsn="ignored", extra_migrations_package="a.b")
        assert isinstance(store, OrchidInMemoryMCPTokenStore)

    async def test_round_trip(self):
        store = OrchidInMemoryMCPTokenStore()
        await store.save_token(_token())

        fetched = await store.get_token("t1", "u1", "srv")
        assert fetched is not None
        assert fetched.access_token == "access"
        assert fetched.updated_at > 0
        assert await store.get_token("t1", "u2", "srv") is None
        assert await store.get_token("t2", "u1", "srv") is None

    async def test_list_tokens_filters_by_user(self):
        store = OrchidInMemoryMCPTokenStore()
        await store.save_token(_token(server_name="one"))
        await store.save_token(_token(server_name="two"))
        await store.save_token(_token(server_name="other", user_id="u2"))

        names = {record.server_name for record in await store.list_tokens("t1", "u1")}
        assert names == {"one", "two"}

    async def test_delete_token(self):
        store = OrchidInMemoryMCPTokenStore()
        await store.save_token(_token())

        assert await store.delete_token("t1", "u1", "srv") is True
        assert await store.delete_token("t1", "u1", "srv") is False
        assert await store.get_token("t1", "u1", "srv") is None

    async def test_cleanup_expired(self):
        store = OrchidInMemoryMCPTokenStore()
        now = time.time()
        await store.save_token(_token(server_name="expired", expires_at=now - 10))
        await store.save_token(_token(server_name="live", expires_at=now + 3600))
        await store.save_token(_token(server_name="no-expiry", expires_at=0.0))

        assert await store.cleanup_expired() == 1
        assert await store.get_token("t1", "u1", "live") is not None
        assert await store.get_token("t1", "u1", "no-expiry") is not None
        assert await store.get_token("t1", "u1", "expired") is None

    async def test_cleanup_expired_honours_before(self):
        store = OrchidInMemoryMCPTokenStore()
        now = time.time()
        await store.save_token(_token(server_name="a", expires_at=now + 100))
        await store.save_token(_token(server_name="b", expires_at=now + 200))

        assert await store.cleanup_expired(before=now + 150) == 1
        assert await store.get_token("t1", "u1", "b") is not None

    async def test_stored_records_are_isolated(self):
        store = OrchidInMemoryMCPTokenStore()
        record = _token()
        await store.save_token(record)
        record.access_token = "mutated"

        fetched = await store.get_token("t1", "u1", "srv")
        assert fetched is not None
        assert fetched.access_token == "access"


class TestInMemoryMCPClientRegistrationStore:
    def test_is_a_registration_store(self):
        assert isinstance(OrchidInMemoryMCPClientRegistrationStore(), OrchidMCPClientRegistrationStore)

    async def test_round_trip(self):
        store = OrchidInMemoryMCPClientRegistrationStore()
        await store.save(_registration())

        fetched = await store.get("srv")
        assert fetched is not None
        assert fetched.client_id == "client-1"
        assert fetched.updated_at > 0
        assert await store.get("other") is None

    async def test_save_upserts(self):
        store = OrchidInMemoryMCPClientRegistrationStore()
        await store.save(_registration(client_id="first"))
        await store.save(_registration(client_id="second"))

        fetched = await store.get("srv")
        assert fetched is not None
        assert fetched.client_id == "second"

    async def test_delete(self):
        store = OrchidInMemoryMCPClientRegistrationStore()
        await store.save(_registration())

        assert await store.delete("srv") is True
        assert await store.delete("srv") is False
        assert await store.get("srv") is None


class TestInMemoryMCPGatewayStateStore:
    def test_implements_all_three_gateway_abcs(self):
        store = OrchidInMemoryMCPGatewayStateStore()
        assert isinstance(store, OrchidMCPGatewayClientStore)
        assert isinstance(store, OrchidMCPGatewayAuthCodeStore)
        assert isinstance(store, OrchidMCPGatewayTokenStore)

    async def test_clients(self):
        store = OrchidInMemoryMCPGatewayStateStore()
        await store.register(
            OrchidMCPGatewayClient(
                client_id="c1",
                redirect_uris=["https://app.example/cb"],
                grant_types=["authorization_code"],
                response_types=["code"],
                client_name="App",
            )
        )

        client = await store.get("c1")
        assert client is not None
        assert client.redirect_uris == ["https://app.example/cb"]
        assert client.client_name == "App"
        assert await store.get("missing") is None

    async def test_auth_code_round_trip_and_upstream_lookup(self):
        store = OrchidInMemoryMCPGatewayStateStore()
        await store.put(_auth_code())

        fetched = await store.get_by_upstream_state("upstream-1")
        assert fetched is not None
        assert fetched.code == "code-1"
        assert await store.get_by_upstream_state("missing") is None

    async def test_auth_code_partial_update_skips_none(self):
        store = OrchidInMemoryMCPGatewayStateStore()
        await store.put(_auth_code(idp_access_token="initial-token"))

        await store.update("code-1", identity={"sub": "u1"}, idp_refresh_token="refresh-1")
        updated = await store.get_by_upstream_state("upstream-1")
        assert updated is not None
        assert updated.identity == {"sub": "u1"}
        assert updated.idp_refresh_token == "refresh-1"
        # Untouched fields survive the partial update.
        assert updated.idp_access_token == "initial-token"
        assert updated.client_state == "client-state"

    async def test_auth_code_update_unknown_code_is_noop(self):
        store = OrchidInMemoryMCPGatewayStateStore()
        await store.update("missing", identity={"sub": "u1"})  # must not raise

    async def test_consume_is_one_shot(self):
        store = OrchidInMemoryMCPGatewayStateStore()
        await store.put(_auth_code())

        first = await store.consume("code-1")
        assert first is not None
        assert await store.consume("code-1") is None
        assert await store.get_by_upstream_state("upstream-1") is None

    async def test_tokens_by_access_and_refresh(self):
        store = OrchidInMemoryMCPGatewayStateStore()
        await store.issue(_gateway_token())

        by_access = await store.get_by_access_token("gw-access")
        by_refresh = await store.get_by_refresh_token("gw-refresh")
        assert by_access is not None and by_access.subject == "user-1"
        assert by_refresh is not None and by_refresh.access_token == "gw-access"
        assert await store.get_by_access_token("missing") is None

    async def test_expired_tokens_are_invisible(self):
        store = OrchidInMemoryMCPGatewayStateStore()
        await store.issue(_gateway_token(expires_at=time.time() - 1))

        assert await store.get_by_access_token("gw-access") is None
        assert await store.get_by_refresh_token("gw-refresh") is None

    async def test_revoke(self):
        store = OrchidInMemoryMCPGatewayStateStore()
        await store.issue(_gateway_token())

        assert await store.revoke("gw-access") is True
        assert await store.revoke("gw-access") is False
        assert await store.get_by_access_token("gw-access") is None


def _auth_code(**overrides) -> OrchidMCPGatewayAuthCode:
    values = {
        "code": "code-1",
        "client_id": "c1",
        "redirect_uri": "https://app.example/cb",
        "code_challenge": "challenge",
        "code_challenge_method": "S256",
        "upstream_state": "upstream-1",
        "upstream_code_verifier": "verifier",
        "scopes": ["openid"],
        "client_state": "client-state",
    }
    values.update(overrides)
    return OrchidMCPGatewayAuthCode(**values)


def _gateway_token(**overrides) -> OrchidMCPGatewayToken:
    values = {
        "access_token": "gw-access",
        "refresh_token": "gw-refresh",
        "client_id": "c1",
        "subject": "user-1",
        "identity": {"sub": "user-1"},
        "scopes": ["openid"],
        "expires_at": time.time() + 3600,
    }
    values.update(overrides)
    return OrchidMCPGatewayToken(**values)
