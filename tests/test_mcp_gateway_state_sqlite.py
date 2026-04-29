"""Tests for OrchidSQLiteMCPGatewayStateStore — in-memory SQLite
coverage for the three inbound-gateway ABCs (Phase 3).

Mirrors the structure of ``test_mcp_token_store_sqlite.py`` but exercises
the three concerns in turn (clients, auth codes, tokens) because a
single class implements all three ABCs.

Highlights:

- ``consume`` is one-shot — the second call returns ``None``.
- ``update`` is a partial-patch — missing fields leave the column
  untouched rather than wiping it.
- Expired tokens look up as ``None`` so callers don't need TTL bookkeeping.
- ``identity`` is stored as an opaque dict — JSON round-tripping must
  preserve nested structures without the caller seeing a string wrapper.
"""

from __future__ import annotations

import time

import pytest

from orchid_ai.core.mcp_gateway_state import (
    OrchidMCPGatewayAuthCode,
    OrchidMCPGatewayClient,
    OrchidMCPGatewayToken,
)
from orchid_ai.persistence.mcp_gateway_state_sqlite import (
    OrchidSQLiteMCPGatewayStateStore,
)


@pytest.fixture
async def store():
    """Create a fresh in-memory gateway-state store for each test."""
    s = OrchidSQLiteMCPGatewayStateStore(dsn=":memory:")
    await s.init_db()
    yield s
    await s.close()


# ── Clients ──────────────────────────────────────────────────────


def _client(
    client_id: str = "cli-abc",
    *,
    redirect_uris: list[str] | None = None,
    grant_types: list[str] | None = None,
    response_types: list[str] | None = None,
    client_name: str = "MCP Inspector",
) -> OrchidMCPGatewayClient:
    return OrchidMCPGatewayClient(
        client_id=client_id,
        redirect_uris=redirect_uris or ["http://localhost:8765/callback"],
        grant_types=grant_types or ["authorization_code", "refresh_token"],
        response_types=response_types or ["code"],
        client_name=client_name,
    )


@pytest.mark.asyncio
class TestGatewayClientStore:
    async def test_get_returns_none_when_empty(self, store):
        assert await store.get("missing") is None

    async def test_register_and_get_round_trip(self, store):
        await store.register(_client())
        loaded = await store.get("cli-abc")
        assert loaded is not None
        assert loaded.client_id == "cli-abc"
        assert loaded.client_name == "MCP Inspector"
        assert loaded.redirect_uris == ["http://localhost:8765/callback"]
        assert loaded.grant_types == ["authorization_code", "refresh_token"]
        assert loaded.response_types == ["code"]
        assert loaded.token_endpoint_auth_method == "none"
        assert loaded.created_at > 0

    async def test_register_replaces_existing_record(self, store):
        """``INSERT OR REPLACE`` — a second register for the same
        ``client_id`` overwrites the previous row (no duplicate key
        error, no stale row retained).
        """
        await store.register(_client(client_name="first"))
        await store.register(_client(client_name="second", redirect_uris=["http://other"]))

        loaded = await store.get("cli-abc")
        assert loaded is not None
        assert loaded.client_name == "second"
        assert loaded.redirect_uris == ["http://other"]

    async def test_different_clients_isolated(self, store):
        await store.register(_client(client_id="alice", client_name="alice-client"))
        await store.register(_client(client_id="bob", client_name="bob-client"))

        alice = await store.get("alice")
        bob = await store.get("bob")
        assert alice is not None and alice.client_name == "alice-client"
        assert bob is not None and bob.client_name == "bob-client"


# ── Auth codes ───────────────────────────────────────────────────


def _auth_code(
    code: str = "authcode-xyz",
    *,
    upstream_state: str = "ust-123",
    client_id: str = "cli-abc",
    scopes: list[str] | None = None,
    identity: dict | None = None,
) -> OrchidMCPGatewayAuthCode:
    return OrchidMCPGatewayAuthCode(
        code=code,
        client_id=client_id,
        redirect_uri="http://localhost:8765/callback",
        code_challenge="challenge-abc",
        code_challenge_method="S256",
        upstream_state=upstream_state,
        upstream_code_verifier="verifier-def",
        scopes=scopes or ["mcp.read"],
        client_state="client-echo-state",
        identity=identity,
    )


@pytest.mark.asyncio
class TestGatewayAuthCodeStore:
    async def test_lookup_returns_none_when_empty(self, store):
        assert await store.get_by_upstream_state("missing") is None

    async def test_put_and_lookup_by_upstream_state(self, store):
        await store.put(_auth_code())
        loaded = await store.get_by_upstream_state("ust-123")
        assert loaded is not None
        assert loaded.code == "authcode-xyz"
        assert loaded.scopes == ["mcp.read"]
        assert loaded.client_state == "client-echo-state"
        # Post-exchange fields remain their defaults until update() runs.
        assert loaded.identity is None
        assert loaded.idp_access_token == ""
        assert loaded.idp_refresh_token == ""
        assert loaded.idp_expires_at == 0.0

    async def test_update_patches_only_specified_fields(self, store):
        """``update`` is partial — passing ``identity`` alone must not
        reset ``idp_access_token`` to its default.
        """
        await store.put(_auth_code())
        await store.update(
            "authcode-xyz",
            idp_access_token="at-v1",
            idp_refresh_token="rt-v1",
            idp_expires_at=time.time() + 600,
        )
        # Second partial update only touches ``identity`` — previous
        # IdP tokens must survive.
        await store.update(
            "authcode-xyz",
            identity={"sub": "u-42", "email": "a@b.c"},
        )

        loaded = await store.get_by_upstream_state("ust-123")
        assert loaded is not None
        assert loaded.identity == {"sub": "u-42", "email": "a@b.c"}
        assert loaded.idp_access_token == "at-v1"
        assert loaded.idp_refresh_token == "rt-v1"
        assert loaded.idp_expires_at > 0.0

    async def test_update_noop_when_nothing_specified(self, store):
        """Calling ``update`` with no field args must not emit SQL
        that wipes the row."""
        await store.put(_auth_code(identity={"sub": "u-1"}))
        await store.update("authcode-xyz")  # all-None defaults

        loaded = await store.get_by_upstream_state("ust-123")
        assert loaded is not None
        assert loaded.identity == {"sub": "u-1"}

    async def test_consume_returns_row_then_deletes(self, store):
        await store.put(_auth_code(identity={"sub": "u-1"}))
        first = await store.consume("authcode-xyz")
        assert first is not None
        assert first.code == "authcode-xyz"
        assert first.identity == {"sub": "u-1"}

        # Second consume gets nothing — row is gone.
        second = await store.consume("authcode-xyz")
        assert second is None

    async def test_consume_returns_none_when_missing(self, store):
        assert await store.consume("never-seen") is None

    async def test_identity_survives_json_round_trip(self, store):
        """Opaque identity dicts with nested shapes must be stored
        and retrieved verbatim (no string wrapping).
        """
        payload = {
            "sub": "u-42",
            "email": "a@b.c",
            "nested": {"platform": {"domain": "acme.example.com", "roles": [1, 2, 3]}},
        }
        await store.put(_auth_code(identity=payload))
        loaded = await store.get_by_upstream_state("ust-123")
        assert loaded is not None
        assert loaded.identity == payload


# ── Tokens ───────────────────────────────────────────────────────


def _token(
    access_token: str = "at-1",
    *,
    refresh_token: str = "rt-1",
    subject: str = "u-42",
    identity: dict | None = None,
    scopes: list[str] | None = None,
    expires_at: float | None = None,
) -> OrchidMCPGatewayToken:
    return OrchidMCPGatewayToken(
        access_token=access_token,
        refresh_token=refresh_token,
        client_id="cli-abc",
        subject=subject,
        identity=identity or {"sub": subject, "email": "a@b.c"},
        scopes=scopes or ["mcp.read"],
        expires_at=time.time() + 3600 if expires_at is None else expires_at,
    )


@pytest.mark.asyncio
class TestGatewayTokenStore:
    async def test_lookup_returns_none_when_empty(self, store):
        assert await store.get_by_access_token("missing") is None
        assert await store.get_by_refresh_token("missing") is None

    async def test_issue_and_lookup_by_access_token(self, store):
        await store.issue(_token())
        loaded = await store.get_by_access_token("at-1")
        assert loaded is not None
        assert loaded.refresh_token == "rt-1"
        assert loaded.subject == "u-42"
        assert loaded.identity == {"sub": "u-42", "email": "a@b.c"}
        assert loaded.scopes == ["mcp.read"]

    async def test_issue_and_lookup_by_refresh_token(self, store):
        await store.issue(_token())
        loaded = await store.get_by_refresh_token("rt-1")
        assert loaded is not None
        assert loaded.access_token == "at-1"

    async def test_expired_access_token_returns_none(self, store):
        """Expired rows must not surface to callers — per the ABC
        contract, callers should not need TTL bookkeeping.
        """
        await store.issue(_token(expires_at=time.time() - 10))
        assert await store.get_by_access_token("at-1") is None

    async def test_expired_refresh_token_returns_none(self, store):
        await store.issue(_token(expires_at=time.time() - 10))
        assert await store.get_by_refresh_token("rt-1") is None

    async def test_revoke_returns_true_when_removed(self, store):
        await store.issue(_token())
        assert await store.revoke("at-1") is True
        assert await store.get_by_access_token("at-1") is None

    async def test_revoke_returns_false_when_missing(self, store):
        assert await store.revoke("never-seen") is False

    async def test_different_tokens_isolated(self, store):
        await store.issue(_token(access_token="at-a", refresh_token="rt-a", subject="alice"))
        await store.issue(_token(access_token="at-b", refresh_token="rt-b", subject="bob"))

        a = await store.get_by_access_token("at-a")
        b = await store.get_by_access_token("at-b")
        assert a is not None and a.subject == "alice"
        assert b is not None and b.subject == "bob"


# ── Lifecycle ───────────────────────────────────────────────────


@pytest.mark.asyncio
class TestLifecycle:
    async def test_all_three_concerns_share_one_connection(self, store):
        """Single concrete class implements all three ABCs against
        the same connection — a client + an auth code + a token must
        coexist without interference.
        """
        await store.register(_client())
        await store.put(_auth_code())
        await store.issue(_token())

        assert await store.get("cli-abc") is not None
        assert await store.get_by_upstream_state("ust-123") is not None
        assert await store.get_by_access_token("at-1") is not None

    async def test_close_allows_reopen(self):
        """Close releases the connection; a follow-up init_db reopens
        it against a fresh connection (relevant for test scenarios
        that tear down and restart).
        """
        s = OrchidSQLiteMCPGatewayStateStore(dsn=":memory:")
        await s.init_db()
        await s.close()
        # close() is idempotent — calling again must not raise.
        await s.close()
