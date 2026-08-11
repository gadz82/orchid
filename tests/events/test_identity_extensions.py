"""Phase-3 identity-resolver extensions.

Covers the additive ``OrchidIdentityResolver.resolve_service_account``
and ``mint_for_user`` methods (with their raising defaults) and the
:class:`OAuthMintingMixin` composition helper.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from orchid_ai.core.events.errors import (
    MintingProbeUnsupportedError,
    OrchidIdentityNotMintableError,
    OrchidServiceAccountUnknownError,
)
from orchid_ai.core.identity import OrchidIdentityResolver
from orchid_ai.core.mcp import OrchidMCPTokenRecord, OrchidMCPTokenStore
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.identity import OAuthMintingMixin, OrchidTokenRefresher

# ── Defaults on OrchidIdentityResolver ──────────────────────


class _BareResolver(OrchidIdentityResolver):
    async def resolve(self, domain: str, bearer_token: str) -> OrchidAuthContext:
        return OrchidAuthContext(access_token=bearer_token)


async def test_default_resolve_service_account_raises_unknown() -> None:
    resolver = _BareResolver()
    with pytest.raises(OrchidServiceAccountUnknownError) as exc_info:
        await resolver.resolve_service_account("digest-bot")
    assert "digest-bot" in str(exc_info.value)


async def test_default_mint_for_user_raises_probe_unsupported() -> None:
    resolver = _BareResolver()
    with pytest.raises(MintingProbeUnsupportedError) as exc_info:
        await resolver.mint_for_user("tenant-1", "u-7")
    # ``MintingProbeUnsupportedError`` is a subclass of
    # ``OrchidIdentityNotMintableError`` — registry's mint probe at
    # boot uses ``isinstance`` to distinguish "no minting at all"
    # from "no token for this user".
    assert isinstance(exc_info.value, OrchidIdentityNotMintableError)
    assert exc_info.value.resolver_class == "_BareResolver"


# ── OAuthMintingMixin ───────────────────────────────────────


class _FakeTokenStore(OrchidMCPTokenStore):
    """Drop-in OrchidMCPTokenStore for the mixin tests."""

    def __init__(self) -> None:
        self.records: dict[tuple[str, str, str], OrchidMCPTokenRecord] = {}

    async def init_db(self) -> None:  # pragma: no cover — fixture only
        return

    async def close(self) -> None:  # pragma: no cover — fixture only
        return

    async def get_token(self, tenant_id: str, user_id: str, server_name: str) -> OrchidMCPTokenRecord | None:
        return self.records.get((server_name, tenant_id, user_id))

    async def save_token(self, record: OrchidMCPTokenRecord) -> None:
        self.records[(record.server_name, record.tenant_id, record.user_id)] = record

    async def delete_token(self, tenant_id: str, user_id: str, server_name: str) -> bool:
        return self.records.pop((server_name, tenant_id, user_id), None) is not None

    async def list_tokens(self, tenant_id: str, user_id: str) -> list[OrchidMCPTokenRecord]:
        return [r for (s, t, u), r in self.records.items() if t == tenant_id and u == user_id]


class _FakeRefresher(OrchidTokenRefresher):
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    async def refresh(self, *, server_name: str, refresh_token: str) -> dict[str, Any]:
        self.calls.append((server_name, refresh_token))
        if self.fail:
            raise RuntimeError("simulated refresh failure")
        return {
            "access_token": "fresh-access",
            "refresh_token": "fresh-refresh",
            "expires_at": time.time() + 3600,
            "scopes": "read write",
        }


class _MintingResolver(OAuthMintingMixin, OrchidIdentityResolver):
    def __init__(self, **kwargs: Any) -> None:
        OAuthMintingMixin.__init__(self, **kwargs)

    async def resolve(self, domain: str, bearer_token: str) -> OrchidAuthContext:
        return OrchidAuthContext(access_token=bearer_token)


async def test_mixin_returns_context_for_fresh_token() -> None:
    store = _FakeTokenStore()
    await store.save_token(
        OrchidMCPTokenRecord(
            server_name="ext-crm",
            tenant_id="t-1",
            user_id="u-7",
            access_token="live-token",
            refresh_token="r",
            expires_at=time.time() + 3600,
            scopes="read",
        )
    )
    resolver = _MintingResolver(token_store=store, server_name="ext-crm")
    auth = await resolver.mint_for_user("t-1", "u-7")
    assert auth.access_token == "live-token"
    assert auth.tenant_key == "t-1"
    assert auth.user_id == "u-7"


async def test_mixin_returns_context_for_no_expiry_token() -> None:
    store = _FakeTokenStore()
    await store.save_token(
        OrchidMCPTokenRecord(
            server_name="ext-crm",
            tenant_id="t-1",
            user_id="u-7",
            access_token="never-expires",
            refresh_token="",
            expires_at=0,  # store contract: 0 = no expiry info
            scopes="",
        )
    )
    resolver = _MintingResolver(token_store=store, server_name="ext-crm")
    auth = await resolver.mint_for_user("t-1", "u-7")
    assert auth.access_token == "never-expires"


async def test_mixin_raises_when_no_token_stored() -> None:
    store = _FakeTokenStore()
    resolver = _MintingResolver(token_store=store, server_name="ext-crm")
    with pytest.raises(OrchidIdentityNotMintableError) as exc_info:
        await resolver.mint_for_user("t-1", "u-7")
    # Concrete user-not-found, not a probe-unsupported.
    assert not isinstance(exc_info.value, MintingProbeUnsupportedError)
    assert exc_info.value.tenant_key == "t-1"
    assert exc_info.value.user_id == "u-7"


async def test_mixin_refreshes_stale_token() -> None:
    store = _FakeTokenStore()
    await store.save_token(
        OrchidMCPTokenRecord(
            server_name="ext-crm",
            tenant_id="t-1",
            user_id="u-7",
            access_token="stale",
            refresh_token="rt",
            expires_at=time.time() - 60,  # expired
            scopes="read",
        )
    )
    refresher = _FakeRefresher()
    resolver = _MintingResolver(token_store=store, server_name="ext-crm", refresher=refresher)
    auth = await resolver.mint_for_user("t-1", "u-7")
    assert auth.access_token == "fresh-access"
    assert refresher.calls == [("ext-crm", "rt")]
    refreshed = await store.get_token("t-1", "u-7", "ext-crm")
    assert refreshed is not None
    assert refreshed.access_token == "fresh-access"
    assert refreshed.refresh_token == "fresh-refresh"


async def test_mixin_raises_when_stale_and_no_refresher() -> None:
    store = _FakeTokenStore()
    await store.save_token(
        OrchidMCPTokenRecord(
            server_name="ext-crm",
            tenant_id="t-1",
            user_id="u-7",
            access_token="stale",
            refresh_token="rt",
            expires_at=time.time() - 60,
            scopes="read",
        )
    )
    resolver = _MintingResolver(token_store=store, server_name="ext-crm", refresher=None)
    with pytest.raises(OrchidIdentityNotMintableError):
        await resolver.mint_for_user("t-1", "u-7")


async def test_mixin_raises_when_refresher_fails() -> None:
    store = _FakeTokenStore()
    await store.save_token(
        OrchidMCPTokenRecord(
            server_name="ext-crm",
            tenant_id="t-1",
            user_id="u-7",
            access_token="stale",
            refresh_token="rt",
            expires_at=time.time() - 60,
            scopes="",
        )
    )
    resolver = _MintingResolver(
        token_store=store,
        server_name="ext-crm",
        refresher=_FakeRefresher(fail=True),
    )
    with pytest.raises(OrchidIdentityNotMintableError):
        await resolver.mint_for_user("t-1", "u-7")
