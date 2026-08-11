"""Tests for the :class:`OrchidAuthConfigProvider` ABC and its
companion :class:`OrchidUpstreamOAuthConfig` dataclass in
:mod:`orchid_ai.core.auth_config`.
"""

from __future__ import annotations

import dataclasses

import pytest

from orchid_ai.core.auth_config import (
    OrchidAuthConfigProvider,
    OrchidAuthExchangeClient,
    OrchidAuthExchangeError,
    OrchidUpstreamOAuthConfig,
    OrchidUpstreamTokenResponse,
)

# ── OrchidAuthConfigProvider is abstract ─────────────────────


def test_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        OrchidAuthConfigProvider()  # type: ignore[abstract]


# ── Concrete subclasses behave correctly ─────────────────────


def test_subclass_returning_none_is_valid() -> None:
    """A provider that declines to emit config is a legitimate answer."""

    class NoneProvider(OrchidAuthConfigProvider):
        def get_oauth_config(self) -> OrchidUpstreamOAuthConfig | None:
            return None

    assert NoneProvider().get_oauth_config() is None


def test_subclass_returning_config_roundtrips() -> None:
    class FakeProvider(OrchidAuthConfigProvider):
        def get_oauth_config(self) -> OrchidUpstreamOAuthConfig:
            return OrchidUpstreamOAuthConfig(
                issuer_url="https://idp.example.com",
                authorization_endpoint="https://idp.example.com/authorize",
                token_endpoint="https://idp.example.com/token",
                client_id="client-abc",
                userinfo_endpoint="https://idp.example.com/userinfo",
                scope="openid profile",
            )

    cfg = FakeProvider().get_oauth_config()
    assert cfg is not None
    assert cfg.issuer_url == "https://idp.example.com"
    assert cfg.userinfo_endpoint == "https://idp.example.com/userinfo"
    assert cfg.scope == "openid profile"


# ── OrchidUpstreamOAuthConfig dataclass shape ────────────────


def test_config_is_frozen() -> None:
    cfg = OrchidUpstreamOAuthConfig(
        issuer_url="i",
        authorization_endpoint="a",
        token_endpoint="t",
        client_id="c",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.client_id = "other"  # type: ignore[misc]


def test_config_optional_fields_default_cleanly() -> None:
    cfg = OrchidUpstreamOAuthConfig(
        issuer_url="i",
        authorization_endpoint="a",
        token_endpoint="t",
        client_id="c",
    )
    assert cfg.userinfo_endpoint is None
    assert cfg.scope == ""
    # JSON-path hints default to None — "use standard OIDC top-level
    # sub/email" — so OIDC-compliant upstreams don't need to touch
    # these fields at all.
    assert cfg.userinfo_sub_path is None
    assert cfg.userinfo_email_path is None
    # Platform domain defaults to None — downstream consumers fall
    # back to their own heuristics (e.g. email-domain derivation).
    assert cfg.auth_domain is None
    # Exchange-via-api defaults to False so legacy deployments keep
    # doing their own code exchange.
    assert cfg.exchange_via_api is False
    # Resolve-via-api defaults to False so legacy deployments keep
    # hitting the upstream userinfo endpoint themselves.
    assert cfg.resolve_via_api is False


def test_config_opts_in_to_exchange_via_api() -> None:
    """Non-default value flips on the server-side exchange proxy flag."""
    cfg = OrchidUpstreamOAuthConfig(
        issuer_url="i",
        authorization_endpoint="a",
        token_endpoint="t",
        client_id="c",
        exchange_via_api=True,
    )
    assert cfg.exchange_via_api is True


def test_config_opts_in_to_resolve_via_api() -> None:
    """Non-default value flips on the server-side identity-resolver flag."""
    cfg = OrchidUpstreamOAuthConfig(
        issuer_url="i",
        authorization_endpoint="a",
        token_endpoint="t",
        client_id="c",
        resolve_via_api=True,
    )
    assert cfg.resolve_via_api is True
    # And the two flags are independent — opting in to one doesn't
    # force the other on.
    assert cfg.exchange_via_api is False


def test_config_opts_in_to_refresh_via_api() -> None:
    """Non-default value flips on the server-side refresh flag."""
    cfg = OrchidUpstreamOAuthConfig(
        issuer_url="i",
        authorization_endpoint="a",
        token_endpoint="t",
        client_id="c",
        refresh_via_api=True,
    )
    assert cfg.refresh_via_api is True
    # All four flags are independent — opting in to one doesn't
    # force the others on.
    assert cfg.exchange_via_api is False
    assert cfg.resolve_via_api is False


def test_config_carries_json_path_hints() -> None:
    """Non-OIDC upstreams set path hints so downstream can pluck claims."""
    cfg = OrchidUpstreamOAuthConfig(
        issuer_url="i",
        authorization_endpoint="a",
        token_endpoint="t",
        client_id="c",
        userinfo_sub_path="data.user_id",
        userinfo_email_path="data.email",
    )
    assert cfg.userinfo_sub_path == "data.user_id"
    assert cfg.userinfo_email_path == "data.email"


def test_config_equality_by_value() -> None:
    a = OrchidUpstreamOAuthConfig(
        issuer_url="i",
        authorization_endpoint="a",
        token_endpoint="t",
        client_id="c",
    )
    b = OrchidUpstreamOAuthConfig(
        issuer_url="i",
        authorization_endpoint="a",
        token_endpoint="t",
        client_id="c",
    )
    assert a == b


def test_config_disallows_extra_kwargs() -> None:
    with pytest.raises(TypeError):
        OrchidUpstreamOAuthConfig(  # type: ignore[call-arg]
            issuer_url="i",
            authorization_endpoint="a",
            token_endpoint="t",
            client_id="c",
            extra="nope",  # type: ignore[call-arg]
        )


# ── OrchidUpstreamTokenResponse dataclass shape ────────────


def test_token_response_is_frozen() -> None:
    r = OrchidUpstreamTokenResponse(access_token="at-1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.access_token = "at-2"  # type: ignore[misc]


def test_token_response_defaults() -> None:
    r = OrchidUpstreamTokenResponse(access_token="at-1")
    assert r.token_type == "Bearer"
    assert r.refresh_token is None
    assert r.expires_in is None
    assert r.scope is None


# ── OrchidAuthExchangeClient ABC ───────────────────────────


def test_exchange_client_is_abstract() -> None:
    with pytest.raises(TypeError):
        OrchidAuthExchangeClient()  # type: ignore[abstract]


@pytest.mark.asyncio
async def test_concrete_exchange_client_roundtrip() -> None:
    """A minimal subclass returning a canned response must satisfy the ABC."""

    class FakeExchange(OrchidAuthExchangeClient):
        async def exchange_code(
            self,
            *,
            code: str,
            redirect_uri: str,
            code_verifier: str | None = None,
            domain: str | None = None,
        ) -> OrchidUpstreamTokenResponse:
            return OrchidUpstreamTokenResponse(
                access_token=f"at-for-{code}",
                refresh_token="rt-xyz",
                expires_in=3600,
                scope="api",
            )

    r = await FakeExchange().exchange_code(
        code="abc",
        redirect_uri="http://localhost/cb",
        code_verifier="v",
    )
    assert r.access_token == "at-for-abc"
    assert r.refresh_token == "rt-xyz"
    assert r.expires_in == 3600
    assert r.scope == "api"


@pytest.mark.asyncio
async def test_refresh_token_defaults_to_not_implemented() -> None:
    """The base ABC keeps exchange-only subclasses instantiable;
    they just can't perform refreshes.  The gate in
    ``/auth-info``'s ``refresh_via_api`` flag relies on this default
    behaviour — an exchange-only deployment must not advertise the
    feature.
    """

    class ExchangeOnly(OrchidAuthExchangeClient):
        async def exchange_code(
            self,
            *,
            code: str,
            redirect_uri: str,
            code_verifier: str | None = None,
            domain: str | None = None,
        ) -> OrchidUpstreamTokenResponse:
            return OrchidUpstreamTokenResponse(access_token="at")

    with pytest.raises(NotImplementedError) as exc:
        await ExchangeOnly().refresh_token(refresh_token="rt-1")
    assert "ExchangeOnly" in str(exc.value)


@pytest.mark.asyncio
async def test_concrete_refresh_roundtrip() -> None:
    """A subclass that overrides ``refresh_token`` works exactly like
    :meth:`exchange_code` — same response dataclass, same error
    handling contract.
    """

    class FullExchange(OrchidAuthExchangeClient):
        async def exchange_code(
            self,
            *,
            code: str,
            redirect_uri: str,
            code_verifier: str | None = None,
            domain: str | None = None,
        ) -> OrchidUpstreamTokenResponse:
            return OrchidUpstreamTokenResponse(access_token="at")

        async def refresh_token(
            self,
            *,
            refresh_token: str,
            domain: str | None = None,
        ) -> OrchidUpstreamTokenResponse:
            return OrchidUpstreamTokenResponse(
                access_token=f"refreshed-for-{refresh_token}",
                refresh_token="rt-rotated",
                expires_in=1200,
            )

    r = await FullExchange().refresh_token(refresh_token="rt-old")
    assert r.access_token == "refreshed-for-rt-old"
    assert r.refresh_token == "rt-rotated"
    assert r.expires_in == 1200


# ── OrchidAuthExchangeError ────────────────────────────────


def test_exchange_error_carries_status_code() -> None:
    err = OrchidAuthExchangeError("invalid_grant", status_code=400)
    assert str(err) == "invalid_grant"
    assert err.status_code == 400


def test_exchange_error_default_status_code_is_zero() -> None:
    err = OrchidAuthExchangeError("unreachable")
    assert err.status_code == 0


def test_exchange_error_is_exception_subclass() -> None:
    with pytest.raises(OrchidAuthExchangeError):
        raise OrchidAuthExchangeError("boom")
