"""Tests for the :class:`OrchidAuthConfigProvider` ABC and its
companion :class:`OrchidUpstreamOAuthConfig` dataclass in
:mod:`orchid_ai.core.auth_config`.
"""

from __future__ import annotations

import dataclasses

import pytest

from orchid_ai.core.auth_config import (
    OrchidAuthConfigProvider,
    OrchidUpstreamOAuthConfig,
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
