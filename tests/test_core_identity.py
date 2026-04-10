"""Tests for IdentityResolver ABC and IdentityError from src/core/identity.py."""

from __future__ import annotations

import pytest

from orchid_ai.core.identity import IdentityError, IdentityResolver
from orchid_ai.core.state import AuthContext


# ── IdentityResolver is abstract ──


def test_identity_resolver_is_abstract():
    with pytest.raises(TypeError):
        IdentityResolver()


# ── Concrete subclass works ──


@pytest.mark.asyncio
async def test_concrete_resolver():
    class StubResolver(IdentityResolver):
        async def resolve(self, domain: str, bearer_token: str) -> AuthContext:
            return AuthContext(
                access_token=bearer_token,
                tenant_key=domain,
                user_id="resolved-user",
            )

    resolver = StubResolver()
    ctx = await resolver.resolve("acme.example.com", "my-token")
    assert ctx.access_token == "my-token"
    assert ctx.tenant_key == "acme.example.com"
    assert ctx.user_id == "resolved-user"


# ── IdentityError ──


def test_identity_error_stores_message_and_status():
    err = IdentityError("bad token", status_code=401)
    assert str(err) == "bad token"
    assert err.status_code == 401


def test_identity_error_default_status_code():
    err = IdentityError("oops")
    assert err.status_code == 0


def test_identity_error_is_exception():
    with pytest.raises(IdentityError):
        raise IdentityError("fail", status_code=500)
