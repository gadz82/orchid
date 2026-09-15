"""Tests for OrchidIdentityResolver ABC and OrchidIdentityError from orchid_ai/core/identity.py."""

from __future__ import annotations

import pytest

from orchid_ai.core.identity import OrchidIdentityError, OrchidIdentityResolver
from orchid_ai.core.state import OrchidAuthContext

# ── OrchidIdentityResolver is abstract ──


def test_identity_resolver_is_abstract():
    with pytest.raises(TypeError):
        OrchidIdentityResolver()


# ── Concrete subclass works ──


@pytest.mark.asyncio
async def test_concrete_resolver():
    class StubResolver(OrchidIdentityResolver):
        async def resolve(self, domain: str, bearer_token: str) -> OrchidAuthContext:
            return OrchidAuthContext(
                access_token=bearer_token,
                tenant_key=domain,
                user_id="resolved-user",
            )

    resolver = StubResolver()
    ctx = await resolver.resolve("acme.example.com", "my-token")
    assert ctx.access_token == "my-token"
    assert ctx.tenant_key == "acme.example.com"
    assert ctx.user_id == "resolved-user"


# ── OrchidIdentityError ──


def test_identity_error_stores_message_and_status():
    err = OrchidIdentityError("bad token", status_code=401)
    assert str(err) == "bad token"
    assert err.status_code == 401


def test_identity_error_default_status_code():
    err = OrchidIdentityError("oops")
    assert err.status_code == 0


def test_identity_error_is_exception():
    with pytest.raises(OrchidIdentityError):
        raise OrchidIdentityError("fail", status_code=500)


# ── Resolver security contract is documented ──


def test_resolver_docstring_states_token_only_tenant_rule():
    """The resolver's docstring carries a security contract: ``tenant_key``
    and ``user_id`` MUST come from the token, never from a client-supplied
    value. Consumer projects rely on that wording when building custom
    resolvers, so guard against accidental wording drift."""
    doc = (OrchidIdentityResolver.__doc__ or "").lower()
    assert "tenant_key" in doc
    assert "user_id" in doc
    assert "must not" in doc or "must " in doc, "contract should use a normative MUST"
    assert "domain" in doc, "domain must be called out as a routing hint, not identity"


def test_resolve_docstring_marks_domain_as_routing_hint():
    doc = (OrchidIdentityResolver.resolve.__doc__ or "").lower()
    assert "routing hint" in doc or "never" in doc
