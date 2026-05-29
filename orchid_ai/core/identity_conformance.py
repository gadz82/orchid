"""
Conformance test base class for :class:`OrchidIdentityResolver` implementations.

Consumer projects inherit from :class:`OrchidIdentityResolverConformanceTests`
in their own test suite to verify their resolver satisfies the security
contract documented on the ABC.

Usage::

    from orchid_ai.core.identity_conformance import OrchidIdentityResolverConformanceTests

    class TestMyResolver(OrchidIdentityResolverConformanceTests):
        def create_resolver(self):
            return MyResolver(client_id="test", ...)

        def valid_bearer_token(self):
            return "valid-test-token"

        def expected_tenant_key(self):
            return "tenant-from-idp"

        def expected_user_id(self):
            return "user-from-idp"
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pytest


class OrchidIdentityResolverConformanceTests(ABC):
    """Base class for identity resolver conformance tests.

    Subclass this in your consumer test suite and implement the
    abstract factory methods.  The conformance tests verify:

    1. ``tenant_key`` does NOT depend on the ``domain`` parameter
       (attacker-controlled routing hint).
    2. ``user_id`` does NOT depend on the ``domain`` parameter.
    3. The resolver raises ``OrchidIdentityError`` for invalid tokens.
    """

    @abstractmethod
    def create_resolver(self):
        """Return a configured resolver instance for testing."""
        ...

    @abstractmethod
    def valid_bearer_token(self) -> str:
        """Return a bearer token that the resolver will accept."""
        ...

    @abstractmethod
    def expected_tenant_key(self) -> str:
        """Return the tenant_key the resolver should produce for the valid token."""
        ...

    @abstractmethod
    def expected_user_id(self) -> str:
        """Return the user_id the resolver should produce for the valid token."""
        ...

    @pytest.mark.asyncio
    async def test_tenant_key_independent_of_domain(self):
        """tenant_key MUST NOT change when domain changes (security contract)."""
        resolver = self.create_resolver()
        token = self.valid_bearer_token()

        ctx1 = await resolver.resolve("legitimate.example.com", token)
        ctx2 = await resolver.resolve("attacker.evil.com", token)

        assert ctx1.tenant_key == ctx2.tenant_key, (
            f"tenant_key changed with domain: "
            f"'{ctx1.tenant_key}' vs '{ctx2.tenant_key}'. "
            f"The resolver MUST derive tenant_key from IdP-attested data only."
        )
        assert ctx1.tenant_key == self.expected_tenant_key()

    @pytest.mark.asyncio
    async def test_user_id_independent_of_domain(self):
        """user_id MUST NOT change when domain changes (security contract)."""
        resolver = self.create_resolver()
        token = self.valid_bearer_token()

        ctx1 = await resolver.resolve("legitimate.example.com", token)
        ctx2 = await resolver.resolve("attacker.evil.com", token)

        assert ctx1.user_id == ctx2.user_id, (
            f"user_id changed with domain: "
            f"'{ctx1.user_id}' vs '{ctx2.user_id}'. "
            f"The resolver MUST derive user_id from IdP-attested data only."
        )
        assert ctx1.user_id == self.expected_user_id()

    @pytest.mark.asyncio
    async def test_invalid_token_raises(self):
        """Invalid tokens MUST raise OrchidIdentityError."""
        from .identity import OrchidIdentityError

        resolver = self.create_resolver()

        with pytest.raises(OrchidIdentityError):
            await resolver.resolve("any.example.com", "invalid-garbage-token-xyz")
