"""
Abstract identity resolution — consumers provide concrete implementations.

The library depends ONLY on this ABC.  Concrete resolvers (e.g. OAuth,
OIDC, SAML) live in consumer projects and are loaded at runtime via
``settings.identity_resolver_class``.

This module uses ONLY stdlib types — safe for ``core/``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .state import OrchidAuthContext


class OrchidIdentityResolver(ABC):
    """
    Resolves a bearer token into an ``OrchidAuthContext``.

    Implementations typically call a platform-specific endpoint
    (e.g. a platform-specific token endpoint, or a standard
    OIDC userinfo endpoint) to validate the token and extract
    identity claims.

    Security contract
    -----------------
    The returned ``OrchidAuthContext.tenant_key`` and ``user_id`` MUST be
    derived **exclusively from data that the upstream IdP attests** —
    typically a verified JWT claim (``iss``, ``sub``, a custom tenant
    claim) or a server-side lookup keyed by the token. Implementations
    MUST NOT trust the ``domain`` argument, request headers, query
    string, or any other client-supplied value when populating these
    fields: those are attacker-controlled, and a mistake here cross-pollutes
    RAG namespaces, chat ownership, and MCP tokens between tenants. The
    ``domain`` parameter is a routing hint for multi-IdP deployments
    only; it never decides ``tenant_key``.
    """

    @abstractmethod
    async def resolve(self, domain: str, bearer_token: str) -> OrchidAuthContext:
        """
        Validate the bearer token and return a populated OrchidAuthContext.

        Parameters
        ----------
        domain : str
            Tenant domain (e.g. ``acme.example.com``), without protocol.
            Used only to pick which IdP to call — never to populate
            ``tenant_key`` directly. See the class-level security
            contract.
        bearer_token : str
            The raw access token (without ``Bearer `` prefix).

        Raises
        ------
        OrchidIdentityError
            If the token is invalid, expired, or the platform is unreachable.
        """
        ...


class OrchidIdentityError(Exception):
    """Raised when identity resolution fails."""

    def __init__(self, message: str, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = status_code
