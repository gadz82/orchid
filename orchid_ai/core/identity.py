"""
Abstract identity resolution — consumers provide concrete implementations.

The library depends ONLY on this ABC.  Concrete resolvers (e.g. OAuth,
OIDC, SAML) live in consumer projects and are loaded at runtime via
``settings.identity_resolver_class``.

This module uses ONLY stdlib types — safe for ``core/``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .state import AuthContext


class IdentityResolver(ABC):
    """
    Resolves a bearer token into an ``AuthContext``.

    Implementations typically call a platform-specific endpoint
    (e.g. a platform-specific token endpoint, or a standard
    OIDC userinfo endpoint) to validate the token and extract
    identity claims.
    """

    @abstractmethod
    async def resolve(self, domain: str, bearer_token: str) -> AuthContext:
        """
        Validate the bearer token and return a populated AuthContext.

        Parameters
        ----------
        domain : str
            Tenant domain (e.g. ``acme.example.com``), without protocol.
        bearer_token : str
            The raw access token (without ``Bearer `` prefix).

        Raises
        ------
        IdentityError
            If the token is invalid, expired, or the platform is unreachable.
        """
        ...


class IdentityError(Exception):
    """Raised when identity resolution fails."""

    def __init__(self, message: str, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = status_code
