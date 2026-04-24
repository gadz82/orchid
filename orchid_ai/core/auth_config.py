"""
Abstract upstream-OAuth discovery — consumers provide concrete
implementations that resolve their platform's well-known endpoints
from whatever tenant-scoped configuration they hold (typically a
domain from ``orchid.yml``).

The output of a provider is an :class:`OrchidUpstreamOAuthConfig`:
**non-secret** public discovery info that orchid-api safely exposes
to downstream OAuth clients (the MCP gateway, Next.js frontends)
over an unauthenticated endpoint.  Never includes ``client_secret``,
access tokens, refresh tokens, or any user-identifying data.

This module uses ONLY stdlib types — safe for ``core/``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class OrchidUpstreamOAuthConfig:
    """
    Public OAuth discovery info for downstream OAuth clients.

    This is EXCLUSIVELY non-secret data — endpoints, the public
    ``client_id``, and the advertised scope.  Safe to return from an
    unauthenticated HTTP endpoint.  Consumers that need the
    corresponding ``client_secret`` MUST read it from their own
    secure configuration (env var, secret store) — it never travels
    alongside this object.

    Fields
    ------
    issuer_url : str
        Root URL of the upstream authorization server (e.g.
        ``https://acme.example.com``).  Downstream OIDC-aware
        clients may use it to hit ``.well-known/openid-configuration``
        for cross-checks.
    authorization_endpoint : str
        Absolute URL of the upstream ``/authorize`` endpoint.
    token_endpoint : str
        Absolute URL of the upstream ``/token`` endpoint.
    client_id : str
        Public OAuth client identifier registered at the upstream for
        the consuming application (MCP gateway, frontends).  An empty
        string means the operator has not wired a client_id yet — the
        downstream consumer must treat it as missing and surface a
        clear configuration error.
    userinfo_endpoint : str | None
        Absolute URL of the upstream userinfo endpoint (OIDC) or
        equivalent profile API.  Optional — not all OAuth2 providers
        expose one.
    scope : str
        Space-separated OAuth scopes the consumer should request when
        launching the authorization dance.  Empty string means "use
        whatever the upstream defaults to".
    """

    issuer_url: str
    authorization_endpoint: str
    token_endpoint: str
    client_id: str
    userinfo_endpoint: str | None = None
    scope: str = ""
    #: Platform domain that downstream consumers should attach to
    #: authenticated requests as ``X-Auth-Domain`` (or equivalent).
    #: Distinct from the user's email domain — for a single-tenant
    #: Docebo deploy this is the tenant host (e.g.
    #: ``mytenant.docebosaas.com``).  The MCP gateway uses this to
    #: tell orchid-api which Docebo platform to validate tokens
    #: against, overriding whatever the identity resolver might
    #: infer from user claims.  ``None`` → consumers fall back to
    #: heuristics (typically email-domain derivation).
    auth_domain: str | None = None
    #: Dotted JSON path at which the downstream consumer should look
    #: for the ``sub`` claim in a userinfo response.  Example:
    #: ``"data.user_id"`` for an API that wraps its payload under
    #: ``data``.  ``None`` (the default) means "use the standard OIDC
    #: top-level ``sub``".  Strings and numbers are both accepted and
    #: coerced to string by the consumer.
    userinfo_sub_path: str | None = None
    #: Dotted JSON path at which the downstream consumer should look
    #: for the user's email in a userinfo response.  Paired with
    #: :attr:`userinfo_sub_path` for non-OIDC upstreams that put the
    #: email under a wrapper object (``"data.email"``).  ``None``
    #: means "use the standard OIDC top-level ``email``".
    userinfo_email_path: str | None = None


class OrchidAuthConfigProvider(ABC):
    """
    Resolves upstream-OAuth discovery info from consumer-owned config.

    Implementations are **pure** — no network calls, no side effects.
    They read their configuration at construction time (typically from
    environment variables seeded by ``orchid.yml``) and return a frozen
    :class:`OrchidUpstreamOAuthConfig`, or ``None`` when the operator
    has not configured OAuth at all.

    Consumers (MCP gateway, Next.js frontends) fetch the resolved
    config over HTTP from orchid-api's ``GET /auth-info`` and configure
    themselves against it.  This removes the duplicated-OAuth-config
    drift between the API layer, the gateway, and the frontends.

    The companion :class:`OrchidIdentityResolver` handles the
    *runtime* side of auth — validating a specific bearer token and
    producing an ``OrchidAuthContext``.  This ABC handles the *static*
    side — telling downstream clients where they should send users to
    obtain a token in the first place.
    """

    @abstractmethod
    def get_oauth_config(self) -> OrchidUpstreamOAuthConfig | None:
        """
        Return non-secret upstream-OAuth discovery info.

        Returns
        -------
        OrchidUpstreamOAuthConfig | None
            ``None`` when OAuth is not configured for this deployment
            (e.g. the provider is instantiated but no domain was
            supplied).  Downstream consumers treat ``None`` as
            "discovery unavailable" and either fall back to their own
            env-var overrides or refuse to start.
        """
        ...
