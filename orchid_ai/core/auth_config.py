"""
Abstract upstream-OAuth discovery + code-exchange — consumers
provide concrete implementations that resolve their platform's
well-known endpoints (from whatever tenant-scoped configuration
they hold, typically a domain from ``orchid.yml``) and,
optionally, perform the secret-bearing authorization-code exchange
on behalf of downstream OAuth clients.

Two ABCs live here:

- :class:`OrchidAuthConfigProvider` returns
  :class:`OrchidUpstreamOAuthConfig`: **non-secret** public
  discovery info that orchid-api safely exposes to downstream OAuth
  clients (the MCP gateway, Next.js frontends) over an
  unauthenticated endpoint.  Never includes ``client_secret``,
  access tokens, refresh tokens, or any user-identifying data.

- :class:`OrchidAuthExchangeClient` wraps the
  ``grant_type=authorization_code`` exchange against the upstream
  IdP's token endpoint.  The implementation holds the upstream
  ``client_secret`` and is exposed over orchid-api's
  ``POST /auth/exchange-code`` so downstream clients can migrate
  from confidential-client (secret on the MCP gateway) to public
  PKCE-only clients (secret held only by orchid-api).  This is the
  Phase 2 boundary in the auth-centralisation roadmap — it removes
  the last copy of ``client_secret`` from the gateway layer.

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
    #: deployment this is the tenant host (e.g.
    #: ``mytenant.example.com``).  The MCP gateway uses this to
    #: tell orchid-api which platform tenant to validate tokens
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
    #: When ``True``, orchid-api advertises a server-side
    #: ``POST /auth/exchange-code`` endpoint that the downstream
    #: OAuth client (MCP gateway, Next.js frontends) should use
    #: instead of calling the upstream IdP's ``token_endpoint``
    #: directly.  Operationally: the ``client_secret`` lives on
    #: orchid-api only; downstream clients become public PKCE
    #: clients and no longer hold secrets.  ``False`` (the default)
    #: preserves Phase 1 behaviour — the downstream client exchanges
    #: the code itself using its own copy of the secret.
    exchange_via_api: bool = False
    #: When ``True``, orchid-api advertises a server-side
    #: ``POST /auth/resolve-identity`` endpoint that the downstream
    #: OAuth client should use to turn an upstream access token into
    #: an :class:`OrchidAuthContext`-flavoured identity payload
    #: (``subject`` / ``bearer`` / ``auth_domain``), instead of
    #: calling the upstream ``userinfo_endpoint`` itself.  Removes
    #: the last piece of upstream-specific config (userinfo URL +
    #: JSON-path hints for non-OIDC shapes) from the downstream —
    #: Phase 4 of the auth-centralisation roadmap.  ``False`` (the
    #: default) preserves pre-Phase-4 behaviour.
    resolve_via_api: bool = False
    #: When ``True``, orchid-api advertises a server-side
    #: ``POST /auth/refresh-token`` endpoint — the refresh grant
    #: equivalent of :attr:`exchange_via_api`.  Downstream OAuth
    #: clients that stashed an upstream ``refresh_token`` post a
    #: refresh request here and let orchid-api's
    #: :class:`OrchidAuthExchangeClient` (with the
    #: ``client_secret``) perform the upstream exchange.  Gated on
    #: the client actually implementing :meth:`refresh_token` —
    #: otherwise orchid-api's endpoint 503s and downstream clients
    #: fall back to direct upstream refresh (if they hold a secret)
    #: or re-authentication.
    refresh_via_api: bool = False


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
    def get_oauth_config(
        self,
        *,
        domain: str | None = None,
    ) -> OrchidUpstreamOAuthConfig | None:
        """
        Return non-secret upstream-OAuth discovery info.

        ``domain`` is an optional **per-request** tenant hint —
        downstream consumers that serve many tenants from a single
        orchid-api deployment can pass the user-supplied platform
        host on each ``GET /auth-info?domain=…`` call so the
        provider builds tenant-scoped URLs (e.g.
        ``https://{domain}/oauth2/authorize``).  Single-tenant
        deployments ignore the parameter and return their fixed
        operator-level config.

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


@dataclass(frozen=True)
class OrchidUpstreamTokenResponse:
    """
    Normalised token response from an upstream OAuth2 / OIDC
    ``grant_type=authorization_code`` (or ``refresh_token``) exchange.

    Mirrors RFC 6749 §5.1 — the downstream consumer passes these
    values into its own internal state (e.g. the MCP gateway stores
    ``access_token`` on the :class:`GatewayTokenRecord` it mints
    for the end-user).
    """

    access_token: str
    token_type: str = "Bearer"
    refresh_token: str | None = None
    #: Seconds until ``access_token`` expires, as reported by the
    #: upstream.  Absolute expiry is computed by the consumer.
    expires_in: int | None = None
    scope: str | None = None


class OrchidAuthExchangeError(Exception):
    """Raised when an upstream code exchange fails.

    ``status_code`` mirrors the upstream HTTP status when the
    failure originated at the IdP (useful for mapping back to an
    appropriate response code in orchid-api's endpoint).  ``0``
    means the exchange failed without reaching the upstream
    (misconfiguration, network, etc.).
    """

    def __init__(self, message: str, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = status_code


class OrchidAuthExchangeClient(ABC):
    """
    Performs the secret-bearing upstream-OAuth code exchange on behalf
    of downstream public clients.

    Lives on the orchid-api side so the downstream consumers (MCP
    gateway, Next.js frontends) can drop their copy of
    ``client_secret`` and become public PKCE clients.  orchid-api
    exposes this via ``POST /auth/exchange-code`` (see
    :mod:`orchid_api.routers.auth_exchange`).

    Implementations read the upstream endpoint + credentials at
    construction time (typically from environment variables
    mirrored from ``orchid.yml``) and perform a single POST to the
    upstream ``token_endpoint`` per ``exchange_code`` call.

    The companion :class:`OrchidAuthConfigProvider` tells the
    downstream whether this client is wired (by setting
    :attr:`OrchidUpstreamOAuthConfig.exchange_via_api` ``True``);
    without that signal the downstream falls back to direct
    exchange with the upstream IdP using its own secret copy.
    """

    @abstractmethod
    async def exchange_code(
        self,
        *,
        code: str,
        redirect_uri: str,
        code_verifier: str | None = None,
        domain: str | None = None,
    ) -> OrchidUpstreamTokenResponse:
        """
        Exchange an upstream authorization code for an access token.

        Parameters
        ----------
        code : str
            The ``code`` value the upstream IdP issued when the user
            completed the browser-based authorization flow.
        redirect_uri : str
            The redirect URI the downstream consumer registered with
            the upstream IdP.  Must match byte-for-byte what was
            sent on ``/authorize``; required for the exchange per
            RFC 6749 §4.1.3.
        code_verifier : str | None
            The PKCE verifier matching the challenge sent on
            ``/authorize``.  Required when the upstream enforces
            PKCE (all MCP 2025-03-26 clients do).
        domain : str | None
            Optional **per-request** tenant hint.  Multi-tenant
            consumers (one orchid-api fronting many tenant hosts)
            forward the user-supplied domain on every exchange so
            the implementation can route to the correct upstream
            ``token_endpoint``.  Single-tenant deployments ignore the
            parameter and use their fixed operator-level config.

        Raises
        ------
        OrchidAuthExchangeError
            If the upstream rejected the exchange (invalid_grant,
            invalid_client, …) or the exchange didn't reach the
            upstream at all.
        """
        ...

    async def refresh_token(
        self,
        *,
        refresh_token: str,
        domain: str | None = None,
    ) -> OrchidUpstreamTokenResponse:
        """
        Exchange an upstream refresh token for a fresh access token.

        Parallel to :meth:`exchange_code` — lives here rather than in
        a separate ABC because the same ``client_secret`` protects
        both grant types against the same ``token_endpoint``, and
        consumers almost always implement them as a pair.

        Phase 4 of the auth-centralisation roadmap.  Default
        implementation raises :class:`NotImplementedError` so
        existing :class:`OrchidAuthExchangeClient` subclasses
        (written for Phase 2) keep instantiating cleanly; an
        operator who hasn't implemented it yet sees a useful error
        if a downstream client actually tries to use it via
        ``POST /auth/refresh-token``.

        Parameters
        ----------
        refresh_token : str
            The opaque upstream refresh token the downstream
            consumer obtained from a prior :meth:`exchange_code` (or
            a prior :meth:`refresh_token`).
        domain : str | None
            Optional **per-request** tenant hint — same semantics as
            on :meth:`exchange_code`.  Multi-tenant consumers forward
            the user's platform host so refreshes route to the right
            upstream ``token_endpoint``.

        Returns
        -------
        OrchidUpstreamTokenResponse
            Fresh access token plus a (possibly rotated) refresh
            token.  OAuth 2.1 recommends upstream implementations
            rotate on every refresh; the consumer should update its
            stored ``refresh_token`` with the new value if present.

        Raises
        ------
        OrchidAuthExchangeError
            Semantically identical to :meth:`exchange_code` —
            ``invalid_grant`` when the refresh token was revoked or
            expired, ``invalid_client`` when the credentials don't
            match, ``status_code=0`` when the request didn't reach
            the upstream.
        NotImplementedError
            When the concrete subclass hasn't implemented the
            refresh grant yet.  Downstream clients can detect this
            via the ``refresh_via_api`` discovery flag — orchid-api
            only advertises the feature when the wired client
            overrides this method.
        """
        raise NotImplementedError(
            "OrchidAuthExchangeClient.refresh_token is not implemented by "
            f"{type(self).__name__}.  Advertising refresh_via_api=True in "
            "OrchidUpstreamOAuthConfig requires a concrete implementation."
        )
