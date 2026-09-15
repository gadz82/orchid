"""MCP Authorization 2025-03-26 discovery chain.

Implements the three RFCs that the MCP authorization spec composes:

1. **Protected Resource Metadata** (RFC 9728).  An MCP server that
   requires OAuth responds to unauthenticated requests with ``401`` +
   ``WWW-Authenticate: Bearer resource_metadata="…"``.  The URL points
   to the server's protected-resource metadata document; its
   ``authorization_servers`` array names one or more OAuth 2.0
   authorization servers that can issue tokens for it.

2. **Authorization Server Metadata** (RFC 8414).  For each candidate
   authorization server, fetch ``/.well-known/oauth-authorization-server``
   (falling back to ``/.well-known/openid-configuration`` when the
   OAuth 2.0 variant is missing).  The document exposes the endpoints
   the client needs: ``authorization_endpoint``, ``token_endpoint``,
   ``registration_endpoint``, plus advertised ``scopes_supported``,
   ``grant_types_supported``, ``code_challenge_methods_supported`` and
   ``token_endpoint_auth_methods_supported``.

3. **Dynamic Client Registration** (RFC 7591).  POST client metadata
   — redirect URIs, grant types, PKCE method, requested auth method —
   to the ``registration_endpoint`` and receive back a ``client_id``
   plus (for confidential clients) a ``client_secret``.  The result is
   persisted to :class:`OrchidMCPClientRegistrationStore` so the same
   registration is reused on every subsequent container lifetime.

The final output is an :class:`OrchidMCPClientRegistration` record.
Callers (the MCP client, the API authorization router) never need to
touch HTTP themselves.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from ..core.mcp import (
    OrchidMCPClientRegistration,
    OrchidMCPClientRegistrationStore,
    OrchidMCPDiscoveryError,
)

logger = logging.getLogger(__name__)


# ── Public service ─────────────────────────────────────────


@dataclass
class OrchidMCPAuthDiscovery:
    """End-to-end RFC 9728 → RFC 8414 → RFC 7591 discovery service.

    Stateless aside from the injected ``store`` (used as an idempotent
    cache so repeated calls for the same server return the same
    registration rather than re-running DCR).  Pure I/O: no business
    rules, no side effects beyond the HTTP + persistence work.
    """

    #: Persistence layer for registrations — one row per MCP server.
    store: OrchidMCPClientRegistrationStore

    #: Public redirect URI where the authorization server sends the user
    #: back after consent.  Corresponds to ``API_BASE_URL + /mcp/auth/callback``.
    redirect_uri: str

    #: User-visible client name registered with the authorization server
    #: (RFC 7591 ``client_name``).  Defaults to a generic string to keep
    #: the framework vendor-neutral; integrators may override.
    client_name: str = "Orchid MCP Client"

    #: Timeout for each HTTP leg, in seconds.
    http_timeout: float = 15.0

    # ── Public entry points ───────────────────────────────

    async def ensure_registration(
        self,
        *,
        server_name: str,
        resource_metadata_url: str,
        manual_config: dict[str, Any] | None = None,
    ) -> OrchidMCPClientRegistration:
        """Return a cached or freshly-minted registration for a server.

        Cache semantics: the first call per MCP server runs the full
        discovery chain and stores the result; subsequent calls return
        the stored record verbatim.  Upstream callers trigger a refresh
        by deleting the store row.

        When ``manual_config`` is provided (from YAML auth config), it
        skips auto-discovery and uses the supplied endpoints + credentials
        directly.  This is for servers that do not implement MCP 2025-03-26
        discovery (e.g. Atlassian Rovo).
        """
        existing = await self.store.get(server_name)
        if existing is not None:
            logger.info(
                "[MCPAuthDiscovery] Using cached registration for '%s' (client_id=%s…)",
                server_name,
                existing.client_id[:8] if existing.client_id else "",
            )
            return existing

        if manual_config and manual_config.get("authorization_endpoint") and manual_config.get("token_endpoint"):
            logger.info(
                "[MCPAuthDiscovery] Using manual OAuth config for '%s' (skipping auto-discovery)",
                server_name,
            )
            return await self._seed_from_manual_config(server_name, manual_config)

        logger.info(
            "[MCPAuthDiscovery] Starting discovery for '%s' via %s",
            server_name,
            resource_metadata_url,
        )

        resource_meta = await self._fetch_protected_resource_metadata(
            server_name,
            resource_metadata_url,
        )
        auth_server_url = self._pick_authorization_server(server_name, resource_meta)
        as_meta = await self._fetch_authorization_server_metadata(server_name, auth_server_url)

        registration_endpoint = as_meta.get("registration_endpoint", "")
        if not registration_endpoint:
            raise OrchidMCPDiscoveryError(
                server_name,
                f"authorization server '{auth_server_url}' does not advertise a "
                f"``registration_endpoint`` — dynamic client registration (RFC 7591) "
                f"is required by MCP 2025-03-26.  Provide a DCR-capable authorization "
                f"server or pre-seed the client_registration store.",
            )

        dcr = await self._register_client(server_name, as_meta, registration_endpoint)

        record = OrchidMCPClientRegistration(
            server_name=server_name,
            authorization_endpoint=as_meta.get("authorization_endpoint", ""),
            token_endpoint=as_meta.get("token_endpoint", ""),
            registration_endpoint=registration_endpoint,
            issuer=as_meta.get("issuer", ""),
            scopes_supported=_scopes_to_string(as_meta.get("scopes_supported", [])),
            token_endpoint_auth_methods_supported=_pick_auth_method(
                as_meta.get("token_endpoint_auth_methods_supported", ["client_secret_post"]),
            ),
            client_id=dcr.get("client_id", ""),
            client_secret=dcr.get("client_secret", ""),
            client_id_issued_at=float(dcr.get("client_id_issued_at", 0) or 0),
            client_secret_expires_at=float(dcr.get("client_secret_expires_at", 0) or 0),
        )
        await self.store.save(record)
        logger.info(
            "[MCPAuthDiscovery] Registered client for '%s' → client_id=%s… (public=%s)",
            server_name,
            record.client_id[:8],
            record.is_public_client,
        )
        return record

    async def _seed_from_manual_config(
        self,
        server_name: str,
        config: dict[str, Any],
    ) -> OrchidMCPClientRegistration:
        """Create a registration record from manually-supplied OAuth config.

        Used for servers that do not implement MCP 2025-03-26 discovery
        (e.g. Atlassian Rovo).  The config dict comes from the YAML
        ``auth`` block and contains ``authorization_endpoint``,
        ``token_endpoint``, ``client_id``, ``client_secret``, etc.
        """
        record = OrchidMCPClientRegistration(
            server_name=server_name,
            authorization_endpoint=config.get("authorization_endpoint", ""),
            token_endpoint=config.get("token_endpoint", ""),
            registration_endpoint=config.get("registration_endpoint", ""),
            issuer=config.get("issuer", ""),
            scopes_supported=config.get("scopes", ""),
            token_endpoint_auth_methods_supported="client_secret_post" if config.get("client_secret") else "none",
            client_id=config.get("client_id", ""),
            client_secret=config.get("client_secret", ""),
            client_id_issued_at=0.0,
            client_secret_expires_at=0.0,
        )
        await self.store.save(record)
        logger.info(
            "[MCPAuthDiscovery] Seeded manual registration for '%s' → client_id=%s…",
            server_name,
            record.client_id[:8] if record.client_id else "(none)",
        )
        return record

    # ── RFC 9728 — protected resource metadata ───────────

    async def _fetch_protected_resource_metadata(
        self,
        server_name: str,
        url: str,
    ) -> dict[str, Any]:
        data = await _get_json(url, self.http_timeout, server_name, "protected resource metadata")
        if not isinstance(data.get("authorization_servers"), list) or not data["authorization_servers"]:
            raise OrchidMCPDiscoveryError(
                server_name,
                f"protected resource metadata at '{url}' is missing a non-empty ``authorization_servers`` array",
            )
        return data

    def _pick_authorization_server(
        self,
        server_name: str,
        resource_meta: dict[str, Any],
    ) -> str:
        servers = [s for s in resource_meta.get("authorization_servers", []) if isinstance(s, str)]
        if not servers:
            raise OrchidMCPDiscoveryError(server_name, "no string entries in authorization_servers")
        # First entry wins — RFC 9728 doesn't define a preference order,
        # and operators list their primary AS first.  Future: honour an
        # integrator-supplied selector.
        return servers[0].rstrip("/")

    # ── RFC 8414 — authorization server metadata ─────────

    async def _fetch_authorization_server_metadata(
        self,
        server_name: str,
        issuer: str,
    ) -> dict[str, Any]:
        candidates = [
            f"{issuer}/.well-known/oauth-authorization-server",
            f"{issuer}/.well-known/openid-configuration",
        ]
        last_error: str = ""
        for url in candidates:
            try:
                return await _get_json(url, self.http_timeout, server_name, "authorization server metadata")
            except OrchidMCPDiscoveryError as exc:
                last_error = exc.reason
                continue
        raise OrchidMCPDiscoveryError(
            server_name,
            f"no authorization server metadata found at {candidates!r} — last error: {last_error}",
        )

    # ── RFC 7591 — dynamic client registration ───────────

    async def _register_client(
        self,
        server_name: str,
        as_meta: dict[str, Any],
        registration_endpoint: str,
    ) -> dict[str, Any]:
        auth_method = _pick_auth_method(
            as_meta.get("token_endpoint_auth_methods_supported", ["client_secret_post"]),
        )
        grant_types = _intersect_preferred(
            as_meta.get("grant_types_supported", ["authorization_code", "refresh_token"]),
            ("authorization_code", "refresh_token"),
        )
        body: dict[str, Any] = {
            "client_name": self.client_name,
            "redirect_uris": [self.redirect_uri],
            "grant_types": list(grant_types),
            "response_types": ["code"],
            "token_endpoint_auth_method": auth_method,
            "application_type": "web",
        }
        scopes = as_meta.get("scopes_supported", [])
        if isinstance(scopes, list) and scopes:
            body["scope"] = _scopes_to_string(scopes)

        import httpx

        logger.debug(
            "[MCPAuthDiscovery] POST %s body=%s",
            registration_endpoint,
            {k: v for k, v in body.items() if k != "client_secret"},
        )
        async with httpx.AsyncClient(timeout=self.http_timeout) as http:
            resp = await http.post(
                registration_endpoint,
                json=body,
                headers={"Accept": "application/json"},
            )
        if resp.status_code >= 400:
            raise OrchidMCPDiscoveryError(
                server_name,
                f"dynamic client registration rejected by '{registration_endpoint}' "
                f"({resp.status_code}): {resp.text[:500]}",
            )
        data = resp.json()
        if "client_id" not in data:
            raise OrchidMCPDiscoveryError(
                server_name,
                f"registration response from '{registration_endpoint}' missing ``client_id``",
            )
        return data


# ── 401 WWW-Authenticate parsing ──────────────────────────


_RESOURCE_METADATA_RE = re.compile(
    r'resource_metadata\s*=\s*"(?P<url>[^"]+)"',
    re.IGNORECASE,
)


def extract_resource_metadata_url(www_authenticate_header: str) -> str | None:
    """Extract ``resource_metadata="…"`` from a ``WWW-Authenticate`` header.

    RFC 9728 defines the parameter name as ``resource_metadata``.  The
    header value follows the standard ``Bearer <params>`` challenge
    syntax; we scan for the first match and return the unquoted URL.
    Returns ``None`` when the header is absent or the parameter is not
    present (callers treat that as "server doesn't support discovery").
    """
    if not www_authenticate_header:
        return None
    match = _RESOURCE_METADATA_RE.search(www_authenticate_header)
    return match.group("url") if match else None


async def probe_mcp_server_for_resource_metadata(
    *,
    mcp_url: str,
    server_name: str,
    timeout: float = 10.0,
) -> str:
    """Send an unauthenticated probe to the MCP server and pull the RFC 9728 URL.

    The MCP 2025-03-26 authorization spec says a server requiring OAuth
    MUST return ``401`` with a ``WWW-Authenticate: Bearer
    resource_metadata="…"`` header on any unauthenticated request.  We
    issue a minimal POST (the streamable-HTTP JSON-RPC transport expects
    POST; the server short-circuits on auth before parsing the body)
    and extract the metadata URL from the response.

    Raises :class:`OrchidMCPDiscoveryError` when:
      * the server does not return 401 (looks like no auth is required);
      * the 401 is missing a ``WWW-Authenticate`` header;
      * the header does not carry a ``resource_metadata`` parameter
        (the server isn't MCP 2025-03-26 compliant — integrators must
        pre-seed :class:`OrchidMCPClientRegistrationStore`).
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=timeout) as http:
            # Empty JSON body is enough — auth checks run before
            # JSON-RPC parsing on every compliant server.
            resp = await http.post(
                mcp_url,
                json={},
                headers={"Accept": "application/json, text/event-stream"},
            )
    except Exception as exc:
        raise OrchidMCPDiscoveryError(
            server_name,
            f"unauthenticated probe of '{mcp_url}' failed: {exc}",
        ) from exc

    if resp.status_code != 401:
        raise OrchidMCPDiscoveryError(
            server_name,
            f"expected 401 from '{mcp_url}' to start MCP 2025-03-26 discovery "
            f"but got {resp.status_code} — server may not require OAuth, or may "
            f"require auth at a layer the spec doesn't cover",
        )

    www_auth = resp.headers.get("www-authenticate", "")
    metadata_url = extract_resource_metadata_url(www_auth)
    if not metadata_url:
        raise OrchidMCPDiscoveryError(
            server_name,
            f"401 from '{mcp_url}' did not advertise ``resource_metadata`` in "
            f"``WWW-Authenticate`` header (got: {www_auth!r}) — server is not "
            f"MCP 2025-03-26 compliant.  Seed OrchidMCPClientRegistrationStore "
            f"manually with the authorization-server endpoints + credentials.",
        )
    return metadata_url


# ── Small helpers (module-private) ─────────────────────────


async def _get_json(
    url: str,
    timeout: float,
    server_name: str,
    what: str,
) -> dict[str, Any]:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=timeout) as http:
            resp = await http.get(url, headers={"Accept": "application/json"})
    except Exception as exc:  # network / TLS / DNS / timeout
        raise OrchidMCPDiscoveryError(server_name, f"{what} fetch failed: {exc}") from exc

    if resp.status_code >= 400:
        raise OrchidMCPDiscoveryError(
            server_name,
            f"{what} fetch returned {resp.status_code} for {url}: {resp.text[:200]}",
        )
    try:
        return resp.json()
    except Exception as exc:
        raise OrchidMCPDiscoveryError(
            server_name,
            f"{what} at {url} is not valid JSON: {exc}",
        ) from exc


def _scopes_to_string(scopes: Any) -> str:
    if isinstance(scopes, str):
        return scopes
    if isinstance(scopes, (list, tuple)):
        return " ".join(str(s) for s in scopes if s)
    return ""


def _pick_auth_method(methods: Any) -> str:
    """Pick the best supported ``token_endpoint_auth_method``.

    Preference order: ``client_secret_post`` > ``client_secret_basic`` >
    ``none`` > first entry advertised.  ``client_secret_post`` wins
    because it avoids the double-client-id problem (Basic + body)
    that several strict IdPs reject with ``invalid_request``.  ``none``
    is the public-client (PKCE-only) form.
    """
    advertised: list[str] = [str(m) for m in methods or []] if isinstance(methods, (list, tuple)) else []
    for preferred in ("client_secret_post", "client_secret_basic", "none"):
        if preferred in advertised:
            return preferred
    return advertised[0] if advertised else "client_secret_post"


def _intersect_preferred(advertised: Any, preferred: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(advertised, (list, tuple)):
        return preferred
    adv_set = {str(a) for a in advertised}
    picked = tuple(p for p in preferred if p in adv_set)
    return picked or preferred


# ── Time helpers ─────────────────────────────────────────


def now_epoch() -> float:
    return time.time()
