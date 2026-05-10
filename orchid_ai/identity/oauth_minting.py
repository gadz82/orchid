"""``OAuthMintingMixin`` — opt-in default for ``mint_for_user``.

Ships with the framework so consumers whose only consumer-facing MCP
servers are ``oauth``-mode don't have to re-derive the
identity-from-stored-token plumbing themselves.

Composition:

```python
from orchid_ai.core.identity import OrchidIdentityResolver
from orchid_ai.identity import OAuthMintingMixin


class MyResolver(OAuthMintingMixin, OrchidIdentityResolver):
    def __init__(
        self,
        *,
        token_store: OrchidMCPTokenStore,
        server_name: str,
        **kwargs,
    ) -> None:
        OAuthMintingMixin.__init__(
            self,
            token_store=token_store,
            server_name=server_name,
        )
        ...

    async def resolve(self, domain: str, bearer: str) -> OrchidAuthContext:
        ...  # consumer's own implementation
```

The mixin reads the user's stored token from
:class:`OrchidMCPTokenStore`.  When the access token is fresh (more
than ``stale_margin_seconds`` from expiry, default 60s) it builds an
``OrchidAuthContext`` directly.  When stale, it delegates to an
optional :class:`OrchidTokenRefresher` (a Protocol — anything with a
``refresh(server_name, refresh_token)`` coroutine works) and persists
the refreshed token back through the store.

Without a refresher, a stale token raises
:class:`OrchidIdentityNotMintableError` for that user — the registry's
mint probe at boot is unaffected because the probe distinguishes
"no specific user" from "no minting at all" (see
:class:`MintingProbeUnsupportedError`).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol, runtime_checkable

from ..core.events.errors import OrchidIdentityNotMintableError
from ..core.mcp import OrchidMCPTokenRecord, OrchidMCPTokenStore
from ..core.state import OrchidAuthContext

_logger = logging.getLogger(__name__)


@runtime_checkable
class OrchidTokenRefresher(Protocol):
    """Narrow protocol for refreshing an OAuth access token.

    Any object with this method works.  In production this is
    typically the same StreamableHttpMCPClient that talks to the
    server in question — its OAuth path already implements RFC 6749
    refresh and persists the new token.  In tests it's a 5-line fake.
    """

    async def refresh(self, *, server_name: str, refresh_token: str) -> dict[str, Any]:
        """Return ``{"access_token", "refresh_token", "expires_at",
        "scopes"}`` after a successful refresh.  Raise on failure —
        the mixin wraps any exception in
        :class:`OrchidIdentityNotMintableError`."""
        ...


class OAuthMintingMixin:
    """Default ``mint_for_user`` for OAuth-style resolvers.

    The mixin is a `composition` mixin (mix BEFORE
    :class:`OrchidIdentityResolver` in the MRO) — it does NOT provide
    the abstract :meth:`resolve` method.  Consumers still implement
    bearer-token resolution themselves.
    """

    def __init__(
        self,
        *,
        token_store: OrchidMCPTokenStore,
        server_name: str,
        refresher: OrchidTokenRefresher | None = None,
        stale_margin_seconds: int = 60,
    ) -> None:
        self._token_store = token_store
        self._server_name = server_name
        self._refresher = refresher
        self._stale_margin = max(0, stale_margin_seconds)

    async def mint_for_user(self, tenant_key: str, user_id: str) -> OrchidAuthContext:
        """Mint an :class:`OrchidAuthContext` for the requested user.

        Steps:

        1. Look up the stored OAuth record via :class:`OrchidMCPTokenStore`.
        2. If the access token is non-stale, build the context.
        3. If stale and a :class:`OrchidTokenRefresher` is configured,
           refresh and persist; build the context from the new record.
        4. Otherwise raise :class:`OrchidIdentityNotMintableError` for
           that ``(tenant_key, user_id)`` pair.

        Service accounts call :meth:`resolve_service_account` (which
        defaults to raising); this mixin only handles the user-bound
        ``act_as_user`` flavour.
        """
        record = await self._token_store.get_token(tenant_key, user_id, self._server_name)
        if record is None:
            raise OrchidIdentityNotMintableError(tenant_key, user_id)

        if self._is_fresh(record):
            return _record_to_auth(record, tenant_key=tenant_key, user_id=user_id)

        if self._refresher is None or not record.refresh_token:
            raise OrchidIdentityNotMintableError(tenant_key, user_id)

        try:
            refreshed = await self._refresher.refresh(
                server_name=self._server_name,
                refresh_token=record.refresh_token,
            )
        except Exception as exc:
            _logger.warning(
                "[OAuthMintingMixin] refresh failed for %s/%s: %s",
                tenant_key,
                user_id,
                exc,
            )
            raise OrchidIdentityNotMintableError(tenant_key, user_id) from exc

        new_record = OrchidMCPTokenRecord(
            server_name=self._server_name,
            tenant_id=tenant_key,
            user_id=user_id,
            access_token=str(refreshed["access_token"]),
            refresh_token=str(refreshed.get("refresh_token") or record.refresh_token),
            expires_at=float(refreshed.get("expires_at") or 0.0),
            scopes=str(refreshed.get("scopes") or record.scopes or ""),
        )
        await self._token_store.save_token(new_record)
        return _record_to_auth(new_record, tenant_key=tenant_key, user_id=user_id)

    def _is_fresh(self, record: OrchidMCPTokenRecord) -> bool:
        """A token is fresh when it has no expiry (``expires_at == 0``,
        which the store contract treats as 'no expiry info') OR
        ``expires_at - stale_margin > now``.  The margin gives downstream
        callers a buffer before the IdP itself decides the token is dead.
        """
        if record.expires_at == 0:
            return True
        return record.expires_at - self._stale_margin > time.time()


def _record_to_auth(
    record: OrchidMCPTokenRecord,
    *,
    tenant_key: str,
    user_id: str,
) -> OrchidAuthContext:
    return OrchidAuthContext(
        access_token=record.access_token,
        tenant_key=tenant_key,
        user_id=user_id,
        expires_at=record.expires_at,
    )
