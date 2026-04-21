"""
OrchidOAuthStateStore — pluggable PKCE + CSRF state for MCP OAuth flows.

Between the ``GET /authorize`` redirect and the ``GET /callback``
redirect, the server must remember per-request data (PKCE verifier,
server name, token endpoint, tenant / user identity).  The default
:class:`OrchidInMemoryOAuthStateStore` keeps this in a process-local dict —
fine for single-instance deployments.

Multi-instance deployments (multiple uvicorn workers behind a load
balancer) need shared storage so the callback can land on a different
worker than the authorize.  Integrators implement :class:`OrchidOAuthStateStore`
on top of Redis / the database of their choice and register it via the
:func:`build_oauth_state_store` factory::

    register_oauth_state_store("redis", my_build_redis_store)

Then point ``oauth_state_store_class`` at ``"redis"`` (or a dotted
class path) in settings.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

from ..utils import import_class

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrchidOAuthPendingState:
    """Per-request state stored between ``/authorize`` and ``/callback``.

    Frozen because the payload is captured at the ``/authorize`` call
    and consumed verbatim at the ``/callback`` — there's no intermediate
    state a store should (or is allowed to) mutate.
    """

    server_name: str
    tenant_id: str
    user_id: str
    code_verifier: str
    token_endpoint: str
    created_at: float


class OrchidOAuthStateStore(ABC):
    """Abstract PKCE / CSRF state store for MCP OAuth flows.

    Implementations must be safe to call from concurrent requests
    (``OrchidInMemoryOAuthStateStore`` relies on a single event loop / GIL).

    Constructor contract
    --------------------
    Subclasses instantiated via :func:`build_oauth_state_store` from a
    dotted class path MUST accept two keyword arguments::

        def __init__(self, *, dsn: str = "", ttl_seconds: float = 600.0) -> None:

    The factory always passes ``ttl_seconds``.  ``dsn`` is forwarded
    only when the caller supplied one — so subclasses with no backing
    store can simply declare a default.  Integrators needing a
    different constructor signature should register an async factory
    via :func:`register_oauth_state_store` instead of relying on the
    dotted-path path.
    """

    @abstractmethod
    async def put(self, state: str, payload: OrchidOAuthPendingState) -> None:
        """Store ``payload`` under the CSRF ``state`` token."""

    @abstractmethod
    async def pop(self, state: str) -> OrchidOAuthPendingState | None:
        """Fetch *and remove* the payload for ``state``; ``None`` if unknown."""

    async def cleanup_expired(self, ttl_seconds: float) -> None:
        """Best-effort eviction of expired entries.

        Default implementation is a no-op — stores with native TTL
        support (Redis) should override; the in-memory store does its
        own sweep on every access.
        """
        return None

    async def close(self) -> None:
        """Release any underlying resources.  Default no-op."""
        return None


class OrchidInMemoryOAuthStateStore(OrchidOAuthStateStore):
    """Single-process OAuth state store — the default.

    Evicts expired entries lazily on every access.  Suitable for
    single-worker deployments (dev, demos, simple prod).
    """

    def __init__(self, *, ttl_seconds: float = 600.0) -> None:
        self._ttl = ttl_seconds
        self._pending: dict[str, OrchidOAuthPendingState] = {}

    async def put(self, state: str, payload: OrchidOAuthPendingState) -> None:
        await self._sweep()
        self._pending[state] = payload

    async def pop(self, state: str) -> OrchidOAuthPendingState | None:
        await self._sweep()
        return self._pending.pop(state, None)

    async def cleanup_expired(self, ttl_seconds: float) -> None:
        await self._sweep(override_ttl=ttl_seconds)

    async def _sweep(self, *, override_ttl: float | None = None) -> None:
        ttl = override_ttl if override_ttl is not None else self._ttl
        now = time.time()
        expired = [k for k, v in self._pending.items() if now - v.created_at > ttl]
        for k in expired:
            del self._pending[k]


# ── Factory + registry ────────────────────────────────────────


_OAUTH_STATE_REGISTRY: dict[str, Callable[..., Coroutine[Any, Any, OrchidOAuthStateStore]]] = {}


def register_oauth_state_store(
    type_name: str,
    factory: Callable[..., Coroutine[Any, Any, OrchidOAuthStateStore]],
) -> None:
    """Register a custom OAuth state store type for :func:`build_oauth_state_store`.

    The factory is an async callable ``(dsn: str, ttl_seconds: float) ->
    OrchidOAuthStateStore``.
    """
    _OAUTH_STATE_REGISTRY[type_name] = factory
    logger.info("[OrchidOAuthStateStore] Registered custom type: %s", type_name)


async def build_oauth_state_store(
    store_type: str = "memory",
    *,
    dsn: str = "",
    ttl_seconds: float = 600.0,
) -> OrchidOAuthStateStore:
    """Resolve ``store_type`` to a concrete :class:`OrchidOAuthStateStore`.

    Resolution order:
        1. Registered types from :func:`register_oauth_state_store`.
        2. The built-in ``"memory"`` type → :class:`OrchidInMemoryOAuthStateStore`.
        3. A dotted class path → ``cls(ttl_seconds=..., dsn=...)``.

    See :class:`OrchidOAuthStateStore` for the constructor contract that
    dotted-path subclasses must satisfy.  If your class needs a
    different signature, register an async factory via
    :func:`register_oauth_state_store` instead.

    Raises ``TypeError`` when a dotted path resolves to a class that is
    not a :class:`OrchidOAuthStateStore` subclass, or whose constructor
    rejects the ``ttl_seconds`` / ``dsn`` kwargs.
    """
    if store_type in _OAUTH_STATE_REGISTRY:
        factory_fn = _OAUTH_STATE_REGISTRY[store_type]
        logger.info("[OrchidOAuthStateStore] Using registered type: %s", store_type)
        return await factory_fn(dsn=dsn, ttl_seconds=ttl_seconds)

    if store_type == "memory":
        return OrchidInMemoryOAuthStateStore(ttl_seconds=ttl_seconds)

    cls = import_class(store_type)
    if not (isinstance(cls, type) and issubclass(cls, OrchidOAuthStateStore)):
        raise TypeError(f"'{store_type}' resolves to {cls!r}, which is not an OrchidOAuthStateStore subclass.")
    kwargs: dict[str, Any] = {"ttl_seconds": ttl_seconds}
    if dsn:
        kwargs["dsn"] = dsn
    logger.info("[OrchidOAuthStateStore] Using custom class: %s", store_type)
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise TypeError(
            f"Could not instantiate '{store_type}' with kwargs={list(kwargs)}. "
            "Dotted-path OrchidOAuthStateStore subclasses must accept `ttl_seconds=` "
            "(and optionally `dsn=`).  Register an async factory via "
            "`register_oauth_state_store` if you need a different constructor."
        ) from exc
