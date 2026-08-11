"""Shared fixtures + small helpers for the events test suite.

Single goal: hand out a ready-to-use bundle of in-memory stores +
queue + dispatcher so each test file doesn't repeat the same wiring.
"""

from __future__ import annotations

import datetime as _dt
import uuid as _uuid
from dataclasses import dataclass

import pytest

from orchid_ai.core.events import (
    OrchidSignalDispatcher,
    SignalEnvelope,
)
from orchid_ai.core.events.errors import (
    MintingProbeUnsupportedError,
    OrchidIdentityNotMintableError,
    OrchidServiceAccountUnknownError,
)
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.events.queues.inmemory import (
    InMemoryJobStore,
    InMemorySignalQueue,
    InMemorySignalStore,
)

# ── Bundles ─────────────────────────────────────────────────


@dataclass
class _Backend:
    """A self-contained set of stores + queue + dispatcher."""

    queue: InMemorySignalQueue
    signal_store: InMemorySignalStore
    job_store: InMemoryJobStore
    dispatcher: OrchidSignalDispatcher


@pytest.fixture
def backend() -> _Backend:
    queue = InMemorySignalQueue()
    signal_store = InMemorySignalStore()
    job_store = InMemoryJobStore()
    dispatcher = OrchidSignalDispatcher(store=signal_store, queue=queue)
    return _Backend(
        queue=queue,
        signal_store=signal_store,
        job_store=job_store,
        dispatcher=dispatcher,
    )


# ── Envelope helpers ────────────────────────────────────────


@pytest.fixture
def make_envelope():
    """Factory fixture — call ``make_envelope(type=..., payload=...)``
    inside a test to get a fully-populated :class:`SignalEnvelope`."""

    def _factory(
        *,
        type: str = "demo.event",
        payload: dict | None = None,
        source: str = "test:fixture",
        tenant_key: str = "tenant-123",
        user_id: str | None = "user-abc",
        dedupe_key: str | None = None,
        identity_claim: dict | None = None,
        correlation_id: str | None = None,
    ) -> SignalEnvelope:
        return SignalEnvelope(
            type=type,
            payload=payload or {},
            source=source,
            occurred_at=_dt.datetime.now(tz=_dt.UTC),
            tenant_key=tenant_key,
            user_id=user_id,
            correlation_id=correlation_id or f"corr-{_uuid.uuid4().hex[:8]}",
            dedupe_key=dedupe_key,
            identity_claim=identity_claim,
        )

    return _factory


# ── Fake identity resolver — sync, deterministic ────────────


class FakeIdentityResolver:
    """Tiny sync resolver that the asyncio processor accepts via
    ``_maybe_await``.

    - ``resolve_service_account(name)`` returns a fresh
      ``OrchidAuthContext`` tagged with ``name`` in ``extra``.
    - ``mint_for_user(tenant_key, user_id)`` returns an
      ``OrchidAuthContext`` for the requested user — except when the
      probe sentinel ``"__probe__"`` is supplied, where it raises
      ``OrchidIdentityNotMintableError`` (i.e. 'I support minting,
      just not for this sentinel user' — which is the *passing*
      behaviour at registration time).
    - ``known_service_accounts`` controls which names are accepted —
      pass ``known_service_accounts=set()`` to make every call raise
      ``OrchidServiceAccountUnknownError``.
    """

    def __init__(
        self,
        *,
        known_service_accounts: set[str] | None = None,
        can_mint: bool = True,
    ) -> None:
        self._known = known_service_accounts
        self._can_mint = can_mint
        self.calls: list[tuple[str, tuple, dict]] = []

    # The base ``resolve`` — the events pipeline never calls it, but
    # it's the canonical resolver method so we keep it on the fake.
    def resolve(self, domain: str, bearer_token: str) -> OrchidAuthContext:
        self.calls.append(("resolve", (domain, bearer_token), {}))
        return OrchidAuthContext(access_token=bearer_token, tenant_key="tenant-123", user_id="user-abc")

    def resolve_service_account(self, name: str) -> OrchidAuthContext:
        self.calls.append(("resolve_service_account", (name,), {}))
        if self._known is not None and name not in self._known:
            raise OrchidServiceAccountUnknownError(name)
        ctx = OrchidAuthContext(
            access_token=f"sa-token:{name}",
            tenant_key="tenant-123",
            user_id="",
        )
        ctx.extra["service_account"] = name
        return ctx

    def mint_for_user(self, tenant_key: str, user_id: str) -> OrchidAuthContext:
        self.calls.append(("mint_for_user", (tenant_key, user_id), {}))
        if not self._can_mint:
            raise MintingProbeUnsupportedError(type(self).__name__)
        if user_id == "__probe__":
            raise OrchidIdentityNotMintableError(tenant_key, user_id)
        return OrchidAuthContext(
            access_token=f"user-token:{tenant_key}:{user_id}",
            tenant_key=tenant_key,
            user_id=user_id,
        )


@pytest.fixture
def fake_resolver() -> FakeIdentityResolver:
    return FakeIdentityResolver(known_service_accounts={"digest-bot", "ops-bot"}, can_mint=True)
