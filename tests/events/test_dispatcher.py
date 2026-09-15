"""Dispatcher behaviour: outbox order, dedup short-circuit, middleware."""

from __future__ import annotations

import pytest

from orchid_ai.core.events import (
    OrchidSignalDispatcher,
    SignalEnvelope,
)
from orchid_ai.core.events.middleware import SignalIngestMiddleware

# ── Order: insert before enqueue ────────────────────────────


async def test_dispatcher_inserts_then_enqueues(backend, make_envelope) -> None:
    env = make_envelope(type="demo.event", payload={"a": 1})
    result = await backend.dispatcher.ingest(env)
    assert result.signal_id is not None
    assert result.queue_msg_id is not None
    assert result.deduplicated is False

    # Signal landed in the store.
    sig = await backend.signal_store.get(result.signal_id)
    assert sig is not None
    assert sig.type == "demo.event"

    # Queue saw exactly one enqueue.
    assert backend.queue.visible_messages == 1


async def test_dispatcher_dedup_returns_existing_id_no_enqueue(backend, make_envelope) -> None:
    env_a = make_envelope(type="demo.event", payload={"v": 1}, dedupe_key="abc")
    env_b = make_envelope(type="demo.event", payload={"v": 2}, dedupe_key="abc")

    r1 = await backend.dispatcher.ingest(env_a)
    r2 = await backend.dispatcher.ingest(env_b)
    assert r1.deduplicated is False
    assert r2.deduplicated is True
    assert r2.signal_id == r1.signal_id
    assert r2.queue_msg_id is None
    # Only one queue message — the dup didn't enqueue.
    assert backend.queue.visible_messages == 1


async def test_dispatcher_no_dedup_when_dedupe_key_missing(backend, make_envelope) -> None:
    env_a = make_envelope(type="demo.event", payload={"v": 1})
    env_b = make_envelope(type="demo.event", payload={"v": 2})
    r1 = await backend.dispatcher.ingest(env_a)
    r2 = await backend.dispatcher.ingest(env_b)
    # Different signals, both enqueued.
    assert r1.signal_id != r2.signal_id
    assert backend.queue.visible_messages == 2


async def test_dispatcher_dedup_scoped_to_source(backend, make_envelope) -> None:
    env_a = make_envelope(source="src-a", dedupe_key="key")
    env_b = make_envelope(source="src-b", dedupe_key="key")
    r1 = await backend.dispatcher.ingest(env_a)
    r2 = await backend.dispatcher.ingest(env_b)
    assert r1.deduplicated is False
    assert r2.deduplicated is False
    assert r1.signal_id != r2.signal_id


# ── Middleware ──────────────────────────────────────────────


class _TaggingMiddleware(SignalIngestMiddleware):
    """Adds ``trace_tag`` into the payload and bumps a per-instance
    counter so tests can assert ordering."""

    def __init__(self, tag: str, log: list[str]) -> None:
        self._tag = tag
        self._log = log

    async def __call__(self, envelope: SignalEnvelope) -> SignalEnvelope:
        self._log.append(self._tag)
        # Frozen dataclass — rebuild with a tagged payload.
        return SignalEnvelope(
            type=envelope.type,
            payload={**envelope.payload, "trace_tag": self._tag},
            source=envelope.source,
            occurred_at=envelope.occurred_at,
            tenant_key=envelope.tenant_key,
            user_id=envelope.user_id,
            correlation_id=envelope.correlation_id,
            dedupe_key=envelope.dedupe_key,
            identity_claim=envelope.identity_claim,
        )


class _BlockingMiddleware(SignalIngestMiddleware):
    """Refuses to ingest anything — used to assert short-circuit."""

    async def __call__(self, envelope: SignalEnvelope) -> SignalEnvelope:
        raise PermissionError("blocked")


async def test_middleware_runs_in_order(backend, make_envelope) -> None:
    log: list[str] = []
    backend.dispatcher = OrchidSignalDispatcher(
        store=backend.signal_store,
        queue=backend.queue,
        middleware=[
            _TaggingMiddleware("first", log),
            _TaggingMiddleware("second", log),
        ],
    )
    env = make_envelope(payload={"x": 1})
    result = await backend.dispatcher.ingest(env)
    assert log == ["first", "second"]
    sig = await backend.signal_store.get(result.signal_id)
    assert sig is not None
    # Last middleware's tag wins (it ran second on the already-tagged
    # payload).
    assert sig.payload["trace_tag"] == "second"


async def test_middleware_can_short_circuit(backend, make_envelope) -> None:
    backend.dispatcher = OrchidSignalDispatcher(
        store=backend.signal_store,
        queue=backend.queue,
        middleware=[_BlockingMiddleware()],
    )
    env = make_envelope()
    with pytest.raises(PermissionError):
        await backend.dispatcher.ingest(env)
    # Nothing landed.
    assert backend.queue.visible_messages == 0
