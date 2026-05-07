"""DispatcherSignalEmitter — every emit becomes a dispatcher ingest."""

from __future__ import annotations

from orchid_ai.events.producers.internal import DispatcherSignalEmitter


async def test_emitter_forwards_to_dispatcher(backend, make_envelope) -> None:
    emitter = DispatcherSignalEmitter(backend.dispatcher)
    env = make_envelope(type="internal.fired", payload={"src": "agent"})
    result = await emitter.emit(env)

    assert result.signal_id is not None
    assert result.queue_msg_id is not None
    sig = await backend.signal_store.get(result.signal_id)
    assert sig is not None
    assert sig.type == "internal.fired"
    assert sig.payload == {"src": "agent"}


async def test_emitter_idempotent_on_dedupe_key(backend, make_envelope) -> None:
    emitter = DispatcherSignalEmitter(backend.dispatcher)
    env_a = make_envelope(dedupe_key="abc")
    env_b = make_envelope(dedupe_key="abc")
    r1 = await emitter.emit(env_a)
    r2 = await emitter.emit(env_b)
    assert r2.deduplicated is True
    assert r1.signal_id == r2.signal_id
