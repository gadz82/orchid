"""DispatcherSignalEmitter — every emit becomes a dispatcher ingest."""

from __future__ import annotations

from orchid_ai.events.producers.internal import DispatcherSignalEmitter, InternalEmissionProducer
from orchid_ai.utils import import_class


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


async def test_internal_emission_producer_is_importable_and_lifecycle_managed(backend, make_envelope) -> None:
    cls = import_class("orchid_ai.events.producers.internal.InternalEmissionProducer")
    producer = cls()

    await producer.start(backend.dispatcher)
    try:
        assert isinstance(producer, InternalEmissionProducer)
        assert isinstance(producer.emitter, DispatcherSignalEmitter)

        env = make_envelope(type="internal.producer.fired")
        result = await producer.emitter.emit(env)
        assert result.signal_id is not None
        assert await backend.signal_store.get(result.signal_id) is not None
    finally:
        await producer.stop()

    assert producer.emitter is None
