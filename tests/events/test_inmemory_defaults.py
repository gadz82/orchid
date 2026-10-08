"""Events bootstrap defaults to the in-memory store + queue when omitted."""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

from orchid_ai.config.schema_events import OrchidEventsConfig, OrchidProcessorConfig
from orchid_ai.core.events.dispatcher import OrchidSignalDispatcher
from orchid_ai.core.events.signal import SignalEnvelope
from orchid_ai.events.backends.inmemory import InMemoryEventStorage
from orchid_ai.events.bootstrap import _build_queue, _build_storage, start_events, stop_events
from orchid_ai.events.queues.inmemory import InMemorySignalQueue


class TestHelperDefaults:
    async def test_build_storage_defaults_to_in_memory(self):
        storage = await _build_storage(SimpleNamespace(store=None))
        assert isinstance(storage, InMemoryEventStorage)

    async def test_build_queue_defaults_to_in_memory(self):
        storage = await _build_storage(SimpleNamespace(store=None))
        queue = await _build_queue(SimpleNamespace(queue=None), storage=storage)
        assert isinstance(queue, InMemorySignalQueue)


class TestDefaultBackendsDispatch:
    async def test_dispatch_round_trip_without_configured_store_or_queue(self):
        storage = await _build_storage(SimpleNamespace(store=None))
        queue = await _build_queue(SimpleNamespace(queue=None), storage=storage)
        dispatcher = OrchidSignalDispatcher(store=storage.signals, queue=queue)

        result = await dispatcher.ingest(
            SignalEnvelope(
                type="smoke.event",
                payload={"k": "v"},
                source="test",
                occurred_at=dt.datetime.now(dt.UTC),
                tenant_key="t1",
            )
        )

        assert result.deduplicated is False
        assert await storage.signals.get(result.signal_id) is not None
        batch = await queue.dequeue(batch_size=1, lease_seconds=30)
        assert [message.signal_id for message in batch] == [result.signal_id]


class TestStartEventsWithDefaults:
    async def test_enabled_config_without_store_or_queue_boots_on_memory(self):
        cfg = OrchidEventsConfig(
            enabled=True,
            processors=[
                OrchidProcessorConfig(
                    class_path="orchid_ai.events.processors.asyncio_pool.AsyncioWorkerPoolProcessor",
                    poll_interval_ms=10_000,
                    drain_timeout_seconds=0.1,
                )
            ],
        )

        runtime = await start_events(
            events_config=cfg,
            chat_storage=None,
            identity_resolver=None,
            session_warmer=None,
            known_agents=set(),
        )
        try:
            assert runtime.enabled is True
            assert isinstance(runtime.storage, InMemoryEventStorage)
            assert isinstance(runtime.signal_queue, InMemorySignalQueue)
            assert runtime.dispatcher is not None
        finally:
            await stop_events(runtime)
