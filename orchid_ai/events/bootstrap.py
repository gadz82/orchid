"""Wire ``OrchidEventsConfig`` → live runtime objects in the lifespan.

Encapsulates the §16.6 lifespan ordering so ``setup_orchid`` /
``teardown_orchid`` stay readable.  When ``events.enabled`` is
``False`` (the default), every helper is a no-op and zero new
runtime objects are created — that's the §rule-8 zero-overhead
opt-out.

What this module owns:

- Resolving ``events.store`` / ``events.queue`` dotted paths into
  live :class:`SQLiteEventStorage` / :class:`PostgresEventStorage`
  instances and a matching :class:`OrchidSignalQueue`.
- Building the trigger registry from
  :attr:`OrchidEventsConfig.triggers`, with the §13 / §25 / §26
  registration-time validations.
- Compiling :attr:`OrchidEventsConfig.ingestion.sources` into
  :class:`SignalSource` rows for the
  :class:`HTTPIngestionProducer`.
- Resolving secret refs (``env:VAR``) for HMAC / bearer validators.
- Building the :class:`OrchidSignalDispatcher`,
  :class:`AsyncioWorkerPoolProcessor`, and the producer list per
  ``events.processors`` / ``events.producers``.
- Starting and stopping everything in the right order.

What it does NOT own: route registration (that's
:func:`orchid_api.main`) and the §26 boot warning when no
admin-role mapping exists (that's :func:`orchid_api.lifecycle`).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from orchid_ai.config.schema_events import (
    OrchidEventsConfig,
    OrchidIngestionSourceConfig,
    OrchidValidatorConfig,
)
from orchid_ai.core.events.dispatcher import OrchidSignalDispatcher
from orchid_ai.core.events.producer import OrchidSignalProducer
from orchid_ai.core.events.queue import OrchidSignalQueue
from orchid_ai.core.events.store import (
    OrchidJobStore,
    OrchidScheduleRecord,
    OrchidSignalStore,
)
from orchid_ai.events.auth.base import SignalAuthValidator
from orchid_ai.events.processors.asyncio_pool import AsyncioWorkerPoolProcessor
from orchid_ai.events.producers.http import (
    HTTPIngestionProducer,
    SignalSource,
    SignalSourceRegistry,
)
from orchid_ai.events.producers.internal import DispatcherSignalEmitter
from orchid_ai.events.registry import (
    InMemoryTriggerRegistry,
    build_registry_from_config,
)
from orchid_ai.events.streaming import BloomEventStream
from orchid_ai.utils import import_class

_logger = logging.getLogger(__name__)


@dataclass
class EventsRuntime:
    """Bag of live event-system objects, populated when
    ``events.enabled: true``.

    The lifespan stashes one of these on :class:`AppContext` so the
    routers (``signals``, ``jobs``, ``runs``, ``schedules``) and the
    eventual ``OrchidAgent.emit_signal`` plumbing can access them
    without re-resolving config every request.
    """

    enabled: bool = False
    dispatcher: OrchidSignalDispatcher | None = None
    signal_store: OrchidSignalStore | None = None
    signal_queue: OrchidSignalQueue | None = None
    job_store: OrchidJobStore | None = None
    schedule_store: Any | None = None
    trigger_store: Any | None = None
    trigger_registry: InMemoryTriggerRegistry | None = None
    storage: Any | None = None  # SQLiteEventStorage | PostgresEventStorage
    processor: AsyncioWorkerPoolProcessor | None = None
    producers: list[OrchidSignalProducer] = field(default_factory=list)
    http_producer: HTTPIngestionProducer | None = None
    event_stream: BloomEventStream | None = None
    signal_emitter: DispatcherSignalEmitter | None = None


# ── Public lifespan helpers ─────────────────────────────────


async def start_events(
    *,
    events_config: OrchidEventsConfig | None,
    chat_storage: Any,
    identity_resolver: Any,
    session_warmer: Any | None,
    known_agents: set[str],
) -> EventsRuntime:
    """Bring the events block online.

    Returns an :class:`EventsRuntime`.  When the config is absent or
    ``events.enabled`` is False, returns a default
    :class:`EventsRuntime` with ``enabled=False`` and every field
    ``None`` — exactly the zero-overhead path.

    Raises any exception from store / queue / producer construction
    or from the trigger-registry validation (so misconfigured YAML
    surfaces at boot, not first-fire-time).
    """
    if events_config is None or not events_config.enabled:
        return EventsRuntime(enabled=False)

    runtime = EventsRuntime(enabled=True)

    # ── Storage backend ──────────────────────────────────
    storage = await _build_storage(events_config)
    runtime.storage = storage
    runtime.signal_store = storage.signals
    runtime.job_store = storage.jobs
    runtime.schedule_store = storage.schedules
    runtime.trigger_store = storage.triggers

    # ── Queue ────────────────────────────────────────────
    runtime.signal_queue = await _build_queue(events_config, storage=storage)

    # ── Trigger registry (validates §13 / §25 / §26) ─────
    runtime.trigger_registry = build_registry_from_config(
        events_config.triggers,
        known_agents=known_agents,
        identity_resolver=identity_resolver,
    )

    # ── Persist trigger versions for retry-replay safety ─
    # Each boot snapshots the trigger config so a delayed retry can
    # replay against the same definition even if the YAML moved on.
    for cfg in events_config.triggers:
        from orchid_ai.core.events.store import OrchidTriggerRecord

        existing = await runtime.trigger_store.latest(cfg.id)
        version = (existing.version + 1) if existing is not None else 1
        if existing is not None and existing.config == cfg.model_dump(exclude_none=False):
            # Same config bytes — skip reinserting.
            continue
        await runtime.trigger_store.insert_version(
            OrchidTriggerRecord(
                trigger_id=cfg.id,
                version=version,
                config=cfg.model_dump(exclude_none=False),
                created_at=_now(),
                deleted_at=None,
            )
        )

    # ── Schedules → schedule store ────────────────────────
    for sched in events_config.schedules:
        await runtime.schedule_store.upsert(
            OrchidScheduleRecord(
                schedule_id=sched.id,
                trigger_id=sched.trigger_id,
                cron=sched.cron,
                interval_seconds=sched.interval_seconds,
                identity_claim=sched.identity.model_dump(),
                last_fire_at=None,
                next_fire_at=None,
                enabled=sched.enabled,
            )
        )

    # ── Dispatcher + emitter ─────────────────────────────
    runtime.dispatcher = OrchidSignalDispatcher(
        store=runtime.signal_store,
        queue=runtime.signal_queue,
    )
    runtime.signal_emitter = DispatcherSignalEmitter(runtime.dispatcher)

    # ── Bloom event stream ───────────────────────────────
    runtime.event_stream = BloomEventStream()

    # ── Processor pool ───────────────────────────────────
    if events_config.processors:
        # First processor wins — multi-processor support is deferred.
        proc_cfg = events_config.processors[0]
        runtime.processor = AsyncioWorkerPoolProcessor(
            concurrency=proc_cfg.concurrency,
            poll_interval_ms=proc_cfg.poll_interval_ms,
            lease_seconds=proc_cfg.lease_seconds,
            max_attempts=proc_cfg.max_attempts,
            drain_timeout_seconds=proc_cfg.drain_timeout_seconds,
        )
        await runtime.processor.start(
            queue=runtime.signal_queue,
            signal_store=runtime.signal_store,
            triggers=runtime.trigger_registry,
            identity_resolver=identity_resolver,
            job_store=runtime.job_store,
            job_runner=_build_runner(chat_storage=chat_storage),
            session_warmer=session_warmer,
            event_stream=runtime.event_stream,
        )

    # ── Producers (HTTP, scheduler, internal) ────────────
    runtime.producers, runtime.http_producer = await _build_producers(events_config, runtime)

    return runtime


async def stop_events(runtime: EventsRuntime | None) -> None:
    """Tear down the events runtime in reverse order.

    Idempotent — safe to call when ``events.enabled`` was False.
    Stops producers first (so no new signals arrive), then drains
    the processor (which acks any in-flight messages and finishes
    the runs they spawned), then closes the queue's owned pool when
    we own one.
    """
    if runtime is None or not runtime.enabled:
        return

    for producer in runtime.producers:
        try:
            await producer.stop()
        except Exception:
            _logger.exception(
                "events stop: producer %s failed during teardown",
                producer.name,
            )

    if runtime.processor is not None:
        try:
            await runtime.processor.stop()
        except Exception:
            _logger.exception("events stop: processor stop failed")

    if runtime.signal_queue is not None and hasattr(runtime.signal_queue, "close"):
        try:
            await runtime.signal_queue.close()
        except Exception:
            _logger.exception("events stop: queue close failed")

    if runtime.storage is not None and hasattr(runtime.storage, "close"):
        try:
            await runtime.storage.close()
        except Exception:
            _logger.exception("events stop: storage close failed")


# ── Private builders ────────────────────────────────────────


async def _build_storage(cfg: OrchidEventsConfig) -> Any:
    if cfg.store is None:
        raise RuntimeError("events.store is required when events.enabled=true")
    cls = import_class(cfg.store.class_path)
    args = dict(cfg.store.extra_args)
    instance = cls(**args)
    if hasattr(instance, "init_db"):
        await instance.init_db()
    return instance


async def _build_queue(cfg: OrchidEventsConfig, *, storage: Any) -> OrchidSignalQueue:
    if cfg.queue is None:
        raise RuntimeError("events.queue is required when events.enabled=true")
    cls = import_class(cfg.queue.class_path)
    extras = {
        k: v
        for k, v in cfg.queue.model_dump().items()
        if k
        not in {
            "class_path",
            # ``OrchidQueueConfig`` allows extra=allow so unknown knobs
            # flow through; the constructor below will reject those it
            # doesn't accept.
        }
    }
    extras.pop("class_path", None)
    # Pass storage's connection / pool when the constructor accepts
    # it (production wiring shares the connection so the dispatcher's
    # outbox commits both the signal and queue rows atomically).
    if hasattr(storage, "_conn") and storage._conn is not None:
        extras.setdefault("conn", storage._conn)
    elif hasattr(storage, "_pool") and storage._pool is not None:
        extras.setdefault("pool", storage._pool)

    # Only pass kwargs the queue actually accepts to avoid TypeError.
    import inspect

    sig = inspect.signature(cls.__init__)
    accepted = {name: value for name, value in extras.items() if name in sig.parameters}
    instance = cls(**accepted)
    if hasattr(instance, "init_db"):
        await instance.init_db()
    return instance


def _build_runner(*, chat_storage: Any) -> Any:
    """Construct the GraphJobRunner.

    Phase-4 ships an injectable invoker that returns a non-empty
    result.  The full LangGraph wiring (closure over the supervisor)
    lands when the v1 examples (Phase 6) need it; until then the
    runner is wired to a synthetic invoker that surfaces the
    JobSpec fields back so the ``bloom.run.finished`` payload
    carries something visible.  Integrators wanting the real graph
    today set ``orchid.config.events.processors[0].extra_args
    .runner_class`` to a custom dotted path that overrides the
    default — slipped to a future spec update.
    """
    from orchid_ai.events.runners.graph_runner import GraphJobRunner

    async def _invoker(run, auth) -> dict:
        return {
            "final_response": (
                f"[bloom-default-invoker] run={run.run_id} agent={run.spec.agent_name} prompt={run.spec.prompt!r}"
            )
        }

    return GraphJobRunner(invoker=_invoker, chat_storage=chat_storage)


async def _build_producers(
    cfg: OrchidEventsConfig, runtime: EventsRuntime
) -> tuple[list[OrchidSignalProducer], HTTPIngestionProducer | None]:
    """Build every producer referenced in ``events.producers``.

    Returns ``(producers, http_producer)`` so the caller can
    register the HTTP producer's router with FastAPI.
    """
    assert runtime.dispatcher is not None  # set just before this is called
    producers: list[OrchidSignalProducer] = []
    http_producer: HTTPIngestionProducer | None = None

    for ref in cfg.producers:
        cls_path = ref.class_path
        # The HTTP producer is special-cased — it needs the
        # signal_sources registry, which we compile from
        # ``events.ingestion.sources``.
        if cls_path.endswith("HTTPIngestionProducer"):
            registry = SignalSourceRegistry(_compile_sources(cfg.ingestion.sources))
            mount = ref.extra_args.get("mount", "/signals")
            max_body = ref.extra_args.get("max_body_bytes", 1_000_000)
            http_producer = HTTPIngestionProducer(registry=registry, mount=mount, max_body_bytes=max_body)
            await http_producer.start(runtime.dispatcher)
            producers.append(http_producer)
            continue

        # Generic producer — the constructor takes its kwargs from
        # ``extra_args``.  Schedule + relay recovery providers
        # deserve a small wiring helper because they need the
        # store / publisher; integrators with custom producers
        # supply their own kwargs.
        cls = import_class(cls_path)
        kwargs = dict(ref.extra_args)
        if cls_path.endswith("SchedulerProducer"):
            kwargs.setdefault("schedule_store", runtime.schedule_store)
        elif cls_path.endswith("RelayRecoveryProducer"):
            kwargs.setdefault("store", runtime.signal_store)
            # ``publisher`` must be supplied via extra_args — it's
            # integrator-specific (Kafka / Redis / SQS).
        # ``InternalEmissionProducer`` is implicit; the dispatcher
        # emitter is exposed via ``runtime.signal_emitter``.

        instance = cls(**kwargs)
        await instance.start(runtime.dispatcher)
        producers.append(instance)

    return producers, http_producer


def _compile_sources(
    configs: list[OrchidIngestionSourceConfig],
) -> list[SignalSource]:
    """Resolve every dotted-path validator + secret_ref into a live
    :class:`SignalSource`."""
    out: list[SignalSource] = []
    for src in configs:
        validator = _build_validator(src.validator)
        out.append(
            SignalSource(
                source_id=src.id,
                validator=validator,
                allowed_types=frozenset(src.allowed_types),
            )
        )
    return out


def _build_validator(cfg: OrchidValidatorConfig) -> SignalAuthValidator:
    cls = import_class(cfg.class_path)
    kwargs = dict(cfg.extra_args)
    if cfg.secret_ref is not None:
        kwargs["secret"] = _resolve_secret(cfg.secret_ref)
    return cls(**kwargs)


def _resolve_secret(ref: str) -> str:
    """Resolve ``env:VAR`` (or other prefixes) to a concrete secret.

    v1 supports ``env:`` only; consumers needing Vault / AWS Secrets
    Manager point ``OrchidValidatorConfig.class_path`` at a custom
    validator that pulls from their store.
    """
    if ref.startswith("env:"):
        var = ref[len("env:") :]
        value = os.environ.get(var)
        if value is None:
            raise RuntimeError(f"events validator secret_ref env:{var} but {var} is unset")
        return value
    raise RuntimeError(f"unsupported secret_ref scheme {ref!r} — only env: is supported in v1")


def _now() -> Any:
    import datetime as _dt

    return _dt.datetime.now(tz=_dt.UTC)
