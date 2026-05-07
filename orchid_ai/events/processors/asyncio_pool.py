"""In-process asyncio worker pool — the default processor.

The pool spawns ``concurrency`` coroutines, each of which:

1. ``dequeue`` a single message (batch_size=1 — keeps the
   per-``parallelism_key`` lock simple).
2. Resolve identity from the signal's claim.
3. Match triggers, build :class:`JobSpec` for each match.
4. For each spec: insert a :class:`JobRun` row, acquire the per-
   ``parallelism_key`` lock, invoke the runner, persist the final
   state.
5. ``ack`` the queue message on success, ``nack`` with backoff on
   transient failure, ``dead_letter`` on terminal failure / max
   attempts.

All of the steps above run on a *separate* path from the producer that
ingested the signal — ingest never blocks on identity resolution or
LangGraph invocation.

Identity resolution is async-or-sync tolerant: this module calls the
resolver behind a small wrapper that ``await``s coroutines and accepts
plain return values, so an in-memory ``FakeIdentityResolver`` (sync)
and a real ``OrchidIdentityResolver`` (async ``resolve``) both work.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import random
import uuid as _uuid
from typing import Any, Awaitable, Callable

from ...core.events.errors import (
    JobRunnerError,
    OrchidIdentityNotMintableError,
    OrchidServiceAccountUnknownError,
)
from ...core.events.job import JobRun, JobStatus, RetryPolicy
from ...core.events.processor import OrchidSignalProcessor
from ...core.events.queue import OrchidSignalQueue, QueuedSignal
from ...core.events.runner import OrchidJobRunner
from ...core.events.signal import Signal
from ...core.events.store import OrchidJobStore, OrchidSignalStore
from ...core.events.trigger import OrchidTrigger, TriggerRegistry
from ..registry import resolve_user_id_for_signal

_logger = logging.getLogger(__name__)

_TerminalStatuses = {
    JobStatus.SUCCEEDED,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
}


class AsyncioWorkerPoolProcessor(OrchidSignalProcessor):
    """Configurable in-process worker pool."""

    def __init__(
        self,
        *,
        concurrency: int = 4,
        poll_interval_ms: int = 200,
        lease_seconds: int = 30,
        max_attempts: int = 5,
        drain_timeout_seconds: float = 10.0,
        clock: Callable[[], _dt.datetime] | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self._concurrency = concurrency
        self._poll_interval = poll_interval_ms / 1000.0
        self._lease_seconds = lease_seconds
        self._max_attempts = max_attempts
        self._drain_timeout = drain_timeout_seconds
        self._clock = clock or _default_clock

        self._stopping = asyncio.Event()
        self._workers: list[asyncio.Task[None]] = []
        self._key_locks: dict[str, asyncio.Lock] = {}
        self._key_locks_guard = asyncio.Lock()

        # Wired in :meth:`start`; set to ``None`` until then so a
        # mistaken call before start raises a clear ``RuntimeError``.
        self._queue: OrchidSignalQueue | None = None
        self._signal_store: OrchidSignalStore | None = None
        self._triggers: TriggerRegistry | None = None
        self._identity_resolver: Any | None = None
        self._job_store: OrchidJobStore | None = None
        self._job_runner: OrchidJobRunner | None = None
        # Optional — the processor warms MCP capability caches for the
        # resolved auth before invoking the runner, mirroring rule #11
        # in the package CLAUDE.md.  When no warmer is supplied (unit
        # tests, demos) the warm step becomes a no-op.
        self._session_warmer: Any | None = None
        # Optional ``BloomEventStream`` — when supplied, the processor
        # publishes ``bloom.run.queued``, ``.started``, and ``.finished``
        # events for every run it advances.  Phase-1 tests work fine
        # without one.
        self._event_stream: Any | None = None

    async def start(
        self,
        *,
        queue: OrchidSignalQueue,
        signal_store: OrchidSignalStore,
        triggers: TriggerRegistry,
        identity_resolver: Any,
        job_store: OrchidJobStore,
        job_runner: OrchidJobRunner,
        session_warmer: Any | None = None,
        event_stream: Any | None = None,
    ) -> None:
        if self._workers:
            raise RuntimeError("processor already started")
        self._queue = queue
        self._signal_store = signal_store
        self._triggers = triggers
        self._identity_resolver = identity_resolver
        self._job_store = job_store
        self._job_runner = job_runner
        self._session_warmer = session_warmer
        self._event_stream = event_stream
        self._stopping.clear()

        for i in range(self._concurrency):
            task = asyncio.create_task(self._worker_loop(i), name=f"orchid-events-worker-{i}")
            self._workers.append(task)

    async def stop(self) -> None:
        self._stopping.set()
        if not self._workers:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._workers, return_exceptions=True),
                timeout=self._drain_timeout,
            )
        except asyncio.TimeoutError:
            for task in self._workers:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self._workers, return_exceptions=True)
        finally:
            self._workers.clear()

    # ── Test-only synchronous draining ────────────────────────

    async def process_until_idle(
        self,
        *,
        queue: OrchidSignalQueue,
        signal_store: OrchidSignalStore,
        triggers: TriggerRegistry,
        identity_resolver: Any,
        job_store: OrchidJobStore,
        job_runner: OrchidJobRunner,
        session_warmer: Any | None = None,
        event_stream: Any | None = None,
        max_iterations: int = 100,
    ) -> int:
        """Drain the queue synchronously by pulling one message at a
        time until ``dequeue`` returns nothing for two consecutive
        rounds OR ``max_iterations`` is hit.  Used by unit tests so we
        don't have to schedule and tear down the worker pool just to
        process a handful of signals.

        Returns the number of messages successfully processed (acked
        OR dead-lettered)."""
        self._queue = queue
        self._signal_store = signal_store
        self._triggers = triggers
        self._identity_resolver = identity_resolver
        self._job_store = job_store
        self._job_runner = job_runner
        self._session_warmer = session_warmer
        self._event_stream = event_stream

        processed = 0
        empties = 0
        for _ in range(max_iterations):
            batch = await queue.dequeue(batch_size=1, lease_seconds=self._lease_seconds)
            if not batch:
                empties += 1
                if empties >= 2:
                    break
                # Visible-after timers are real-time, not virtual.
                await asyncio.sleep(0)
                continue
            empties = 0
            await self._handle_message(batch[0])
            processed += 1
        return processed

    # ── Worker loop ────────────────────────────────────────────

    async def _worker_loop(self, worker_id: int) -> None:
        assert self._queue is not None
        while not self._stopping.is_set():
            try:
                batch = await self._queue.dequeue(batch_size=1, lease_seconds=self._lease_seconds)
            except Exception:
                _logger.exception("worker %d dequeue failed", worker_id)
                await asyncio.sleep(self._poll_interval)
                continue

            if not batch:
                await asyncio.sleep(self._poll_interval)
                continue

            await self._handle_message(batch[0])

    # ── One message ────────────────────────────────────────────

    async def _handle_message(self, msg: QueuedSignal) -> None:
        assert self._queue is not None
        assert self._signal_store is not None
        assert self._triggers is not None
        assert self._job_store is not None
        assert self._job_runner is not None

        signal = await self._signal_store.get(msg.signal_id)
        if signal is None:
            _logger.warning(
                "queued message %s references missing signal %s — acking",
                msg.queue_msg_id,
                msg.signal_id,
            )
            await self._queue.ack(msg.queue_msg_id)
            return

        matches = list(self._triggers.find_matches(signal))
        if not matches:
            await self._queue.ack(msg.queue_msg_id)
            return

        any_failed_terminal = False
        any_failed_retryable = False

        for trigger in matches:
            outcome = await self._run_trigger(trigger=trigger, signal=signal, attempt=msg.attempt)
            if outcome == _TriggerOutcome.SUCCESS:
                continue
            if outcome == _TriggerOutcome.RETRY:
                any_failed_retryable = True
            else:
                any_failed_terminal = True

        if any_failed_retryable:
            backoff = _compute_backoff_seconds(attempt=msg.attempt)
            await self._queue.nack(msg.queue_msg_id, retry_after_seconds=backoff)
            return

        if any_failed_terminal and msg.attempt >= self._max_attempts:
            await self._queue.dead_letter(msg.queue_msg_id, reason="terminal trigger failure")
            return

        # Either everything succeeded, or every failure was terminal but
        # we still have queue attempts left — terminal failures still
        # result in ack because there's nothing useful to retry: a job
        # with status=FAILED is already persisted, and reprocessing it
        # would just produce another identical failure.
        await self._queue.ack(msg.queue_msg_id)

    async def _run_trigger(self, *, trigger: OrchidTrigger, signal: Signal, attempt: int) -> "_TriggerOutcome":
        assert self._job_store is not None
        assert self._job_runner is not None

        try:
            spec = trigger.build_job_spec(signal)
        except Exception:
            _logger.exception(
                "trigger %s failed to build job spec for signal %s",
                trigger.trigger_id,
                signal.signal_id,
            )
            return _TriggerOutcome.TERMINAL_FAILURE

        try:
            auth = await self._materialise_auth(spec.identity_claim, signal=signal)
        except OrchidServiceAccountUnknownError as exc:
            _logger.error(
                "trigger %s identity claim unresolved (unknown service account): %s",
                trigger.trigger_id,
                exc,
            )
            return _TriggerOutcome.TERMINAL_FAILURE
        except OrchidIdentityNotMintableError as exc:
            _logger.error("trigger %s could not mint identity: %s", trigger.trigger_id, exc)
            return _TriggerOutcome.TERMINAL_FAILURE
        except Exception:
            _logger.exception(
                "trigger %s identity resolution raised — treating as retryable",
                trigger.trigger_id,
            )
            return _TriggerOutcome.RETRY

        # Idempotency: a queue redelivery for the same
        # ``(trigger_id, signal_id)`` whose previous attempt is in a
        # terminal status must NOT spawn a new run.  Only ``RETRY_SCHEDULED``
        # warrants a new attempt — that's the explicit retry path.
        previous = await self._job_store.find_latest(trigger_id=spec.trigger_id, signal_id=spec.signal_id)
        if previous is not None and previous.status in _TerminalStatuses:
            return (
                _TriggerOutcome.SUCCESS if previous.status == JobStatus.SUCCEEDED else _TriggerOutcome.TERMINAL_FAILURE
            )

        attempt_number = previous.attempt_number + 1 if previous is not None else 1

        run = JobRun(
            run_id=_uuid.uuid4(),
            spec=spec,
            attempt_number=attempt_number,
            status=JobStatus.PENDING,
            queued_at=self._clock(),
        )
        run = await self._job_store.insert(run)
        await self._publish_event(_make_queued_event(run), run=run)

        run.status = JobStatus.RUNNING
        run.started_at = self._clock()
        await self._job_store.update(run)
        await self._publish_event(_make_started_event(run), run=run)

        # Warm MCP capability caches for the resolved auth before
        # we hand off to the runner (rule #11 in the package's
        # AGENTS.md).  Best-effort: a warm failure is logged but
        # doesn't block the run — the supervisor's tool calls will
        # still issue discovery RPCs lazily, just slower.
        if self._session_warmer is not None:
            try:
                warm = self._session_warmer.warm_for_user(auth)
                if hasattr(warm, "__await__"):
                    await warm
            except Exception:
                _logger.exception(
                    "session warmer failed for trigger %s — proceeding with cold MCP caches",
                    trigger.trigger_id,
                )

        async with await self._lock_for_key(spec.parallelism_key):
            try:
                await self._job_runner.run(run, auth=auth)
            except JobRunnerError as exc:
                _logger.warning("runner raised JobRunnerError for run %s: %s", run.run_id, exc)
                run.status = JobStatus.RETRY_SCHEDULED if exc.retryable else JobStatus.FAILED
                run.error = repr(exc)
                run.finished_at = self._clock()
            except Exception as exc:
                _logger.exception("runner raised unexpected exception for run %s", run.run_id)
                run.status = JobStatus.FAILED
                run.error = repr(exc)
                run.finished_at = self._clock()

            if run.finished_at is None:
                run.finished_at = self._clock()
            await self._job_store.update(run)
            await self._publish_event(_make_finished_event(run), run=run)

        if run.status == JobStatus.SUCCEEDED:
            return _TriggerOutcome.SUCCESS
        if run.status == JobStatus.RETRY_SCHEDULED:
            return _TriggerOutcome.RETRY
        return _TriggerOutcome.TERMINAL_FAILURE

    async def _materialise_auth(self, claim: dict[str, Any], *, signal: Signal) -> Any:
        """Turn an identity claim into an ``OrchidAuthContext`` via the
        configured resolver.  Tolerant of sync-or-async resolver
        methods so tests can plug in a 3-line ``FakeResolver`` without
        wrapping every method in ``async``."""
        assert self._identity_resolver is not None
        mode = claim.get("mode")
        if mode == "service_account":
            return await _maybe_await(self._identity_resolver.resolve_service_account, claim["name"])
        if mode == "addressed_to_user":
            base = await _maybe_await(
                self._identity_resolver.resolve_service_account,
                claim["service_account"],
            )
            user_id = resolve_user_id_for_signal(claim, signal=signal)
            # Tag the auth context so downstream RAG can scope to this
            # user.  The base ``OrchidAuthContext`` exposes a generic
            # ``extra`` dict for exactly this kind of breadcrumb.
            try:
                base.extra["addressed_user_id"] = user_id
            except Exception:
                # Custom auth contexts that don't carry an ``extra``
                # dict — leave them untouched.  The processor still
                # has a usable auth.
                pass
            return base
        if mode == "act_as_user":
            user_id = resolve_user_id_for_signal(claim, signal=signal)
            if user_id is None:
                raise OrchidIdentityNotMintableError(
                    tenant_key=signal.tenant_key,
                    user_id="<missing>",
                )
            return await _maybe_await(
                self._identity_resolver.mint_for_user,
                signal.tenant_key,
                user_id,
            )
        raise ValueError(f"unknown identity claim mode: {mode!r}")

    async def _publish_event(self, event: Any, *, run: JobRun | None = None) -> None:
        """Publish a :class:`BloomEvent` (or anything with the same
        shape) when an event stream is wired.  Best-effort — stream
        publish failures log but never block the run.

        Phase 4.5 §LS6/LS7 — dual-publish branch:

        1. **Run channel** (``f"run:{run_id}"``): authoritative path
           consumed by ``GET /runs/{run_id}/stream``.  Always
           attempted when an event is provided.
        2. **Chat channel** (``f"chat:{chat_id}"``): only when ``run``
           is supplied AND its spec carries a ``chat_binding``.  The
           event is mapped to a :class:`ChatBloomEvent` via
           :func:`_wrap_for_chat`; events that don't apply to chat
           consumers (e.g. ``bloom.signal.ingested``) yield ``None``
           and are silently skipped.

        The two paths are wrapped in **independent** ``try/except``
        blocks so a chat-channel failure never affects the run-channel
        publish (and vice-versa).  The run-channel publish runs first
        so the operator-facing path is preserved even if the chat
        channel is misconfigured.
        """
        if self._event_stream is None or event is None:
            return

        # 1. Run channel — authoritative.
        try:
            await self._event_stream.publish_run(event)
        except Exception:
            _logger.exception(
                "BloomEventStream publish to run channel failed for event %r run_id=%s — continuing",
                getattr(event, "type", type(event).__name__),
                getattr(event, "run_id", None),
            )

        # 2. Chat channel — only for chat-bound runs.
        if run is None:
            return
        binding = run.spec.chat_binding if run.spec is not None else None
        if not binding:
            return

        # Map run-event → chat-event.  A None return means "not
        # applicable on the chat channel" (e.g. signal-ingested
        # noise) — that's a silent skip, not an error.
        from ..streaming import _wrap_for_chat as _wrap

        try:
            chat_event = _wrap(event, run=run, binding=binding)
        except Exception:
            _logger.exception(
                "_wrap_for_chat raised on event %r run_id=%s — skipping chat publish",
                getattr(event, "type", type(event).__name__),
                getattr(event, "run_id", None),
            )
            return
        if chat_event is None:
            return

        try:
            await self._event_stream.publish(f"chat:{binding['chat_id']}", chat_event)
        except Exception:
            _logger.exception(
                "BloomEventStream publish to chat channel failed chat_id=%s run_id=%s — continuing",
                binding.get("chat_id"),
                getattr(event, "run_id", None),
            )

    async def _lock_for_key(self, key: str) -> asyncio.Lock:
        async with self._key_locks_guard:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._key_locks[key] = lock
        return lock


class _TriggerOutcome:
    SUCCESS = "success"
    RETRY = "retry"
    TERMINAL_FAILURE = "terminal"


def _default_clock() -> _dt.datetime:
    return _dt.datetime.now(tz=_dt.UTC)


def _compute_backoff_seconds(*, attempt: int) -> int:
    base = min(60, 2 ** max(0, attempt - 1))
    jitter = random.uniform(0.5, 1.0)
    return max(1, int(base * jitter))


async def _maybe_await(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` and await it iff the return value is awaitable."""
    result = fn(*args, **kwargs)
    if hasattr(result, "__await__"):
        return await result  # type: ignore[no-any-return]
    return result


def _make_queued_event(run: JobRun) -> Any:
    """Build a ``bloom.run.queued`` event lazily so the streaming
    module is not imported when the events block is disabled."""
    from ..streaming import queued_event

    return queued_event(
        run_id=run.run_id,
        trigger_id=run.spec.trigger_id,
        queued_at=run.queued_at,
    )


def _make_started_event(run: JobRun) -> Any:
    from ..streaming import started_event

    return started_event(
        run_id=run.run_id,
        attempt_number=run.attempt_number,
        started_at=run.started_at or _default_clock(),
    )


def _make_finished_event(run: JobRun) -> Any:
    from ..streaming import finished_event

    return finished_event(
        run_id=run.run_id,
        status=run.status.value,
        finished_at=run.finished_at or _default_clock(),
        result=run.result,
        error=run.error,
    )


# Suppress an unused-import lint for ``RetryPolicy`` — it's part of the
# public surface of the module via type annotations on JobRun and we
# import it eagerly to keep the type names available to IDE tooling.
_unused: type[RetryPolicy] = RetryPolicy  # noqa: F841
_unused_awaitable: type[Awaitable] = Awaitable  # noqa: F841
