"""Asyncio worker pool — end-to-end smoke + idempotency + retry / DLQ."""

from __future__ import annotations

import datetime as _dt
from typing import Any


from orchid_ai.config.schema_events import (
    ActAsUserIdentity,
    OrchidTriggerConfig,
    OrchidTriggerEmitConfig,
    OrchidTriggerMatchConfig,
    ServiceAccountIdentity,
)
from orchid_ai.core.events.errors import JobRunnerError
from orchid_ai.core.events.job import JobRun, JobStatus
from orchid_ai.core.events.runner import OrchidJobRunner
from orchid_ai.events.processors.asyncio_pool import AsyncioWorkerPoolProcessor
from orchid_ai.events.registry import build_registry_from_config


# ── Tiny fake runners ───────────────────────────────────────


class _RecordingRunner(OrchidJobRunner):
    def __init__(self) -> None:
        self.calls: list[tuple[JobRun, Any]] = []

    async def run(self, run: JobRun, *, auth: Any) -> None:
        self.calls.append((run, auth))
        run.result = {"ok": True, "auth_user": auth.user_id, "agent": run.spec.agent_name}
        run.status = JobStatus.SUCCEEDED
        run.finished_at = _dt.datetime.now(tz=_dt.UTC)


class _FailingRunner(OrchidJobRunner):
    def __init__(self, *, retryable: bool, fail_n: int = 999) -> None:
        self.retryable = retryable
        self.calls = 0
        self.fail_n = fail_n

    async def run(self, run: JobRun, *, auth: Any) -> None:
        self.calls += 1
        if self.calls > self.fail_n:
            run.result = {"ok": True}
            run.status = JobStatus.SUCCEEDED
            run.finished_at = _dt.datetime.now(tz=_dt.UTC)
            return
        raise JobRunnerError("fail", retryable=self.retryable)


def _trigger(
    *, id: str = "t1", agent: str = "notifications", when: str | None = None, identity=None
) -> OrchidTriggerConfig:
    return OrchidTriggerConfig(
        id=id,
        on=OrchidTriggerMatchConfig(signal="demo.event", when=when),
        emits=OrchidTriggerEmitConfig(
            agent=agent,
            prompt_template="hi",
            identity=identity or ServiceAccountIdentity(name="digest-bot"),
        ),
    )


# ── Smoke: ingest → match → run → ack ───────────────────────


async def test_processor_runs_a_matching_trigger_end_to_end(backend, make_envelope, fake_resolver) -> None:
    triggers = build_registry_from_config([_trigger()], known_agents={"notifications"}, identity_resolver=fake_resolver)
    runner = _RecordingRunner()
    processor = AsyncioWorkerPoolProcessor()

    env = make_envelope(type="demo.event", payload={"k": "v"})
    await backend.dispatcher.ingest(env)

    processed = await processor.process_until_idle(
        queue=backend.queue,
        signal_store=backend.signal_store,
        triggers=triggers,
        identity_resolver=fake_resolver,
        job_store=backend.job_store,
        job_runner=runner,
    )
    assert processed == 1

    runs = await backend.job_store.list()
    assert len(runs) == 1
    run = runs[0]
    assert run.status == JobStatus.SUCCEEDED
    assert run.attempt_number == 1
    assert run.spec.trigger_id == "t1"
    assert run.spec.parallelism_key.startswith("sa:")
    assert run.result == {"ok": True, "auth_user": "", "agent": "notifications"}
    # Queue empty + nothing dead-lettered.
    assert backend.queue.visible_messages == 0
    assert backend.queue.in_flight == 0
    assert not backend.queue.dead_letters


async def test_processor_acks_when_no_trigger_matches(backend, make_envelope, fake_resolver) -> None:
    triggers = build_registry_from_config(
        [_trigger(when="payload.priority == 'high'")],
        known_agents={"notifications"},
        identity_resolver=fake_resolver,
    )
    runner = _RecordingRunner()
    processor = AsyncioWorkerPoolProcessor()
    await backend.dispatcher.ingest(make_envelope(payload={"priority": "low"}))

    await processor.process_until_idle(
        queue=backend.queue,
        signal_store=backend.signal_store,
        triggers=triggers,
        identity_resolver=fake_resolver,
        job_store=backend.job_store,
        job_runner=runner,
    )
    assert runner.calls == []
    assert backend.queue.visible_messages == 0
    assert backend.queue.in_flight == 0


# ── Idempotency: queue redelivery does not double-fire ──────


async def test_processor_idempotent_on_redelivery(backend, make_envelope, fake_resolver) -> None:
    triggers = build_registry_from_config([_trigger()], known_agents={"notifications"}, identity_resolver=fake_resolver)
    runner = _RecordingRunner()
    processor = AsyncioWorkerPoolProcessor()

    result = await backend.dispatcher.ingest(make_envelope())
    # Drain once.
    await processor.process_until_idle(
        queue=backend.queue,
        signal_store=backend.signal_store,
        triggers=triggers,
        identity_resolver=fake_resolver,
        job_store=backend.job_store,
        job_runner=runner,
    )
    runs_after_first = await backend.job_store.list()
    assert len(runs_after_first) == 1

    # Simulate redelivery — re-enqueue the same signal id and drain.
    await backend.queue.enqueue(result.signal_id)
    await processor.process_until_idle(
        queue=backend.queue,
        signal_store=backend.signal_store,
        triggers=triggers,
        identity_resolver=fake_resolver,
        job_store=backend.job_store,
        job_runner=runner,
    )
    runs_after_second = await backend.job_store.list()
    # Phase 1: in-memory store dedupes by (trigger_id, signal_id,
    # attempt_number=1) — no new run row appears.
    assert len(runs_after_second) == 1


# ── Retry / dead-letter ─────────────────────────────────────


async def test_retryable_failure_nacks_with_backoff(backend, make_envelope, fake_resolver) -> None:
    triggers = build_registry_from_config([_trigger()], known_agents={"notifications"}, identity_resolver=fake_resolver)
    runner = _FailingRunner(retryable=True)
    processor = AsyncioWorkerPoolProcessor()

    await backend.dispatcher.ingest(make_envelope())
    await processor.process_until_idle(
        queue=backend.queue,
        signal_store=backend.signal_store,
        triggers=triggers,
        identity_resolver=fake_resolver,
        job_store=backend.job_store,
        job_runner=runner,
    )
    # Queue: nacked back into in-memory queue with backoff > 0,
    # so it shows up in ``in_flight`` (not visible) until visibility.
    assert not backend.queue.dead_letters
    runs = await backend.job_store.list()
    assert runs and runs[0].status == JobStatus.RETRY_SCHEDULED


async def test_terminal_failure_persists_failed_run(backend, make_envelope, fake_resolver) -> None:
    triggers = build_registry_from_config([_trigger()], known_agents={"notifications"}, identity_resolver=fake_resolver)
    runner = _FailingRunner(retryable=False)
    processor = AsyncioWorkerPoolProcessor()

    await backend.dispatcher.ingest(make_envelope())
    await processor.process_until_idle(
        queue=backend.queue,
        signal_store=backend.signal_store,
        triggers=triggers,
        identity_resolver=fake_resolver,
        job_store=backend.job_store,
        job_runner=runner,
    )
    runs = await backend.job_store.list()
    assert len(runs) == 1
    assert runs[0].status == JobStatus.FAILED
    # Acked — terminal failure produces no retry, so the queue entry
    # is gone.
    assert backend.queue.visible_messages == 0
    assert backend.queue.in_flight == 0


# ── Identity resolution ─────────────────────────────────────


async def test_processor_resolves_act_as_user_identity(backend, make_envelope, fake_resolver) -> None:
    triggers = build_registry_from_config(
        [
            _trigger(
                identity=ActAsUserIdentity(user_id_from="signal.user_id"),
            )
        ],
        known_agents={"notifications"},
        identity_resolver=fake_resolver,
    )
    runner = _RecordingRunner()
    processor = AsyncioWorkerPoolProcessor()

    env = make_envelope(user_id="alice")
    await backend.dispatcher.ingest(env)
    await processor.process_until_idle(
        queue=backend.queue,
        signal_store=backend.signal_store,
        triggers=triggers,
        identity_resolver=fake_resolver,
        job_store=backend.job_store,
        job_runner=runner,
    )
    assert runner.calls
    _run, auth = runner.calls[0]
    assert auth.user_id == "alice"
    # mint_for_user was called with the right (tenant, user) pair.
    minted = [c for c in fake_resolver.calls if c[0] == "mint_for_user"]
    assert any(call[1] == ("tenant-123", "alice") for call in minted)


async def test_processor_unknown_service_account_terminates(backend, make_envelope) -> None:
    from tests.events.conftest import FakeIdentityResolver

    resolver = FakeIdentityResolver(known_service_accounts=set())
    triggers = build_registry_from_config(
        [_trigger(identity=ServiceAccountIdentity(name="ghost"))],
        known_agents={"notifications"},
        identity_resolver=resolver,
    )
    runner = _RecordingRunner()
    processor = AsyncioWorkerPoolProcessor()

    await backend.dispatcher.ingest(make_envelope())
    await processor.process_until_idle(
        queue=backend.queue,
        signal_store=backend.signal_store,
        triggers=triggers,
        identity_resolver=resolver,
        job_store=backend.job_store,
        job_runner=runner,
    )
    # Runner never invoked, no JobRun row inserted (we bailed before
    # the insert), queue acked.
    assert runner.calls == []
    runs = await backend.job_store.list()
    assert runs == []
    assert backend.queue.visible_messages == 0
    assert backend.queue.in_flight == 0
