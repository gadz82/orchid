"""Decision-backed event classification and activation.

The ``events`` point is advisory and opt-in twice (``events.enabled`` and
``events.decision.enabled``). A confident ``process``/``skip``/``dedup`` answer
maps onto the existing worker-pool path; every failure, refusal,
low-confidence answer or invalid response runs the deterministic processors so
no event is lost. Gated signals are acknowledged exactly once and never reach
identity resolution or trigger matching.

No test reaches the network: the decision suites use the stub provider or a
local fake, and ``tests/decision/conftest.py`` makes constructing a real
``httpx.AsyncClient`` fail loudly.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import socket
import uuid as _uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from orchid_ai.config.schema_decision import OrchidDecisionPointConfig
from orchid_ai.config.schema_events import (
    OrchidEventsConfig,
    OrchidEventsDecisionConfig,
    OrchidProcessorConfig,
    OrchidTriggerConfig,
    OrchidTriggerEmitConfig,
    OrchidTriggerMatchConfig,
    ServiceAccountIdentity,
)
from orchid_ai.core.decision import (
    OrchidDecisionAnswer,
    OrchidDecisionError,
    OrchidDecisionPointSpec,
    OrchidDecisionQuestion,
    OrchidDecisionResult,
    OrchidDecisionState,
)
from orchid_ai.core.events.dispatcher import OrchidSignalDispatcher
from orchid_ai.core.events.errors import JobRunnerError
from orchid_ai.core.events.job import JobRun, JobStatus
from orchid_ai.core.events.queue import QueuedSignal
from orchid_ai.core.events.runner import OrchidJobRunner
from orchid_ai.core.events.signal import Signal, SignalEnvelope
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.decision.registry import DECISION_POINT_REGISTRY, get_decision_point, register_decision_point
from orchid_ai.decision.stub import StubDecisionModel
from orchid_ai.events.bootstrap import start_events, stop_events
from orchid_ai.events.decision import (
    DEFAULT_EVENT_METADATA_KEYS,
    EVENT_ACTIONS,
    MAX_CLASS_LABEL_CHARS,
    MAX_CLASSIFICATION_LABELS,
    MAX_ENVELOPE_NAME_CHARS,
    MAX_METADATA_VALUE_CHARS,
    MAX_SUMMARY_CHARS,
    build_event_envelope,
    decide_event_action,
)
from orchid_ai.events.processors.asyncio_pool import AsyncioWorkerPoolProcessor
from orchid_ai.events.queues.inmemory import (
    InMemoryJobStore,
    InMemorySignalQueue,
    InMemorySignalStore,
)
from orchid_ai.events.registry import build_registry_from_config
from orchid_ai.graph.decision_bindings import DecisionBinding

SENTINEL = "SENTINEL-EVENT-PAYLOAD-DO-NOT-LEAK"
PROCESSOR_CLASS = "orchid_ai.events.processors.asyncio_pool.AsyncioWorkerPoolProcessor"


# ── Helpers ─────────────────────────────────────────────────


def _binding(
    provider: Any,
    *,
    threshold: float = 0.75,
    options: dict[str, Any] | None = None,
) -> DecisionBinding:
    return DecisionBinding(
        spec=get_decision_point("events"),
        config=OrchidDecisionPointConfig(
            model="stub",
            confidence_threshold=threshold,
            options=options or {},
        ),
        model=provider,
    )


def _signal(
    *,
    type: str = "demo.event",
    payload: dict[str, Any] | None = None,
    source: str = "test:fixture",
    tenant_key: str = "tenant-123",
    user_id: str | None = "user-abc",
    correlation_id: str | None = "corr-1",
    dedupe_key: str | None = None,
) -> Signal:
    now = _dt.datetime.now(tz=_dt.UTC)
    return Signal(
        type=type,
        payload=payload or {},
        source=source,
        occurred_at=now,
        tenant_key=tenant_key,
        signal_id=_uuid.uuid4(),
        persisted_at=now,
        user_id=user_id,
        correlation_id=correlation_id,
        dedupe_key=dedupe_key,
    )


def _decision_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == "orchid.decision"]


class _RecordingProvider:
    """Return one fixed action (and optional classification) and record state."""

    def __init__(
        self,
        action: str = "process",
        *,
        confidence: float = 0.95,
        classification: str | None = None,
        classification_confidence: float = 0.95,
    ) -> None:
        self.action = action
        self.confidence = confidence
        self.classification = classification
        self.classification_confidence = classification_confidence
        self.states: list[OrchidDecisionState] = []
        self.question_sets: list[list[OrchidDecisionQuestion]] = []

    async def decide(self, state, questions, *, timeout=None):
        self.states.append(state)
        materialised = list(questions)
        self.question_sets.append(materialised)
        answers: dict[str, OrchidDecisionAnswer] = {
            "action": OrchidDecisionAnswer(
                name="action",
                kind="choice",
                value=self.action,
                confidence=self.confidence,
            )
        }
        if self.classification is not None and any(q.name == "classification" for q in materialised):
            answers["classification"] = OrchidDecisionAnswer(
                name="classification",
                kind="choice",
                value=self.classification,
                confidence=self.classification_confidence,
            )
        return OrchidDecisionResult(answers=answers, provider="recorder", model="recorder")


class _ErrorProvider:
    def __init__(self, code: str) -> None:
        self.code = code

    async def decide(self, state, questions, *, timeout=None):
        raise OrchidDecisionError("provider failed", code=self.code, provider="error-provider")


class _RaisingProvider:
    async def decide(self, state, questions, *, timeout=None):
        raise RuntimeError(SENTINEL)


class _TimeoutProvider:
    async def decide(self, state, questions, *, timeout=None):
        raise TimeoutError


class _EmptyProvider:
    async def decide(self, state, questions, *, timeout=None):
        return OrchidDecisionResult(answers={}, provider="empty", model="empty")


class _BooleanProvider:
    async def decide(self, state, questions, *, timeout=None):
        return OrchidDecisionResult(
            answers={
                "action": OrchidDecisionAnswer(
                    name="action",
                    kind="boolean",
                    value=True,
                    confidence=0.95,
                )
            },
            provider="boolean",
            model="boolean",
        )


class _RefusalProvider:
    async def decide(self, state, questions, *, timeout=None):
        return OrchidDecisionResult(
            answers={
                "action": OrchidDecisionAnswer(
                    name="action",
                    kind="choice",
                    value=None,
                    confidence=0.9,
                    refusal="not enough context",
                )
            },
            provider="refusal",
            model="refusal",
        )


class _SequenceProvider:
    """Return one action per call, repeating the last action when exhausted."""

    def __init__(self, actions: list[str]) -> None:
        if not actions:
            raise ValueError("a sequence provider needs at least one action")
        self.actions = actions
        self.calls = 0

    async def decide(self, state, questions, *, timeout=None):
        action = self.actions[min(self.calls, len(self.actions) - 1)]
        self.calls += 1
        return OrchidDecisionResult(
            answers={
                "action": OrchidDecisionAnswer(
                    name="action",
                    kind="choice",
                    value=action,
                    confidence=0.95,
                )
            },
            provider="sequence",
            model="sequence",
        )


class _NoneProvider:
    """A binding whose provider violates the result contract."""

    async def decide(self, state, questions, *, timeout=None):
        return None


class _BadAnswersProvider:
    """A duck-typed result whose ``answers`` is not a mapping."""

    async def decide(self, state, questions, *, timeout=None):
        return SimpleNamespace(provider="bad", answers=[])


class _BrokenBinding:
    """A custom binding that fails before the provider call.

    Exercises the processor's defensive catch-all: ``confidence_threshold``
    raises, so a broken protocol object must not kill the worker or drop the
    event.
    """

    point = "events"
    provider = "stub"

    @property
    def options(self) -> dict[str, Any]:
        return {}

    @property
    def confidence_threshold(self) -> float:
        raise RuntimeError(SENTINEL)


# ── Processor harness ───────────────────────────────────────


class _CountingQueue(InMemorySignalQueue):
    """In-memory queue that records every ack/nack for delivery assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.acks: list[str] = []
        self.nacks: list[str] = []

    async def ack(self, queue_msg_id: str) -> None:
        self.acks.append(queue_msg_id)
        await super().ack(queue_msg_id)

    async def nack(self, queue_msg_id: str, *, retry_after_seconds: int) -> None:
        self.nacks.append(queue_msg_id)
        await super().nack(queue_msg_id, retry_after_seconds=retry_after_seconds)


@dataclass
class _Backend:
    queue: _CountingQueue
    signal_store: InMemorySignalStore
    job_store: InMemoryJobStore
    dispatcher: OrchidSignalDispatcher


def _backend() -> _Backend:
    queue = _CountingQueue()
    signal_store = InMemorySignalStore()
    job_store = InMemoryJobStore()
    return _Backend(
        queue=queue,
        signal_store=signal_store,
        job_store=job_store,
        dispatcher=OrchidSignalDispatcher(store=signal_store, queue=queue),
    )


def _envelope(*, type: str = "demo.event", payload: dict[str, Any] | None = None) -> SignalEnvelope:
    return SignalEnvelope(
        type=type,
        payload=payload or {},
        source="test:fixture",
        occurred_at=_dt.datetime.now(tz=_dt.UTC),
        tenant_key="tenant-123",
        user_id="user-abc",
        correlation_id="corr-1",
    )


class _Resolver:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def resolve_service_account(self, name: str) -> OrchidAuthContext:
        self.calls.append(f"service_account:{name}")
        return OrchidAuthContext(access_token=f"sa:{name}", tenant_key="tenant-123", user_id="")

    def mint_for_user(self, tenant_key: str, user_id: str) -> OrchidAuthContext:
        self.calls.append(f"mint:{tenant_key}:{user_id}")
        return OrchidAuthContext(access_token=f"user:{user_id}", tenant_key=tenant_key, user_id=user_id)


class _RecordingRunner(OrchidJobRunner):
    def __init__(self) -> None:
        self.calls: list[JobRun] = []

    async def run(self, run: JobRun, *, auth: Any) -> None:
        self.calls.append(run)
        run.status = JobStatus.SUCCEEDED
        run.result = {"ok": True}
        run.finished_at = _dt.datetime.now(tz=_dt.UTC)


class _FlakyRunner(OrchidJobRunner):
    """Fail the first ``fail_n`` attempts with a retryable runner error."""

    def __init__(self, *, fail_n: int = 1) -> None:
        self.fail_n = fail_n
        self.calls: list[JobRun] = []

    async def run(self, run: JobRun, *, auth: Any) -> None:
        self.calls.append(run)
        if len(self.calls) <= self.fail_n:
            raise JobRunnerError("transient failure", retryable=True)
        run.status = JobStatus.SUCCEEDED
        run.result = {"ok": True}
        run.finished_at = _dt.datetime.now(tz=_dt.UTC)


def _triggers():
    trigger = OrchidTriggerConfig(
        id="t1",
        on=OrchidTriggerMatchConfig(signal="demo.event"),
        emits=OrchidTriggerEmitConfig(
            agent="notifications",
            prompt_template="hi",
            identity=ServiceAccountIdentity(name="digest-bot"),
        ),
    )
    return build_registry_from_config(
        [trigger],
        known_agents={"notifications"},
        identity_resolver=_Resolver(),
    )


async def _drain(
    backend: _Backend,
    *,
    binding: Any | None,
    runner: _RecordingRunner,
    resolver: _Resolver | None = None,
    max_iterations: int = 10,
) -> tuple[AsyncioWorkerPoolProcessor, int]:
    processor = AsyncioWorkerPoolProcessor()
    processed = await processor.process_until_idle(
        queue=backend.queue,
        signal_store=backend.signal_store,
        triggers=_triggers(),
        identity_resolver=resolver or _Resolver(),
        job_store=backend.job_store,
        job_runner=runner,
        decision_binding=binding,
        max_iterations=max_iterations,
    )
    return processor, processed


# ── Gate: action mapping and observability ──────────────────


@pytest.mark.parametrize("action", ["process", "skip", "dedup"])
async def test_confident_action_is_accepted_with_one_decision_line(
    action: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _RecordingProvider(action)

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_event_action(_binding(provider), _signal())

    assert decision.action == action
    assert decision.source == "decision"
    assert decision.accepted is True
    assert decision.fallback_reason is None
    assert decision.provider == "recorder"
    if action == "process":
        assert decision.should_process is True
        assert decision.gated is False
    else:
        assert decision.gated is True
        assert decision.should_process is False

    lines = _decision_lines(caplog)
    assert len(lines) == 1
    for fragment in (
        "decision point=events",
        "path=decision",
        "provider=recorder",
        "outcome=accepted",
        "fallback_reason=null",
    ):
        assert fragment in lines[0]

    # The provider receives the declared action options, in order.
    assert tuple(provider.question_sets[0][0].options) == EVENT_ACTIONS


async def test_low_confidence_falls_back_to_deterministic_processors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = StubDecisionModel(mode="low_confidence", answers={"action": {"choice": "skip"}})

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_event_action(_binding(provider, threshold=0.8), _signal())

    assert decision.action == "process"
    assert decision.source == "fallback"
    assert decision.fallback_reason == "low_confidence"
    lines = _decision_lines(caplog)
    assert len(lines) == 1
    assert "path=llm_fallback" in lines[0]
    assert "outcome=fallback" in lines[0]
    assert "fallback_reason=low_confidence" in lines[0]


async def test_provider_error_falls_back_to_process(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_event_action(_binding(_ErrorProvider("provider_unavailable")), _signal())

    assert decision.action == "process"
    assert decision.source == "fallback"
    assert decision.fallback_reason == "provider_unavailable"
    assert "fallback_reason=provider_unavailable" in _decision_lines(caplog)[0]


async def test_unexpected_provider_error_logs_only_the_exception_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        caplog.at_level(logging.WARNING, logger="orchid_ai.events.decision"),
        caplog.at_level(logging.INFO, logger="orchid.decision"),
    ):
        decision = await decide_event_action(_binding(_RaisingProvider()), _signal())

    assert decision.action == "process"
    assert decision.fallback_reason == "provider_error"
    assert SENTINEL not in caplog.text
    assert "RuntimeError" in caplog.text
    assert len(_decision_lines(caplog)) == 1


@pytest.mark.parametrize(
    ("provider", "reason"),
    [
        (_TimeoutProvider(), "timeout"),
        (_RefusalProvider(), "refusal"),
        (_EmptyProvider(), "invalid_response"),
        (_BooleanProvider(), "invalid_response"),
        (_RecordingProvider("archive"), "unknown_action"),
    ],
)
async def test_invalid_or_failed_answers_fall_back_to_process(
    provider: Any,
    reason: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_event_action(_binding(provider), _signal())

    assert decision.action == "process"
    assert decision.source == "fallback"
    assert decision.fallback_reason == reason
    lines = _decision_lines(caplog)
    assert len(lines) == 1
    assert f"fallback_reason={reason}" in lines[0]


async def test_disabled_point_does_not_call_a_provider(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        decision = await decide_event_action(None, _signal())

    assert decision.action == "process"
    assert decision.source == "disabled"
    assert decision.should_process is True
    assert decision.provider == ""
    assert _decision_lines(caplog) == []


@pytest.mark.parametrize("provider", [_NoneProvider(), _BadAnswersProvider()])
async def test_malformed_provider_result_emits_a_fallback_line(
    provider: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_event_action(_binding(provider), _signal())

    assert decision.action == "process"
    assert decision.source == "fallback"
    assert decision.fallback_reason == "invalid_response"
    lines = _decision_lines(caplog)
    assert len(lines) == 1
    assert "path=llm_fallback" in lines[0]
    assert "outcome=fallback" in lines[0]
    assert "fallback_reason=invalid_response" in lines[0]


async def test_structured_line_uses_the_binding_point(caplog: pytest.LogCaptureFixture) -> None:
    spec = OrchidDecisionPointSpec(
        name="events_custom",
        scope="fleet",
        description="consumer-registered event gate for the test",
    )
    register_decision_point(spec)
    try:
        accepted_binding = DecisionBinding(
            spec=spec,
            config=OrchidDecisionPointConfig(model="stub", confidence_threshold=0.75, options={}),
            model=_RecordingProvider("process"),
        )
        fallback_binding = DecisionBinding(
            spec=spec,
            config=OrchidDecisionPointConfig(model="stub", confidence_threshold=0.75, options={}),
            model=_ErrorProvider("provider_unavailable"),
        )

        with caplog.at_level(logging.INFO, logger="orchid.decision"):
            accepted = await decide_event_action(accepted_binding, _signal())
            fallback = await decide_event_action(fallback_binding, _signal())
    finally:
        DECISION_POINT_REGISTRY.pop("events_custom", None)

    assert accepted.accepted is True
    assert fallback.fallback_reason == "provider_unavailable"
    lines = _decision_lines(caplog)
    assert len(lines) == 2
    assert "point=events_custom" in lines[0]
    assert "outcome=accepted" in lines[0]
    assert "point=events_custom" in lines[1]
    assert "outcome=fallback" in lines[1]
    assert "point=events " not in " ".join(lines)


# ── Gate: optional classification ───────────────────────────


async def test_classification_is_accepted_when_configured_and_confident() -> None:
    provider = _RecordingProvider("skip", classification="urgent")

    decision = await decide_event_action(
        _binding(provider, options={"classes": ["routine", "urgent"]}),
        _signal(),
    )

    assert decision.action == "skip"
    assert decision.classification == "urgent"
    classification_question = provider.question_sets[0][1]
    assert classification_question.name == "classification"
    assert classification_question.required is False
    assert tuple(classification_question.options) == ("routine", "urgent")


async def test_unknown_or_low_confidence_classification_is_dropped() -> None:
    unknown = _RecordingProvider("process", classification="other")
    low = _RecordingProvider("process", classification="urgent", classification_confidence=0.1)

    unknown_decision = await decide_event_action(
        _binding(unknown, options={"classes": ["routine", "urgent"]}),
        _signal(),
    )
    low_decision = await decide_event_action(
        _binding(low, options={"classes": ["routine", "urgent"]}),
        _signal(),
    )

    assert unknown_decision.action == "process"
    assert unknown_decision.accepted is True
    assert unknown_decision.classification is None
    assert low_decision.action == "process"
    assert low_decision.accepted is True
    assert low_decision.classification is None


async def test_invalid_classes_option_disables_classification_only() -> None:
    provider = _RecordingProvider("process", classification="urgent")

    decision = await decide_event_action(_binding(provider, options={"classes": ["only-one"]}), _signal())

    assert decision.action == "process"
    assert decision.classification is None
    assert [q.name for q in provider.question_sets[0]] == ["action"]


async def test_class_labels_are_sanitised_and_deduplicated() -> None:
    provider = _RecordingProvider("process", classification="urgent")

    decision = await decide_event_action(
        _binding(provider, options={"classes": ["routine", "urgent", "urgent", 7, "   "]}),
        _signal(),
    )
    raw_string = _RecordingProvider("process", classification="routine")
    string_decision = await decide_event_action(_binding(raw_string, options={"classes": "routine"}), _signal())

    assert decision.classification == "urgent"
    assert tuple(provider.question_sets[0][1].options) == ("routine", "urgent")
    assert string_decision.classification is None
    assert [q.name for q in raw_string.question_sets[0]] == ["action"]


async def test_class_labels_are_truncated_and_capped() -> None:
    long_label = "a" * (MAX_CLASS_LABEL_CHARS + 50)
    labels = [long_label, *[f"label-{index:02d}" for index in range(30)]]
    provider = _RecordingProvider("process", classification="a" * MAX_CLASS_LABEL_CHARS)

    decision = await decide_event_action(_binding(provider, options={"classes": labels}), _signal())

    question = provider.question_sets[0][1]
    assert len(question.options) == MAX_CLASSIFICATION_LABELS
    assert question.options[0] == "a" * MAX_CLASS_LABEL_CHARS
    assert decision.classification == "a" * MAX_CLASS_LABEL_CHARS


# ── Gate: bounded envelope ──────────────────────────────────


def test_envelope_is_bounded_and_never_carries_payload_values() -> None:
    signal = _signal(
        type="demo.event\nwith-control-chars-" + "t" * 1000,
        source="test:fixture-" + "s" * 1000,
        payload={"secret": SENTINEL, "nested": {"token": SENTINEL}},
        dedupe_key=SENTINEL,
    )

    envelope = build_event_envelope(signal)
    state = envelope.to_state()
    serialised = json.dumps(state.payload) + state.text

    assert SENTINEL not in serialised
    assert "secret" not in serialised
    assert "nested" not in serialised
    assert "payload" not in state.payload
    assert len(envelope.type) <= MAX_ENVELOPE_NAME_CHARS
    assert len(envelope.source) <= MAX_ENVELOPE_NAME_CHARS
    assert len(envelope.summary) <= MAX_SUMMARY_CHARS
    assert "\n" not in envelope.type
    assert set(envelope.metadata).issubset(DEFAULT_EVENT_METADATA_KEYS)
    # The dedupe key itself is not sent; only its presence.
    assert envelope.metadata["dedupe_key_present"] is True
    assert all(len(str(value)) <= MAX_METADATA_VALUE_CHARS for value in envelope.metadata.values())


def test_custom_summary_is_bounded_and_collapsed() -> None:
    hostile = "line1\nline2\x00\x9b\u2028\u2029line3" + "x" * 1000

    envelope = build_event_envelope(_signal(), summary=hostile)
    state = envelope.to_state()

    assert len(envelope.summary) <= MAX_SUMMARY_CHARS
    assert state.text == envelope.summary
    for marker in ("\n", "\x00", "\x9b", "\u2028", "\u2029"):
        assert marker not in envelope.summary
    assert envelope.summary.startswith("line1 line2 line3")


def test_envelope_metadata_keys_are_allowlisted() -> None:
    signal = _signal(payload={"secret": SENTINEL})

    envelope = build_event_envelope(signal, metadata_keys=["payload", "tenant_key", "not_a_field"])
    missing = build_event_envelope(_signal(user_id=None), metadata_keys=["user_id"])

    assert dict(envelope.metadata) == {"tenant_key": "tenant-123"}
    assert SENTINEL not in json.dumps(envelope.to_state().payload)
    assert dict(missing.metadata) == {}


async def test_provider_state_does_not_include_the_event_payload() -> None:
    provider = _RecordingProvider("process")
    signal = _signal(payload={"secret": SENTINEL})

    await decide_event_action(_binding(provider), signal)

    state = provider.states[0]
    assert SENTINEL not in json.dumps(state.payload) + state.text
    assert state.payload["signal_id"] == str(signal.signal_id)
    assert state.payload["type"] == "demo.event"
    assert state.payload["source"] == "test:fixture"


# ── Processor: delivery and ack semantics ───────────────────


async def test_processor_process_runs_the_trigger_and_acks_once() -> None:
    backend = _backend()
    runner = _RecordingRunner()
    provider = _RecordingProvider("process")
    await backend.dispatcher.ingest(_envelope(payload={"k": "v"}))

    _processor, processed = await _drain(backend, binding=_binding(provider), runner=runner)

    assert processed == 1
    assert len(runner.calls) == 1
    runs = list(await backend.job_store.list())
    assert len(runs) == 1
    assert runs[0].status == JobStatus.SUCCEEDED
    assert len(backend.queue.acks) == 1
    assert backend.queue.nacks == []
    assert backend.queue.visible_messages == 0
    assert backend.queue.in_flight == 0


@pytest.mark.parametrize("action", ["skip", "dedup"])
async def test_processor_gated_action_acks_once_and_runs_nothing(
    action: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    backend = _backend()
    runner = _RecordingRunner()
    resolver = _Resolver()
    provider = _RecordingProvider(action)
    await backend.dispatcher.ingest(_envelope(payload={"secret": SENTINEL}))

    with caplog.at_level(logging.INFO, logger="orchid_ai.events.processors.asyncio_pool"):
        _processor, processed = await _drain(
            backend,
            binding=_binding(provider),
            runner=runner,
            resolver=resolver,
        )

    assert processed == 1
    assert runner.calls == []
    assert list(await backend.job_store.list()) == []
    assert len(backend.queue.acks) == 1
    assert backend.queue.nacks == []
    assert backend.queue.visible_messages == 0
    assert backend.queue.in_flight == 0
    # Identity resolution and trigger matching never ran.
    assert resolver.calls == []
    assert f"action={action}" in caplog.text
    assert SENTINEL not in caplog.text


async def test_gated_message_is_not_redelivered_or_acked_twice() -> None:
    backend = _backend()
    runner = _RecordingRunner()
    await backend.dispatcher.ingest(_envelope())

    first_processor, first = await _drain(
        backend,
        binding=_binding(_RecordingProvider("skip")),
        runner=runner,
    )
    _second_processor, second = await _drain(
        backend,
        binding=_binding(_RecordingProvider("skip")),
        runner=runner,
    )

    assert first == 1
    assert second == 0
    assert len(backend.queue.acks) == 1
    assert runner.calls == []
    assert first_processor.decision_binding is not None


async def test_processor_without_binding_runs_the_deterministic_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    backend = _backend()
    runner = _RecordingRunner()
    await backend.dispatcher.ingest(_envelope())

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        _processor, processed = await _drain(backend, binding=None, runner=runner)

    assert processed == 1
    assert len(runner.calls) == 1
    assert len(backend.queue.acks) == 1
    assert _decision_lines(caplog) == []


async def test_processor_provider_error_still_processes_the_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    backend = _backend()
    runner = _RecordingRunner()
    await backend.dispatcher.ingest(_envelope())

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        _processor, processed = await _drain(
            backend,
            binding=_binding(_ErrorProvider("provider_unavailable")),
            runner=runner,
        )

    assert processed == 1
    assert len(runner.calls) == 1
    assert len(backend.queue.acks) == 1
    assert backend.queue.nacks == []
    lines = _decision_lines(caplog)
    assert len(lines) == 1
    assert "outcome=fallback" in lines[0]
    assert "fallback_reason=provider_unavailable" in lines[0]


async def test_processor_low_confidence_still_processes_the_event() -> None:
    backend = _backend()
    runner = _RecordingRunner()
    await backend.dispatcher.ingest(_envelope())

    provider = StubDecisionModel(mode="low_confidence", answers={"action": {"choice": "skip"}})
    _processor, processed = await _drain(backend, binding=_binding(provider, threshold=0.8), runner=runner)

    assert processed == 1
    assert len(runner.calls) == 1
    assert len(backend.queue.acks) == 1


async def test_retry_bypasses_the_gate_when_the_provider_flips_to_skip() -> None:
    backend = _backend()
    runner = _FlakyRunner(fail_n=1)
    provider = _SequenceProvider(["process", "skip"])
    processor = AsyncioWorkerPoolProcessor()

    # Wire the processor without starting worker tasks; the queue is empty so
    # this only assigns the collaborators used by the direct deliveries below.
    wired = await processor.process_until_idle(
        queue=backend.queue,
        signal_store=backend.signal_store,
        triggers=_triggers(),
        identity_resolver=_Resolver(),
        job_store=backend.job_store,
        job_runner=runner,
        decision_binding=_binding(provider),
    )
    assert wired == 0

    await backend.dispatcher.ingest(_envelope())
    first = (await backend.queue.dequeue(batch_size=1, lease_seconds=30))[0]
    assert first.attempt == 1
    await processor._handle_message(first)

    assert provider.calls == 1
    assert len(runner.calls) == 1
    assert len(backend.queue.nacks) == 1
    assert backend.queue.acks == []

    retry = QueuedSignal(
        queue_msg_id=first.queue_msg_id,
        signal_id=first.signal_id,
        enqueued_at=first.enqueued_at,
        lease_until=first.lease_until,
        attempt=2,
    )
    await processor._handle_message(retry)

    # The gate is not re-evaluated on redelivery: the provider's second answer
    # (skip) must not suppress the retry.
    assert provider.calls == 1
    assert len(runner.calls) == 2
    assert len(backend.queue.acks) == 1
    runs = {run.attempt_number: run for run in await backend.job_store.list()}
    assert sorted(runs) == [1, 2]
    assert runs[1].status == JobStatus.RETRY_SCHEDULED
    assert runs[2].status == JobStatus.SUCCEEDED


async def test_processor_catch_all_processes_when_the_binding_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    backend = _backend()
    runner = _RecordingRunner()
    await backend.dispatcher.ingest(_envelope(payload={"secret": SENTINEL}))

    with caplog.at_level(logging.WARNING, logger="orchid_ai.events.processors.asyncio_pool"):
        _processor, processed = await _drain(backend, binding=_BrokenBinding(), runner=runner)

    assert processed == 1
    assert len(runner.calls) == 1
    assert len(backend.queue.acks) == 1
    assert "RuntimeError" in caplog.text
    assert SENTINEL not in caplog.text


async def test_processor_gate_does_not_reach_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend()
    runner = _RecordingRunner()
    await backend.dispatcher.ingest(_envelope())

    def _network_attempt(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the event decision path must not open a network connection")

    monkeypatch.setattr(socket, "getaddrinfo", _network_attempt)
    monkeypatch.setattr(socket, "create_connection", _network_attempt)

    provider = StubDecisionModel(answers={"action": {"choice": "process", "confidence": 0.99}})
    _processor, processed = await _drain(backend, binding=_binding(provider), runner=runner)

    assert processed == 1
    assert len(runner.calls) == 1


# ── Config: additive, disabled by default, validated ────────


def test_events_decision_defaults_to_disabled() -> None:
    config = OrchidEventsConfig()

    assert config.decision.enabled is False
    assert config.decision.point == "events"


def test_events_decision_parses_when_events_are_disabled() -> None:
    config = OrchidEventsConfig.model_validate({"enabled": False, "decision": {"enabled": True, "point": "events"}})

    assert config.enabled is False
    assert config.decision.enabled is True


def test_events_decision_rejects_unknown_point() -> None:
    with pytest.raises(ValidationError, match="unknown decision point"):
        OrchidEventsConfig.model_validate({"decision": {"enabled": True, "point": "not_a_point"}})


def test_events_decision_rejects_agent_scoped_point() -> None:
    with pytest.raises(ValidationError, match="fleet-scoped"):
        OrchidEventsConfig.model_validate({"decision": {"enabled": True, "point": "tools"}})


def test_events_decision_rejects_empty_point() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        OrchidEventsDecisionConfig.model_validate({"point": "   "})


def test_events_decision_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        OrchidEventsDecisionConfig.model_validate({"enabled": True, "confidence": 0.9})


# ── Bootstrap: binding injection ────────────────────────────


def _events_config(*, enabled: bool = True, decision_enabled: bool = True, point: str = "events") -> OrchidEventsConfig:
    return OrchidEventsConfig(
        enabled=enabled,
        processors=[OrchidProcessorConfig.model_validate({"class": PROCESSOR_CLASS})],
        decision={"enabled": decision_enabled, "point": point},
    )


async def test_start_events_disabled_ignores_the_decision_binding() -> None:
    binding = _binding(StubDecisionModel(answers={"action": {"choice": "skip"}}))

    runtime = await start_events(
        events_config=None,
        chat_storage=None,
        identity_resolver=_Resolver(),
        session_warmer=None,
        known_agents=set(),
        decision_binding=binding,
    )
    disabled = await start_events(
        events_config=_events_config(enabled=False),
        chat_storage=None,
        identity_resolver=_Resolver(),
        session_warmer=None,
        known_agents=set(),
        decision_binding=binding,
    )

    assert runtime.enabled is False
    assert runtime.processor is None
    assert runtime.decision_binding is None
    assert disabled.enabled is False
    assert disabled.processor is None
    assert disabled.decision_binding is None


async def test_start_events_threads_the_binding_when_both_switches_are_enabled() -> None:
    binding = _binding(StubDecisionModel(answers={"action": {"choice": "skip"}}))
    runtime = await start_events(
        events_config=_events_config(),
        chat_storage=None,
        identity_resolver=_Resolver(),
        session_warmer=None,
        known_agents=set(),
        decision_binding=binding,
    )
    try:
        assert runtime.enabled is True
        assert runtime.decision_binding is binding
        assert runtime.processor is not None
        assert runtime.processor.decision_binding is binding
    finally:
        await stop_events(runtime)


async def test_start_events_ignores_the_binding_when_the_point_is_disabled() -> None:
    binding = _binding(StubDecisionModel(answers={"action": {"choice": "skip"}}))
    runtime = await start_events(
        events_config=_events_config(decision_enabled=False),
        chat_storage=None,
        identity_resolver=_Resolver(),
        session_warmer=None,
        known_agents=set(),
        decision_binding=binding,
    )
    try:
        assert runtime.decision_binding is None
        assert runtime.processor is not None
        assert runtime.processor.decision_binding is None
    finally:
        await stop_events(runtime)


async def test_start_events_warns_when_decision_is_enabled_without_a_binding(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="orchid_ai.events.bootstrap"):
        runtime = await start_events(
            events_config=_events_config(),
            chat_storage=None,
            identity_resolver=_Resolver(),
            session_warmer=None,
            known_agents=set(),
            decision_binding=None,
        )
    try:
        assert runtime.decision_binding is None
        assert runtime.processor is not None
        assert runtime.processor.decision_binding is None
        assert "no decision binding" in caplog.text
    finally:
        await stop_events(runtime)


async def test_start_events_rejects_a_binding_for_the_wrong_point(
    caplog: pytest.LogCaptureFixture,
) -> None:
    wrong_point = DecisionBinding(
        spec=get_decision_point("memory"),
        config=OrchidDecisionPointConfig(model="stub", confidence_threshold=0.75, options={}),
        model=StubDecisionModel(answers={"action": {"choice": "keep"}}),
    )
    with caplog.at_level(logging.WARNING, logger="orchid_ai.events.bootstrap"):
        runtime = await start_events(
            events_config=_events_config(),
            chat_storage=None,
            identity_resolver=_Resolver(),
            session_warmer=None,
            known_agents=set(),
            decision_binding=wrong_point,
        )
    try:
        assert runtime.decision_binding is None
        assert "does not match" in caplog.text or "is for point" in caplog.text
    finally:
        await stop_events(runtime)
