"""Tests for the emit pipeline step and its identity-only signal envelope."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from orchid_ai.config.schema_pipeline import OrchidPipelineDefinitionConfig, OrchidPipelineEmitJobConfig
from orchid_ai.core.pipeline import (
    OrchidPipelineEventsDisabledError,
    OrchidPipelineJobContext,
    OrchidPipelineRunContext,
    OrchidPipelineSerializationError,
    OrchidPipelineStepError,
    OrchidPipelineTimeoutError,
)
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.pipeline.builder import PipelineExecution
from orchid_ai.pipeline.definition import CompiledJob, CompiledPipeline, PipelineDependencies, PipelineLimits
from orchid_ai.pipeline.graph import DataGraph
from orchid_ai.pipeline.mapping import build_mapping_context
from orchid_ai.pipeline.registry import resolve_pipeline_step
from orchid_ai.pipeline.runner import OrchidPipelineRunner
from orchid_ai.pipeline.session import PipelineStepContext
from orchid_ai.pipeline.steps.emit import EmitStep

START = datetime(2026, 11, 19, 9, 0, tzinfo=UTC)


class _Emitter:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.envelopes: list[Any] = []

    async def emit(self, envelope: Any) -> Any:
        self.envelopes.append(envelope)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            signal_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
            queue_msg_id="queue-1",
            deduplicated=False,
        )


def _config(**overrides: Any) -> OrchidPipelineEmitJobConfig:
    values: dict[str, Any] = {
        "kind": "emit",
        "signal": "notify",
        "payload": {"message": "{{ input.message }}"},
    }
    values.update(overrides)
    return OrchidPipelineEmitJobConfig.model_validate(values)


def _job(config: OrchidPipelineEmitJobConfig | None = None) -> CompiledJob:
    resolved = config or _config()
    return CompiledJob(id="emit", kind="emit", config=resolved, timeout_seconds=resolved.timeout_seconds)


def _ctx(
    *,
    config: OrchidPipelineEmitJobConfig | None = None,
    emitter: Any = "default",
    input: dict[str, Any] | None = None,
) -> PipelineStepContext:
    values = input or {"message": "hello"}
    run = OrchidPipelineRunContext(
        run_id="run-1",
        pipeline="demo",
        pipeline_version="1",
        mode="direct",
        tenant_key="tenant-1",
        user_id="user-1",
        started_at=START,
    )
    if emitter == "default":
        emitter = _Emitter()
    return PipelineStepContext(
        run=run,
        job=OrchidPipelineJobContext(job_id="emit", kind="emit"),
        input=values,
        graph=DataGraph().view(),
        auth=OrchidAuthContext(access_token="super-secret", tenant_key="tenant-1", user_id="user-1"),
        signal_emitter=emitter,
        mapping_context=build_mapping_context(input=values, run=run),
    )


def _run(job: CompiledJob, ctx: PipelineStepContext) -> Any:
    return asyncio.run(EmitStep(job).execute(ctx))


# ── Envelope shape ─────────────────────────────────────────


def test_emit_builds_the_documented_envelope() -> None:
    emitter = _Emitter()
    config = _config(
        payload={"message": "{{ input.message }}"},
        dedupe_key="{{ run.id }}:{{ input.message }}",
        identity={"mode": "service_account", "name": "digest-bot"},
        chat_binding={"chat_id": "chat-1", "mode": "append_final_message"},
    )
    ctx = _ctx(config=config, emitter=emitter, input={"message": "hello"})

    result = _run(_job(config), ctx)

    assert len(emitter.envelopes) == 1
    envelope = emitter.envelopes[0]
    assert envelope.type == "notify"
    assert envelope.payload == {"message": "hello"}
    assert envelope.source == "pipeline:demo"
    assert envelope.tenant_key == "tenant-1"
    assert envelope.user_id == "user-1"
    assert envelope.correlation_id == "run-1"
    assert envelope.dedupe_key == "run-1:hello"
    assert envelope.identity_claim == {"mode": "service_account", "name": "digest-bot"}
    assert envelope.chat_binding == {
        "chat_id": "chat-1",
        "mode": "append_final_message",
        "on_failure": "post_error",
        "source_message_id": None,
    }
    assert envelope.occurred_at.tzinfo is not None
    assert "super-secret" not in str(envelope)
    assert result.output["signal_id"] == "00000000-0000-0000-0000-000000000001"
    assert result.output["deduplicated"] is False


def test_emit_defaults_to_an_act_as_user_identity() -> None:
    emitter = _Emitter()
    ctx = _ctx(emitter=emitter)

    _run(_job(), ctx)

    assert emitter.envelopes[0].identity_claim == {"mode": "act_as_user", "user_id": "user-1"}


def test_emit_payload_renders_native_values() -> None:
    emitter = _Emitter()
    config = _config(payload={"count": "{{ input.count }}", "items": "{{ input.items }}"})
    ctx = _ctx(config=config, emitter=emitter, input={"count": 3, "items": [1, 2]})

    _run(_job(config), ctx)

    assert emitter.envelopes[0].payload == {"count": 3, "items": [1, 2]}


# ── Failure modes ──────────────────────────────────────────


def test_emit_without_an_emitter_is_events_disabled() -> None:
    with pytest.raises(OrchidPipelineEventsDisabledError):
        _run(_job(), _ctx(emitter=None))


def test_emit_provider_failure_is_sanitized() -> None:
    emitter = _Emitter(error=RuntimeError("secret sk-test"))

    with pytest.raises(OrchidPipelineStepError) as excinfo:
        _run(_job(), _ctx(emitter=emitter))

    message = str(excinfo.value)
    assert "failed to emit signal 'notify'" in message
    assert "secret" not in message


def test_emit_payload_must_be_json_serializable() -> None:
    config = _config(payload={"bad": "{{ input.bad }}"})
    ctx = _ctx(config=config, input={"bad": {1, 2}})

    with pytest.raises(OrchidPipelineSerializationError, match="not JSON-serializable"):
        _run(_job(config), ctx)


def test_emit_payload_must_render_to_an_object() -> None:
    config = OrchidPipelineEmitJobConfig.model_construct(
        kind="emit",
        signal="notify",
        payload=["{{ input.message }}"],
        dedupe_key=None,
        identity=None,
        chat_binding=None,
    )
    ctx = _ctx(config=config)

    with pytest.raises(OrchidPipelineStepError, match="payload must render to an object"):
        _run(_job(config), ctx)


def test_emit_timeout_is_enforced() -> None:
    class _SlowEmitter:
        async def emit(self, envelope: Any) -> Any:
            await asyncio.sleep(0.2)
            return SimpleNamespace(
                signal_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
                queue_msg_id="queue-1",
                deduplicated=False,
            )

    config = _config(timeout_seconds=0.01)

    with pytest.raises(OrchidPipelineTimeoutError, match="timed out after 0.010"):
        _run(_job(config), _ctx(config=config, emitter=_SlowEmitter()))


def test_emit_runs_through_the_pipeline_runner() -> None:
    emitter = _Emitter()
    config = _config(payload={"message": "{{ input.message }}"}, dedupe_key="{{ run.id }}")
    job = _job(config)
    pipeline = CompiledPipeline(
        name="demo",
        version="1",
        description="",
        definition=OrchidPipelineDefinitionConfig(),
        jobs={job.id: job},
        limits=PipelineLimits.from_config(),
    )

    result = asyncio.run(
        OrchidPipelineRunner(pipeline, execution=PipelineExecution(signal_emitter=emitter)).run(
            {"message": "hello"},
            user_id="user-1",
        )
    )

    assert result.status == "succeeded"
    assert result.jobs["emit"]["output"]["signal"] == "notify"
    assert emitter.envelopes[0].payload == {"message": "hello"}
    assert emitter.envelopes[0].dedupe_key == result.run_id


def test_emit_run_deadline_is_rejected_before_the_call() -> None:
    ctx = _ctx()
    now = datetime.now(UTC)
    ctx.run = replace(
        ctx.run,
        started_at=now - timedelta(seconds=10),
        deadline=now - timedelta(seconds=1),
    )

    with pytest.raises(OrchidPipelineTimeoutError, match="deadline has passed"):
        _run(_job(), ctx)


def test_emit_events_disabled_through_the_runner() -> None:
    config = _config()
    job = _job(config)
    pipeline = CompiledPipeline(
        name="demo",
        version="1",
        description="",
        definition=OrchidPipelineDefinitionConfig(),
        jobs={job.id: job},
        limits=PipelineLimits.from_config(),
    )

    result = asyncio.run(OrchidPipelineRunner(pipeline).run({"message": "hello"}, user_id="user-1"))

    assert result.status == "failed"
    assert result.jobs["emit"]["error_code"] == "pipeline_events_disabled"


# ── Registration ───────────────────────────────────────────


def test_emit_kind_is_registered() -> None:
    step = resolve_pipeline_step("emit", _job(), PipelineDependencies())

    assert isinstance(step, EmitStep)
