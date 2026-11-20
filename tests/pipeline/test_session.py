"""Tests for the pipeline run session (``orchid_ai.pipeline.session``)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from orchid_ai.config.schema_pipeline import (
    OrchidPipelineDefinitionConfig,
    OrchidPipelineDeterministicJobConfig,
    OrchidPipelineShapeConfig,
)
from orchid_ai.core.pipeline import (
    OrchidGraphMutation,
    OrchidGraphRelation,
    OrchidPipelineConfigError,
    OrchidPipelineEventsDisabledError,
    OrchidPipelineGraphConflictError,
    OrchidPipelineJobContext,
    OrchidPipelineLimitError,
    OrchidPipelineObserver,
    OrchidPipelineRunContext,
    OrchidPipelineShapeError,
    OrchidPipelineTimeoutError,
    OrchidPipelineWriteNotAllowedError,
)
from orchid_ai.pipeline.definition import CompiledJob, CompiledPipeline, PipelineLimits
from orchid_ai.pipeline.graph import DataGraph
from orchid_ai.pipeline.schema import compile_shapes
from orchid_ai.pipeline.session import PipelineJobResult, PipelineRunSession, PipelineStepContext

START = datetime(2026, 11, 17, 9, 0, tzinfo=UTC)


class _Clock:
    def __init__(self, start: datetime = START) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


class _ListObserver(OrchidPipelineObserver):
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event: Any) -> None:
        self.events.append(event)


class _BrokenObserver(OrchidPipelineObserver):
    def __init__(self) -> None:
        self.calls = 0

    async def emit(self, event: Any) -> None:
        self.calls += 1
        raise RuntimeError("observer down")


def _run(**overrides: Any) -> OrchidPipelineRunContext:
    values: dict[str, Any] = {
        "run_id": "run-1",
        "pipeline": "demo",
        "pipeline_version": "1",
        "mode": "direct",
        "tenant_key": "tenant",
        "user_id": "user",
        "started_at": START,
    }
    values.update(overrides)
    return OrchidPipelineRunContext(**values)


def _job(
    job_id: str = "run",
    *,
    writes: list[dict[str, Any]] | None = None,
    links: list[dict[str, Any]] | None = None,
    on_error: str = "fail",
    allow_dynamic_writes: bool = True,
    depends_on: list[str] | None = None,
) -> CompiledJob:
    config = OrchidPipelineDeterministicJobConfig.model_validate(
        {
            "kind": "deterministic",
            "uses": "pkg.step",
            "depends_on": depends_on or [],
            "writes": writes or [],
            "links": links or [],
            "on_error": on_error,
        }
    )
    return CompiledJob(
        id=job_id,
        kind="deterministic",
        config=config,
        depends_on=tuple(config.depends_on),
        writes=tuple(config.writes),
        links=tuple(config.links),
        retry=config.retry or _default_retry(),
        on_error=on_error,
        allow_dynamic_writes=allow_dynamic_writes,
    )


def _default_retry() -> Any:
    from orchid_ai.config.schema_pipeline import OrchidPipelineRetryConfig

    return OrchidPipelineRetryConfig()


def _limits(**overrides: Any) -> PipelineLimits:
    return replace(PipelineLimits.from_config(), **overrides)


def _pipeline(
    *jobs: CompiledJob,
    shapes: Any = None,
    limits: PipelineLimits | None = None,
) -> CompiledPipeline:
    job_map = {job.id: job for job in (jobs or (_job(),))}
    return CompiledPipeline(
        name="demo",
        version="1",
        description="",
        definition=OrchidPipelineDefinitionConfig(),
        jobs=job_map,
        limits=limits or PipelineLimits.from_config(),
        shapes=shapes,
    )


def _create(object_id: str, data: dict[str, Any] | None = None) -> OrchidGraphMutation:
    return OrchidGraphMutation(op="create", object_id=object_id, object_type="Account", data=data or {"id": object_id})


def _link(relation_type: str, source: str, target: str) -> OrchidGraphMutation:
    relation = OrchidGraphRelation(
        id=f"{relation_type}:{source}:{target}",
        type=relation_type,
        source_id=source,
        target_id=target,
    )
    return OrchidGraphMutation(op="link", relation=relation)


def _session(
    pipeline: CompiledPipeline | None = None,
    *,
    run: OrchidPipelineRunContext | None = None,
    input: dict[str, Any] | None = None,
    observer: OrchidPipelineObserver | None = None,
    clock: Any = None,
    logger: logging.Logger | None = None,
) -> PipelineRunSession:
    return PipelineRunSession(
        definition=pipeline or _pipeline(),
        run=run or _run(),
        input=input or {"account_id": "account:1"},
        observer=observer,
        clock=clock,
        logger=logger,
    )


# ── Construction ─────────────────────────────────────────────


def test_session_builds_an_empty_graph() -> None:
    session = _session()

    assert session.graph.version == 0
    assert session.snapshot() == {"version": 0, "objects": {}, "relations": {}}
    assert session.job_results == {}
    assert session.fatal is False
    assert session.cancelled is False


def test_session_rejects_invalid_arguments() -> None:
    with pytest.raises(TypeError, match="definition"):
        PipelineRunSession(definition="x", run=_run(), input={})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="OrchidPipelineRunContext"):
        PipelineRunSession(definition=_pipeline(), run="x", input={})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="input"):
        PipelineRunSession(definition=_pipeline(), run=_run(), input=[])  # type: ignore[arg-type]


def test_unknown_job_lookup_raises() -> None:
    with pytest.raises(KeyError, match="no job 'nope'"):
        _session().job("nope")


# ── Commit allowlist ────────────────────────────────────────


def test_commit_applies_a_declared_write() -> None:
    pipeline = _pipeline(_job(writes=[{"id": "account:{{ input.account_id }}", "shape": "Account"}]))

    commit = asyncio.run(_session(pipeline).commit("run", [_create("account:account:1")]))

    assert commit.object_ids == ("account:account:1",)
    assert commit.version == 1


def test_commit_rejects_an_undeclared_object() -> None:
    pipeline = _pipeline(_job(writes=[{"id": "account:{{ input.account_id }}", "shape": "Account"}]))
    session = _session(pipeline)

    with pytest.raises(OrchidPipelineWriteNotAllowedError, match="account:other"):
        asyncio.run(session.commit("run", [_create("account:other")]))

    assert session.graph.version == 0
    assert session.graph.objects == {}


def test_commit_rejects_an_undeclared_relation() -> None:
    pipeline = _pipeline(
        _job(
            links=[
                {
                    "type": "has_account",
                    "from": "account:{{ input.account_id }}",
                    "to": "usage:{{ input.account_id }}",
                }
            ]
        )
    )
    session = _session(pipeline)

    with pytest.raises(OrchidPipelineWriteNotAllowedError, match="has_account"):
        asyncio.run(session.commit("run", [_link("has_account", "account:account:1", "other:1")]))


def test_commit_allows_a_declared_relation() -> None:
    pipeline = _pipeline(
        _job(
            writes=[
                {"id": "account:{{ input.account_id }}", "shape": "Account"},
                {"id": "usage:{{ input.account_id }}", "shape": "Usage"},
            ],
            links=[
                {
                    "type": "has_account",
                    "from": "account:{{ input.account_id }}",
                    "to": "usage:{{ input.account_id }}",
                }
            ],
        )
    )
    session = _session(pipeline)

    commit = asyncio.run(
        session.commit(
            "run",
            [
                _create("account:account:1"),
                _create("usage:account:1"),
                _link("has_account", "account:account:1", "usage:account:1"),
            ],
        )
    )

    assert commit.relation_ids == ("has_account:account:account:1:usage:account:1",)


def test_commit_allows_dynamic_writes_without_declarations() -> None:
    session = _session(_pipeline(_job()))

    asyncio.run(session.commit("run", [_create("anything:1")]))

    assert "anything:1" in session.graph.objects


def test_commit_denies_dynamic_writes_when_disabled() -> None:
    session = _session(_pipeline(_job(allow_dynamic_writes=False)))

    with pytest.raises(OrchidPipelineWriteNotAllowedError, match="declared objects: \\(none\\)"):
        asyncio.run(session.commit("run", [_create("anything:1")]))


def test_write_when_false_removes_the_target_from_the_allowlist() -> None:
    pipeline = _pipeline(_job(writes=[{"id": "account:{{ input.account_id }}", "shape": "Account", "when": "`false`"}]))
    session = _session(pipeline)

    with pytest.raises(OrchidPipelineWriteNotAllowedError):
        asyncio.run(session.commit("run", [_create("account:account:1")]))


def test_write_when_expression_gates_the_allowlist() -> None:
    pipeline = _pipeline(
        _job(
            writes=[
                {
                    "id": "account:{{ input.account_id }}",
                    "shape": "Account",
                    "when": "input.account_id == 'account:1'",
                }
            ]
        )
    )
    session = _session(pipeline)

    asyncio.run(session.commit("run", [_create("account:account:1")]))

    assert session.graph.version == 1


def test_commit_rejects_an_undeclared_alias() -> None:
    pipeline = _pipeline(_job(writes=[{"id": "account:1", "shape": "Account"}]))
    session = _session(pipeline)

    with pytest.raises(OrchidPipelineWriteNotAllowedError, match="alias 'account'"):
        asyncio.run(session.commit("run", [_create("account:1")], aliases={"account": "account:1"}))


def test_commit_registers_a_declared_alias() -> None:
    pipeline = _pipeline(_job(writes=[{"id": "account:1", "shape": "Account", "as": "account"}]))
    session = _session(pipeline)

    asyncio.run(session.commit("run", [_create("account:1")], aliases={"account": "account:1"}))

    assert session.graph.alias("account") == "account:1"
    assert dict(session.aliases_for("run")) == {"account": "account:1"}


def test_commit_is_atomic() -> None:
    pipeline = _pipeline(
        _job(writes=[{"id": "account:1", "shape": "Account"}, {"id": "missing:1", "shape": "Account"}])
    )
    session = _session(pipeline)
    bad_update = OrchidGraphMutation(op="update", object_id="missing:1", data={"x": 1})

    with pytest.raises(OrchidPipelineGraphConflictError):
        asyncio.run(session.commit("run", [_create("account:1"), bad_update]))

    assert session.graph.objects == {}
    assert session.graph.version == 0


def test_concurrent_commits_are_serialized() -> None:
    pipeline = _pipeline(
        _job(writes=[{"id": "account:1", "shape": "Account"}, {"id": "account:2", "shape": "Account"}])
    )
    session = _session(pipeline)

    async def run_both() -> None:
        await asyncio.gather(
            session.commit("run", [_create("account:1")]),
            session.commit("run", [_create("account:2")]),
        )

    asyncio.run(run_both())

    assert set(session.graph.objects) == {"account:1", "account:2"}
    assert session.graph.version == 2


def test_commit_applies_shape_validation_and_rolls_back() -> None:
    shapes = compile_shapes({"Account": OrchidPipelineShapeConfig(fields={"id": {"type": "string", "required": True}})})
    pipeline = _pipeline(_job(writes=[{"id": "account:1", "shape": "Account"}]), shapes=shapes)
    session = _session(pipeline)
    bad = OrchidGraphMutation(op="create", object_id="account:1", object_type="Account", data={"id": 3})

    with pytest.raises(OrchidPipelineShapeError):
        asyncio.run(session.commit("run", [bad]))

    assert session.graph.objects == {}


# ── Limits, deadline, cancellation ──────────────────────────


def test_llm_call_cap_is_enforced() -> None:
    limits = _limits(max_llm_calls=2)
    pipeline = _pipeline(limits=limits)
    session = _session(pipeline)

    asyncio.run(session.record_job_result(PipelineJobResult(job_id="run", kind="llm", status="succeeded", llm_calls=2)))
    session.check_limits()

    asyncio.run(session.record_job_result(PipelineJobResult(job_id="run", kind="llm", status="succeeded", llm_calls=1)))
    with pytest.raises(OrchidPipelineLimitError, match="max_llm_calls=2"):
        session.check_limits()


def test_object_cap_is_enforced_by_the_graph() -> None:
    limits = _limits(max_objects=1)
    pipeline = _pipeline(_job(), limits=limits)
    session = _session(pipeline)

    asyncio.run(session.commit("run", [_create("account:1")]))

    with pytest.raises(OrchidPipelineLimitError, match="max_objects"):
        asyncio.run(session.commit("run", [_create("account:2")]))
    assert set(session.graph.objects) == {"account:1"}


def test_check_limits_reports_relation_and_byte_caps() -> None:
    limits = _limits(max_relations=1)
    session = _session(_pipeline(_job(), limits=limits))
    asyncio.run(session.commit("run", [_create("account:1"), _create("account:2")]))
    asyncio.run(session.commit("run", [_link("has_account", "account:1", "account:2")]))
    session.limits = _limits(max_relations=0)
    with pytest.raises(OrchidPipelineLimitError, match="max_relations=0"):
        session.check_limits()

    session.limits = _limits(max_graph_bytes=1)
    with pytest.raises(OrchidPipelineLimitError, match="max_graph_bytes=1"):
        session.check_limits()


def test_llm_calls_property_reports_the_counter() -> None:
    session = _session()
    asyncio.run(session.record_job_result(PipelineJobResult(job_id="run", kind="llm", status="succeeded", llm_calls=3)))

    assert session.llm_calls == 3


def test_link_when_false_removes_the_relation_from_the_allowlist() -> None:
    pipeline = _pipeline(
        _job(
            links=[
                {
                    "type": "has_account",
                    "from": "account:1",
                    "to": "usage:1",
                    "when": "`false`",
                }
            ]
        )
    )
    session = _session(pipeline)

    with pytest.raises(OrchidPipelineWriteNotAllowedError, match="declared relations"):
        asyncio.run(session.commit("run", [_link("has_account", "account:1", "usage:1")]))


def test_deadline_is_enforced() -> None:
    clock = _Clock()
    run = _run(deadline=START + timedelta(seconds=30))
    session = _session(run=run, clock=clock)

    session.check_deadline()
    clock.now = START + timedelta(seconds=30)
    with pytest.raises(OrchidPipelineTimeoutError, match="deadline"):
        session.check_deadline()


def test_cancellation_is_sticky() -> None:
    session = _session()

    assert session.cancelled is False
    session.cancel()
    session.cancel()

    assert session.cancelled is True


# ── Job results ─────────────────────────────────────────────


def test_record_job_result_stores_the_outcome() -> None:
    session = _session()
    result = PipelineJobResult(
        job_id="run",
        kind="deterministic",
        status="succeeded",
        output={"ok": True},
        duration_ms=12.5,
        objects_written=1,
        relations_written=2,
        llm_calls=0,
        objects={"account": "account:1"},
    )

    asyncio.run(session.record_job_result(result))

    assert session.job_results["run"] == result
    assert session.job_results["run"].to_dict()["output"] == {"ok": True}
    assert session.stats()["jobs"] == 1


def test_failed_job_sets_the_fatal_flag() -> None:
    session = _session(_pipeline(_job(on_error="fail")))

    asyncio.run(
        session.record_job_result(
            PipelineJobResult(job_id="run", kind="deterministic", status="failed", error="boom", error_code="x")
        )
    )

    assert session.fatal is True
    assert session.errors[0]["code"] == "x"
    assert session.errors[0]["message"] == "boom"


def test_continue_policy_does_not_set_fatal() -> None:
    session = _session(_pipeline(_job(on_error="continue")))

    asyncio.run(session.record_job_result(PipelineJobResult(job_id="run", kind="deterministic", status="failed")))

    assert session.fatal is False
    assert len(session.errors) == 1


def test_error_messages_are_bounded() -> None:
    session = _session(_pipeline(_job()))

    asyncio.run(
        session.record_job_result(
            PipelineJobResult(job_id="run", kind="deterministic", status="failed", error="x" * 5000)
        )
    )

    assert len(session.errors[0]["message"]) <= 500
    assert session.errors[0]["message"].endswith("…")


def test_errors_are_capped() -> None:
    session = _session(_pipeline(_job(on_error="continue")))

    async def record_many() -> None:
        for _ in range(60):
            await session.record_job_result(
                PipelineJobResult(job_id="run", kind="deterministic", status="failed", error="boom")
            )

    asyncio.run(record_many())

    assert len(session.errors) == 50


def test_failed_result_dict_carries_the_bounded_error() -> None:
    result = PipelineJobResult(
        job_id="run",
        kind="deterministic",
        status="failed",
        error="boom",
        error_code="pipeline_shape_error",
        retryable=True,
    )

    payload = result.to_dict()

    assert payload["error"] == "boom"
    assert payload["error_code"] == "pipeline_shape_error"
    assert payload["retryable"] is True


def test_skipped_result_records_the_reason() -> None:
    session = _session(_pipeline(_job()))

    asyncio.run(
        session.record_job_result(
            PipelineJobResult(job_id="run", kind="deterministic", status="skipped", reason="when is false")
        )
    )

    assert session.job_results["run"].reason == "when is false"


def test_mapping_context_exposes_job_results_and_aliases() -> None:
    session = _session(_pipeline(_job(writes=[{"id": "account:1", "shape": "Account", "as": "account"}])))
    asyncio.run(session.commit("run", [_create("account:1")], aliases={"account": "account:1"}))
    asyncio.run(
        session.record_job_result(
            PipelineJobResult(job_id="run", kind="deterministic", status="succeeded", output={"headline": "ok"})
        )
    )

    context = session.mapping_context()

    assert context.jobs["run"]["output"] == {"headline": "ok"}
    assert context.aliases["account"]["id"] == "account:1"


def test_stats_include_graph_and_run_counters() -> None:
    session = _session(_pipeline(_job()))
    asyncio.run(session.commit("run", [_create("account:1")]))

    stats = session.stats()

    assert stats["objects"] == 1
    assert stats["run_id"] == "run-1"
    assert stats["pipeline"] == "demo"
    assert stats["objects_written"] == 1


# ── Observability ───────────────────────────────────────────


def test_events_arrive_in_order_with_run_id() -> None:
    observer = _ListObserver()
    session = _session(_pipeline(_job()), observer=observer)

    async def run() -> None:
        await session.job_started("run")
        await session.commit("run", [_create("account:1")])
        await session.record_job_result(
            PipelineJobResult(job_id="run", kind="deterministic", status="succeeded", objects_written=1)
        )

    asyncio.run(run())

    types = [event.type for event in observer.events]
    assert types == [
        "pipeline.job.started",
        "pipeline.object.written",
        "pipeline.job.finished",
    ]
    assert all(event.run_id == "run-1" for event in observer.events)
    assert all(event.timestamp is not None for event in observer.events)


def test_written_events_are_bounded() -> None:
    observer = _ListObserver()
    writes = [{"id": f"account:{index}", "shape": "Account"} for index in range(150)]
    session = _session(_pipeline(_job(writes=writes)), observer=observer)

    asyncio.run(session.commit("run", [_create(f"account:{index}") for index in range(150)]))

    written = [event for event in observer.events if event.type == "pipeline.object.written"]
    assert len(written) == 100


def test_observer_failures_never_break_the_run() -> None:
    observer = _BrokenObserver()
    session = _session(_pipeline(_job()), observer=observer)

    asyncio.run(session.emit("pipeline.started", mode="direct"))
    asyncio.run(session.record_job_result(PipelineJobResult(job_id="run", kind="deterministic", status="succeeded")))

    assert observer.calls == 2
    assert "run" in session.job_results


def test_structured_log_line_names_the_job(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("pipeline-session-test")
    caplog.set_level(logging.INFO, logger=logger.name)
    session = _session(_pipeline(_job()), logger=logger)

    asyncio.run(
        session.record_job_result(
            PipelineJobResult(
                job_id="run",
                kind="deterministic",
                status="succeeded",
                duration_ms=3.5,
                objects_written=1,
                relations_written=2,
                llm_calls=0,
            )
        )
    )

    message = caplog.records[0].getMessage()
    assert "pipeline=demo" in message
    assert "run=run-1" in message
    assert "job=run" in message
    assert "kind=deterministic" in message
    assert "status=succeeded" in message
    assert "duration_ms=3.500" in message
    assert "objects_written=1" in message
    assert "error_code=-" in message


# ── Step context ────────────────────────────────────────────


def _step_context(**overrides: Any) -> PipelineStepContext:
    values: dict[str, Any] = {
        "run": _run(),
        "job": OrchidPipelineJobContext(job_id="run", kind="deterministic"),
        "input": {},
        "graph": DataGraph().view(),
        "auth": object(),
    }
    values.update(overrides)
    return PipelineStepContext(**values)


def test_stable_id_is_deterministic_per_run() -> None:
    first = _step_context().stable_id("account", "1")
    second = _step_context().stable_id("account", "1")
    other_run = _step_context(run=_run(run_id="run-2")).stable_id("account", "1")

    assert first == second
    assert first != other_run


def test_call_tool_injects_auth() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    async def caller(name: str, **kwargs: Any) -> Any:
        calls.append((name, kwargs))
        return {"ok": True}

    context = _step_context(tool_caller=caller)

    assert asyncio.run(context.call_tool("lookup", query="x")) == {"ok": True}
    assert calls[0][0] == "lookup"
    assert calls[0][1]["query"] == "x"
    assert calls[0][1]["auth_context"] is context.auth


def test_call_tool_without_a_caller_is_a_configuration_error() -> None:
    with pytest.raises(OrchidPipelineConfigError, match="no built-in tool caller"):
        asyncio.run(_step_context().call_tool("lookup"))


def test_emit_signal_without_events_is_an_error() -> None:
    with pytest.raises(OrchidPipelineEventsDisabledError, match="events layer is disabled"):
        asyncio.run(_step_context().emit_signal(object()))


def test_emit_signal_delegates_to_the_emitter() -> None:
    envelopes: list[Any] = []

    class _Emitter:
        async def emit(self, envelope: Any) -> Any:
            envelopes.append(envelope)
            return "sent"

    envelope = object()
    context = _step_context(signal_emitter=_Emitter())

    assert asyncio.run(context.emit_signal(envelope)) == "sent"
    assert envelopes == [envelope]


def test_step_context_exposes_params() -> None:
    context = _step_context(params={"limit": 3})

    assert context.params == {"limit": 3}


# ── Review regressions (P-006 CR round 1) ───────────────────


def test_shape_version_is_stamped_from_the_compiled_shape() -> None:
    shapes = compile_shapes(
        {
            "Account": OrchidPipelineShapeConfig(
                version="2",
                fields={"id": {"type": "string", "required": True}},
            )
        }
    )
    pipeline = _pipeline(_job(writes=[{"id": "account:1", "shape": "Account"}]), shapes=shapes)
    session = _session(pipeline)

    asyncio.run(session.commit("run", [_create("account:1")]))

    assert session.graph.objects["account:1"].shape_version == "2"
    assert session.input_shape_version == "1"


def test_input_shape_version_is_exposed() -> None:
    shapes = compile_shapes({"Input": OrchidPipelineShapeConfig(version="4", fields={"id": {"type": "string"}})})
    pipeline = CompiledPipeline(
        name="demo",
        version="1",
        description="",
        definition=OrchidPipelineDefinitionConfig(),
        jobs={},
        limits=PipelineLimits.from_config(),
        shapes=shapes,
        input_shape=shapes.get("Input"),
    )

    assert _session(pipeline).input_shape_version == "4"


def test_deadline_is_derived_from_the_run_timeout() -> None:
    session = _session(_pipeline(limits=_limits(timeout_seconds=30.0)))

    assert session.deadline == START + timedelta(seconds=30)


def test_explicit_run_deadline_wins_over_the_timeout() -> None:
    deadline = START + timedelta(seconds=5)
    session = _session(run=_run(deadline=deadline), pipeline=_pipeline(limits=_limits(timeout_seconds=30.0)))

    assert session.deadline == deadline


def test_input_is_read_only_for_a_step() -> None:
    session = _session()
    context = _step_context(input=session.input)

    with pytest.raises(TypeError):
        session.input["account_id"] = "tampered"  # type: ignore[index]
    with pytest.raises(TypeError):
        context.input["account_id"] = "tampered"  # type: ignore[index]


def test_input_is_deep_copied_from_the_caller() -> None:
    payload = {"account": {"id": "account:1"}}
    session = _session(input=payload)
    payload["account"]["id"] = "changed"

    assert session.input["account"]["id"] == "account:1"
    assert session.input is not payload


def test_failure_events_redact_bearer_tokens() -> None:
    observer = _ListObserver()
    session = _session(_pipeline(_job()), observer=observer)
    error = "Authorization: Bearer sk-secret-123 request failed"

    asyncio.run(
        session.record_job_result(
            PipelineJobResult(job_id="run", kind="deterministic", status="failed", error=error, error_code="x")
        )
    )

    assert "sk-secret-123" not in session.errors[0]["message"]
    failed = [event for event in observer.events if event.type == "pipeline.job.failed"]
    assert "sk-secret-123" not in failed[0].data["error"]
    assert "[redacted]" in failed[0].data["error"]


def test_raw_authorization_header_is_redacted() -> None:
    session = _session(_pipeline(_job()))

    asyncio.run(
        session.record_job_result(
            PipelineJobResult(job_id="run", kind="deterministic", status="failed", error="authorization: abc12345678")
        )
    )

    assert "abc123" not in session.errors[0]["message"]


def test_written_events_snapshot_the_committed_revision() -> None:
    observer = _ListObserver()
    pipeline = _pipeline(
        _job("writer", writes=[{"id": "account:1", "shape": "Account"}, {"id": "account:2", "shape": "Account"}]),
        _job("updater", writes=[{"id": "account:2", "shape": "Account"}]),
    )
    session = _session(pipeline, observer=observer)

    async def run_both() -> None:
        await asyncio.gather(
            session.commit("writer", [_create("account:1"), _create("account:2")]),
            session.commit("updater", [OrchidGraphMutation(op="update", object_id="account:2", data={"x": 1})]),
        )

    asyncio.run(run_both())

    writer_events = [
        event for event in observer.events if event.type == "pipeline.object.written" and event.data["job"] == "writer"
    ]
    assert {event.data["object_id"]: event.data["revision"] for event in writer_events} == {
        "account:1": 1,
        "account:2": 1,
    }


def test_negative_counters_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        PipelineJobResult(job_id="run", kind="deterministic", status="succeeded", llm_calls=-1)
    with pytest.raises(TypeError, match="must be an integer"):
        PipelineJobResult(job_id="run", kind="deterministic", status="succeeded", attempts=1.5)  # type: ignore[arg-type]


def test_unknown_status_is_rejected() -> None:
    with pytest.raises(ValueError, match="job status"):
        PipelineJobResult(job_id="run", kind="deterministic", status="done")  # type: ignore[arg-type]


def test_idempotency_key_defaults_to_the_run_id() -> None:
    assert _step_context().idempotency_key == "run-1"


def test_idempotency_key_uses_trigger_and_signal_for_bloom() -> None:
    context = _step_context(run=_run(trigger_id="trigger-1", signal_id="signal-1"))

    assert context.idempotency_key == "trigger-1:signal-1"


def test_explicit_idempotency_key_wins() -> None:
    assert _step_context(idempotency_key="custom").idempotency_key == "custom"


def test_stored_job_result_error_is_redacted() -> None:
    session = _session(_pipeline(_job()))
    error = "Authorization: Bearer sk-secret-123 request failed"

    asyncio.run(
        session.record_job_result(
            PipelineJobResult(job_id="run", kind="deterministic", status="failed", error=error, error_code="x")
        )
    )

    stored = session.job_results["run"]
    assert stored.error is not None
    assert "sk-secret-123" not in stored.error
    assert "sk-secret-123" not in stored.to_dict()["error"]
    assert "sk-secret-123" not in str(session.mapping_context().jobs)


def test_redaction_keeps_prose_readable() -> None:
    from orchid_ai.pipeline.session import _redact

    assert _redact("Bearer authentication failed") == "Bearer authentication failed"
    assert _redact("the bearer token is missing") == "the bearer token is missing"
    assert _redact("authorization denied") == "authorization denied"
    assert _redact("request failed with 500") == "request failed with 500"
    assert _redact("Bearer sk-secret-123") == "Bearer [redacted]"


@pytest.mark.parametrize(
    "value",
    [
        "ghp_abcdefghijklmnopqrstuvwxyzABCDEFGH",
        "sk-abcdefghijklmnopqrstuvwxyzABCDEFGH",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.signature",
        "xoxb-abcdefghijklmnop",
        "AKIAIOSFODNN7EXAMPLE",
    ],
)
def test_redaction_covers_common_token_families(value: str) -> None:
    from orchid_ai.pipeline.session import _redact

    assert value not in _redact(f"request failed: Bearer {value}")


def test_redaction_covers_basic_auth() -> None:
    from orchid_ai.pipeline.session import _redact

    assert _redact("Authorization: Basic YWxhZGRpbjpvcGVuc2VzYW1l") == "Authorization: Basic [redacted]"
    assert _redact("authorization: Basic dXNlcjpwYXNz") == "authorization: Basic [redacted]"


def test_skipped_reason_is_redacted() -> None:
    session = _session(_pipeline(_job()))

    asyncio.run(
        session.record_job_result(
            PipelineJobResult(
                job_id="run",
                kind="deterministic",
                status="skipped",
                reason="Bearer sk-secret-123",
            )
        )
    )

    stored = session.job_results["run"]
    assert "sk-secret-123" not in stored.reason
    assert "sk-secret-123" not in stored.to_dict()["reason"]


def test_result_fields_are_type_checked() -> None:
    with pytest.raises(TypeError, match="duration_ms must be a number"):
        PipelineJobResult(job_id="run", kind="deterministic", status="succeeded", duration_ms=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite and non-negative"):
        PipelineJobResult(job_id="run", kind="deterministic", status="succeeded", duration_ms=float("nan"))
    with pytest.raises(TypeError, match="error must be a string"):
        PipelineJobResult(job_id="run", kind="deterministic", status="failed", error=123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="reason must be a string"):
        PipelineJobResult(job_id="run", kind="deterministic", status="skipped", reason=3)  # type: ignore[arg-type]


def test_redaction_covers_quoted_and_slack_tokens() -> None:
    from orchid_ai.pipeline.session import _redact

    assert "ghp_" not in _redact("Authorization: 'ghp_abcdefghijklmnopqrstuvwxyz'")
    assert "ghp_" not in _redact('Bearer "ghp_abcdefghijklmnopqrstuvwxyz"')
    assert "xoxc-" not in _redact("Authorization: xoxc-abcdefghijklmnopqrstuv")


def test_redaction_keeps_hyphenated_prose_readable() -> None:
    from orchid_ai.pipeline.session import _redact

    assert _redact("Bearer authentication-failed") == "Bearer authentication-failed"
    assert _redact("Bearer the-request-timed-out") == "Bearer the-request-timed-out"
