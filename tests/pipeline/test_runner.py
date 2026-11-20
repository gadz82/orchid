"""End-to-end tests for the direct pipeline runner."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest

from orchid_ai.config.schema_pipeline import (
    OrchidPipelineDefinitionConfig,
    OrchidPipelineDeterministicJobConfig,
    OrchidPipelineInputConfig,
    OrchidPipelineRetryConfig,
    OrchidPipelineShapeConfig,
)
from orchid_ai.core.pipeline import (
    OrchidPipelineAuthError,
    OrchidPipelineInputError,
    OrchidPipelineObserver,
    OrchidPipelineRetryableError,
    OrchidPipelineRunContext,
    OrchidPipelineStep,
    OrchidPipelineStepError,
    OrchidPipelineStepResult,
)
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.pipeline.definition import CompiledJob, CompiledPipeline, PipelineLimits
from orchid_ai.pipeline.registry import register_pipeline_step, unregister_pipeline_step
from orchid_ai.pipeline.runner import OrchidPipelineRunner
from orchid_ai.pipeline.schema import compile_shapes


def _shapes() -> Any:
    return compile_shapes(
        {
            "Input": OrchidPipelineShapeConfig(
                fields={"account_id": {"type": "string", "required": True}, "region": {"type": "string"}},
            ),
            "Account": OrchidPipelineShapeConfig(
                fields={"id": {"type": "string", "required": True}, "name": {"type": "string"}},
            ),
            "Report": OrchidPipelineShapeConfig(fields={"headline": {"type": "string"}}),
        }
    )


def _job(
    job_id: str,
    handler: Any,
    *,
    depends_on: list[str] | None = None,
    writes: list[dict[str, Any]] | None = None,
    links: list[dict[str, Any]] | None = None,
    when: str | None = None,
    on_error: str = "fail",
    retry: OrchidPipelineRetryConfig | None = None,
    timeout_seconds: float | None = None,
    kind: str = "deterministic",
    branches: dict[str, str] | None = None,
    fallback: str | None = None,
    join: str = "all",
    metadata: dict[str, Any] | None = None,
) -> CompiledJob:
    config = OrchidPipelineDeterministicJobConfig.model_validate(
        {
            "kind": "deterministic",
            "uses": "pkg.step",
            "depends_on": depends_on or [],
            "writes": writes or [],
            "links": links or [],
            "when": when,
            "on_error": on_error,
        }
    )
    return CompiledJob(
        id=job_id,
        kind=kind,
        config=config,
        handler=handler,
        depends_on=tuple(config.depends_on),
        join=join,
        when=config.when,
        writes=tuple(config.writes),
        links=tuple(config.links),
        retry=retry or OrchidPipelineRetryConfig(),
        on_error=on_error,
        timeout_seconds=timeout_seconds,
        branches=branches or {},
        fallback=fallback,
        metadata=metadata or {},
    )


def _pipeline(*jobs: CompiledJob, input_shape: bool = False, limits: PipelineLimits | None = None) -> CompiledPipeline:
    definition = OrchidPipelineDefinitionConfig()
    shapes = _shapes()
    return CompiledPipeline(
        name="demo",
        version="1",
        description="",
        definition=definition,
        jobs={job.id: job for job in jobs},
        limits=limits or PipelineLimits.from_config(),
        shapes=shapes,
        input_shape=shapes.get("Input") if input_shape else None,
    )


def _runner(pipeline: CompiledPipeline | None = None) -> OrchidPipelineRunner:
    return OrchidPipelineRunner(pipeline or _pipeline(_job("run", _echo)))


async def _echo(ctx: Any, **params: Any) -> dict[str, Any]:
    return {"echo": True}


def _account_write() -> dict[str, Any]:
    return {"id": "account:{{ input.account_id }}", "shape": "Account", "from": "result.account", "as": "account"}


# ── Happy path ──────────────────────────────────────────────


def test_run_executes_a_deterministic_pipeline() -> None:
    pipeline = _pipeline(
        _job(
            "load", lambda ctx: {"account": {"id": ctx.input["account_id"], "name": "Acme"}}, writes=[_account_write()]
        ),
        _job(
            "report",
            lambda ctx: {"headline": "ok"},
            depends_on=["load"],
            writes=[{"id": "report:1", "shape": "Report"}],
        ),
        input_shape=True,
    )

    result = asyncio.run(_runner(pipeline).run({"account_id": "a1"}, user_id="u1", tenant_id="t1", include_graph=True))

    assert result.status == "succeeded"
    assert result.run_id
    assert result.pipeline == "demo"
    assert result.graph_stats["objects"] == 3
    assert result.graph_stats["objects_written"] == 2
    assert result.jobs["load"]["status"] == "succeeded"
    assert result.jobs["load"]["objects"] == {"account": "account:a1"}
    assert set(result.graph["objects"]) == {"__input__", "account:a1", "report:1"}
    # No output block: the output is the serialized graph snapshot.
    assert result.output["objects"]["account:a1"]["type"] == "Account"
    assert result.summary == ""
    assert result.summary_source == "none"
    assert result.error is None


def test_include_graph_false_returns_an_empty_graph() -> None:
    result = asyncio.run(_runner().run({}, user_id="u1"))

    assert result.graph == {}
    assert result.graph_stats["objects"] == 1  # the reserved input object


def test_job_output_is_available_to_later_jobs() -> None:
    pipeline = _pipeline(
        _job("first", lambda ctx: {"headline": "from first"}),
        _job(
            "second",
            _echo,
            depends_on=["first"],
            writes=[{"id": "report:1", "shape": "Report", "from": "jobs.first.output"}],
        ),
    )

    result = asyncio.run(_runner(pipeline).run({}, user_id="u1"))

    assert result.status == "succeeded", result.jobs
    assert result.graph_stats["by_type"].get("Report") == 1


def test_llm_usage_is_counted_on_the_job_and_stats() -> None:
    async def handler(ctx: Any) -> OrchidPipelineStepResult:
        return OrchidPipelineStepResult(output={"ok": True}, usage={"llm_calls": 2})

    result = asyncio.run(_runner(_pipeline(_job("run", handler))).run({}, user_id="u1"))

    assert result.jobs["run"]["llm_calls"] == 2
    assert result.graph_stats["llm_calls"] == 2


def test_input_defaults_are_applied() -> None:
    seen: dict[str, Any] = {}

    def handler(ctx: Any) -> dict[str, Any]:
        seen.update(ctx.input)
        return {}

    pipeline = _pipeline(_job("run", handler))
    pipeline.definition.input = OrchidPipelineInputConfig(
        fields={"account_id": {"type": "string", "required": True}, "region": {"type": "string"}},
        defaults={"region": "eu"},
    )
    shapes = _shapes()
    pipeline.input_shape = shapes.get("Input")

    asyncio.run(_runner(pipeline).run({"account_id": "a1"}, user_id="u1"))

    assert seen == {"account_id": "a1", "region": "eu"}


def test_input_validation_failure_raises() -> None:
    pipeline = _pipeline(_job("run", _echo), input_shape=True)

    with pytest.raises(OrchidPipelineInputError, match="input is invalid"):
        asyncio.run(_runner(pipeline).run({}, user_id="u1"))


def test_input_must_be_a_mapping() -> None:
    with pytest.raises(OrchidPipelineInputError, match="must be a mapping"):
        asyncio.run(_runner().run("nope", user_id="u1"))  # type: ignore[arg-type]


def test_auth_is_derived_from_ids_and_visible_to_steps() -> None:
    seen: dict[str, Any] = {}

    def handler(ctx: Any) -> dict[str, Any]:
        seen["tenant"] = ctx.run.tenant_key
        seen["user"] = ctx.run.user_id
        seen["auth"] = ctx.auth
        return {}

    asyncio.run(_runner(_pipeline(_job("run", handler))).run({}, user_id="alice", tenant_id="acme"))

    assert seen["tenant"] == "acme"
    assert seen["user"] == "alice"
    assert seen["auth"].tenant_key == "acme"


def test_explicit_auth_wins() -> None:
    auth = OrchidAuthContext(access_token="token", tenant_key="explicit", user_id="owner")
    seen: dict[str, Any] = {}

    def handler(ctx: Any) -> dict[str, Any]:
        seen["auth"] = ctx.auth
        return {}

    asyncio.run(_runner(_pipeline(_job("run", handler))).run({}, auth=auth, tenant_id="ignored"))

    assert seen["auth"] is auth


def test_run_requires_some_identity() -> None:
    with pytest.raises(OrchidPipelineAuthError, match="authenticated caller"):
        asyncio.run(_runner().run({}, user_id="", tenant_id="", access_token=""))


def test_seed_objects_are_visible_to_steps() -> None:
    seen: dict[str, Any] = {}

    def handler(ctx: Any) -> dict[str, Any]:
        seeded = ctx.graph.get("account:seeded").data
        seen["id"] = seeded.get("id")
        return {}

    result = asyncio.run(
        _runner(_pipeline(_job("run", handler))).run(
            {},
            user_id="u1",
            seed=[{"id": "account:seeded", "type": "Account", "data": {"id": "seeded"}}],
        )
    )

    assert result.status == "succeeded"
    assert seen["id"] == "seeded"


def test_seed_objects_are_shape_validated() -> None:
    with pytest.raises(OrchidPipelineInputError, match=r"seed\[0\] is invalid"):
        asyncio.run(
            _runner().run(
                {},
                user_id="u1",
                seed=[{"id": "account:bad", "type": "Account", "data": {}}],
            )
        )


# ── Status matrix ───────────────────────────────────────────


def test_fatal_failure_skips_dependents() -> None:
    def fail(ctx: Any) -> dict[str, Any]:
        raise OrchidPipelineStepError("boom", job_id="first", kind="deterministic")

    pipeline = _pipeline(
        _job("first", fail),
        _job("second", _echo, depends_on=["first"]),
    )

    result = asyncio.run(_runner(pipeline).run({}, user_id="u1"))

    assert result.status == "failed"
    assert result.jobs["first"]["status"] == "failed"
    assert result.jobs["second"]["status"] == "skipped"
    assert "boom" in (result.error or "")


def test_continue_policy_yields_partial() -> None:
    def fail(ctx: Any) -> dict[str, Any]:
        raise OrchidPipelineStepError("boom")

    pipeline = _pipeline(
        _job("first", fail, on_error="continue"),
        _job("second", _echo, depends_on=["first"]),
    )

    result = asyncio.run(_runner(pipeline).run({}, user_id="u1"))

    assert result.status == "partial"
    assert result.jobs["first"]["status"] == "failed"
    assert result.jobs["second"]["status"] == "succeeded"


def test_when_false_skips_without_writing_and_dependents_run() -> None:
    pipeline = _pipeline(
        _job("first", lambda ctx: {"account": {"id": "a1"}}, when="`false`", writes=[_account_write()]),
        _job("second", _echo, depends_on=["first"]),
    )

    result = asyncio.run(_runner(pipeline).run({"account_id": "a1"}, user_id="u1"))

    assert result.status == "succeeded"
    assert result.jobs["first"]["status"] == "skipped"
    assert "when expression" in result.jobs["first"]["reason"]
    assert result.jobs["second"]["status"] == "succeeded"
    assert result.graph_stats["by_type"].get("Account") is None


def test_skipped_jobs_still_count_as_success() -> None:
    result = asyncio.run(_runner(_pipeline(_job("run", _echo, when="`false`"))).run({}, user_id="u1"))

    assert result.status == "succeeded"


def test_retryable_failure_is_retried() -> None:
    attempts = {"count": 0}

    def flaky(ctx: Any) -> dict[str, Any]:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise OrchidPipelineRetryableError("try again", job_id="run")
        return {"ok": True}

    pipeline = _pipeline(_job("run", flaky, retry=OrchidPipelineRetryConfig(max=1, initial_delay_seconds=0.001)))

    result = asyncio.run(_runner(pipeline).run({}, user_id="u1"))

    assert result.status == "succeeded"
    assert result.jobs["run"]["attempts"] == 2
    assert attempts["count"] == 2


def test_retry_exhaustion_fails_the_job() -> None:
    def always_retryable(ctx: Any) -> dict[str, Any]:
        raise OrchidPipelineRetryableError("nope", job_id="run")

    pipeline = _pipeline(
        _job("run", always_retryable, retry=OrchidPipelineRetryConfig(max=2, initial_delay_seconds=0.001))
    )

    result = asyncio.run(_runner(pipeline).run({}, user_id="u1"))

    assert result.status == "failed"
    assert result.jobs["run"]["attempts"] == 3


def test_non_retryable_failure_is_not_retried() -> None:
    attempts = {"count": 0}

    def broken(ctx: Any) -> dict[str, Any]:
        attempts["count"] += 1
        raise OrchidPipelineStepError("no retry")

    pipeline = _pipeline(_job("run", broken, retry=OrchidPipelineRetryConfig(max=3, initial_delay_seconds=0.001)))

    result = asyncio.run(_runner(pipeline).run({}, user_id="u1"))

    assert result.status == "failed"
    assert attempts["count"] == 1
    assert result.jobs["run"]["attempts"] == 1


def test_timeout_fails_the_job() -> None:
    async def slow(ctx: Any) -> dict[str, Any]:
        await asyncio.sleep(0.5)
        return {}

    pipeline = _pipeline(_job("run", slow, timeout_seconds=0.01))

    result = asyncio.run(_runner(pipeline).run({}, user_id="u1"))

    assert result.status == "failed"
    assert result.jobs["run"]["error_code"] == "pipeline_timeout"


def test_invalid_write_value_fails_the_job() -> None:
    pipeline = _pipeline(
        _job("run", lambda ctx: "not-a-mapping", writes=[{"id": "account:a1", "shape": "Account"}]),
    )

    result = asyncio.run(_runner(pipeline).run({}, user_id="u1"))

    assert result.status == "failed"
    assert result.jobs["run"]["status"] == "failed"
    assert "must return None, a dict" in (result.jobs["run"]["error"] or "")


def test_unsupported_return_value_fails_the_job() -> None:
    result = asyncio.run(_runner(_pipeline(_job("run", lambda ctx: 42))).run({}, user_id="u1"))

    assert result.status == "failed"
    assert "must return None, a dict" in (result.jobs["run"]["error"] or "")


# ── Concurrency ─────────────────────────────────────────────


def test_concurrent_runs_are_isolated() -> None:
    seen: list[str] = []

    async def handler(ctx: Any) -> dict[str, Any]:
        seen.append(ctx.run.run_id)
        await asyncio.sleep(0.01)
        return {"account": {"id": ctx.input["account_id"], "name": ctx.input["account_id"]}}

    pipeline = _pipeline(_job("run", handler, writes=[_account_write()]))
    runner = _runner(pipeline)

    async def run_both() -> tuple[Any, Any]:
        return await asyncio.gather(
            runner.run({"account_id": "a1"}, user_id="u1", include_graph=True),
            runner.run({"account_id": "a2"}, user_id="u1", include_graph=True),
        )

    first, second = asyncio.run(run_both())

    assert first.run_id != second.run_id
    assert set(seen) == {first.run_id, second.run_id}
    assert set(first.graph["objects"]) == {"__input__", "account:a1"}
    assert set(second.graph["objects"]) == {"__input__", "account:a2"}


# ── Events and streaming ───────────────────────────────────


def test_observer_receives_lifecycle_events_in_order() -> None:
    events: list[str] = []

    class _Observer(OrchidPipelineObserver):
        async def emit(self, event: Any) -> None:
            events.append(event.type)

    asyncio.run(_runner().run({}, user_id="u1", observer=_Observer()))

    assert events[0] == "pipeline.started"
    assert "pipeline.job.started" in events
    assert "pipeline.job.finished" in events
    assert events[-1] == "pipeline.finished"


def test_stream_yields_events_and_a_terminal_result() -> None:
    async def collect() -> list[Any]:
        runner = _runner()
        return [event async for event in runner.stream({}, user_id="u1")]

    events = asyncio.run(collect())

    assert events[0].type == "pipeline.started"
    assert events[-1].type == "pipeline.finished"
    assert events[-1].data["status"] == "succeeded"
    assert events[-1].data["result"]["status"] == "succeeded"


def test_stream_forwards_events_to_a_caller_observer() -> None:
    seen: list[str] = []

    class _Observer(OrchidPipelineObserver):
        async def emit(self, event: Any) -> None:
            seen.append(event.type)

    async def collect() -> None:
        runner = _runner()
        async for _ in runner.stream({}, user_id="u1", observer=_Observer()):
            pass

    asyncio.run(collect())

    assert seen[-1] == "pipeline.finished"


# ── Decision routing ───────────────────────────────────────


class _DecisionStep(OrchidPipelineStep):
    kind = "decision"

    def __init__(self, label: str) -> None:
        self._label = label

    async def execute(self, ctx: Any) -> OrchidPipelineStepResult:
        return OrchidPipelineStepResult(output={"label": self._label, "confidence": 0.9})


@pytest.fixture
def decision_kind() -> Any:
    def factory(job: CompiledJob, deps: Any) -> OrchidPipelineStep:
        return _DecisionStep(job.metadata.get("label", "retain"))

    register_pipeline_step("decision", factory, override=True)
    yield
    unregister_pipeline_step("decision")


def test_decision_runs_only_the_selected_branch(decision_kind: Any) -> None:
    decision = _job(
        "choose",
        None,
        kind="decision",
        branches={"retain": "left", "expand": "right"},
        fallback="retain",
        metadata={"label": "retain"},
    )
    pipeline = _pipeline(decision, _job("left", _echo), _job("right", _echo))

    result = asyncio.run(_runner(pipeline).run({}, user_id="u1"))

    assert result.jobs["choose"]["status"] == "succeeded"
    assert result.jobs["left"]["status"] == "succeeded"
    assert "right" not in result.jobs


def test_join_any_after_branches_converges(decision_kind: Any) -> None:
    decision = _job(
        "choose",
        None,
        kind="decision",
        branches={"retain": "left", "expand": "right"},
        fallback="retain",
        metadata={"label": "left"},
    )
    join = _job("report", _echo, depends_on=["left", "right"], join="any")
    pipeline = _pipeline(decision, _job("left", _echo), _job("right", _echo), join)

    result = asyncio.run(_runner(pipeline).run({}, user_id="u1"))

    assert result.jobs["report"]["status"] == "succeeded"


# ── Cancellation ───────────────────────────────────────────


def test_cancelled_session_reports_cancelled() -> None:
    pipeline = _pipeline(_job("run", _echo))
    run = OrchidPipelineRunContext(
        run_id="run-1",
        pipeline="demo",
        pipeline_version="1",
        mode="direct",
        tenant_key="t",
        user_id="u",
        started_at=datetime.now(UTC),
    )
    from orchid_ai.pipeline.session import PipelineRunSession

    session = PipelineRunSession(definition=pipeline, run=run, input={})
    session.cancel()

    from orchid_ai.pipeline.builder import final_status

    assert final_status(session) == "cancelled"


# ── Additional runner coverage ──────────────────────────────


def test_runner_exposes_the_pipeline() -> None:
    pipeline = _pipeline(_job("run", _echo))

    assert OrchidPipelineRunner(pipeline).pipeline is pipeline


def test_runner_requires_a_compiled_pipeline() -> None:
    with pytest.raises(TypeError, match="CompiledPipeline"):
        OrchidPipelineRunner("nope")  # type: ignore[arg-type]


def test_chat_mode_sets_the_chat_id() -> None:
    seen: dict[str, Any] = {}

    def handler(ctx: Any) -> dict[str, Any]:
        seen["chat_id"] = ctx.run.chat_id
        seen["mode"] = ctx.run.mode
        return {}

    asyncio.run(_runner(_pipeline(_job("run", handler))).run({}, user_id="u1", chat_id="c-1", mode="chat"))

    assert seen == {"chat_id": "c-1", "mode": "chat"}


def test_direct_mode_ignores_the_chat_id() -> None:
    seen: dict[str, Any] = {}

    def handler(ctx: Any) -> dict[str, Any]:
        seen["chat_id"] = ctx.run.chat_id
        return {}

    asyncio.run(_runner(_pipeline(_job("run", handler))).run({}, user_id="u1", chat_id="c-1"))

    assert seen["chat_id"] is None


def test_run_timeout_is_clamped_to_the_pipeline_limit() -> None:
    seen: dict[str, Any] = {}

    def handler(ctx: Any) -> dict[str, Any]:
        seen["seconds"] = (ctx.run.deadline - ctx.run.started_at).total_seconds()
        return {}

    pipeline = _pipeline(_job("run", handler), limits=replace(PipelineLimits.from_config(), timeout_seconds=0.05))
    asyncio.run(_runner(pipeline).run({}, user_id="u1", timeout=10.0))

    assert seen["seconds"] == pytest.approx(0.05)


def test_run_id_can_be_injected() -> None:
    result = asyncio.run(_runner().run({}, user_id="u1", run_id="run-fixed"))

    assert result.run_id == "run-fixed"


def test_oversized_graph_is_omitted() -> None:
    pipeline = _pipeline(
        _job(
            "run",
            lambda ctx: {"id": "a1", "name": "x" * 400},
            writes=[{"id": "account:a1", "shape": "Account"}],
        ),
        limits=replace(PipelineLimits.from_config(), max_graph_bytes=300),
    )

    result = asyncio.run(_runner(pipeline).run({}, user_id="u1", include_graph=True))

    assert result.graph == {}
    assert result.metadata["graph_omitted"] == "size"


def test_unexpected_graph_error_becomes_a_failed_result(monkeypatch: pytest.MonkeyPatch) -> None:
    from orchid_ai.pipeline import runner as runner_module

    class _BrokenGraph:
        async def ainvoke(self, state: Any, config: Any) -> Any:
            raise RuntimeError("graph exploded")

    monkeypatch.setattr(runner_module, "build_pipeline_graph", lambda *a, **k: _BrokenGraph())

    result = asyncio.run(_runner().run({}, user_id="u1"))

    assert result.status == "failed"
    assert "graph exploded" in (result.error or "")


def test_stream_surfaces_a_runner_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from orchid_ai.pipeline import runner as runner_module

    class _BrokenGraph:
        async def ainvoke(self, state: Any, config: Any) -> Any:
            raise RuntimeError("graph exploded")

    monkeypatch.setattr(runner_module, "build_pipeline_graph", lambda *a, **k: _BrokenGraph())

    async def collect() -> list[Any]:
        return [event async for event in _runner().stream({}, user_id="u1")]

    events = asyncio.run(collect())

    assert events[-1].data["status"] == "failed"


def test_stream_can_be_closed_early() -> None:
    async def consume() -> None:
        stream = _runner().stream({}, user_id="u1")
        event = await stream.__anext__()
        assert event.type == "pipeline.started"
        await stream.aclose()

    asyncio.run(consume())


def test_broken_stream_observer_does_not_stop_events() -> None:
    class _Broken(OrchidPipelineObserver):
        async def emit(self, event: Any) -> None:
            raise RuntimeError("observer down")

    async def collect() -> list[Any]:
        return [event async for event in _runner().stream({}, user_id="u1", observer=_Broken())]

    events = asyncio.run(collect())

    assert events[-1].type == "pipeline.finished"


def test_status_helper_reports_partial_and_failed() -> None:
    from orchid_ai.pipeline.builder import final_status
    from orchid_ai.pipeline.session import PipelineJobResult, PipelineRunSession

    pipeline = _pipeline(_job("fail", _echo, on_error="continue"), _job("fatal", _echo))
    run = OrchidPipelineRunContext(
        run_id="run-1",
        pipeline="demo",
        pipeline_version="1",
        mode="direct",
        tenant_key="t",
        user_id="u",
        started_at=datetime.now(UTC),
    )
    session = PipelineRunSession(definition=pipeline, run=run, input={})

    async def record(job_id: str, status: str) -> None:
        await session.record_job_result(PipelineJobResult(job_id=job_id, kind="deterministic", status=status))

    asyncio.run(record("fail", "failed"))
    assert final_status(session) == "partial"
    asyncio.run(record("fatal", "failed"))
    assert final_status(session) == "failed"


def test_result_error_prefers_the_explicit_state_error() -> None:
    from orchid_ai.pipeline.runner import _result_error
    from orchid_ai.pipeline.session import PipelineRunSession

    session = PipelineRunSession(
        definition=_pipeline(_job("run", _echo)),
        run=OrchidPipelineRunContext(
            run_id="run-1",
            pipeline="demo",
            pipeline_version="1",
            mode="direct",
            tenant_key="t",
            user_id="u",
            started_at=datetime.now(UTC),
        ),
        input={},
    )

    assert _result_error(session, {"error": "explicit"}) == "explicit"
    assert _result_error(session, {}) is None
    session.cancel()
    assert _result_error(session, {}) == "pipeline run was cancelled"


# ── Review regressions (P-007 CR round 1) ──────────────────


def test_deadline_marks_the_run_cancelled() -> None:
    async def slow(ctx: Any) -> dict[str, Any]:
        # Run past the deadline deterministically, whatever the machine speed.
        remaining = (ctx.run.deadline - datetime.now(UTC)).total_seconds()
        await asyncio.sleep(max(0.0, remaining) + 0.02)
        return {}

    pipeline = _pipeline(
        _job("slow", slow),
        _job("after", _echo, depends_on=["slow"]),
        limits=replace(PipelineLimits.from_config(), timeout_seconds=0.5),
    )
    result = asyncio.run(_runner(pipeline).run({}, user_id="u1"))

    assert result.status == "cancelled"
    assert result.jobs["slow"]["status"] == "succeeded"
    assert result.jobs["after"]["status"] == "cancelled"
    assert "deadline" in (result.jobs["after"]["reason"] or "")


def test_non_json_output_fails_the_job() -> None:
    pipeline = _pipeline(_job("run", lambda ctx: {"obj": object()}))

    result = asyncio.run(_runner(pipeline).run({}, user_id="u1"))

    assert result.status == "failed"
    assert "not JSON-serializable" in (result.jobs["run"]["error"] or "")


def test_non_mapping_seed_is_an_input_error() -> None:
    with pytest.raises(OrchidPipelineInputError, match="seed\\[0\\] must be a mapping"):
        asyncio.run(_runner().run({}, user_id="u1", seed=["not-a-mapping"]))  # type: ignore[list-item]


def test_unexpected_graph_error_is_redacted(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    from orchid_ai.pipeline import runner as runner_module

    token = "sk-live-abcdefghijklmnopqrstuvwxyz0123456789"

    class _TokenGraph:
        async def ainvoke(self, state: Any, config: Any) -> Any:
            raise RuntimeError(f"upstream 401: Authorization: Bearer {token}")

    monkeypatch.setattr(runner_module, "build_pipeline_graph", lambda *a, **k: _TokenGraph())
    caplog.set_level("ERROR", logger="orchid_ai.pipeline.runner")
    events: list[Any] = []

    class _Observer(OrchidPipelineObserver):
        async def emit(self, event: Any) -> None:
            events.append(event)

    result = asyncio.run(_runner().run({}, user_id="u1", observer=_Observer()))

    finished = [event for event in events if event.type == "pipeline.finished"]
    assert token not in (result.error or "")
    assert token not in str(result.to_dict(include_graph=False))
    assert token not in str([event.to_dict() for event in finished])
    assert token not in caplog.text


# ── Review regressions (P-007 CR round 2) ──────────────────


def test_non_json_output_does_not_commit_writes() -> None:
    pipeline = _pipeline(
        _job(
            "write",
            lambda ctx: {"headline": "ok", "obj": object()},
            writes=[{"id": "report:1", "shape": "Report", "from": "result"}],
        ),
    )

    result = asyncio.run(_runner(pipeline).run({}, user_id="u1", include_graph=True))

    assert result.status == "failed"
    assert "report:1" not in result.graph.get("objects", {})


def test_datetime_output_is_normalized_to_iso() -> None:
    pipeline = _pipeline(_job("run", lambda ctx: {"at": datetime.now(UTC)}))

    result = asyncio.run(_runner(pipeline).run({}, user_id="u1"))

    assert result.status == "succeeded"
    assert isinstance(result.jobs["run"]["output"]["at"], str)


def test_set_output_is_normalized_to_a_list() -> None:
    pipeline = _pipeline(_job("run", lambda ctx: {"items": {3, 1, 2}}))

    result = asyncio.run(_runner(pipeline).run({}, user_id="u1"))

    assert result.status == "succeeded"
    assert sorted(result.jobs["run"]["output"]["items"]) == [1, 2, 3]


def test_nan_output_fails_without_breaking_the_result() -> None:
    pipeline = _pipeline(_job("run", lambda ctx: {"value": float("nan")}))

    result = asyncio.run(_runner(pipeline).run({}, user_id="u1"))

    assert result.status == "failed"
    assert "not JSON-serializable" in (result.jobs["run"]["error"] or "")
    assert result.to_dict(include_graph=False)["status"] == "failed"


def test_decision_end_path_still_honours_the_deadline() -> None:
    class _SlowDecision(OrchidPipelineStep):
        kind = "decision"

        async def execute(self, ctx: Any) -> OrchidPipelineStepResult:
            remaining = (ctx.run.deadline - datetime.now(UTC)).total_seconds()
            await asyncio.sleep(max(0.0, remaining) + 0.02)
            return OrchidPipelineStepResult(output={"label": "unknown"})

    decision = _job("choose", None, kind="decision", branches={"retain": "left"})
    object.__setattr__(decision, "handler", _SlowDecision())
    pipeline = _pipeline(
        decision,
        _job("left", _echo),
        limits=replace(PipelineLimits.from_config(), timeout_seconds=0.5),
    )

    result = asyncio.run(_runner(pipeline).run({}, user_id="u1"))

    assert result.status == "cancelled"


def test_decision_without_branches_can_use_a_job_fallback(decision_kind: Any) -> None:
    decision = _job(
        "choose",
        None,
        kind="decision",
        branches={},
        fallback="fallback",
        metadata={"label": "unmatched"},
    )
    pipeline = _pipeline(decision, _job("fallback", _echo))

    result = asyncio.run(_runner(pipeline).run({}, user_id="u1"))

    assert result.jobs["fallback"]["status"] == "succeeded"


def test_fallback_label_wins_over_a_job_with_the_same_name(decision_kind: Any) -> None:
    from orchid_ai.config.schema_pipeline import OrchidPipelineDecisionJobConfig

    config = OrchidPipelineDecisionJobConfig.model_validate(
        {
            "kind": "decision",
            "decision_model": "triage",
            "question": "Which?",
            "labels": ["retain", "left"],
            "branches": {"retain": "left"},
            "fallback": "left",
        }
    )
    decision = CompiledJob(
        id="choose",
        kind="decision",
        config=config,
        branches={"retain": "left"},
        fallback="left",
        metadata={"label": "unmatched"},
    )
    pipeline = _pipeline(decision, _job("left", _echo))

    result = asyncio.run(_runner(pipeline).run({}, user_id="u1"))

    # ``left`` is a declared label with no branch mapping: no target, so the
    # decision ends the run instead of running the job named ``left``.
    assert "left" not in result.jobs
    assert result.status == "succeeded"


def test_fatal_failure_outranks_a_deadline_cancellation() -> None:
    started = asyncio.Event()

    async def fail(ctx: Any) -> dict[str, Any]:
        # Wait until the parallel slow job is running, then fail fatally.
        await started.wait()
        raise OrchidPipelineStepError("boom")

    async def slow(ctx: Any) -> dict[str, Any]:
        started.set()
        remaining = (ctx.run.deadline - datetime.now(UTC)).total_seconds()
        await asyncio.sleep(max(0.0, remaining) + 0.02)
        return {}

    pipeline = _pipeline(
        _job("fail", fail),
        _job("slow", slow),
        limits=replace(PipelineLimits.from_config(), timeout_seconds=0.3),
    )

    result = asyncio.run(_runner(pipeline).run({}, user_id="u1"))

    assert result.status == "failed"
    assert result.jobs["fail"]["status"] == "failed"
    assert result.jobs["slow"]["status"] == "succeeded"


def test_graph_error_after_a_job_is_reported_as_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    from orchid_ai.pipeline import runner as runner_module
    from orchid_ai.pipeline.session import PipelineJobResult

    class _Graph:
        async def ainvoke(self, state: Any, config: Any) -> Any:
            session = config["configurable"]["pipeline_session"]
            await session.record_job_result(PipelineJobResult(job_id="run", kind="deterministic", status="succeeded"))
            raise RuntimeError("graph exploded after job")

    monkeypatch.setattr(runner_module, "build_pipeline_graph", lambda *a, **k: _Graph())

    result = asyncio.run(_runner().run({}, user_id="u1"))

    assert result.status == "failed"
    assert result.jobs["run"]["status"] == "succeeded"
    assert "graph exploded after job" in (result.error or "")


def test_graph_interrupt_is_reported_as_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    from orchid_ai.pipeline import runner as runner_module

    class _Graph:
        async def ainvoke(self, state: Any, config: Any) -> Any:
            return {"__interrupt__": [object()], "job_results": {}}

    monkeypatch.setattr(runner_module, "build_pipeline_graph", lambda *a, **k: _Graph())

    result = asyncio.run(_runner().run({}, user_id="u1"))

    assert result.status == "failed"
    assert result.error == "pipeline interrupts are not supported in v1"


def test_empty_graph_error_message_is_still_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    from orchid_ai.pipeline import runner as runner_module

    class _Graph:
        async def ainvoke(self, state: Any, config: Any) -> Any:
            raise RuntimeError()

    monkeypatch.setattr(runner_module, "build_pipeline_graph", lambda *a, **k: _Graph())

    result = asyncio.run(_runner().run({}, user_id="u1"))

    assert result.status == "failed"
    assert result.error == "pipeline graph failed"


def test_real_graph_interrupt_is_reported_as_failed() -> None:
    from langgraph.errors import GraphInterrupt

    def interrupt(ctx: Any) -> dict[str, Any]:
        raise GraphInterrupt(())

    pipeline = _pipeline(_job("run", interrupt))

    result = asyncio.run(_runner(pipeline).run({}, user_id="u1"))

    assert result.status == "failed"
    assert result.error == "pipeline interrupts are not supported in v1"
    assert result.jobs["run"]["status"] == "failed"
    assert result.jobs["run"]["error_code"] == "pipeline_interrupt_unsupported"
