"""Tests for the pipeline graph builder and its topology."""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime
from types import ModuleType
from typing import Any

import pytest

from orchid_ai.config.schema_pipeline import (
    OrchidPipelineDecisionJobConfig,
    OrchidPipelineDefinitionConfig,
    OrchidPipelineDeterministicJobConfig,
    OrchidPipelineRetryConfig,
)
from orchid_ai.core.pipeline import (
    OrchidPipelineAuthError,
    OrchidPipelineConfigError,
    OrchidPipelineRunContext,
    OrchidPipelineStep,
    OrchidPipelineStepResult,
)
from orchid_ai.core.run_config import with_auth
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.pipeline.builder import build_pipeline_graph, pipeline_session_from_config
from orchid_ai.pipeline.definition import CompiledJob, CompiledPipeline, PipelineLimits
from orchid_ai.pipeline.session import PipelineRunSession


def echo(ctx: Any) -> dict[str, Any]:
    return {"ok": True}


class _FakeStep(OrchidPipelineStep):
    kind = "fake"

    async def execute(self, ctx: Any) -> OrchidPipelineStepResult:  # pragma: no cover - not executed here
        return OrchidPipelineStepResult(output={})


@pytest.fixture
def steps_module(monkeypatch: pytest.MonkeyPatch) -> str:
    module = ModuleType("pipeline_builder_fakes")
    module.echo = echo  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pipeline_builder_fakes", module)
    return "pipeline_builder_fakes"


def _job(job_id: str, *, depends_on: list[str] | None = None, **kwargs: Any) -> CompiledJob:
    writes = kwargs.pop("writes", [])
    links = kwargs.pop("links", [])
    when = kwargs.pop("when", None)
    kind = kwargs.pop("kind", "deterministic")
    config = OrchidPipelineDeterministicJobConfig.model_validate(
        {
            "kind": "deterministic",
            "uses": "pipeline_builder_fakes.echo",
            "depends_on": depends_on or [],
            "writes": writes,
            "links": links,
            "when": when,
        }
    )
    return CompiledJob(
        id=job_id,
        kind=kind,
        config=config,
        handler=kwargs.pop("handler", echo if kind == "deterministic" else None),
        depends_on=tuple(config.depends_on),
        when=config.when,
        writes=tuple(config.writes),
        links=tuple(config.links),
        retry=kwargs.pop("retry", None) or OrchidPipelineRetryConfig(),
        timeout_seconds=kwargs.pop("timeout_seconds", None),
        branches=kwargs.pop("branches", {}),
        fallback=kwargs.pop("fallback", None),
        metadata=kwargs.pop("metadata", {}),
        **kwargs,
    )


def _pipeline(*jobs: CompiledJob, relations: Any = None) -> CompiledPipeline:
    job_map = {job.id: job for job in (jobs or (_job("run"),))}
    return CompiledPipeline(
        name="demo",
        version="1",
        description="",
        definition=OrchidPipelineDefinitionConfig(),
        jobs=job_map,
        limits=PipelineLimits.from_config(),
        relations=relations or {},
    )


def _session(pipeline: CompiledPipeline) -> OrchidPipelineRunContext:
    run = OrchidPipelineRunContext(
        run_id="run-1",
        pipeline=pipeline.name,
        pipeline_version="1",
        mode="direct",
        tenant_key="t",
        user_id="u",
        started_at=datetime.now(UTC),
    )
    return run


# ── Topology ────────────────────────────────────────────────


def test_graph_is_cached_on_the_definition(steps_module: str) -> None:
    pipeline = _pipeline()

    first = build_pipeline_graph(pipeline)
    second = build_pipeline_graph(pipeline)

    assert first is second
    assert pipeline.graph is first


def test_graph_has_no_checkpointer(steps_module: str) -> None:
    pipeline = _pipeline(_job("a"), _job("b", depends_on=["a"]))

    graph = build_pipeline_graph(pipeline)

    assert getattr(graph, "checkpointer", None) is None


def test_graph_contains_start_jobs_finish_and_edges(steps_module: str) -> None:
    pipeline = _pipeline(_job("a"), _job("b", depends_on=["a"]))

    graph = build_pipeline_graph(pipeline)
    drawing = graph.get_graph()

    assert {"__pipeline_start__", "__finish__", "a", "b"} <= set(drawing.nodes)
    edges = {(edge.source, edge.target) for edge in drawing.edges}
    assert ("__pipeline_start__", "a") in edges
    assert ("a", "b") in edges
    assert ("b", "__finish__") in edges


def test_parallel_roots_both_start(steps_module: str) -> None:
    pipeline = _pipeline(_job("a"), _job("b"))

    edges = {(edge.source, edge.target) for edge in build_pipeline_graph(pipeline).get_graph().edges}

    assert ("__pipeline_start__", "a") in edges
    assert ("__pipeline_start__", "b") in edges
    assert ("a", "__finish__") in edges
    assert ("b", "__finish__") in edges


def test_join_any_gets_a_synthetic_join_node(steps_module: str) -> None:
    decision = CompiledJob(
        id="choose",
        kind="decision",
        config=OrchidPipelineDecisionJobConfig.model_validate(
            {
                "kind": "decision",
                "decision_model": "triage",
                "question": "Which?",
                "labels": ["a", "b"],
                "branches": {"a": "left", "b": "right"},
            }
        ),
        handler=_FakeStep(),
        branches={"a": "left", "b": "right"},
    )
    left = _job("left")
    right = _job("right")
    join = _job("join", depends_on=["left", "right"])
    object.__setattr__(join, "join", "any")
    pipeline = _pipeline(decision, left, right, join)

    graph = build_pipeline_graph(pipeline)
    nodes = set(graph.get_graph().nodes)

    assert "__join__join" in nodes
    edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}
    assert ("left", "__join__join") in edges
    assert ("right", "__join__join") in edges


def test_decision_job_has_conditional_edges(steps_module: str) -> None:
    decision = CompiledJob(
        id="choose",
        kind="decision",
        config=OrchidPipelineDecisionJobConfig.model_validate(
            {
                "kind": "decision",
                "decision_model": "triage",
                "question": "Which?",
                "labels": ["a", "b"],
                "branches": {"a": "left", "b": "right"},
            }
        ),
        handler=_FakeStep(),
        branches={"a": "left", "b": "right"},
    )
    pipeline = _pipeline(decision, _job("left"), _job("right"))

    drawing = build_pipeline_graph(pipeline).get_graph()
    conditional = {(edge.source, edge.target) for edge in drawing.edges if edge.conditional}

    assert ("choose", "left") in conditional
    assert ("choose", "right") in conditional


def test_unknown_kind_fails_at_build_time(steps_module: str) -> None:
    job = CompiledJob(
        id="run",
        kind="not_a_registered_kind",
        config=OrchidPipelineDeterministicJobConfig.model_validate(
            {"kind": "deterministic", "uses": "pipeline_builder_fakes.echo"}
        ),
    )

    with pytest.raises(OrchidPipelineConfigError, match="no step is registered for that kind"):
        build_pipeline_graph(_pipeline(job))


def test_builder_requires_a_compiled_pipeline() -> None:
    with pytest.raises(TypeError, match="CompiledPipeline"):
        build_pipeline_graph("not-a-pipeline")  # type: ignore[arg-type]


# ── Start node ──────────────────────────────────────────────


def _invoke_config(session: PipelineRunSession, *, auth: OrchidAuthContext | None) -> dict[str, Any]:
    base = {"configurable": {"pipeline_session": session}}
    return with_auth(auth, thread_id=session.run.run_id, base=base)


def test_start_node_requires_auth(steps_module: str) -> None:
    pipeline = _pipeline()
    graph = build_pipeline_graph(pipeline)
    run = _session(pipeline)
    session = PipelineRunSession(definition=pipeline, run=run, input={})

    with pytest.raises(OrchidPipelineAuthError, match="authenticated caller"):
        asyncio.run(graph.ainvoke({"run": {}, "input": {}}, config=_invoke_config(session, auth=None)))

    assert session.job_results == {}


def test_start_node_seeds_input_and_emits_started(steps_module: str) -> None:
    from orchid_ai.core.pipeline import OrchidPipelineObserver

    events: list[Any] = []

    class _Observer(OrchidPipelineObserver):
        async def emit(self, event: Any) -> None:
            events.append(event)

    pipeline = _pipeline(_job("run"))
    graph = build_pipeline_graph(pipeline)
    run = _session(pipeline)
    session = PipelineRunSession(definition=pipeline, run=run, input={"account": "a1"}, observer=_Observer())

    asyncio.run(
        graph.ainvoke(
            {"run": {}, "input": {}}, config=_invoke_config(session, auth=OrchidAuthContext(access_token="t"))
        )
    )

    assert session.graph.objects["__input__"].data == {"account": "a1"}
    assert events[0].type == "pipeline.started"


def test_start_node_validates_seed_objects(steps_module: str) -> None:
    from orchid_ai.core.pipeline import OrchidPipelineInputError

    pipeline = _pipeline(_job("run"))
    graph = build_pipeline_graph(pipeline)
    run = _session(pipeline)
    session = PipelineRunSession(definition=pipeline, run=run, input={})

    with pytest.raises(OrchidPipelineInputError, match="requires a 'data' mapping"):
        asyncio.run(
            graph.ainvoke(
                {"run": {}, "input": {}, "seed": [{"id": "a", "type": "Account"}]},
                config=_invoke_config(session, auth=OrchidAuthContext(access_token="t")),
            )
        )


def test_missing_session_in_config_is_a_config_error(steps_module: str) -> None:
    pipeline = _pipeline()
    graph = build_pipeline_graph(pipeline)

    with pytest.raises(OrchidPipelineConfigError, match="without a run session"):
        asyncio.run(graph.ainvoke({"run": {}, "input": {}}, config={"configurable": {}}))


def test_pipeline_session_from_config_is_type_checked() -> None:
    with pytest.raises(OrchidPipelineConfigError, match="without a run session"):
        pipeline_session_from_config({"configurable": {"pipeline_session": "nope"}})


def test_join_router_skips_when_no_dependency_ran(steps_module: str) -> None:
    from orchid_ai.pipeline.builder import _make_join_router

    job = _job("join", depends_on=["a", "b"])
    object.__setattr__(job, "join", "any")
    router = _make_join_router(job)

    assert router({"job_results": {}}) == "skip"
    assert router({"job_results": {"a": {"status": "skipped"}}}) == "run"
    assert router({"job_results": {"a": {}, "b": {}}}) == "run"


def test_decision_router_uses_label_fallback_and_finish() -> None:
    from orchid_ai.pipeline.builder import _make_decision_router

    job = CompiledJob(
        id="choose",
        kind="decision",
        config=OrchidPipelineDecisionJobConfig.model_validate(
            {
                "kind": "decision",
                "decision_model": "triage",
                "question": "Which?",
                "labels": ["a", "b"],
                "branches": {"a": "left", "b": "right"},
                "fallback": "b",
            }
        ),
        branches={"a": "left", "b": "right"},
        fallback="b",
    )
    router = _make_decision_router(job)

    assert router({"job_results": {"choose": {"output": {"label": "a"}}}}) == "left"
    assert router({"job_results": {"choose": {"output": {"label": "unknown"}}}}) == "right"
    assert router({"job_results": {}}) == "right"

    no_fallback = CompiledJob(
        id="choose",
        kind="decision",
        config=OrchidPipelineDecisionJobConfig.model_validate(
            {
                "kind": "decision",
                "decision_model": "triage",
                "question": "Which?",
                "labels": ["a"],
                "branches": {"a": "left"},
            }
        ),
    )
    assert _make_decision_router(no_fallback)({"job_results": {}}) == "__finish__"


# ── Additional builder coverage ─────────────────────────────


@pytest.mark.parametrize(
    ("seed", "expected"),
    [
        pytest.param(["not-a-mapping"], "must be a mapping", id="not-mapping"),
        pytest.param([{"type": "Account", "data": {}}], "non-empty 'id'", id="missing-id"),
        pytest.param([{"id": "a", "data": {}}], "non-empty 'type'", id="missing-type"),
        pytest.param([{"id": "a", "type": "Account"}], "requires a 'data' mapping", id="missing-data"),
    ],
)
def test_seed_validation_messages(steps_module: str, seed: Any, expected: str) -> None:
    from orchid_ai.core.pipeline import OrchidPipelineInputError

    pipeline = _pipeline(_job("run"))
    graph = build_pipeline_graph(pipeline)
    run = _session(pipeline)
    session = PipelineRunSession(definition=pipeline, run=run, input={})

    with pytest.raises(OrchidPipelineInputError, match=expected):
        asyncio.run(
            graph.ainvoke(
                {"run": {}, "input": {}, "seed": seed},
                config=_invoke_config(session, auth=OrchidAuthContext(access_token="t")),
            )
        )


def test_cancelled_session_marks_jobs_cancelled(steps_module: str) -> None:
    pipeline = _pipeline(_job("run"))
    graph = build_pipeline_graph(pipeline)
    run = _session(pipeline)
    session = PipelineRunSession(definition=pipeline, run=run, input={})
    session.cancel()

    state = asyncio.run(
        graph.ainvoke(
            {"run": {}, "input": {}}, config=_invoke_config(session, auth=OrchidAuthContext(access_token="t"))
        )
    )

    assert state["job_results"]["run"]["status"] == "cancelled"
    assert state["status"] == "cancelled"


def test_expired_deadline_cancels_the_job(steps_module: str) -> None:
    pipeline = _pipeline(_job("run"))
    graph = build_pipeline_graph(pipeline)
    run = OrchidPipelineRunContext(
        run_id="run-1",
        pipeline="demo",
        pipeline_version="1",
        mode="direct",
        tenant_key="t",
        user_id="u",
        started_at=datetime.now(UTC),
        deadline=datetime.now(UTC),
    )
    session = PipelineRunSession(definition=pipeline, run=run, input={})

    state = asyncio.run(
        graph.ainvoke(
            {"run": {}, "input": {}}, config=_invoke_config(session, auth=OrchidAuthContext(access_token="t"))
        )
    )

    assert state["job_results"]["run"]["status"] == "cancelled"
    assert state["status"] == "cancelled"


def test_invalid_when_expression_fails_the_job(steps_module: str) -> None:
    job = _job("run")
    object.__setattr__(job, "when", "input.x + 1")
    pipeline = _pipeline(job)
    graph = build_pipeline_graph(pipeline)
    run = _session(pipeline)
    session = PipelineRunSession(definition=pipeline, run=run, input={})

    state = asyncio.run(
        graph.ainvoke(
            {"run": {}, "input": {}}, config=_invoke_config(session, auth=OrchidAuthContext(access_token="t"))
        )
    )

    assert state["job_results"]["run"]["status"] == "failed"
    assert state["job_results"]["run"]["error_code"] == "pipeline_mapping_error"


def test_write_expected_revision_expression_is_resolved(steps_module: str) -> None:
    job = _job("run", writes=[{"id": "account:a1", "shape": "Account", "expected_revision": "input.rev"}])
    object.__setattr__(job, "handler", lambda ctx: {"id": "a1"})
    pipeline = _pipeline(job)
    pipeline.definition = OrchidPipelineDefinitionConfig()

    pipeline.input_shape = None

    result = asyncio.run(
        __import__("orchid_ai.pipeline.runner", fromlist=["OrchidPipelineRunner"])
        .OrchidPipelineRunner(pipeline)
        .run({"rev": 1}, user_id="u")
    )

    assert result.jobs["run"]["status"] == "failed"
    assert "expected_revision" in (result.jobs["run"]["error"] or "") or "must not set expected_revision" in (
        result.jobs["run"]["error"] or ""
    )


def test_link_when_false_skips_the_relation(steps_module: str) -> None:
    from orchid_ai.pipeline.runner import OrchidPipelineRunner

    job = _job("run", links=[{"type": "has_account", "from": "a:1", "to": "b:1", "when": "`false`"}])
    object.__setattr__(job, "handler", lambda ctx: {})
    pipeline = _pipeline(job)

    result = asyncio.run(OrchidPipelineRunner(pipeline).run({}, user_id="u"))

    assert result.status == "succeeded"
    assert result.jobs["run"]["relations_written"] == 0


def test_relation_endpoint_type_mismatch_fails_the_job(steps_module: str) -> None:
    from orchid_ai.config.schema_pipeline import OrchidPipelineRelationConfig
    from orchid_ai.pipeline.runner import OrchidPipelineRunner

    relation = OrchidPipelineRelationConfig.model_validate({"from": ["Account"], "to": ["Report"]})
    job = _job("run", links=[{"type": "has_account", "from": "report:1", "to": "account:1"}])
    object.__setattr__(job, "handler", lambda ctx: {})
    pipeline = _pipeline(job, relations={"has_account": relation})

    result = asyncio.run(
        OrchidPipelineRunner(pipeline).run(
            {},
            user_id="u",
            seed=[
                {"id": "account:1", "type": "Account", "data": {"id": "a1"}},
                {"id": "report:1", "type": "Report", "data": {"headline": "x"}},
            ],
        )
    )

    assert result.jobs["run"]["status"] == "failed"
    assert "expected one of" in (result.jobs["run"]["error"] or "")


@pytest.mark.parametrize(
    ("backoff", "expected"),
    [
        pytest.param("fixed", 1.0, id="fixed"),
        pytest.param("linear", 2.0, id="linear"),
        pytest.param("exponential", 2.0, id="exponential"),
    ],
)
def test_backoff_delays(steps_module: str, backoff: str, expected: float) -> None:
    from orchid_ai.pipeline.builder import _backoff_delay

    job = _job(
        "run",
        retry=OrchidPipelineRetryConfig(max=2, backoff=backoff, jitter=False, initial_delay_seconds=1.0),
    )

    assert _backoff_delay(job, 2) == expected


def test_decision_fallback_can_name_a_job(steps_module: str) -> None:
    from orchid_ai.pipeline.builder import _make_decision_router

    job = CompiledJob(
        id="choose",
        kind="decision",
        config=OrchidPipelineDecisionJobConfig.model_validate(
            {
                "kind": "decision",
                "decision_model": "triage",
                "question": "Which?",
                "labels": ["a"],
                "branches": {"a": "left"},
                "fallback": "fallback_job",
            }
        ),
        branches={"a": "left"},
        fallback="fallback_job",
    )

    assert _make_decision_router(job)({"job_results": {}}) == "fallback_job"


# ── Review regressions (P-007 CR round 1) ──────────────────


def test_fallback_job_id_is_not_a_root_and_runs_only_when_selected(decision_kind: Any) -> None:
    fallback = _job("fallback")
    left = _job("left")
    decision = _job(
        "choose",
        kind="decision",
        branches={"retain": "left"},
        fallback="fallback",
        metadata={"label": "unknown"},
    )
    pipeline = _pipeline(decision, left, fallback)
    from orchid_ai.pipeline.runner import OrchidPipelineRunner

    result = asyncio.run(OrchidPipelineRunner(pipeline).run({}, user_id="u"))

    assert result.jobs["fallback"]["status"] == "succeeded"
    assert "left" not in result.jobs


def test_fallback_job_id_does_not_run_when_a_branch_is_selected(decision_kind: Any) -> None:
    fallback = _job("fallback")
    left = _job("left")
    decision = _job(
        "choose",
        kind="decision",
        branches={"retain": "left"},
        fallback="fallback",
        metadata={"label": "retain"},
    )
    pipeline = _pipeline(decision, left, fallback)
    from orchid_ai.pipeline.runner import OrchidPipelineRunner

    result = asyncio.run(OrchidPipelineRunner(pipeline).run({}, user_id="u"))

    assert result.jobs["left"]["status"] == "succeeded"
    assert "fallback" not in result.jobs


def test_branch_target_with_an_unrelated_dependency_is_rejected(steps_module: str) -> None:
    helper = _job("helper")
    left = _job("left", depends_on=["helper"])
    decision = CompiledJob(
        id="choose",
        kind="decision",
        config=OrchidPipelineDecisionJobConfig.model_validate(
            {
                "kind": "decision",
                "decision_model": "triage",
                "question": "Which?",
                "labels": ["a"],
                "branches": {"a": "left"},
            }
        ),
        handler=_FakeStep(),
        branches={"a": "left"},
    )

    with pytest.raises(OrchidPipelineConfigError, match="must depend only on its decision job"):
        build_pipeline_graph(_pipeline(decision, helper, left))


@pytest.fixture
def decision_kind() -> Any:
    from orchid_ai.pipeline.registry import register_pipeline_step, unregister_pipeline_step

    def factory(job: CompiledJob, deps: Any) -> OrchidPipelineStep:
        from orchid_ai.core.pipeline import OrchidPipelineStepResult

        class _Decision(OrchidPipelineStep):
            kind = "decision"

            async def execute(self, ctx: Any) -> OrchidPipelineStepResult:
                return OrchidPipelineStepResult(output={"label": job.metadata.get("label", "retain")})

        return _Decision()

    register_pipeline_step("decision", factory, override=True)
    yield
    unregister_pipeline_step("decision")
