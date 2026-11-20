"""Tests for the four pipeline interpretation modes."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from orchid_ai.config.schema_pipeline import (
    OrchidPipelineDefinitionConfig,
    OrchidPipelineDeterministicJobConfig,
    OrchidPipelineInterpretationConfig,
    OrchidPipelineShapeConfig,
)
from orchid_ai.core.pipeline import OrchidGraphObject, OrchidPipelineRunContext
from orchid_ai.pipeline.definition import CompiledJob, CompiledPipeline, PipelineLimits
from orchid_ai.pipeline.interpretation import OrchidPipelineInterpreter
from orchid_ai.pipeline.schema import compile_shapes
from orchid_ai.pipeline.session import PipelineJobResult, PipelineRunSession

START = datetime(2026, 11, 18, 12, 0, tzinfo=UTC)


class _FakeModel:
    def __init__(self, *, content: str = "model summary", error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls: list[list[Any]] = []

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return AIMessage(content=self.content)


def _shapes() -> Any:
    return compile_shapes(
        {
            "Account": OrchidPipelineShapeConfig(
                fields={
                    "name": {"type": "string"},
                    "status": {"type": "string"},
                    "score": {"type": "number"},
                }
            )
        }
    )


def _job(job_id: str, *, kind: str = "deterministic") -> CompiledJob:
    config = OrchidPipelineDeterministicJobConfig.model_validate({"kind": "deterministic", "uses": "pkg.step"})
    return CompiledJob(id=job_id, kind=kind, config=config)


def _pipeline(
    *,
    interpretation: OrchidPipelineInterpretationConfig | None = None,
    jobs: tuple[CompiledJob, ...] = (),
    limits: PipelineLimits | None = None,
) -> CompiledPipeline:
    definition = OrchidPipelineDefinitionConfig(
        interpretation=interpretation or OrchidPipelineInterpretationConfig(),
    )
    return CompiledPipeline(
        name="demo",
        version="1",
        description="",
        definition=definition,
        jobs={job.id: job for job in jobs},
        limits=limits or PipelineLimits.from_config(),
        shapes=_shapes(),
    )


def _session(pipeline: CompiledPipeline, *, objects: tuple[OrchidGraphObject, ...] = ()) -> PipelineRunSession:
    run = OrchidPipelineRunContext(
        run_id="run-1",
        pipeline="demo",
        pipeline_version="1",
        mode="direct",
        tenant_key="tenant-secret",
        user_id="user-secret",
        started_at=START,
        metadata={"token": "super-secret"},
    )
    session = PipelineRunSession(definition=pipeline, run=run, input={})
    if objects:
        session.graph.seed(objects)
    return session


async def _record(
    session: PipelineRunSession,
    job_id: str,
    *,
    status: str = "succeeded",
    kind: str = "deterministic",
    output: Any = None,
    llm_calls: int = 0,
) -> None:
    result = PipelineJobResult(
        job_id=job_id,
        kind=kind,
        status=status,
        output=output,
        duration_ms=1.0,
        llm_calls=llm_calls,
        objects_written=1 if output is not None else 0,
        error="boom" if status == "failed" else None,
        error_code="pipeline_step_error" if status == "failed" else "",
    )
    await session.record_job_result(result)


# ── none / template / caller ───────────────────────────────


async def test_none_mode_returns_an_empty_summary() -> None:
    pipeline = _pipeline()
    session = _session(pipeline)

    summary = await OrchidPipelineInterpreter(pipeline, session).interpret(
        output={"headline": "ok"},
        status="succeeded",
        graph_stats=session.stats(),
    )

    assert summary.text == ""
    assert summary.source == "none"
    assert summary.metadata == {}


async def test_template_mode_renders_output_stats_and_run() -> None:
    pipeline = _pipeline(
        interpretation=OrchidPipelineInterpretationConfig(
            mode="template",
            template="{{ run.id }} {{ run.mode }}: {{ output.headline }} ({{ graph_stats.objects }} objects)",
        )
    )
    session = _session(
        pipeline,
        objects=(OrchidGraphObject(id="account:a1", type="Account", data={"name": "Acme"}),),
    )

    summary = await OrchidPipelineInterpreter(pipeline, session).interpret(
        output={"headline": "growth"},
        status="succeeded",
        graph_stats=session.stats(),
    )

    assert summary.source == "template"
    assert summary.text == "run-1 direct: growth (1 objects)"


async def test_template_does_not_expose_run_metadata() -> None:
    pipeline = _pipeline(
        interpretation=OrchidPipelineInterpretationConfig(
            mode="template",
            template="{{ run.metadata.token }}",
        )
    )
    session = _session(pipeline)

    summary = await OrchidPipelineInterpreter(pipeline, session).interpret(
        output={},
        status="succeeded",
        graph_stats=session.stats(),
    )

    assert summary.source == "none"
    assert summary.metadata["interpretation_fallback"] == "template_failed"


async def test_reserved_input_object_is_excluded_from_the_caller_digest() -> None:
    pipeline = _pipeline(interpretation=OrchidPipelineInterpretationConfig(mode="caller"))
    session = _session(
        pipeline,
        objects=(OrchidGraphObject(id="account:a1", type="Account", data={"name": "Acme"}),),
    )
    session.graph.set_input({"account_id": "a1"})

    summary = await OrchidPipelineInterpreter(pipeline, session).interpret(
        output={},
        status="succeeded",
        graph_stats=session.stats(),
    )

    assert "__pipeline_input__" not in summary.text
    assert "Objects: 1 (Account=1)." in summary.text


async def test_caller_mode_builds_a_deterministic_digest() -> None:
    pipeline = _pipeline(
        interpretation=OrchidPipelineInterpretationConfig(mode="caller"),
        jobs=(_job("load"), _job("triage", kind="decision"), _job("report")),
    )
    session = _session(
        pipeline,
        objects=(
            OrchidGraphObject(id="account:a1", type="Account", data={"name": "Acme"}),
            OrchidGraphObject(id="report:1", type="Report", data={"headline": "ok"}),
        ),
    )
    await _record(session, "load")
    await _record(session, "triage", kind="decision", output={"label": "approve"})
    await _record(session, "report", status="failed")

    summary = await OrchidPipelineInterpreter(pipeline, session).interpret(
        output={"headline": "ok"},
        status="partial",
        graph_stats=session.stats(),
    )

    assert summary.source == "caller"
    assert summary.text.startswith("Pipeline 'demo' partial.")
    assert "Jobs: 3 (failed=1, succeeded=2)." in summary.text
    assert "Objects: 2 (Account=1, Report=1)." in summary.text
    assert "Decisions: triage=approve." in summary.text
    assert "Errors: 1." in summary.text


# ── LLM mode ───────────────────────────────────────────────


async def test_llm_mode_returns_the_model_summary() -> None:
    pipeline = _pipeline(
        interpretation=OrchidPipelineInterpretationConfig(mode="llm", model="test/model"),
        jobs=(_job("load"),),
    )
    session = _session(pipeline)
    await _record(session, "load")
    model = _FakeModel(content="  A concise summary.  ")

    summary = await OrchidPipelineInterpreter(pipeline, session, model_factory=lambda ref: model).interpret(
        output={"headline": "growth"},
        status="succeeded",
        graph_stats=session.stats(),
    )

    assert summary.source == "llm"
    assert summary.text == "A concise summary."
    assert len(model.calls) == 1
    assert session.llm_calls == 1


async def test_llm_failure_falls_back_to_the_template() -> None:
    pipeline = _pipeline(
        interpretation=OrchidPipelineInterpretationConfig(
            mode="llm",
            model="test/model",
            template="fallback {{ output.headline }}",
        )
    )
    session = _session(pipeline)
    model = _FakeModel(error=RuntimeError("secret sk-test"))

    summary = await OrchidPipelineInterpreter(pipeline, session, model_factory=lambda ref: model).interpret(
        output={"headline": "growth"},
        status="succeeded",
        graph_stats=session.stats(),
    )

    assert summary.source == "template"
    assert summary.text == "fallback growth"
    assert summary.metadata["interpretation_fallback"] == "model_failed"
    assert session.llm_calls == 1


async def test_llm_failure_without_a_template_returns_none() -> None:
    pipeline = _pipeline(interpretation=OrchidPipelineInterpretationConfig(mode="llm", model="test/model"))
    session = _session(pipeline)
    model = _FakeModel(error=RuntimeError("secret sk-test"))

    summary = await OrchidPipelineInterpreter(pipeline, session, model_factory=lambda ref: model).interpret(
        output={},
        status="succeeded",
        graph_stats=session.stats(),
    )

    assert summary.source == "none"
    assert summary.text == ""
    assert summary.metadata["interpretation_fallback"] == "model_failed"


async def test_llm_call_is_skipped_when_the_cap_is_reached() -> None:
    pipeline = _pipeline(
        interpretation=OrchidPipelineInterpretationConfig(
            mode="llm",
            model="test/model",
            template="fallback",
        ),
        limits=PipelineLimits(max_llm_calls=1),
    )
    session = _session(pipeline)
    session.record_llm_call()
    model = _FakeModel()

    summary = await OrchidPipelineInterpreter(pipeline, session, model_factory=lambda ref: model).interpret(
        output={},
        status="succeeded",
        graph_stats=session.stats(),
    )

    assert summary.source == "template"
    assert summary.text == "fallback"
    assert summary.metadata["interpretation_fallback"] == "llm_calls"
    assert model.calls == []
    assert session.llm_calls == 1


async def test_llm_prompt_respects_the_prompt_cap() -> None:
    pipeline = _pipeline(
        interpretation=OrchidPipelineInterpretationConfig(
            mode="llm",
            model="test/model",
            template="fallback",
        ),
        limits=PipelineLimits(max_prompt_chars=64),
    )
    session = _session(pipeline)
    model = _FakeModel()

    summary = await OrchidPipelineInterpreter(pipeline, session, model_factory=lambda ref: model).interpret(
        output={"headline": "growth"},
        status="succeeded",
        graph_stats=session.stats(),
    )

    assert summary.source == "template"
    assert summary.text == "fallback"
    assert summary.metadata["interpretation_fallback"] == "prompt_cap"
    assert model.calls == []


async def test_llm_prompt_excludes_the_reserved_input_object() -> None:
    pipeline = _pipeline(
        interpretation=OrchidPipelineInterpretationConfig(mode="llm", model="test/model"),
        jobs=(_job("load"),),
    )
    session = _session(
        pipeline,
        objects=(OrchidGraphObject(id="account:a1", type="Account", data={"name": "Acme"}),),
    )
    session.graph.set_input({"account_id": "a1"})
    await _record(session, "load")
    model = _FakeModel()

    await OrchidPipelineInterpreter(pipeline, session, model_factory=lambda ref: model).interpret(
        output={"headline": "growth"},
        status="succeeded",
        graph_stats=session.stats(),
    )

    prompt = model.calls[0][0].content
    assert "__input__" not in prompt
    assert "__pipeline_input__" not in prompt


async def test_llm_prompt_is_bounded_and_excludes_the_full_graph() -> None:
    pipeline = _pipeline(
        interpretation=OrchidPipelineInterpretationConfig(
            mode="llm",
            model="test/model",
            max_objects=1,
        ),
        jobs=(_job("load"),),
        limits=PipelineLimits(max_prompt_chars=4000),
    )
    session = _session(
        pipeline,
        objects=(
            OrchidGraphObject(
                id="account:a1",
                type="Account",
                data={
                    "name": "Acme",
                    "status": "active",
                    "score": 42,
                    "huge": "x" * 1000,
                    "secret_extra": "do-not-send",
                },
            ),
            OrchidGraphObject(
                id="account:a2",
                type="Account",
                data={"name": "Beta", "status": "paused", "score": 1},
            ),
        ),
    )
    await _record(session, "load")
    model = _FakeModel(content="summary")

    await OrchidPipelineInterpreter(pipeline, session, model_factory=lambda ref: model).interpret(
        output={"headline": "growth"},
        status="succeeded",
        graph_stats=session.stats(),
    )

    prompt = model.calls[0][0].content
    assert len(prompt) <= 4000
    assert "account:a1" in prompt
    assert "account:a2" not in prompt  # max_objects=1
    assert "do-not-send" not in prompt  # only the first 3-5 shape fields
    assert "tenant-secret" not in prompt
    assert "user-secret" not in prompt
    assert "super-secret" not in prompt


async def test_llm_without_a_model_falls_back() -> None:
    pipeline = _pipeline()
    pipeline.definition.interpretation = OrchidPipelineInterpretationConfig.model_construct(
        mode="llm",
        model=None,
        template="fallback",
        max_objects=20,
    )
    session = _session(pipeline)

    def forbidden_factory(ref: str) -> Any:
        raise AssertionError("no model should be built without a model reference")

    summary = await OrchidPipelineInterpreter(pipeline, session, model_factory=forbidden_factory).interpret(
        output={},
        status="succeeded",
        graph_stats=session.stats(),
    )

    assert summary.source == "template"
    assert summary.metadata["interpretation_fallback"] == "no_model"


async def test_llm_model_build_failure_falls_back() -> None:
    pipeline = _pipeline(
        interpretation=OrchidPipelineInterpretationConfig(
            mode="llm",
            model="test/model",
            template="fallback",
        )
    )
    session = _session(pipeline)

    def failing_factory(ref: str) -> Any:
        raise RuntimeError("secret sk-test")

    summary = await OrchidPipelineInterpreter(pipeline, session, model_factory=failing_factory).interpret(
        output={},
        status="succeeded",
        graph_stats=session.stats(),
    )

    assert summary.source == "template"
    assert summary.metadata["interpretation_fallback"] == "model_unavailable"
    assert session.llm_calls == 0


async def test_llm_empty_response_falls_back() -> None:
    pipeline = _pipeline(
        interpretation=OrchidPipelineInterpretationConfig(
            mode="llm",
            model="test/model",
            template="fallback",
        )
    )
    session = _session(pipeline)
    model = _FakeModel(content=" ")

    summary = await OrchidPipelineInterpreter(pipeline, session, model_factory=lambda ref: model).interpret(
        output={},
        status="succeeded",
        graph_stats=session.stats(),
    )

    assert summary.source == "template"
    assert summary.metadata["interpretation_fallback"] == "empty_response"
    assert session.llm_calls == 1


async def test_llm_fallback_template_sees_the_counted_attempt() -> None:
    pipeline = _pipeline(
        interpretation=OrchidPipelineInterpretationConfig(
            mode="llm",
            model="test/model",
            template="fallback {{ graph_stats.llm_calls }}",
        )
    )
    session = _session(pipeline)
    model = _FakeModel(error=RuntimeError("boom"))

    summary = await OrchidPipelineInterpreter(pipeline, session, model_factory=lambda ref: model).interpret(
        output={},
        status="succeeded",
        graph_stats=session.stats(),
    )

    assert summary.source == "template"
    assert summary.text == "fallback 1"


async def test_llm_response_content_parts_are_joined() -> None:
    class _PartModel:
        async def ainvoke(self, messages: list[Any]) -> Any:
            from types import SimpleNamespace

            return SimpleNamespace(
                content=[{"type": "text", "text": "part one, "}, {"type": "text", "text": "part two"}]
            )

    pipeline = _pipeline(interpretation=OrchidPipelineInterpretationConfig(mode="llm", model="test/model"))
    session = _session(pipeline)

    summary = await OrchidPipelineInterpreter(pipeline, session, model_factory=lambda ref: _PartModel()).interpret(
        output={}, status="succeeded", graph_stats=session.stats()
    )

    assert summary.text == "part one, part two"


async def test_long_output_is_bounded_in_the_prompt() -> None:
    pipeline = _pipeline(
        interpretation=OrchidPipelineInterpretationConfig(mode="llm", model="test/model"),
        limits=PipelineLimits(max_prompt_chars=4000),
    )
    session = _session(pipeline)
    model = _FakeModel()

    await OrchidPipelineInterpreter(pipeline, session, model_factory=lambda ref: model).interpret(
        output={"blob": "x" * 5000},
        status="succeeded",
        graph_stats=session.stats(),
    )

    prompt = model.calls[0][0].content
    assert len(prompt) <= 4000
    assert "truncated" in prompt


async def test_record_llm_call_validates_the_count() -> None:
    pipeline = _pipeline()
    session = _session(pipeline)

    session.record_llm_call(2)
    assert session.llm_calls == 2
    assert session.stats()["llm_calls"] == 2

    with pytest.raises(ValueError):
        session.record_llm_call(0)
    with pytest.raises(ValueError):
        session.record_llm_call(True)


def test_interpreter_requires_a_real_session() -> None:
    pipeline = _pipeline()

    with pytest.raises(TypeError, match="PipelineRunSession"):
        OrchidPipelineInterpreter(pipeline, object())  # type: ignore[arg-type]


def test_interpret_rejects_an_unknown_mode() -> None:
    pipeline = _pipeline()
    session = _session(pipeline)

    with pytest.raises(ValueError, match="unsupported interpretation mode"):
        asyncio.run(
            OrchidPipelineInterpreter(pipeline, session).interpret(
                output={},
                status="succeeded",
                graph_stats=session.stats(),
                mode="nope",  # type: ignore[arg-type]
            )
        )
