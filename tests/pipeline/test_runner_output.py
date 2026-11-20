"""Runner result assembly: output projection, summary and graph opt-in."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from orchid_ai.config.schema_pipeline import (
    OrchidPipelineDefinitionConfig,
    OrchidPipelineDeterministicJobConfig,
    OrchidPipelineExposeConfig,
    OrchidPipelineInterpretationConfig,
    OrchidPipelineOutputConfig,
    OrchidPipelineShapeConfig,
)
from orchid_ai.core.pipeline import OrchidPipelineStep, OrchidPipelineStepResult
from orchid_ai.pipeline.definition import CompiledJob, CompiledPipeline, PipelineLimits
from orchid_ai.pipeline.runner import OrchidPipelineRunner
from orchid_ai.pipeline.schema import compile_shapes


def _shapes() -> Any:
    return compile_shapes(
        {
            "Input": OrchidPipelineShapeConfig(
                fields={"account_id": {"type": "string", "required": True}},
            ),
            "Account": OrchidPipelineShapeConfig(
                fields={"id": {"type": "string", "required": True}, "name": {"type": "string"}},
            ),
            "Report": OrchidPipelineShapeConfig(fields={"headline": {"type": "string"}}),
            "Projection": OrchidPipelineShapeConfig(
                fields={
                    "account_id": {"type": "string", "required": True},
                    "name": {"type": "string", "required": True},
                }
            ),
            "Brief": OrchidPipelineShapeConfig(fields={"headline": {"type": "string", "required": True}}),
            "SummaryOut": OrchidPipelineShapeConfig(fields={"summary": {"type": "string", "required": True}}),
        }
    )


def _job(job_id: str, handler: Any, *, writes: list[dict[str, Any]] | None = None) -> CompiledJob:
    config = OrchidPipelineDeterministicJobConfig.model_validate(
        {"kind": "deterministic", "uses": "pkg.step", "writes": writes or []}
    )
    return CompiledJob(
        id=job_id,
        kind="deterministic",
        config=config,
        handler=handler,
        writes=tuple(config.writes),
    )


def _pipeline(
    *jobs: CompiledJob,
    output: OrchidPipelineOutputConfig | None = None,
    interpretation: OrchidPipelineInterpretationConfig | None = None,
    expose: OrchidPipelineExposeConfig | None = None,
    limits: PipelineLimits | None = None,
) -> CompiledPipeline:
    shapes = _shapes()
    definition = OrchidPipelineDefinitionConfig(
        output=output,
        interpretation=interpretation or OrchidPipelineInterpretationConfig(),
        expose=expose or OrchidPipelineExposeConfig(),
    )
    return CompiledPipeline(
        name="demo",
        version="1",
        description="",
        definition=definition,
        jobs={job.id: job for job in jobs},
        limits=limits or PipelineLimits.from_config(),
        shapes=shapes,
        input_shape=shapes.get("Input"),
        output_shape=shapes.get(output.shape) if output is not None and output.shape is not None else None,
    )


def _run(pipeline: CompiledPipeline, **kwargs: Any) -> Any:
    return asyncio.run(OrchidPipelineRunner(pipeline).run({"account_id": "a1"}, user_id="u1", **kwargs))


def _account(ctx: Any) -> dict[str, Any]:
    return {"account": {"id": ctx.input["account_id"], "name": "Acme"}}


def _headline(ctx: Any) -> dict[str, Any]:
    return {"headline": "growth"}


# ── Output projection ──────────────────────────────────────


def test_mapping_projects_the_declared_shape() -> None:
    output = OrchidPipelineOutputConfig(
        shape="Projection",
        mapping={
            "account_id": "jobs.load.output.account.id",
            "name": "jobs.load.output.account.name",
        },
    )
    result = _run(_pipeline(_job("load", _account), output=output))

    assert result.status == "succeeded"
    assert result.output == {"account_id": "a1", "name": "Acme"}


def test_from_projects_a_single_object() -> None:
    output = OrchidPipelineOutputConfig(shape="Account", from_="jobs.load.output.account")
    result = _run(_pipeline(_job("load", _account), output=output))

    assert result.status == "succeeded"
    assert result.output == {"id": "a1", "name": "Acme"}


def test_mapping_template_values_are_rendered() -> None:
    output = OrchidPipelineOutputConfig(
        shape="Projection",
        mapping={
            "account_id": "jobs.load.output.account.id",
            "name": "{{ jobs.load.output.account.name }}",
        },
    )
    result = _run(_pipeline(_job("load", _account), output=output))

    assert result.output == {"account_id": "a1", "name": "Acme"}


def test_output_shape_failure_fails_the_run() -> None:
    output = OrchidPipelineOutputConfig(
        shape="Projection",
        mapping={"account_id": "jobs.load.output.account.id"},
    )
    result = _run(_pipeline(_job("load", _account), output=output))

    assert result.status == "failed"
    assert result.output is None
    assert result.metadata["error_code"] == "pipeline_output_error"
    assert "output projection failed" in (result.error or "")
    assert "shape 'Projection'" in (result.error or "")


def test_output_mapping_error_fails_the_run() -> None:
    output = OrchidPipelineOutputConfig(
        shape="Projection",
        mapping={
            "account_id": "jobs.missing.output.account.id",
            "name": "jobs.load.output.account.name",
        },
    )
    result = _run(_pipeline(_job("load", _account), output=output))

    assert result.status == "failed"
    assert result.metadata["error_code"] == "pipeline_output_error"
    assert result.output is None


# ── Graph snapshot / opt-in ────────────────────────────────


def test_no_output_block_returns_the_graph_snapshot() -> None:
    result = _run(_pipeline(_job("load", _account)))

    assert result.status == "succeeded"
    assert result.output["objects"]["__input__"]["type"] == "__pipeline_input__"
    assert result.graph == {}
    assert result.graph_stats["objects"] >= 1


def test_include_graph_returns_the_snapshot() -> None:
    result = _run(_pipeline(_job("load", _account)), include_graph=True)

    assert result.graph["objects"]["__input__"]["type"] == "__pipeline_input__"
    assert result.graph_stats["objects"] == len(result.graph["objects"])


def test_graph_is_omitted_over_the_size_cap() -> None:
    baseline = _run(_pipeline(_job("load", _account)), include_graph=True)
    # The serialized snapshot carries more metadata than the graph's own byte
    # counter, so a cap just above the used graph bytes still omits it.
    limits = PipelineLimits(max_graph_bytes=int(baseline.graph_stats["bytes"]) + 1)
    result = _run(_pipeline(_job("load", _account), limits=limits), include_graph=True)

    assert result.graph == {}
    assert result.metadata["graph_omitted"] == "size"
    assert result.graph_stats["objects"] >= 1


# ── Job metrics / interpretation ───────────────────────────


def test_job_report_keeps_metrics() -> None:
    job = _job(
        "report",
        _headline,
        writes=[{"id": "report:1", "shape": "Report"}],
    )
    result = _run(_pipeline(job))

    entry = result.jobs["report"]
    assert entry["status"] == "succeeded"
    assert isinstance(entry["duration_ms"], (int, float))
    assert entry["llm_calls"] == 0
    assert entry["objects_written"] == 1


def test_template_interpretation_is_attached_to_the_result() -> None:
    output = OrchidPipelineOutputConfig(shape="Brief", mapping={"headline": "jobs.report.output.headline"})
    interpretation = OrchidPipelineInterpretationConfig(
        mode="template",
        template="{{ output.headline }} / {{ graph_stats.objects }} objects / {{ run.id }}",
    )
    result = _run(_pipeline(_job("report", _headline), output=output, interpretation=interpretation))

    assert result.summary_source == "template"
    assert result.summary.startswith("growth / ")
    assert result.summary.endswith(f"/ {result.run_id}")


def test_interpretation_can_be_overridden_per_run() -> None:
    result = _run(_pipeline(_job("report", _headline)), interpretation="caller")

    assert result.summary_source == "caller"
    assert "Pipeline 'demo' succeeded." in result.summary


def test_chat_mode_defaults_to_the_caller_digest() -> None:
    result = _run(
        _pipeline(_job("report", _headline)),
        mode="chat",
        chat_id="c-1",
    )

    assert result.summary_source == "caller"
    assert "Pipeline 'demo' succeeded." in result.summary


def test_output_mapping_can_read_the_caller_summary() -> None:
    output = OrchidPipelineOutputConfig(shape="SummaryOut", mapping={"summary": "summary"})
    result = _run(
        _pipeline(
            _job("report", _headline),
            output=output,
            interpretation=OrchidPipelineInterpretationConfig(mode="caller"),
        )
    )

    assert result.status == "succeeded"
    assert result.output == {"summary": result.summary}
    assert result.summary_source == "caller"
    assert result.summary


def test_direct_runs_ignore_the_exposed_tool_interpretation() -> None:
    expose = OrchidPipelineExposeConfig(
        as_tool=True,
        description="Tool.",
        interpretation="caller",
    )
    result = _run(_pipeline(_job("report", _headline), expose=expose))

    assert result.summary_source == "none"
    assert result.summary == ""


def test_chat_runs_use_the_exposed_tool_interpretation() -> None:
    expose = OrchidPipelineExposeConfig(
        as_tool=True,
        description="Tool.",
        interpretation="caller",
    )
    result = _run(_pipeline(_job("report", _headline), expose=expose), mode="chat", chat_id="c-1")

    assert result.summary_source == "caller"


def test_graph_failure_preserves_the_error_and_skips_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingGraph:
        async def ainvoke(self, state: Any, config: Any) -> Any:
            raise RuntimeError("graph blew up")

    monkeypatch.setattr(
        "orchid_ai.pipeline.runner.build_pipeline_graph",
        lambda *args, **kwargs: _FailingGraph(),
    )
    output = OrchidPipelineOutputConfig(
        shape="Projection",
        mapping={"account_id": "input.account_id", "name": "{{ input.account_id }}"},
    )
    result = _run(_pipeline(_job("load", _account), output=output))

    assert result.status == "failed"
    assert result.output is None
    assert "graph blew up" in (result.error or "")
    assert result.metadata.get("error_code") != "pipeline_output_error"


def test_job_failure_skips_output_projection() -> None:
    def _boom(ctx: Any) -> dict[str, Any]:
        raise RuntimeError("job blew up")

    output = OrchidPipelineOutputConfig(
        shape="Projection",
        mapping={"account_id": "input.account_id", "name": "{{ input.account_id }}"},
    )
    result = _run(_pipeline(_job("load", _boom), output=output))

    assert result.status == "failed"
    assert result.output is None
    assert result.metadata.get("error_code") != "pipeline_output_error"


def test_programmatic_summary_mapping_is_rejected_for_template_mode() -> None:
    output = OrchidPipelineOutputConfig(shape="SummaryOut", mapping={"summary": "@.summary"})
    result = _run(_pipeline(_job("report", _headline), output=output), interpretation="template")

    assert result.status == "failed"
    assert result.output is None
    assert result.metadata["error_code"] == "pipeline_output_error"
    assert "interpretation mode 'template'" in (result.error or "")


def test_cancelled_run_skips_output_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    class _CancellingGraph:
        async def ainvoke(self, state: Any, config: Any) -> Any:
            config["configurable"]["pipeline_session"].cancel()
            return {}

    monkeypatch.setattr(
        "orchid_ai.pipeline.runner.build_pipeline_graph",
        lambda *args, **kwargs: _CancellingGraph(),
    )
    output = OrchidPipelineOutputConfig(
        shape="Projection",
        mapping={"account_id": "input.account_id", "name": "{{ input.account_id }}"},
    )
    result = _run(_pipeline(_job("load", _account), output=output))

    assert result.status == "cancelled"
    assert result.output is None
    assert "cancelled" in (result.error or "")


def test_no_output_llm_prompt_does_not_receive_the_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[Any]] = []

    class _Model:
        async def ainvoke(self, messages: list[Any]) -> Any:
            from langchain_core.messages import AIMessage

            calls.append(messages)
            return AIMessage(content="summary")

    monkeypatch.setattr(
        "orchid_ai.pipeline.interpretation._default_model_factory",
        lambda ref: _Model(),
    )
    result = _run(
        _pipeline(
            _job("load", _account),
            interpretation=OrchidPipelineInterpretationConfig(mode="llm", model="test/model"),
        )
    )

    prompt = calls[0][0].content
    assert result.output["objects"]["__input__"]["type"] == "__pipeline_input__"
    assert "__input__" not in prompt
    assert "__pipeline_input__" not in prompt
    assert '"relations":{' not in prompt


def test_llm_fallback_template_sees_the_counted_call(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Model:
        async def ainvoke(self, messages: list[Any]) -> Any:
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "orchid_ai.pipeline.interpretation._default_model_factory",
        lambda ref: _Model(),
    )
    result = _run(
        _pipeline(
            _job("report", _headline),
            interpretation=OrchidPipelineInterpretationConfig(
                mode="llm",
                model="test/model",
                template="fallback {{ graph_stats.llm_calls }}",
            ),
        )
    )

    assert result.summary == "fallback 1"
    assert result.summary_source == "template"
    assert result.graph_stats["llm_calls"] == 1


class _UsageStep(OrchidPipelineStep):
    async def execute(self, ctx: Any) -> OrchidPipelineStepResult:
        return OrchidPipelineStepResult(output={"headline": "growth"}, usage={"llm_calls": 1})


def test_interpretation_llm_call_counts_toward_the_run_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[Any]] = []

    class _Model:
        async def ainvoke(self, messages: list[Any]) -> Any:
            from langchain_core.messages import AIMessage

            calls.append(messages)
            return AIMessage(content="model summary")

    monkeypatch.setattr(
        "orchid_ai.pipeline.interpretation._default_model_factory",
        lambda ref: _Model(),
    )
    interpretation = OrchidPipelineInterpretationConfig(mode="llm", model="test/model")
    result = _run(_pipeline(_job("report", _headline), interpretation=interpretation))

    assert result.summary_source == "llm"
    assert result.summary == "model summary"
    assert result.graph_stats["llm_calls"] == 1
    assert calls


def test_interpretation_llm_is_skipped_at_the_run_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_factory(ref: str) -> Any:
        raise AssertionError("the interpretation model must not be built at the cap")

    monkeypatch.setattr("orchid_ai.pipeline.interpretation._default_model_factory", forbidden_factory)
    interpretation = OrchidPipelineInterpretationConfig(
        mode="llm",
        model="test/model",
        template="fallback",
    )
    limits = PipelineLimits(max_llm_calls=1)
    result = _run(_pipeline(_job("report", _UsageStep()), interpretation=interpretation, limits=limits))

    assert result.graph_stats["llm_calls"] == 1  # the job's call
    assert result.summary_source == "template"
    assert result.summary == "fallback"
    assert result.metadata["interpretation_fallback"] == "llm_calls"
