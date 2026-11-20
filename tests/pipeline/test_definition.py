"""Tests for the pipeline definition compiler (``orchid_ai.pipeline.definition``)."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from orchid_ai.config.schema_pipeline import (
    OrchidPipelineAgentJobConfig,
    OrchidPipelineCustomJobConfig,
    OrchidPipelineDecisionJobConfig,
    OrchidPipelineDefinitionConfig,
    OrchidPipelineDefinitionLimitsConfig,
    OrchidPipelineDeterministicJobConfig,
    OrchidPipelineExposeConfig,
    OrchidPipelineInputConfig,
    OrchidPipelineLimitsConfig,
    OrchidPipelineLLMJobConfig,
    OrchidPipelineMapJobConfig,
    OrchidPipelineObjectWriteConfig,
    OrchidPipelineOutputConfig,
    OrchidPipelineRetryConfig,
    OrchidPipelinesConfig,
    OrchidPipelineSubPipelineJobConfig,
)
from orchid_ai.core.pipeline import OrchidPipelineConfigError, OrchidPipelineStep, OrchidPipelineStepResult
from orchid_ai.pipeline.definition import (
    CompiledJob,
    CompiledPipeline,
    PipelineDependencies,
    PipelineLimits,
    compile_pipeline,
    compile_pipelines,
)


class _FakeStep(OrchidPipelineStep):
    kind = "fake"

    async def execute(self, ctx: Any) -> OrchidPipelineStepResult:  # pragma: no cover - never executed
        return OrchidPipelineStepResult(output={})


class _NeedsArgumentsStep(OrchidPipelineStep):
    kind = "needs_arguments"

    def __init__(self, required: str) -> None:
        self.required = required

    async def execute(self, ctx: Any) -> OrchidPipelineStepResult:  # pragma: no cover - never executed
        return OrchidPipelineStepResult(output={})


class _NotAStep:
    pass


def echo(value: Any = None) -> dict[str, Any]:
    """A module-level function a deterministic job can point at."""
    return {"value": value}


@pytest.fixture
def fakes_module(monkeypatch: pytest.MonkeyPatch) -> str:
    module_name = "pipeline_definition_fakes"
    module = ModuleType(module_name)
    module.echo = echo  # type: ignore[attr-defined]
    module.FakeStep = _FakeStep  # type: ignore[attr-defined]
    module.NeedsArgumentsStep = _NeedsArgumentsStep  # type: ignore[attr-defined]
    module.NotAStep = _NotAStep  # type: ignore[attr-defined]
    module.NOT_CALLABLE = 42  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, module)
    return module_name


def _deterministic(**overrides: Any) -> OrchidPipelineDeterministicJobConfig:
    values: dict[str, Any] = {"kind": "deterministic", "uses": "pipeline_definition_fakes.echo"}
    values.update(overrides)
    return OrchidPipelineDeterministicJobConfig(**values)


def _definition(**overrides: Any) -> OrchidPipelineDefinitionConfig:
    values: dict[str, Any] = {"jobs": {"run": _deterministic()}}
    values.update(overrides)
    return OrchidPipelineDefinitionConfig(**values)


def _definition_with(job: Any, **overrides: Any) -> OrchidPipelineDefinitionConfig:
    """Build a valid definition, then swap in a job without re-validation."""
    definition = _definition(**overrides)
    _invalidate(definition, jobs={"run": job})
    return definition


def _block(**overrides: Any) -> OrchidPipelinesConfig:
    values: dict[str, Any] = {"definitions": {"demo": _definition()}}
    values.update(overrides)
    return OrchidPipelinesConfig(**values)


def _decision(**overrides: Any) -> OrchidPipelineDecisionJobConfig:
    values: dict[str, Any] = {
        "kind": "decision",
        "decision_model": "triage",
        "question": "Which?",
        "labels": ["retain", "expand"],
        "branches": {"retain": "run"},
        "fallback": None,
    }
    values.update(overrides)
    return OrchidPipelineDecisionJobConfig(**values)


def _invalidate(model: Any, **fields: Any) -> Any:
    """Set fields on a validated model without re-running its validators."""
    for name, value in fields.items():
        setattr(model, name, value)
    return model


# ── Limits ───────────────────────────────────────────────────


def test_limits_default_to_the_global_caps() -> None:
    limits = PipelineLimits.from_config()

    assert limits.max_jobs == OrchidPipelineLimitsConfig().max_jobs
    assert limits.timeout_seconds is None


def test_definition_overrides_only_tighten() -> None:
    limits = PipelineLimits.from_config(
        OrchidPipelineLimitsConfig(max_jobs=10, max_objects=100),
        OrchidPipelineDefinitionLimitsConfig(max_jobs=4, max_objects=500),
    )

    assert limits.max_jobs == 4
    # A larger override is clamped defensively even though config validation
    # rejects it before it reaches the compiler.
    assert limits.max_objects == 100


def test_limits_to_dict_lists_every_cap() -> None:
    payload = PipelineLimits.from_config().to_dict()

    assert set(payload) == {
        "timeout_seconds",
        "max_jobs",
        "max_objects",
        "max_relations",
        "max_object_bytes",
        "max_graph_bytes",
        "max_llm_calls",
        "max_prompt_chars",
        "max_context_chars",
        "max_map_items",
        "max_parallel",
        "max_subpipeline_depth",
    }


# ── Deterministic and custom jobs ────────────────────────────


def test_deterministic_job_resolves_the_callable(fakes_module: str) -> None:
    compiled = compile_pipeline("demo", _definition())

    job = compiled.jobs["run"]
    assert job.kind == "deterministic"
    assert job.handler is echo
    assert compiled.enabled is True
    assert compiled.ordered_jobs == (job,)


def test_deterministic_import_failure_names_the_path() -> None:
    definition = _definition(jobs={"run": _deterministic(uses="pipeline_definition_fakes.missing")})

    with pytest.raises(OrchidPipelineConfigError, match="cannot import"):
        compile_pipeline("demo", definition)


def test_non_callable_uses_is_rejected(fakes_module: str) -> None:
    definition = _definition(jobs={"run": _deterministic(uses="pipeline_definition_fakes.NOT_CALLABLE")})

    with pytest.raises(OrchidPipelineConfigError, match="is not callable"):
        compile_pipeline("demo", definition)


def test_custom_job_builds_one_shared_step_instance(fakes_module: str) -> None:
    job = OrchidPipelineCustomJobConfig.model_validate(
        {"kind": "custom", "class": "pipeline_definition_fakes.FakeStep"}
    )

    compiled = compile_pipeline("demo", _definition(jobs={"run": job}))

    assert isinstance(compiled.jobs["run"].handler, _FakeStep)


def test_custom_job_must_subclass_the_step_abc(fakes_module: str) -> None:
    job = OrchidPipelineCustomJobConfig.model_validate(
        {"kind": "custom", "class": "pipeline_definition_fakes.NotAStep"}
    )

    with pytest.raises(OrchidPipelineConfigError, match="must subclass OrchidPipelineStep"):
        compile_pipeline("demo", _definition(jobs={"run": job}))


def test_custom_job_must_be_instantiable_without_arguments(fakes_module: str) -> None:
    job = OrchidPipelineCustomJobConfig.model_validate(
        {"kind": "custom", "class": "pipeline_definition_fakes.NeedsArgumentsStep"}
    )

    with pytest.raises(OrchidPipelineConfigError, match="cannot be instantiated"):
        compile_pipeline("demo", _definition(jobs={"run": job}))


def test_params_and_aliases_are_exposed(fakes_module: str) -> None:
    job = _deterministic(
        params={"value": 1},
        writes=[{"id": "account:1", "shape": "Account", "as": "account"}],
    )
    definition = _definition(jobs={"run": job})

    compiled = compile_pipeline("demo", definition, pipelines=_block(shapes=_shapes()))

    assert compiled.jobs["run"].params == {"value": 1}
    assert compiled.jobs["run"].object_aliases == ("account",)


# ── Defaults inheritance ─────────────────────────────────────


def test_defaults_fill_retry_and_timeout(fakes_module: str) -> None:
    block = _block(defaults={"retry": {"max": 3}, "timeout_seconds": 12.5})

    compiled = compile_pipeline("demo", block.definitions["demo"], pipelines=block)

    assert compiled.jobs["run"].retry.max == 3
    assert compiled.jobs["run"].timeout_seconds == 12.5


def test_job_values_win_over_defaults(fakes_module: str) -> None:
    block = _block(defaults={"retry": {"max": 3}})
    block.definitions["demo"].jobs["run"] = _deterministic(retry=OrchidPipelineRetryConfig(max=1))

    compiled = compile_pipeline("demo", block.definitions["demo"], pipelines=block)

    assert compiled.jobs["run"].retry.max == 1


# ── Defensive reference validation ───────────────────────────


def test_unknown_dependency_is_rejected(fakes_module: str) -> None:
    definition = _definition(jobs={"run": _deterministic(depends_on=["nope"])})

    with pytest.raises(OrchidPipelineConfigError, match="depends on unknown job 'nope'"):
        compile_pipeline("demo", definition)


def test_self_dependency_is_rejected(fakes_module: str) -> None:
    definition = _definition(jobs={"run": _deterministic(depends_on=["run"])})

    with pytest.raises(OrchidPipelineConfigError, match="must not depend on itself"):
        compile_pipeline("demo", definition)


def test_map_template_must_exist(fakes_module: str) -> None:
    definition = _definition(
        jobs={
            "run": _deterministic(),
            "fan": OrchidPipelineMapJobConfig.model_validate(
                {"kind": "map", "over": "input.items", "as": "item", "job": "nope"}
            ),
        }
    )

    with pytest.raises(OrchidPipelineConfigError, match="maps over unknown template job 'nope'"):
        compile_pipeline("demo", definition)


def test_map_template_must_not_be_a_map(fakes_module: str) -> None:
    definition = _definition(
        jobs={
            "run": _deterministic(),
            "one": OrchidPipelineMapJobConfig.model_validate(
                {"kind": "map", "over": "input.items", "as": "i", "job": "inner"}
            ),
            "inner": OrchidPipelineMapJobConfig.model_validate(
                {"kind": "map", "over": "input.items", "as": "j", "job": "run"}
            ),
        }
    )

    with pytest.raises(OrchidPipelineConfigError, match="which is itself a map job"):
        compile_pipeline("demo", definition)


def test_map_template_must_not_be_a_branching_decision(fakes_module: str) -> None:
    definition = _definition(
        jobs={
            "run": _deterministic(),
            "fan": OrchidPipelineMapJobConfig.model_validate(
                {"kind": "map", "over": "input.items", "as": "item", "job": "choose"}
            ),
            "choose": _decision(),
        }
    )

    with pytest.raises(OrchidPipelineConfigError, match="branching decision job"):
        compile_pipeline("demo", definition)


def test_decision_branch_target_must_exist(fakes_module: str) -> None:
    decision = _invalidate(_decision(), branches={"retain": "run", "expand": "nope"})
    definition = _definition(jobs={"run": _deterministic(), "choose": decision})

    with pytest.raises(OrchidPipelineConfigError, match="targets unknown job 'nope'"):
        compile_pipeline("demo", definition)


def test_decision_fallback_must_be_a_label_or_job(fakes_module: str) -> None:
    decision = _invalidate(_decision(), fallback="nope")

    with pytest.raises(OrchidPipelineConfigError, match="neither a label nor a job id"):
        compile_pipeline("demo", _definition(jobs={"run": _deterministic(), "choose": decision}))


def test_decision_fallback_label_is_accepted(fakes_module: str) -> None:
    decision = _invalidate(_decision(), fallback="expand")
    compiled = compile_pipeline("demo", _definition(jobs={"run": _deterministic(), "choose": decision}))

    assert compiled.jobs["choose"].fallback == "expand"


def test_subpipeline_target_must_exist(fakes_module: str) -> None:
    job = OrchidPipelineSubPipelineJobConfig.model_validate({"kind": "subpipeline", "pipeline": "missing"})

    with pytest.raises(OrchidPipelineConfigError, match="unknown sub-pipeline 'missing'"):
        compile_pipeline("demo", _definition(jobs={"run": job}))


def test_disabled_subpipeline_is_rejected(fakes_module: str) -> None:
    block = _block()
    block.definitions["demo"].jobs["run"] = OrchidPipelineSubPipelineJobConfig.model_validate(
        {"kind": "subpipeline", "pipeline": "child"}
    )
    block.definitions["child"] = OrchidPipelineDefinitionConfig(enabled=False, jobs={"run": _deterministic()})

    with pytest.raises(OrchidPipelineConfigError, match="references disabled sub-pipeline 'child'"):
        compile_pipeline("demo", block.definitions["demo"], pipelines=block)


def test_duplicate_aliases_across_jobs_are_rejected(fakes_module: str) -> None:
    first = _deterministic(writes=[{"id": "account:1", "shape": "Account", "as": "account"}])
    second = _deterministic(writes=[{"id": "account:2", "shape": "Account", "as": "account"}])

    with pytest.raises(OrchidPipelineConfigError, match="not mutually exclusive"):
        compile_pipeline("demo", _definition(jobs={"one": first, "two": second}))


def test_dependency_references_are_checked_when_maps_are_supplied(fakes_module: str) -> None:
    job = OrchidPipelineAgentJobConfig.model_validate({"kind": "agent", "agent": "missing", "prompt": "Go"})

    with pytest.raises(OrchidPipelineConfigError, match="unknown agent 'missing'"):
        compile_pipeline("demo", _definition(jobs={"run": job}), deps=PipelineDependencies(agents={}))


def test_decision_model_is_checked_when_a_map_is_supplied(fakes_module: str) -> None:
    with pytest.raises(OrchidPipelineConfigError, match="unknown decision model 'triage'"):
        compile_pipeline(
            "demo",
            _definition(jobs={"run": _deterministic(), "choose": _decision()}),
            deps=PipelineDependencies(decision_models={}),
        )


def test_tool_references_are_checked_when_a_tool_map_is_supplied(fakes_module: str) -> None:
    job = OrchidPipelineLLMJobConfig.model_validate(
        {"kind": "llm", "model": "m", "prompt": "Go", "tools": ["missing_tool"]}
    )

    with pytest.raises(OrchidPipelineConfigError, match="unknown tool 'missing_tool'"):
        compile_pipeline("demo", _definition(jobs={"run": job}), deps=PipelineDependencies(tools={}))


def test_references_are_not_checked_without_dependency_maps(fakes_module: str) -> None:
    definition = _definition(
        jobs={"run": OrchidPipelineAgentJobConfig.model_validate({"kind": "agent", "agent": "x", "prompt": "Go"})}
    )

    compiled = compile_pipeline("demo", definition)

    assert compiled.jobs["run"].kind == "agent"


# ── Build-time shape knowledge ───────────────────────────────


def _shapes() -> dict[str, Any]:
    return {
        "Account": {"fields": {"id": {"type": "string", "required": True}}},
        "Out": {"fields": {"summary": {"type": "string"}}},
    }


def test_output_mapping_keys_must_be_shape_fields(fakes_module: str) -> None:
    block = _block(shapes=_shapes())
    block.definitions["demo"].output = OrchidPipelineOutputConfig(
        shape="Out",
        mapping={"summary": "summary", "extra": "input.x"},
    )

    with pytest.raises(OrchidPipelineConfigError, match="output mapping references unknown field 'extra'"):
        compile_pipeline("demo", block.definitions["demo"], pipelines=block)


def test_input_default_keys_must_be_shape_fields(fakes_module: str) -> None:
    block = _block(shapes=_shapes())
    block.definitions["demo"].input = OrchidPipelineInputConfig(shape="Account", defaults={"extra": 1})

    with pytest.raises(OrchidPipelineConfigError, match="input defaults reference unknown field 'extra'"):
        compile_pipeline("demo", block.definitions["demo"], pipelines=block)


def test_valid_output_mapping_compiles(fakes_module: str) -> None:
    block = _block(shapes=_shapes())
    block.definitions["demo"].output = OrchidPipelineOutputConfig(shape="Out", mapping={"summary": "summary"})

    compiled = compile_pipeline("demo", block.definitions["demo"], pipelines=block)

    assert compiled.output_shape is not None
    assert compiled.output_shape.name == "Out"


def test_summary_output_mapping_requires_a_deterministic_mode(fakes_module: str) -> None:
    block = _block(shapes=_shapes())
    block.definitions["demo"].interpretation.mode = "template"
    block.definitions["demo"].interpretation.template = "{{ graph_stats.jobs }}"
    block.definitions["demo"].output = OrchidPipelineOutputConfig(shape="Out", mapping={"summary": "summary"})

    with pytest.raises(OrchidPipelineConfigError, match="references 'summary'"):
        compile_pipeline("demo", block.definitions["demo"], pipelines=block)


def test_summary_reference_guard_uses_the_jmespath_tree() -> None:
    from orchid_ai.pipeline.mapping import references_context_key

    for expression in (
        "summary",
        "{{ summary }}",
        "{{  summary }}",
        "length(summary)",
        "{summary: summary}",
        "@.summary",
        "{summary: @.summary}",
        "@ | summary",
        "length(@.summary)",
    ):
        assert references_context_key(expression, "summary"), expression
    for expression in (
        "summary_text",
        r"\{{ summary }}",
        "jobs.summary",
        "items[*].summary",
        "&summary",
        "foo | summary",
        "jobs.load.output | summary",
        "@.foo.summary",
    ):
        assert not references_context_key(expression, "summary"), expression


def test_summary_from_reference_requires_a_deterministic_mode(fakes_module: str) -> None:
    block = _block(shapes=_shapes())
    block.definitions["demo"].interpretation.mode = "llm"
    block.definitions["demo"].interpretation.model = "test/model"
    block.definitions["demo"].output = OrchidPipelineOutputConfig(
        shape="Out",
        from_="{summary: @.summary}",
    )

    with pytest.raises(OrchidPipelineConfigError, match="output 'from' references 'summary'"):
        compile_pipeline("demo", block.definitions["demo"], pipelines=block)


def test_summary_at_mapping_requires_a_deterministic_mode(fakes_module: str) -> None:
    block = _block(shapes=_shapes())
    block.definitions["demo"].interpretation.mode = "template"
    block.definitions["demo"].interpretation.template = "{{ graph_stats.jobs }}"
    block.definitions["demo"].output = OrchidPipelineOutputConfig(
        shape="Out",
        mapping={"summary": "@.summary"},
    )

    with pytest.raises(OrchidPipelineConfigError, match="output mapping field 'summary' references 'summary'"):
        compile_pipeline("demo", block.definitions["demo"], pipelines=block)


def test_piped_nested_summary_is_not_rejected(fakes_module: str) -> None:
    block = _block(shapes=_shapes())
    block.definitions["demo"].interpretation.mode = "template"
    block.definitions["demo"].interpretation.template = "{{ graph_stats.jobs }}"
    block.definitions["demo"].output = OrchidPipelineOutputConfig(
        shape="Out",
        mapping={"summary": "jobs.load.output | summary"},
    )

    compiled = compile_pipeline("demo", block.definitions["demo"], pipelines=block)

    assert compiled.output_shape is not None


def test_escaped_summary_literal_is_not_rejected(fakes_module: str) -> None:
    block = _block(shapes=_shapes())
    block.definitions["demo"].interpretation.mode = "template"
    block.definitions["demo"].interpretation.template = "{{ graph_stats.jobs }}"
    block.definitions["demo"].output = OrchidPipelineOutputConfig(
        shape="Out",
        mapping={"summary": r"\{{ summary }}"},
    )

    compiled = compile_pipeline("demo", block.definitions["demo"], pipelines=block)

    assert compiled.output_shape is not None


def test_summary_output_mapping_compiles_for_caller_mode(fakes_module: str) -> None:
    block = _block(shapes=_shapes())
    block.definitions["demo"].interpretation.mode = "caller"
    block.definitions["demo"].output = OrchidPipelineOutputConfig(shape="Out", mapping={"summary": "summary"})

    compiled = compile_pipeline("demo", block.definitions["demo"], pipelines=block)

    assert compiled.output_shape is not None


def test_invalid_prompt_template_is_rejected_at_build_time(fakes_module: str) -> None:
    job = _invalidate(
        OrchidPipelineLLMJobConfig.model_validate({"kind": "llm", "model": "m", "prompt": "Go"}),
        prompt="{{ input.x + 1 }}",
    )

    with pytest.raises(OrchidPipelineConfigError, match="not a valid JMESPath expression"):
        compile_pipeline("demo", _definition(jobs={"run": job}))


def test_invalid_write_expression_is_rejected_at_build_time(fakes_module: str) -> None:
    write = _invalidate(
        OrchidPipelineObjectWriteConfig.model_validate({"id": "account:1", "shape": "Account"}),
        from_="input.x + 1",
    )
    job = _invalidate(_deterministic(), writes=[write])

    with pytest.raises(OrchidPipelineConfigError, match="not a valid JMESPath expression"):
        compile_pipeline("demo", _definition(jobs={"run": job}))


def test_unknown_job_output_shape_is_rejected(fakes_module: str) -> None:
    job = OrchidPipelineLLMJobConfig.model_validate(
        {"kind": "llm", "model": "m", "prompt": "Go", "output": {"shape": "Nope"}}
    )
    block = _block(shapes=_shapes())

    with pytest.raises(OrchidPipelineConfigError, match="references unknown shape 'Nope'"):
        compile_pipeline("demo", _definition(jobs={"run": job}), pipelines=block)


# ── Compilation entry points ─────────────────────────────────


def test_compile_pipelines_skips_disabled_items(fakes_module: str) -> None:
    block = _block()

    assert compile_pipelines(None) == {}
    assert compile_pipelines(OrchidPipelinesConfig(enabled=False, definitions=block.definitions)) == {}

    block.definitions["off"] = OrchidPipelineDefinitionConfig(enabled=False, jobs={"run": _deterministic()})

    assert set(compile_pipelines(block)) == {"demo"}


def test_compile_pipelines_shares_the_shape_registry(fakes_module: str) -> None:
    block = _block(shapes=_shapes())

    compiled = compile_pipelines(block)

    assert compiled["demo"].shapes is not None
    assert compiled["demo"].relations == {}


def test_compiled_graph_slot_is_set_once() -> None:
    compiled = _minimal_pipeline()

    compiled.set_graph("graph")

    assert compiled.graph == "graph"
    with pytest.raises(OrchidPipelineConfigError, match="already set"):
        compiled.set_graph("other")


def test_compile_requires_a_name() -> None:
    with pytest.raises(OrchidPipelineConfigError, match="non-empty"):
        compile_pipeline("", _definition())


def test_expose_and_enabled_properties() -> None:
    definition = _definition(expose=OrchidPipelineExposeConfig(as_tool=True, description="Run the demo"))
    compiled = CompiledPipeline(
        name="demo",
        version="1",
        description="Demo",
        definition=definition,
        jobs={},
        limits=PipelineLimits.from_config(),
    )

    assert compiled.enabled is True
    assert compiled.exposed is True


def _minimal_pipeline() -> CompiledPipeline:
    job = CompiledJob(id="run", kind="deterministic", config=_deterministic())
    return CompiledPipeline(
        name="demo",
        version="1",
        description="",
        definition=_definition(),
        jobs={"run": job},
        limits=PipelineLimits.from_config(),
    )


def test_job_count_is_checked_against_the_global_cap(fakes_module: str) -> None:
    block = OrchidPipelinesConfig(limits={"max_jobs": 1})
    definition = _definition(jobs={"one": _deterministic(), "two": _deterministic()})

    with pytest.raises(OrchidPipelineConfigError, match="exceeding limits.max_jobs=1"):
        compile_pipeline("demo", definition, pipelines=block)


def test_agent_input_templates_are_revalidated(fakes_module: str) -> None:
    job = _invalidate(
        OrchidPipelineAgentJobConfig.model_validate({"kind": "agent", "agent": "a", "prompt": "Go"}),
        input={"account": "{{ input.x + 1 }}"},
    )

    with pytest.raises(OrchidPipelineConfigError, match="not a valid JMESPath expression"):
        compile_pipeline("demo", _definition_with(job))


def test_emit_payload_templates_are_revalidated(fakes_module: str) -> None:
    from orchid_ai.config.schema_pipeline import OrchidPipelineEmitJobConfig

    job = _invalidate(
        OrchidPipelineEmitJobConfig.model_validate({"kind": "emit", "signal": "report.ready"}),
        payload={"report_id": "{{ input.x + 1 }}"},
    )

    with pytest.raises(OrchidPipelineConfigError, match="not a valid JMESPath expression"):
        compile_pipeline("demo", _definition_with(job))


def test_link_data_templates_are_revalidated(fakes_module: str) -> None:
    from orchid_ai.config.schema_pipeline import OrchidPipelineRelationWriteConfig

    link = OrchidPipelineRelationWriteConfig.model_validate({"type": "has_account", "from": "a:1", "to": "b:1"})
    _invalidate(link, data={"note": "{{ input.x + 1 }}"})
    job = _invalidate(_deterministic(), links=[link])

    with pytest.raises(OrchidPipelineConfigError, match="not a valid JMESPath expression"):
        compile_pipeline("demo", _definition_with(job))


def test_dataclass_shape_fields_are_checked_at_build_time(fakes_module: str) -> None:
    from dataclasses import dataclass

    @dataclass
    class DataclassOut:
        summary: str

    class _Module:
        pass

    module = ModuleType("pipeline_definition_shape_fakes")
    module.DataclassOut = DataclassOut  # type: ignore[attr-defined]
    sys.modules["pipeline_definition_shape_fakes"] = module
    block = _block(shapes={"Out": {"class": "pipeline_definition_shape_fakes.DataclassOut"}})
    block.definitions["demo"].output = OrchidPipelineOutputConfig(shape="Out", mapping={"missing": "input.x"})

    with pytest.raises(OrchidPipelineConfigError, match="output mapping references unknown field 'missing'"):
        compile_pipeline("demo", block.definitions["demo"], pipelines=block)


# ── Review regressions (P-006 CR round 1) ───────────────────


def test_definition_timeout_without_a_global_cap_is_used() -> None:
    limits = PipelineLimits.from_config(
        OrchidPipelineLimitsConfig(timeout_seconds=None),
        OrchidPipelineDefinitionLimitsConfig(timeout_seconds=300.0),
    )

    assert limits.timeout_seconds == 300.0


def test_global_timeout_cap_applies_when_the_definition_has_none() -> None:
    limits = PipelineLimits.from_config(OrchidPipelineLimitsConfig(timeout_seconds=10.0))

    assert limits.timeout_seconds == 10.0


def test_inline_input_fields_compile_to_a_shape(fakes_module: str) -> None:
    definition = _definition(input={"fields": {"account_id": {"type": "string", "required": True}}})

    compiled = compile_pipeline("demo", definition)

    assert compiled.input_shape is not None
    assert compiled.input_shape.name == "demo_input"
    assert "account_id" in compiled.input_shape.json_schema()["properties"]


def test_inline_input_shape_rejects_an_invalid_value(fakes_module: str) -> None:
    from orchid_ai.core.pipeline import OrchidPipelineShapeError

    definition = _definition(input={"fields": {"account_id": {"type": "string", "required": True}}})
    compiled = compile_pipeline("demo", definition)

    assert compiled.input_shape is not None
    with pytest.raises(OrchidPipelineShapeError):
        compiled.input_shape.validate({"account_id": 3})


def test_inline_input_with_an_unknown_ref_is_a_config_error(fakes_module: str) -> None:
    definition = _definition(input={"fields": {"account": {"ref": "Missing"}}})

    with pytest.raises(OrchidPipelineConfigError, match="cannot be compiled"):
        compile_pipeline("demo", definition)


def test_duplicate_aliases_on_mutually_exclusive_branches_are_allowed(fakes_module: str) -> None:
    left = _deterministic(writes=[{"id": "account:left", "shape": "Account", "as": "shared"}])
    right = _deterministic(writes=[{"id": "account:right", "shape": "Account", "as": "shared"}])
    decision = _decision(branches={"retain": "left", "expand": "right"})
    definition = _definition(jobs={"choose": decision, "left": left, "right": right})

    compiled = compile_pipeline("demo", definition, pipelines=_block(shapes=_shapes()))

    assert set(compiled.jobs) == {"choose", "left", "right"}


def test_programmatic_cycle_is_rejected(fakes_module: str) -> None:
    first = _deterministic(depends_on=["two"])
    second = _deterministic(depends_on=["one"])
    definition = _definition(jobs={"one": first, "two": second})

    with pytest.raises(OrchidPipelineConfigError, match="contains a cycle: one -> two -> one"):
        compile_pipeline("demo", definition)


def test_join_any_requires_two_dependencies(fakes_module: str) -> None:
    job = _invalidate(_deterministic(depends_on=["run"]), join="any")
    definition = _definition(jobs={"run": _deterministic(), "fan": job})

    with pytest.raises(OrchidPipelineConfigError, match="join: any but declares fewer than two"):
        compile_pipeline("demo", definition)


def test_parallel_static_writes_are_rejected(fakes_module: str) -> None:
    first = _deterministic(writes=[{"id": "account:1", "shape": "Account"}])
    second = _deterministic(writes=[{"id": "account:1", "shape": "Account"}])

    with pytest.raises(OrchidPipelineConfigError, match="both write static object 'account:1'"):
        compile_pipeline("demo", _definition(jobs={"one": first, "two": second}))


def test_parallel_static_writes_can_be_allowed(fakes_module: str) -> None:
    first = _deterministic(writes=[{"id": "account:1", "shape": "Account"}])
    second = _deterministic(writes=[{"id": "account:1", "shape": "Account"}])
    block = _block(allow_parallel_writes=True, shapes=_shapes())

    compiled = compile_pipeline(
        "demo",
        _definition(jobs={"one": first, "two": second}),
        pipelines=block,
    )

    assert set(compiled.jobs) == {"one", "two"}


def test_sequential_static_writes_are_allowed(fakes_module: str) -> None:
    first = _deterministic(writes=[{"id": "account:1", "shape": "Account"}])
    second = _deterministic(depends_on=["one"], writes=[{"id": "account:1", "shape": "Account"}])

    compiled = compile_pipeline(
        "demo",
        _definition(jobs={"one": first, "two": second}),
        pipelines=_block(shapes=_shapes()),
    )

    assert set(compiled.jobs) == {"one", "two"}


def test_write_shapes_must_be_declared_when_shapes_exist(fakes_module: str) -> None:
    job = _deterministic(writes=[{"id": "account:1", "shape": "Missing"}])

    with pytest.raises(OrchidPipelineConfigError, match="writes to unknown shape 'Missing'"):
        compile_pipeline("demo", _definition(jobs={"run": job}), pipelines=_block(shapes=_shapes()))


def test_link_relations_must_be_declared_when_relations_exist(fakes_module: str) -> None:
    job = _deterministic(links=[{"type": "has_missing", "from": "a:1", "to": "b:1"}])
    block = _block(shapes=_shapes(), relations={"has_account": {"from": ["Account"], "to": ["Out"]}})

    with pytest.raises(OrchidPipelineConfigError, match="links unknown relation 'has_missing'"):
        compile_pipeline("demo", _definition(jobs={"run": job}), pipelines=block)


def test_untyped_relations_can_be_allowed(fakes_module: str) -> None:
    job = _deterministic(links=[{"type": "has_missing", "from": "a:1", "to": "b:1"}])
    block = _block(allow_untyped_relations=True)

    compiled = compile_pipeline("demo", _definition(jobs={"run": job}), pipelines=block)

    assert compiled.jobs["run"].links


def test_inherited_timeout_is_clamped_to_the_hard_cap(fakes_module: str) -> None:
    block = _block(limits={"timeout_seconds": 10.0}, defaults={"timeout_seconds": 1000.0})

    compiled = compile_pipeline("demo", block.definitions["demo"], pipelines=block)

    assert compiled.jobs["run"].timeout_seconds == 10.0


def test_subpipeline_depth_is_rechecked(fakes_module: str) -> None:
    block = _block(limits={"max_subpipeline_depth": 1})

    def child(next_name: str | None) -> OrchidPipelineDefinitionConfig:
        job = (
            OrchidPipelineSubPipelineJobConfig.model_validate({"kind": "subpipeline", "pipeline": next_name})
            if next_name
            else _deterministic()
        )
        return OrchidPipelineDefinitionConfig(jobs={"run": job})

    block.definitions["demo"] = child("one")
    block.definitions["one"] = child("two")
    block.definitions["two"] = child(None)

    with pytest.raises(OrchidPipelineConfigError, match="exceeds max_subpipeline_depth=1"):
        compile_pipeline("demo", block.definitions["demo"], pipelines=block)


def test_subpipeline_input_templates_are_revalidated(fakes_module: str) -> None:
    job = _invalidate(
        OrchidPipelineSubPipelineJobConfig.model_validate({"kind": "subpipeline", "pipeline": "run"}),
        input={"account": "{{ input.x + 1 }}"},
    )
    definition = _definition_with(job)

    with pytest.raises(OrchidPipelineConfigError, match="not a valid JMESPath expression"):
        compile_pipeline("demo", definition)


def test_write_expected_revision_expression_is_revalidated(fakes_module: str) -> None:
    write = _invalidate(
        OrchidPipelineObjectWriteConfig.model_validate({"id": "account:1", "shape": "Account"}),
        expected_revision="input.x + 1",
    )
    job = _invalidate(_deterministic(), writes=[write])

    with pytest.raises(OrchidPipelineConfigError, match="not a valid JMESPath expression"):
        compile_pipeline("demo", _definition_with(job))


# ── Review regressions (P-006 CR round 2) ───────────────────


def test_fallback_target_is_exclusive_with_branch_targets(fakes_module: str) -> None:
    left = _deterministic(writes=[{"id": "account:left", "shape": "Account", "as": "shared"}])
    right = _deterministic(writes=[{"id": "account:right", "shape": "Account", "as": "shared"}])
    decision = _decision(branches={"retain": "left"}, fallback="right")
    definition = _definition(jobs={"choose": decision, "left": left, "right": right})

    compiled = compile_pipeline("demo", definition, pipelines=_block(shapes=_shapes()))

    assert set(compiled.jobs) == {"choose", "left", "right"}


def test_fallback_static_writes_can_share_an_id_with_a_branch_target(fakes_module: str) -> None:
    left = _deterministic(writes=[{"id": "account:1", "shape": "Account"}])
    right = _deterministic(writes=[{"id": "account:1", "shape": "Account"}])
    decision = _decision(branches={"retain": "left"}, fallback="right")

    compiled = compile_pipeline(
        "demo",
        _definition(jobs={"choose": decision, "left": left, "right": right}),
        pipelines=_block(shapes=_shapes()),
    )

    assert set(compiled.jobs) == {"choose", "left", "right"}


def test_subpipeline_depth_is_one_based(fakes_module: str) -> None:
    block = _block(limits={"max_subpipeline_depth": 1})
    block.definitions["demo"] = _definition(
        jobs={"run": OrchidPipelineSubPipelineJobConfig.model_validate({"kind": "subpipeline", "pipeline": "child"})}
    )
    block.definitions["child"] = _definition()

    with pytest.raises(OrchidPipelineConfigError, match="exceeds max_subpipeline_depth=1"):
        compile_pipeline("demo", block.definitions["demo"], pipelines=block)


def test_subpipeline_at_the_depth_limit_compiles(fakes_module: str) -> None:
    block = _block(limits={"max_subpipeline_depth": 2})
    block.definitions["demo"] = _definition(
        jobs={"run": OrchidPipelineSubPipelineJobConfig.model_validate({"kind": "subpipeline", "pipeline": "child"})}
    )
    block.definitions["child"] = _definition()

    compiled = compile_pipeline("demo", block.definitions["demo"], pipelines=block)

    assert compiled.jobs["run"].subpipeline == "child"


def test_deep_subpipeline_chain_respects_the_limit(fakes_module: str) -> None:
    block = _block(limits={"max_subpipeline_depth": 2})
    block.definitions["demo"] = _definition(
        jobs={"run": OrchidPipelineSubPipelineJobConfig.model_validate({"kind": "subpipeline", "pipeline": "one"})}
    )
    block.definitions["one"] = _definition(
        jobs={"run": OrchidPipelineSubPipelineJobConfig.model_validate({"kind": "subpipeline", "pipeline": "two"})}
    )
    block.definitions["two"] = _definition(
        jobs={"run": OrchidPipelineSubPipelineJobConfig.model_validate({"kind": "subpipeline", "pipeline": "three"})}
    )
    block.definitions["three"] = _definition()

    with pytest.raises(OrchidPipelineConfigError, match="exceeds max_subpipeline_depth=2"):
        compile_pipeline("demo", block.definitions["demo"], pipelines=block)


def test_programmatic_subpipeline_cycle_is_rejected(fakes_module: str) -> None:
    block = _block()
    block.definitions["demo"] = _definition(
        jobs={"run": OrchidPipelineSubPipelineJobConfig.model_validate({"kind": "subpipeline", "pipeline": "child"})}
    )
    block.definitions["child"] = _definition(
        jobs={"run": OrchidPipelineSubPipelineJobConfig.model_validate({"kind": "subpipeline", "pipeline": "demo"})}
    )

    with pytest.raises(OrchidPipelineConfigError, match="sub-pipeline cycle detected"):
        compile_pipeline("demo", block.definitions["demo"], pipelines=block)


def test_shape_reference_fails_when_no_shapes_are_declared(fakes_module: str) -> None:
    job = _deterministic(writes=[{"id": "account:1", "shape": "Account"}])

    with pytest.raises(OrchidPipelineConfigError, match="writes to unknown shape 'Account'"):
        compile_pipeline("demo", _definition(jobs={"run": job}))


def test_output_shape_fails_when_no_shapes_are_declared(fakes_module: str) -> None:
    from orchid_ai.config.schema_pipeline import OrchidPipelineOutputConfig

    definition = _definition(output=OrchidPipelineOutputConfig(shape="Out", mapping={"summary": "summary"}))

    with pytest.raises(OrchidPipelineConfigError, match="references unknown shape 'Out'"):
        compile_pipeline("demo", definition)


def test_named_input_shape_unknown_is_a_config_error(fakes_module: str) -> None:
    definition = _definition(input={"shape": "Missing"})

    with pytest.raises(OrchidPipelineConfigError, match="references unknown shape 'Missing'"):
        compile_pipeline("demo", definition)


def test_join_all_over_mutually_exclusive_branches_is_rejected(fakes_module: str) -> None:
    left = _deterministic()
    right = _deterministic()
    end = _deterministic(depends_on=["left", "right"])
    decision = _decision(branches={"retain": "left", "expand": "right"})
    definition = _definition(jobs={"choose": decision, "left": left, "right": right, "end": end})

    with pytest.raises(OrchidPipelineConfigError, match="uses join: all over mutually exclusive branches"):
        compile_pipeline("demo", definition, pipelines=_block(shapes=_shapes()))


def test_branch_target_must_not_list_the_decision(fakes_module: str) -> None:
    left = _deterministic(depends_on=["choose"])
    _invalidate(left, join="any")
    decision = _decision(branches={"retain": "left", "expand": "right"})
    right = _deterministic(depends_on=["choose"])

    with pytest.raises(OrchidPipelineConfigError, match="must not list decision job 'choose'"):
        compile_pipeline(
            "demo",
            _definition(jobs={"choose": decision, "left": left, "right": right}),
            pipelines=_block(shapes=_shapes()),
        )


def test_branch_target_feeding_join_all_is_rejected(fakes_module: str) -> None:
    left = _deterministic()
    right = _deterministic()
    end = _deterministic(depends_on=["left", "right"])
    decision = _decision(branches={"retain": "left", "expand": "right"})

    with pytest.raises(OrchidPipelineConfigError, match="must use join: any"):
        compile_pipeline(
            "demo",
            _definition(jobs={"choose": decision, "left": left, "right": right, "end": end}),
            pipelines=_block(shapes=_shapes()),
        )


def test_fallback_job_may_depend_on_its_decision(fakes_module: str) -> None:
    fallback = _deterministic(depends_on=["choose"])
    decision = _decision(branches={"retain": "left"}, fallback="fallback")
    left = _deterministic()
    definition = _definition(jobs={"choose": decision, "left": left, "fallback": fallback})

    compiled = compile_pipeline("demo", definition, pipelines=_block(shapes=_shapes()))

    assert compiled.jobs["fallback"].depends_on == ("choose",)


def test_fallback_feeding_a_plain_dependent_is_allowed(fakes_module: str) -> None:
    fallback = _deterministic()
    report = _deterministic(depends_on=["fallback"])
    decision = _decision(branches={"retain": "left"}, fallback="fallback")
    left = _deterministic()
    definition = _definition(jobs={"choose": decision, "left": left, "fallback": fallback, "report": report})

    compiled = compile_pipeline("demo", definition, pipelines=_block(shapes=_shapes()))

    assert compiled.jobs["report"].depends_on == ("fallback",)


# ── Review regressions (P-007 CR round 2) ──────────────────


def test_branch_target_with_an_unrelated_dependency_is_rejected_at_compile(fakes_module: str) -> None:
    decision = _decision(branches={"retain": "left"})
    left = _deterministic(depends_on=["helper"])
    helper = _deterministic()
    definition = _definition(jobs={"choose": decision, "left": left, "helper": helper})

    with pytest.raises(OrchidPipelineConfigError, match="must depend only on its decision job"):
        compile_pipeline("demo", definition, pipelines=_block(shapes=_shapes()))


def test_fallback_may_depend_on_its_decision_at_compile(fakes_module: str) -> None:
    decision = _decision(branches={"retain": "left"}, fallback="fallback")
    left = _deterministic()
    fallback = _deterministic(depends_on=["choose"])
    definition = _definition(jobs={"choose": decision, "left": left, "fallback": fallback})

    compiled = compile_pipeline("demo", definition, pipelines=_block(shapes=_shapes()))

    assert set(compiled.jobs) == {"choose", "left", "fallback"}


def test_no_branch_decision_with_a_job_fallback_compiles(fakes_module: str) -> None:
    decision = _decision(labels=["retain"], branches={}, fallback="fallback")
    fallback = _deterministic()
    definition = _definition(jobs={"choose": decision, "fallback": fallback})

    compiled = compile_pipeline("demo", definition, pipelines=_block(shapes=_shapes()))

    assert compiled.jobs["choose"].fallback == "fallback"
