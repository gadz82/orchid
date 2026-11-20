"""Tests for the pipeline registry and step-kind extension point."""

from __future__ import annotations

from typing import Any

import pytest

from orchid_ai.config.schema_pipeline import (
    OrchidPipelineDefinitionConfig,
    OrchidPipelineDeterministicJobConfig,
    OrchidPipelineExposeConfig,
    OrchidPipelinesConfig,
    OrchidPipelineShapeConfig,
)
from orchid_ai.core.pipeline import OrchidPipelineConfigError, OrchidPipelineStep, OrchidPipelineStepResult
from orchid_ai.pipeline.definition import CompiledJob, CompiledPipeline, PipelineDependencies, PipelineLimits
from orchid_ai.pipeline.registry import (
    BUILTIN_JOB_KINDS,
    OrchidPipelineInfo,
    OrchidPipelineJobInfo,
    OrchidPipelineToolSpec,
    PipelineRegistry,
    is_builtin_job_kind,
    register_pipeline_step,
    registered_step_kinds,
    resolve_pipeline_step,
    unregister_pipeline_step,
)
from orchid_ai.pipeline.schema import compile_shapes


class _FakeStep(OrchidPipelineStep):
    kind = "fake"

    def __init__(self, source: str = "") -> None:
        self.source = source

    async def execute(self, ctx: Any) -> OrchidPipelineStepResult:  # pragma: no cover - never executed
        return OrchidPipelineStepResult(output={})


@pytest.fixture(autouse=True)
def clean_registry() -> Any:
    from orchid_ai.pipeline import registry

    before = dict(registry.PIPELINE_STEP_REGISTRY)
    registry.PIPELINE_STEP_REGISTRY.clear()
    yield
    registry.PIPELINE_STEP_REGISTRY.clear()
    registry.PIPELINE_STEP_REGISTRY.update(before)


def _job(job_id: str = "run") -> CompiledJob:
    config = OrchidPipelineDeterministicJobConfig.model_validate(
        {"kind": "deterministic", "uses": "pkg.step", "depends_on": []}
    )
    return CompiledJob(id=job_id, kind="deterministic", config=config)


def _pipeline(
    name: str = "demo",
    *,
    description: str = "Demo pipeline",
    version: str = "1",
    expose: OrchidPipelineExposeConfig | None = None,
    shapes: Any = None,
    jobs: tuple[CompiledJob, ...] = (),
) -> CompiledPipeline:
    shape_registry = shapes
    input_shape = shape_registry.get("Input") if shape_registry is not None and "Input" in shape_registry else None
    output_shape = shape_registry.get("Out") if shape_registry is not None and "Out" in shape_registry else None
    definition = OrchidPipelineDefinitionConfig(
        description=description,
        version=version,
        expose=expose or OrchidPipelineExposeConfig(),
    )
    return CompiledPipeline(
        name=name,
        version=version,
        description=description,
        definition=definition,
        jobs={job.id: job for job in (jobs or (_job(),))},
        limits=PipelineLimits.from_config(),
        shapes=shape_registry,
        input_shape=input_shape,
        output_shape=output_shape,
    )


# ── Step registry ───────────────────────────────────────────


def test_register_and_resolve_a_step() -> None:
    calls: list[tuple[Any, Any]] = []

    def factory(config: Any, deps: Any) -> _FakeStep:
        calls.append((config, deps))
        return _FakeStep(source="factory")

    register_pipeline_step("fake", factory)

    deps = PipelineDependencies()
    step = resolve_pipeline_step("fake", "config", deps)

    assert isinstance(step, _FakeStep)
    assert step.source == "factory"
    assert calls == [("config", deps)]
    assert registered_step_kinds() == ("fake",)


def test_duplicate_registration_requires_override() -> None:
    register_pipeline_step("fake", lambda config, deps: _FakeStep(source="first"))

    with pytest.raises(OrchidPipelineConfigError, match="already registered"):
        register_pipeline_step("fake", lambda config, deps: _FakeStep(source="second"))

    register_pipeline_step("fake", lambda config, deps: _FakeStep(source="second"), override=True)
    assert resolve_pipeline_step("fake", None, PipelineDependencies()).source == "second"


def test_unregister_ignores_unknown_kinds() -> None:
    register_pipeline_step("fake", lambda config, deps: _FakeStep())
    unregister_pipeline_step("fake")
    unregister_pipeline_step("fake")

    assert registered_step_kinds() == ()


def test_register_validates_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        register_pipeline_step("", lambda config, deps: _FakeStep())
    with pytest.raises(TypeError, match="callable"):
        register_pipeline_step("fake", None)  # type: ignore[arg-type]


def test_resolve_unknown_kind_names_the_kind() -> None:
    with pytest.raises(OrchidPipelineConfigError, match="no step is registered for job kind 'missing'"):
        resolve_pipeline_step("missing", None, PipelineDependencies())


def test_factory_errors_are_wrapped() -> None:
    def broken(config: Any, deps: Any) -> _FakeStep:
        raise RuntimeError("boom")

    register_pipeline_step("broken", broken)

    with pytest.raises(OrchidPipelineConfigError, match="cannot build the step for kind 'broken': boom"):
        resolve_pipeline_step("broken", None, PipelineDependencies())


def test_factory_configuration_errors_pass_through() -> None:
    def broken(config: Any, deps: Any) -> _FakeStep:
        raise OrchidPipelineConfigError("bad config")

    register_pipeline_step("broken", broken)

    with pytest.raises(OrchidPipelineConfigError, match="bad config"):
        resolve_pipeline_step("broken", None, PipelineDependencies())


def test_builtin_kinds_are_declared() -> None:
    assert BUILTIN_JOB_KINDS == {
        "deterministic",
        "llm",
        "agent",
        "decision",
        "map",
        "subpipeline",
        "emit",
        "custom",
    }
    assert is_builtin_job_kind("deterministic") is True
    assert is_builtin_job_kind("my_step") is False


# ── Pipeline registry ──────────────────────────────────────


def test_registry_resolves_pipelines() -> None:
    registry = PipelineRegistry([_pipeline("alpha"), _pipeline("beta")])

    assert len(registry) == 2
    assert registry.names() == ("alpha", "beta")
    assert "alpha" in registry
    assert registry.has("alpha") is True
    assert registry.has("missing") is False
    assert registry.get("alpha").name == "alpha"  # type: ignore[union-attr]
    assert registry.get("missing") is None
    assert registry.info("missing") is None


def test_registry_rejects_duplicate_names() -> None:
    with pytest.raises(OrchidPipelineConfigError, match="duplicate pipeline 'alpha'"):
        PipelineRegistry([_pipeline("alpha"), _pipeline("alpha")])


def test_registry_rejects_non_pipeline_values() -> None:
    with pytest.raises(TypeError, match="CompiledPipeline"):
        PipelineRegistry(["not-a-pipeline"])  # type: ignore[list-item]


def test_info_is_redacted_and_typed() -> None:
    pipeline = _pipeline(
        description="Demo",
        version="3",
        expose=OrchidPipelineExposeConfig(as_tool=True, description="Run it"),
        jobs=(_job("first"), _job("second")),
    )

    info = PipelineRegistry([pipeline]).info("demo")

    assert isinstance(info, OrchidPipelineInfo)
    assert info.name == "demo"
    assert info.description == "Demo"
    assert info.version == "3"
    assert {job.id for job in info.jobs} == {"first", "second"}
    assert all(isinstance(job, OrchidPipelineJobInfo) for job in info.jobs)
    payload = info.to_dict()
    assert "prompt" not in str(payload)
    assert payload["expose"]["as_tool"] is True
    assert payload["async"]["enabled"] is False


def test_info_exposes_shape_schemas() -> None:
    shapes = compile_shapes(
        {
            "Input": OrchidPipelineShapeConfig(fields={"account_id": {"type": "string", "required": True}}),
            "Out": OrchidPipelineShapeConfig(fields={"summary": {"type": "string"}}),
        }
    )
    pipeline = _pipeline(shapes=shapes)

    info = PipelineRegistry([pipeline]).info("demo")

    assert info is not None
    assert info.input_schema is not None
    assert "account_id" in str(info.input_schema)
    assert info.output_schema is not None


def test_list_returns_every_pipeline_info() -> None:
    registry = PipelineRegistry([_pipeline("alpha"), _pipeline("beta")])

    listed = registry.list()

    assert [info.name for info in listed] == ["alpha", "beta"]


# ── Tool specs ─────────────────────────────────────────────


def test_tool_specs_only_include_exposed_pipelines() -> None:
    exposed = _pipeline(
        "alpha",
        expose=OrchidPipelineExposeConfig(
            as_tool=True,
            description="Assess an account",
            tool_name="pipeline_assess",
            requires_approval=True,
            parallel_safe=True,
            require_role="analyst",
        ),
    )
    hidden = _pipeline("beta")

    specs = PipelineRegistry([exposed, hidden]).tool_specs()

    assert len(specs) == 1
    spec = specs[0]
    assert isinstance(spec, OrchidPipelineToolSpec)
    assert spec.name == "pipeline_assess"
    assert spec.pipeline == "alpha"
    assert spec.description == "Assess an account"
    assert spec.requires_approval is True
    assert spec.parallel_safe is True
    assert spec.require_role == "analyst"
    assert spec.max_tool_result_chars == 8000


def test_tool_specs_default_name_and_description() -> None:
    exposed = _pipeline(
        "alpha",
        description="Fallback",
        expose=OrchidPipelineExposeConfig(as_tool=True, description="Explicit"),
    )
    unnamed = _pipeline(
        "beta",
        description="Fallback description",
        expose=OrchidPipelineExposeConfig(as_tool=True, description="Explicit"),
    )
    unnamed.definition.expose.tool_name = None
    unnamed.definition.expose.description = None

    specs = {spec.pipeline: spec for spec in PipelineRegistry([exposed, unnamed]).tool_specs()}

    assert specs["alpha"].name == "pipeline_alpha"
    assert specs["beta"].description == "Fallback description"


# ── from_config ────────────────────────────────────────────


def test_from_config_compiles_enabled_definitions(monkeypatch: pytest.MonkeyPatch) -> None:
    from orchid_ai.pipeline import registry as registry_module

    monkeypatch.setattr(registry_module, "compile_pipelines", lambda *a, **k: {"demo": _pipeline("demo")})

    registry = PipelineRegistry.from_config(OrchidPipelinesConfig())

    assert registry.names() == ("demo",)


def test_from_config_returns_empty_for_absent_or_disabled_block() -> None:
    assert len(PipelineRegistry.from_config(None)) == 0
    assert len(PipelineRegistry.from_config(OrchidPipelinesConfig(enabled=False))) == 0


# ── Review regressions (P-006 CR round 1) ───────────────────


def test_default_tool_name_is_sanitized_and_truncated() -> None:
    pipeline = _pipeline(
        "a" * 60,
        expose=OrchidPipelineExposeConfig(as_tool=True, description="Long name"),
    )

    spec = PipelineRegistry([pipeline]).tool_specs()[0]

    assert len(spec.name) <= 64
    assert spec.name.startswith("pipeline_")


def test_default_tool_name_replaces_unsupported_characters() -> None:
    from orchid_ai.pipeline.registry import _default_tool_name

    assert _default_tool_name("a.b c") == "pipeline_a_b_c"


def test_duplicate_tool_names_are_rejected() -> None:
    first = _pipeline(
        "alpha",
        expose=OrchidPipelineExposeConfig(as_tool=True, description="Alpha", tool_name="pipeline_beta"),
    )
    second = _pipeline("beta", expose=OrchidPipelineExposeConfig(as_tool=True, description="Beta"))

    with pytest.raises(OrchidPipelineConfigError, match="expose the same tool name 'pipeline_beta'"):
        PipelineRegistry([first, second])


def test_invalid_explicit_tool_name_is_rejected() -> None:
    pipeline = _pipeline(
        "alpha",
        expose=OrchidPipelineExposeConfig(as_tool=True, description="Alpha"),
    )
    pipeline.definition.expose.tool_name = "bad name"

    with pytest.raises(OrchidPipelineConfigError, match="invalid tool name"):
        PipelineRegistry([pipeline])
