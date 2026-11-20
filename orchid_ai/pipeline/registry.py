"""Pipeline registry: compiled pipelines, the step-kind extension point and tool specs.

The registry is the read surface the API, CLI and agent tools use:

* ``list()`` / ``get()`` / ``has()`` / ``info()`` resolve compiled pipelines and
  return redacted metadata (no prompts, parameters or credentials; the exposed
  tool name is public by design);
* :data:`PIPELINE_STEP_REGISTRY` plus :func:`register_pipeline_step` are the
  Open/Closed extension point for job kinds. Built-in kinds are declared here;
  their step modules register the factory when the runner imports them, so this
  module never depends on a step it does not own;
* ``tool_specs()`` returns the sanitized specs for definitions exposed as agent
  tools. Building the actual ``OrchidTool`` objects is the lifecycle's job
  (ADR-034); the registry only describes them.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..config.schema_pipeline import (
    OrchidPipelinesConfig,
)
from ..core.pipeline import OrchidPipelineConfigError, OrchidPipelineStep
from .definition import CompiledJob, CompiledPipeline, PipelineDependencies, compile_pipelines

__all__ = [
    "BUILTIN_JOB_KINDS",
    "PIPELINE_STEP_REGISTRY",
    "OrchidPipelineInfo",
    "OrchidPipelineJobInfo",
    "OrchidPipelineToolSpec",
    "PipelineRegistry",
    "is_builtin_job_kind",
    "register_pipeline_step",
    "registered_step_kinds",
    "resolve_pipeline_step",
    "unregister_pipeline_step",
]

#: Job kinds the configuration schema accepts without a custom registration.
BUILTIN_JOB_KINDS: frozenset[str] = frozenset(
    {"deterministic", "llm", "agent", "decision", "map", "subpipeline", "emit", "custom"}
)

#: Factory signature: ``(job_config, dependencies) -> OrchidPipelineStep``.
PipelineStepFactory = Callable[[Any, PipelineDependencies], OrchidPipelineStep]

#: Kind → factory. Step modules register themselves when imported.
PIPELINE_STEP_REGISTRY: dict[str, PipelineStepFactory] = {}


def is_builtin_job_kind(kind: str) -> bool:
    """True for the eight kinds the configuration schema declares."""
    return kind in BUILTIN_JOB_KINDS


def register_pipeline_step(kind: str, factory: PipelineStepFactory, *, override: bool = False) -> None:
    """Register a step factory for ``kind``.

    ``override`` is for reloads and tests: without it a second registration of
    the same kind raises, so two plugins cannot silently shadow each other.
    """
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("step kind must be a non-empty string")
    if not callable(factory):
        raise TypeError("step factory must be callable")
    if kind in PIPELINE_STEP_REGISTRY and not override:
        raise OrchidPipelineConfigError(
            f"a step factory is already registered for kind {kind!r}; pass override=True to replace it"
        )
    PIPELINE_STEP_REGISTRY[kind] = factory


def unregister_pipeline_step(kind: str) -> None:
    """Remove a registered factory; unknown kinds are ignored."""
    PIPELINE_STEP_REGISTRY.pop(kind, None)


def registered_step_kinds() -> tuple[str, ...]:
    """Registered kinds in sorted order."""
    return tuple(sorted(PIPELINE_STEP_REGISTRY))


def resolve_pipeline_step(
    kind: str,
    config: Any,
    deps: PipelineDependencies,
) -> OrchidPipelineStep:
    """Build the step for ``kind`` or fail with an actionable configuration error."""
    factory = PIPELINE_STEP_REGISTRY.get(kind)
    if factory is None:
        raise OrchidPipelineConfigError(
            f"no step is registered for job kind {kind!r}; import the step module or "
            f"call register_pipeline_step({kind!r}, factory) before building the pipeline"
        )
    try:
        return factory(config, deps)
    except OrchidPipelineConfigError:
        raise
    except Exception as exc:
        raise OrchidPipelineConfigError(f"cannot build the step for kind {kind!r}: {exc}") from exc


# ── Redacted metadata ────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class OrchidPipelineJobInfo:
    """One job in the public pipeline description (no prompts, no parameters)."""

    id: str
    kind: str
    depends_on: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "depends_on": list(self.depends_on)}


@dataclass(frozen=True, slots=True)
class OrchidPipelineInfo:
    """Public, redacted pipeline description returned by list/info."""

    name: str
    description: str
    version: str
    jobs: tuple[OrchidPipelineJobInfo, ...] = ()
    input_schema: Mapping[str, Any] | None = None
    output_schema: Mapping[str, Any] | None = None
    expose: Mapping[str, Any] = field(default_factory=dict)
    async_: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "jobs": [job.to_dict() for job in self.jobs],
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "expose": dict(self.expose),
            "async": dict(self.async_),
        }


@dataclass(frozen=True, slots=True)
class OrchidPipelineToolSpec:
    """Sanitized description of a definition exposed as an agent-callable tool."""

    name: str
    pipeline: str
    description: str
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    require_role: str | None = None
    requires_approval: bool = False
    parallel_safe: bool = False
    max_tool_result_chars: int = 8000


def _job_info(job: CompiledJob) -> OrchidPipelineJobInfo:
    return OrchidPipelineJobInfo(id=job.id, kind=job.kind, depends_on=job.depends_on)


def _expose_summary(pipeline: CompiledPipeline) -> dict[str, Any]:
    expose = pipeline.definition.expose
    return {
        "as_tool": expose.as_tool,
        "tool_name": expose.tool_name,
        "requires_approval": expose.requires_approval,
        "parallel_safe": expose.parallel_safe,
    }


def _async_summary(pipeline: CompiledPipeline) -> dict[str, Any]:
    async_config = pipeline.definition.async_
    return {
        "enabled": async_config.enabled,
        "signal": async_config.signal,
        "trigger_id": async_config.trigger_id,
        "mode": async_config.mode,
    }


def _pipeline_info(pipeline: CompiledPipeline) -> OrchidPipelineInfo:
    return OrchidPipelineInfo(
        name=pipeline.name,
        description=pipeline.description,
        version=pipeline.version,
        jobs=tuple(_job_info(job) for job in pipeline.ordered_jobs),
        input_schema=pipeline.input_shape.json_schema() if pipeline.input_shape is not None else None,
        output_schema=pipeline.output_shape.json_schema() if pipeline.output_shape is not None else None,
        expose=_expose_summary(pipeline),
        async_=_async_summary(pipeline),
    )


_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_MAX_TOOL_NAME_CHARS = 64


def _default_tool_name(pipeline_name: str) -> str:
    """Sanitize and truncate the generated ``pipeline_<name>`` tool name."""
    candidate = re.sub(r"[^A-Za-z0-9_-]", "_", f"pipeline_{pipeline_name}")
    return candidate[:_MAX_TOOL_NAME_CHARS]


def _tool_spec(pipeline: CompiledPipeline) -> OrchidPipelineToolSpec:
    expose = pipeline.definition.expose
    name = expose.tool_name or _default_tool_name(pipeline.name)
    if not _TOOL_NAME_RE.fullmatch(name):
        raise OrchidPipelineConfigError(
            f"pipeline {pipeline.name!r} exposes an invalid tool name {name!r}; "
            "it must match [A-Za-z0-9][A-Za-z0-9_-]{0,63}"
        )
    return OrchidPipelineToolSpec(
        name=name,
        pipeline=pipeline.name,
        description=expose.description or pipeline.description,
        input_schema=pipeline.input_shape.json_schema() if pipeline.input_shape is not None else {},
        require_role=expose.require_role,
        requires_approval=expose.requires_approval,
        parallel_safe=expose.parallel_safe,
        max_tool_result_chars=expose.max_tool_result_chars,
    )


# ── Registry ─────────────────────────────────────────────────


# ORCHID-MEM: 01a1eb1b-d999-7d5a-bc8c-e07cc257abff — Pipeline registry; redacted pipeline metadata, the step-kind extension point and exposed-tool specs.
def _validate_tool_specs(pipelines: Mapping[str, CompiledPipeline]) -> None:
    """Reject invalid or duplicate exposed tool names when the registry is built."""
    seen: dict[str, str] = {}
    for pipeline in pipelines.values():
        if not pipeline.definition.expose.as_tool:
            continue
        spec = _tool_spec(pipeline)
        owner = seen.get(spec.name)
        if owner is not None:
            raise OrchidPipelineConfigError(
                f"pipelines {owner!r} and {pipeline.name!r} expose the same tool name {spec.name!r}"
            )
        seen[spec.name] = pipeline.name


class PipelineRegistry:
    """Immutable name → compiled pipeline lookup.

    ``from_config`` is the production constructor: it compiles every enabled
    definition once, sharing the shape registry and the dependency maps.
    """

    def __init__(self, pipelines: Iterable[CompiledPipeline] = ()) -> None:
        compiled: dict[str, CompiledPipeline] = {}
        for pipeline in pipelines:
            if not isinstance(pipeline, CompiledPipeline):
                raise TypeError("pipelines must be CompiledPipeline values")
            if pipeline.name in compiled:
                raise OrchidPipelineConfigError(f"duplicate pipeline {pipeline.name!r} in the registry")
            compiled[pipeline.name] = pipeline
        self._pipelines = compiled
        self._resolver = compiled.get
        _validate_tool_specs(compiled)

    @classmethod
    def from_config(
        cls,
        pipelines: OrchidPipelinesConfig | None,
        *,
        deps: PipelineDependencies | None = None,
        resolve_import: Callable[[str], Any] | None = None,
    ) -> PipelineRegistry:
        """Compile every enabled definition of a ``pipelines:`` block."""
        compiled = compile_pipelines(pipelines, deps=deps, resolve_import=resolve_import)
        return cls(compiled.values())

    @property
    def resolver(self) -> Callable[[str], CompiledPipeline | None]:
        """Name → compiled pipeline lookup for sub-pipeline dependencies.

        Pass this as ``PipelineDependencies.pipeline_resolver`` when building
        graphs outside the lifecycle; the lifecycle wires the equivalent
        resolver automatically.
        """
        return self._resolver

    def __len__(self) -> int:
        return len(self._pipelines)

    def __contains__(self, name: object) -> bool:
        return name in self._pipelines

    def names(self) -> tuple[str, ...]:
        """Registered pipeline names in configuration order."""
        return tuple(self._pipelines)

    def list(self) -> list[OrchidPipelineInfo]:
        """Redacted descriptions of every registered pipeline."""
        return [_pipeline_info(pipeline) for pipeline in self._pipelines.values()]

    def get(self, name: str) -> CompiledPipeline | None:
        """Return the compiled pipeline, or ``None`` when it is not registered."""
        return self._pipelines.get(name)

    def has(self, name: str) -> bool:
        """True when ``name`` is registered."""
        return name in self._pipelines

    def info(self, name: str) -> OrchidPipelineInfo | None:
        """Redacted description of one pipeline, or ``None``."""
        pipeline = self._pipelines.get(name)
        return _pipeline_info(pipeline) if pipeline is not None else None

    def tool_specs(self) -> list[OrchidPipelineToolSpec]:
        """Sanitized specs for every definition exposed as an agent tool.

        Names are sanitized and truncated to the 64-character tool-name limit;
        a duplicate name is a configuration error, not a silent shadow. P-006
        ships the description only: the lifecycle (ADR-034) turns these specs
        into ``OrchidTool`` objects and swaps them transactionally.
        """
        return [_tool_spec(pipeline) for pipeline in self._pipelines.values() if pipeline.definition.expose.as_tool]
