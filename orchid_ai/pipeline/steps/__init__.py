"""Built-in pipeline step kinds.

Importing this package registers the kinds implemented in-tree. Later step
modules (LLM, decision, agent, map, sub-pipeline, emit) register themselves the
same way, so a build that references an unimplemented kind fails with a clear
configuration error instead of a runtime surprise.
"""

from __future__ import annotations

from typing import Any

from ..definition import CompiledJob
from ..registry import PIPELINE_STEP_REGISTRY, register_pipeline_step
from .agent import AgentStep
from .decision import DecisionStep
from .deterministic import DeterministicStep
from .emit import EmitStep
from .llm import LLMStep
from .subpipeline import SubPipelineStep

__all__ = [
    "AgentStep",
    "DecisionStep",
    "DeterministicStep",
    "EmitStep",
    "LLMStep",
    "SubPipelineStep",
    "register_builtin_steps",
]


def _build_deterministic(job: CompiledJob, deps: Any) -> DeterministicStep:
    return DeterministicStep(job)


def _build_llm(job: CompiledJob, deps: Any) -> LLMStep:
    return LLMStep(job)


def _build_decision(job: CompiledJob, deps: Any) -> DecisionStep:
    return DecisionStep(job, models=getattr(deps, "decision_models", None) or {})


def _build_agent(job: CompiledJob, deps: Any) -> AgentStep:
    return AgentStep(
        job,
        agents=getattr(deps, "agents", None) or {},
        agent_configs=getattr(deps, "agent_configs", None) or {},
        agent_executors=getattr(deps, "agent_executors", None) or {},
    )


def _build_emit(job: CompiledJob, deps: Any) -> EmitStep:
    return EmitStep(job)


def _build_subpipeline(job: CompiledJob, deps: Any) -> SubPipelineStep:
    return SubPipelineStep(job, resolver=getattr(deps, "pipeline_resolver", None))


def register_builtin_steps() -> None:
    """Register every in-tree step kind without displacing an explicit factory.

    A test or integrator that registers a kind before building keeps its
    factory; production starts with an empty registry. Idempotent across
    reloads.
    """
    for kind, factory in (
        ("deterministic", _build_deterministic),
        ("llm", _build_llm),
        ("decision", _build_decision),
        ("agent", _build_agent),
        ("emit", _build_emit),
        ("subpipeline", _build_subpipeline),
    ):
        if kind not in PIPELINE_STEP_REGISTRY:
            register_pipeline_step(kind, factory)


register_builtin_steps()
