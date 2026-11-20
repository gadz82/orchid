"""Tests for the sub-pipeline step and its inherited run context."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from orchid_ai.config.schema_pipeline import (
    OrchidPipelineDefinitionConfig,
    OrchidPipelineDeterministicJobConfig,
    OrchidPipelineLLMJobConfig,
    OrchidPipelineOutputConfig,
    OrchidPipelinesConfig,
    OrchidPipelineShapeConfig,
    OrchidPipelineSubPipelineJobConfig,
)
from orchid_ai.core.pipeline import (
    OrchidGraphMutation,
    OrchidGraphRelation,
    OrchidPipelineGraphConflictError,
    OrchidPipelineLimitError,
    OrchidPipelineRunContext,
    OrchidPipelineStep,
    OrchidPipelineStepError,
    OrchidPipelineStepResult,
)
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.pipeline.definition import CompiledJob, CompiledPipeline, PipelineDependencies, PipelineLimits
from orchid_ai.pipeline.registry import PipelineRegistry
from orchid_ai.pipeline.runner import OrchidPipelineRunner
from orchid_ai.pipeline.schema import compile_shapes
from orchid_ai.pipeline.steps.subpipeline import SubPipelineStep


def _shapes() -> Any:
    return compile_shapes(
        {
            "Out": OrchidPipelineShapeConfig(
                fields={"value": {"type": "string", "required": True}},
            ),
            "Thing": OrchidPipelineShapeConfig(
                fields={"name": {"type": "string", "required": True}},
            ),
        }
    )


def _child(
    handler: Any,
    *,
    name: str = "child",
    writes: list[dict[str, Any]] | None = None,
    links: list[dict[str, Any]] | None = None,
    output_mapping: dict[str, str] | None = None,
    limits: PipelineLimits | None = None,
) -> CompiledPipeline:
    config = OrchidPipelineDeterministicJobConfig.model_validate(
        {"kind": "deterministic", "uses": "pkg.load", "writes": writes or [], "links": links or []}
    )
    job = CompiledJob(
        id="load",
        kind="deterministic",
        config=config,
        handler=handler,
        writes=tuple(config.writes),
        links=tuple(config.links),
    )
    shapes = _shapes()
    output = OrchidPipelineOutputConfig(shape="Out", mapping=output_mapping) if output_mapping is not None else None
    return CompiledPipeline(
        name=name,
        version="1",
        description="",
        definition=OrchidPipelineDefinitionConfig(output=output),
        jobs={job.id: job},
        limits=limits or PipelineLimits.from_config(),
        shapes=shapes,
        output_shape=shapes.get("Out") if output is not None else None,
    )


def _parent(
    child_name: str = "child",
    *,
    input_mapping: dict[str, Any] | None = None,
    merge_graph: bool = False,
    output_from: str = "output",
    limits: PipelineLimits | None = None,
    job_id: str = "call",
    timeout_seconds: float | None = None,
    writes: list[dict[str, Any]] | None = None,
) -> CompiledPipeline:
    config = OrchidPipelineSubPipelineJobConfig.model_validate(
        {
            "kind": "subpipeline",
            "pipeline": child_name,
            "input": input_mapping,
            "merge_graph": merge_graph,
            "output_from": output_from,
            "timeout_seconds": timeout_seconds,
            "writes": writes or [],
        }
    )
    job = CompiledJob(
        id=job_id,
        kind="subpipeline",
        config=config,
        subpipeline=child_name,
        timeout_seconds=timeout_seconds,
        writes=tuple(config.writes),
    )
    return CompiledPipeline(
        name="parent",
        version="1",
        description="",
        definition=OrchidPipelineDefinitionConfig(),
        jobs={job.id: job},
        limits=limits or PipelineLimits.from_config(),
        shapes=_shapes(),
    )


def _sub_pipeline(name: str, child_name: str, *, limits: PipelineLimits | None = None) -> CompiledPipeline:
    config = OrchidPipelineSubPipelineJobConfig.model_validate({"kind": "subpipeline", "pipeline": child_name})
    job = CompiledJob(
        id="call",
        kind="subpipeline",
        config=config,
        subpipeline=child_name,
    )
    return CompiledPipeline(
        name=name,
        version="1",
        description="",
        definition=OrchidPipelineDefinitionConfig(),
        jobs={"call": job},
        limits=limits or PipelineLimits.from_config(),
        shapes=_shapes(),
    )


def _run(
    parent: CompiledPipeline,
    children: dict[str, CompiledPipeline],
    *,
    auth: OrchidAuthContext | None = None,
    timeout: float | None = None,
    mode: str = "direct",
    chat_id: str | None = None,
    seed: list[dict[str, Any]] | None = None,
) -> Any:
    resolved_auth = auth or OrchidAuthContext(access_token="token", tenant_key="tenant", user_id="user")
    runner = OrchidPipelineRunner(
        parent,
        deps=PipelineDependencies(pipeline_resolver=children.get),
    )
    return asyncio.run(
        runner.run(
            {},
            auth=resolved_auth,
            mode=mode,
            chat_id=chat_id,
            timeout=timeout,
            seed=seed or (),
        )
    )


# ── Child execution and context ─────────────────────────────


def test_child_runs_with_inherited_auth_and_output() -> None:
    seen: list[Any] = []

    async def child_handler(ctx: Any) -> dict[str, Any]:
        seen.append(ctx.auth)
        return {"value": "child-output"}

    child = _child(child_handler, output_mapping={"value": "jobs.load.output.value"})
    auth = OrchidAuthContext(access_token="token", tenant_key="tenant", user_id="user")

    result = _run(_parent(), {"child": child}, auth=auth)

    assert result.status == "succeeded"
    assert result.jobs["call"]["output"] == {"value": "child-output"}
    assert seen == [auth]
    assert result.jobs["call"]["metadata"]["subpipeline"] == "child"


def test_child_input_mapping_is_rendered() -> None:
    async def child_handler(ctx: Any) -> dict[str, Any]:
        return {"value": ctx.input["message"]}

    child = _child(child_handler, output_mapping={"value": "jobs.load.output.value"})
    parent = _parent(input_mapping={"message": "{{ input.message }}"})
    runner = OrchidPipelineRunner(parent, deps=PipelineDependencies(pipeline_resolver={"child": child}.get))

    result = asyncio.run(runner.run({"message": "hello"}, auth=OrchidAuthContext(access_token="t", user_id="u")))

    assert result.jobs["call"]["output"] == {"value": "hello"}


def test_output_from_selects_a_child_result_path() -> None:
    async def child_handler(ctx: Any) -> dict[str, Any]:
        return {"value": "deep"}

    child = _child(child_handler, output_mapping={"value": "jobs.load.output.value"})
    parent = _parent(output_from="jobs.load.output.value")

    result = _run(parent, {"child": child})

    assert result.status == "succeeded"
    assert result.jobs["call"]["output"] == "deep"


def test_chat_scope_is_inherited() -> None:
    seen: list[str | None] = []

    async def child_handler(ctx: Any) -> dict[str, Any]:
        seen.append(ctx.run.chat_id)
        return {"value": "chat"}

    child = _child(child_handler, output_mapping={"value": "jobs.load.output.value"})

    result = _run(_parent(), {"child": child}, mode="chat", chat_id="chat-1")

    assert result.status == "succeeded"
    assert seen == ["chat-1"]


def test_unknown_child_is_a_config_error() -> None:
    result = _run(_parent(), {})

    assert result.status == "failed"
    assert "unknown pipeline 'child'" in result.jobs["call"]["error"]


def test_child_failure_fails_the_parent_job() -> None:
    async def child_handler(ctx: Any) -> dict[str, Any]:
        raise OrchidPipelineStepError("child exploded", job_id="load", kind="deterministic")

    child = _child(child_handler, output_mapping={"value": "jobs.load.output.value"})

    result = _run(_parent(), {"child": child})

    assert result.status == "failed"
    assert result.jobs["call"]["status"] == "failed"
    assert "status 'failed'" in result.jobs["call"]["error"]


def test_missing_output_from_path_fails_the_job() -> None:
    async def child_handler(ctx: Any) -> dict[str, Any]:
        return {"value": "x"}

    child = _child(child_handler, output_mapping={"value": "jobs.load.output.value"})
    result = _run(_parent(output_from="missing"), {"child": child})

    assert result.jobs["call"]["status"] == "failed"
    assert "output_from" in result.jobs["call"]["error"]


# ── Graph merge ────────────────────────────────────────────


def _writer(ctx: Any) -> dict[str, Any]:
    return {"value": "written", "name": "child-object"}


def test_graph_merge_is_explicit_and_prefixed() -> None:
    child = _child(
        _writer,
        output_mapping={"value": "jobs.load.output.value"},
        writes=[{"id": "thing:1", "shape": "Thing", "from": "{name: result.name}"}],
    )

    unmerged = _run(_parent(merge_graph=False), {"child": child})
    assert "call:thing:1" not in unmerged.output["objects"]

    merged = _run(_parent(merge_graph=True), {"child": child})
    assert merged.status == "succeeded"
    assert merged.output["objects"]["call:thing:1"]["data"] == {"name": "child-object"}


def test_end_to_end_relation_merge_uses_prefixed_endpoints() -> None:
    child = _child(
        _writer,
        output_mapping={"value": "jobs.load.output.value"},
        writes=[
            {"id": "thing:1", "shape": "Thing", "from": "{name: 'a'}"},
            {"id": "thing:2", "shape": "Thing", "from": "{name: 'b'}"},
        ],
        links=[{"type": "linked", "from": "thing:1", "to": "thing:2"}],
    )

    result = _run(_parent(merge_graph=True), {"child": child})

    assert result.status == "succeeded"
    assert "call:thing:1" in result.output["objects"]
    assert "call:thing:2" in result.output["objects"]
    assert result.output["relations"]["linked:call:thing:1:call:thing:2"]["source_id"] == "call:thing:1"


def test_graph_merge_coexists_with_declared_parent_writes() -> None:
    child = _child(
        _writer,
        output_mapping={"value": "jobs.load.output.value"},
        writes=[{"id": "thing:1", "shape": "Thing", "from": "{name: 'child'}"}],
    )
    parent = _parent(
        merge_graph=True,
        writes=[{"id": "out:1", "shape": "Thing", "from": "{name: result.value}"}],
    )

    result = _run(parent, {"child": child})

    assert result.status == "succeeded"
    assert result.output["objects"]["out:1"]["data"] == {"name": "written"}
    assert result.output["objects"]["call:thing:1"]["data"] == {"name": "child"}
    assert result.jobs["call"]["metadata"].get("merge_objects") is None
    assert "_merge_objects" not in result.jobs["call"]["metadata"]


def test_graph_merge_collision_fails_the_job() -> None:
    child = _child(
        _writer,
        output_mapping={"value": "jobs.load.output.value"},
        writes=[{"id": "thing:1", "shape": "Thing", "from": "{name: result.name}"}],
    )
    seed = [{"id": "call:thing:1", "type": "Thing", "data": {"name": "parent"}}]

    result = _run(_parent(merge_graph=True), {"child": child}, seed=seed)

    assert result.status == "failed"
    assert result.jobs["call"]["status"] == "failed"
    assert "merge collision" in result.jobs["call"]["error"]


class _GraphView:
    def __init__(self, relations: list[Any] | None = None) -> None:
        self._relations = list(relations or [])

    def get(self, object_id: str) -> Any:
        return None

    def relations(self, **kwargs: Any) -> list[Any]:
        return self._relations


def test_relation_merge_is_prefixed() -> None:
    step = SubPipelineStep(_parent().jobs["call"])
    result = SimpleNamespace(
        graph={
            "objects": {
                "thing:1": {"type": "Thing", "data": {"name": "a"}},
                "thing:2": {"type": "Thing", "data": {"name": "b"}},
            },
            "relations": {
                "linked:1": {
                    "type": "linked",
                    "source_id": "thing:1",
                    "target_id": "thing:2",
                    "data": {},
                    "metadata": {},
                }
            },
        },
        metadata={},
    )

    mutations = step._merge_mutations(SimpleNamespace(graph=_GraphView()), result)

    relation = next(mutation.relation for mutation in mutations if mutation.relation is not None)
    assert relation.id == "linked:call:thing:1:call:thing:2"
    assert relation.source_id == "call:thing:1"
    assert relation.target_id == "call:thing:2"


def test_relation_merge_collision_fails() -> None:
    step = SubPipelineStep(_parent().jobs["call"])
    existing = OrchidGraphRelation(
        id="linked:call:thing:1:call:thing:2",
        type="linked",
        source_id="call:thing:1",
        target_id="call:thing:2",
    )
    result = SimpleNamespace(
        graph={
            "objects": {
                "thing:1": {"type": "Thing", "data": {"name": "a"}},
                "thing:2": {"type": "Thing", "data": {"name": "b"}},
            },
            "relations": {
                "linked:1": {
                    "type": "linked",
                    "source_id": "thing:1",
                    "target_id": "thing:2",
                }
            },
        },
        metadata={},
    )

    with pytest.raises(OrchidPipelineGraphConflictError, match="merge collision on relation"):
        step._merge_mutations(SimpleNamespace(graph=_GraphView([existing])), result)


class _BypassStep(OrchidPipelineStep):
    async def execute(self, ctx: Any) -> OrchidPipelineStepResult:
        return OrchidPipelineStepResult(
            output={"value": "x"},
            mutations=(
                OrchidGraphMutation(
                    op="create",
                    object_id="victim:1",
                    object_type="Thing",
                    data={"name": "x"},
                ),
            ),
            metadata={"_merge_objects": ("victim:1",)},
        )


def test_merge_allowlist_metadata_cannot_be_used_by_a_non_subpipeline_job() -> None:
    config = OrchidPipelineDeterministicJobConfig.model_validate(
        {"kind": "deterministic", "uses": "pkg.step", "allow_dynamic_writes": False}
    )
    job = CompiledJob(
        id="plain",
        kind="deterministic",
        config=config,
        handler=_BypassStep(),
        allow_dynamic_writes=False,
    )
    pipeline = CompiledPipeline(
        name="demo",
        version="1",
        description="",
        definition=OrchidPipelineDefinitionConfig(),
        jobs={"plain": job},
        limits=PipelineLimits.from_config(),
        shapes=_shapes(),
    )

    result = _run(pipeline, {})

    assert result.status == "failed"
    assert "may not write object 'victim:1'" in result.jobs["plain"]["error"]


# ── Depth, usage, deadline ─────────────────────────────────


class _UsageStep(OrchidPipelineStep):
    def __init__(self, calls: int = 7) -> None:
        self._calls = calls

    async def execute(self, ctx: Any) -> OrchidPipelineStepResult:
        return OrchidPipelineStepResult(
            output={"value": "usage"},
            usage={"llm_calls": self._calls, "prompt_tokens": 3, "completion_tokens": 2},
        )


def test_child_llm_usage_counts_toward_the_parent() -> None:
    child = _child(_UsageStep(), output_mapping={"value": "jobs.load.output.value"})

    result = _run(_parent(), {"child": child})

    assert result.status == "succeeded"
    assert result.jobs["call"]["llm_calls"] == 7
    assert result.jobs["call"]["prompt_tokens"] == 3
    assert result.graph_stats["llm_calls"] == 7


def test_failed_child_usage_reaches_the_parent_job() -> None:
    use_config = OrchidPipelineDeterministicJobConfig.model_validate({"kind": "deterministic", "uses": "pkg.use"})
    use_job = CompiledJob(id="use", kind="deterministic", config=use_config, handler=_UsageStep())
    boom_config = OrchidPipelineDeterministicJobConfig.model_validate({"kind": "deterministic", "uses": "pkg.boom"})

    async def boom(ctx: Any) -> dict[str, Any]:
        raise OrchidPipelineStepError("child boom", job_id="boom", kind="deterministic")

    boom_job = CompiledJob(id="boom", kind="deterministic", config=boom_config, handler=boom, depends_on=("use",))
    child = CompiledPipeline(
        name="child",
        version="1",
        description="",
        definition=OrchidPipelineDefinitionConfig(),
        jobs={"use": use_job, "boom": boom_job},
        limits=PipelineLimits.from_config(),
        shapes=_shapes(),
    )

    result = _run(_parent(), {"child": child})

    assert result.status == "failed"
    assert result.jobs["call"]["status"] == "failed"
    assert result.jobs["call"]["llm_calls"] == 7
    assert result.jobs["call"]["prompt_tokens"] == 3


def test_parent_llm_cap_is_enforced_across_a_child() -> None:
    child = _child(_UsageStep(), output_mapping={"value": "jobs.load.output.value"})
    parent = _parent(limits=PipelineLimits(max_llm_calls=1))

    result = _run(parent, {"child": child})

    assert result.status == "failed"
    assert result.jobs["call"]["status"] == "failed"
    assert "max_llm_calls=1" in result.jobs["call"]["error"]
    assert result.jobs["call"]["llm_calls"] == 7


def test_timeout_preserves_the_child_usage_spent_so_far() -> None:
    use_config = OrchidPipelineDeterministicJobConfig.model_validate({"kind": "deterministic", "uses": "pkg.use"})
    use_job = CompiledJob(id="use", kind="deterministic", config=use_config, handler=_UsageStep(6))
    slow_config = OrchidPipelineDeterministicJobConfig.model_validate(
        {"kind": "deterministic", "uses": "pkg.slow", "depends_on": ["use"]}
    )

    async def slow(ctx: Any) -> dict[str, Any]:
        await asyncio.sleep(0.3)
        return {"value": "late"}

    slow_job = CompiledJob(id="slow", kind="deterministic", config=slow_config, handler=slow, depends_on=("use",))
    child = CompiledPipeline(
        name="child",
        version="1",
        description="",
        definition=OrchidPipelineDefinitionConfig(),
        jobs={"use": use_job, "slow": slow_job},
        limits=PipelineLimits.from_config(),
        shapes=_shapes(),
    )

    result = _run(_parent(timeout_seconds=0.05), {"child": child})

    assert result.status == "failed"
    assert result.jobs["call"]["status"] == "failed"
    assert result.jobs["call"]["llm_calls"] == 6


def test_timeout_preserves_an_in_flight_llm_call(monkeypatch: pytest.MonkeyPatch) -> None:
    class _SlowModel:
        async def ainvoke(self, messages: Any, **kwargs: Any) -> AIMessage:
            await asyncio.sleep(0.3)
            return AIMessage(content="late")

    monkeypatch.setattr("orchid_ai.llm_factory.build_chat_model", lambda *args, **kwargs: _SlowModel())

    llm_config = OrchidPipelineLLMJobConfig.model_validate({"kind": "llm", "model": "test/model", "prompt": "Go"})
    slow_job = CompiledJob(id="slow", kind="llm", config=llm_config)
    child = CompiledPipeline(
        name="child",
        version="1",
        description="",
        definition=OrchidPipelineDefinitionConfig(),
        jobs={"slow": slow_job},
        limits=PipelineLimits.from_config(),
        shapes=_shapes(),
    )

    result = _run(_parent(timeout_seconds=0.05), {"child": child})

    assert result.status == "failed"
    assert result.jobs["call"]["status"] == "failed"
    assert result.jobs["call"]["llm_calls"] >= 1


def test_output_failure_preserves_the_child_usage() -> None:
    child = _child(_UsageStep(6), output_mapping={"value": "jobs.load.output.value"})
    result = _run(_parent(output_from="missing"), {"child": child})

    assert result.jobs["call"]["status"] == "failed"
    assert result.jobs["call"]["llm_calls"] == 6


def test_parent_llm_cap_counts_prior_jobs() -> None:
    prior_config = OrchidPipelineDeterministicJobConfig.model_validate({"kind": "deterministic", "uses": "pkg.prior"})
    prior = CompiledJob(id="prior", kind="deterministic", config=prior_config, handler=_UsageStep(6))
    sub_config = OrchidPipelineSubPipelineJobConfig.model_validate(
        {"kind": "subpipeline", "pipeline": "child", "depends_on": ["prior"]}
    )
    sub = CompiledJob(
        id="call",
        kind="subpipeline",
        config=sub_config,
        subpipeline="child",
        depends_on=("prior",),
    )
    parent = CompiledPipeline(
        name="parent",
        version="1",
        description="",
        definition=OrchidPipelineDefinitionConfig(),
        jobs={"prior": prior, "call": sub},
        limits=PipelineLimits(max_llm_calls=10),
        shapes=_shapes(),
    )
    child = _child(_UsageStep(6), output_mapping={"value": "jobs.load.output.value"})

    result = _run(parent, {"child": child})

    assert result.status == "failed"
    assert result.jobs["call"]["status"] == "failed"
    assert "max_llm_calls=10" in result.jobs["call"]["error"]


def test_parallel_children_are_capped_at_finish() -> None:
    configs = []
    for job_id in ("call_a", "call_b"):
        config = OrchidPipelineSubPipelineJobConfig.model_validate({"kind": "subpipeline", "pipeline": "child"})
        configs.append(
            CompiledJob(
                id=job_id,
                kind="subpipeline",
                config=config,
                subpipeline="child",
            )
        )
    parent = CompiledPipeline(
        name="parent",
        version="1",
        description="",
        definition=OrchidPipelineDefinitionConfig(),
        jobs={job.id: job for job in configs},
        limits=PipelineLimits(max_llm_calls=10),
        shapes=_shapes(),
    )
    child = _child(_UsageStep(6), output_mapping={"value": "jobs.load.output.value"})

    result = _run(parent, {"child": child})

    assert result.status == "failed"
    assert "max_llm_calls=10" in (result.error or "")
    assert result.graph_stats["llm_calls"] == 12


def test_depth_limit_fails_the_parent_job() -> None:
    async def grandchild_handler(ctx: Any) -> dict[str, Any]:
        return {"value": "grandchild"}

    grandchild = _child(
        grandchild_handler,
        name="grandchild",
        output_mapping={"value": "jobs.load.output.value"},
    )
    child_config = OrchidPipelineSubPipelineJobConfig.model_validate({"kind": "subpipeline", "pipeline": "grandchild"})
    child_job = CompiledJob(
        id="nested",
        kind="subpipeline",
        config=child_config,
        subpipeline="grandchild",
    )
    child = CompiledPipeline(
        name="child",
        version="1",
        description="",
        definition=OrchidPipelineDefinitionConfig(),
        jobs={"nested": child_job},
        limits=PipelineLimits(max_subpipeline_depth=1),
        shapes=_shapes(),
    )
    result = _run(
        _parent(limits=PipelineLimits(max_subpipeline_depth=1)),
        {"child": child, "grandchild": grandchild},
    )

    assert result.status == "failed"
    assert result.jobs["call"]["status"] == "failed"
    assert "max_subpipeline_depth=1" in result.jobs["call"]["error"]


def test_child_job_timeout_is_enforced() -> None:
    async def child_handler(ctx: Any) -> dict[str, Any]:
        await asyncio.sleep(0.3)
        return {"value": "late"}

    child = _child(child_handler, output_mapping={"value": "jobs.load.output.value"})
    started = time.perf_counter()

    result = _run(_parent(timeout_seconds=0.05), {"child": child})

    assert result.status == "failed"
    assert result.jobs["call"]["status"] == "failed"
    assert "timed out after 0.050" in result.jobs["call"]["error"]
    assert time.perf_counter() - started < 0.5


def test_runner_depth_guard_matches_the_one_based_config_cap() -> None:
    async def child_handler(ctx: Any) -> dict[str, Any]:
        return {"value": "x"}

    child = _child(child_handler, output_mapping={"value": "jobs.load.output.value"})
    runner = OrchidPipelineRunner(
        child,
        deps=PipelineDependencies(pipeline_resolver={"child": child}.get),
    )
    auth = OrchidAuthContext(access_token="t", user_id="u")

    at_cap = asyncio.run(runner.run_pipeline(child, {}, auth=auth, subpipeline_depth=2))
    assert at_cap.status == "succeeded"

    with pytest.raises(OrchidPipelineLimitError, match="max_subpipeline_depth=3"):
        asyncio.run(runner.run_pipeline(child, {}, auth=auth, subpipeline_depth=3))


def test_root_depth_cap_allows_a_valid_nested_chain() -> None:
    async def leaf_handler(ctx: Any) -> dict[str, Any]:
        return {"value": "leaf"}

    leaf = _child(leaf_handler, name="cc", output_mapping={"value": "jobs.load.output.value"})
    bb = _sub_pipeline("bb", "cc", limits=PipelineLimits(max_subpipeline_depth=1))
    aa = _sub_pipeline("aa", "bb", limits=PipelineLimits(max_subpipeline_depth=3))

    result = _run(aa, {"bb": bb, "cc": leaf})

    assert result.status == "succeeded"
    assert result.jobs["call"]["status"] == "succeeded"


def test_per_definition_depth_cap_applies_when_pipeline_is_the_root() -> None:
    async def leaf_handler(ctx: Any) -> dict[str, Any]:
        return {"value": "leaf"}

    leaf = _child(leaf_handler, name="cc", output_mapping={"value": "jobs.load.output.value"})
    bb = _sub_pipeline("bb", "cc", limits=PipelineLimits(max_subpipeline_depth=1))

    result = _run(bb, {"bb": bb, "cc": leaf})

    assert result.status == "failed"
    assert result.jobs["call"]["status"] == "failed"
    assert "max_subpipeline_depth=1" in result.jobs["call"]["error"]


def test_child_inherits_the_parent_deadline() -> None:
    async def child_handler(ctx: Any) -> dict[str, Any]:
        await asyncio.sleep(0.2)
        return {"value": "late"}

    child = _child(child_handler, output_mapping={"value": "jobs.load.output.value"})
    started = time.perf_counter()

    result = _run(_parent(), {"child": child}, timeout=0.05)

    assert result.status == "failed"
    assert result.jobs["call"]["status"] == "failed"
    assert "timed out" in result.jobs["call"]["error"]
    assert time.perf_counter() - started < 0.5


def test_registry_exposes_a_subpipeline_resolver() -> None:
    block = OrchidPipelinesConfig.model_validate(
        {
            "definitions": {
                "child": {
                    "jobs": {
                        "emit": {
                            "kind": "emit",
                            "signal": "notify",
                            "payload": {"message": "hello"},
                        }
                    }
                }
            }
        }
    )

    registry = PipelineRegistry.from_config(block, deps=PipelineDependencies())

    child = registry.resolver("child")
    assert child is not None
    assert child.name == "child"
    assert registry.resolver("missing") is None


def test_run_context_depth_fields_are_validated() -> None:
    base = {
        "run_id": "run",
        "pipeline": "pipeline",
        "pipeline_version": "1",
        "mode": "direct",
        "tenant_key": "t",
        "user_id": "u",
        "started_at": datetime.now(UTC),
    }

    with pytest.raises(TypeError):
        OrchidPipelineRunContext(**base, subpipeline_depth=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        OrchidPipelineRunContext(**base, subpipeline_depth=-1)
    with pytest.raises(TypeError):
        OrchidPipelineRunContext(**base, subpipeline_max_depth=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        OrchidPipelineRunContext(**base, subpipeline_max_depth=0)


def test_child_runner_reuses_the_parent_runner_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    original = OrchidPipelineRunner.__init__
    instances: list[Any] = []

    def spy(self: Any, *args: Any, **kwargs: Any) -> None:
        instances.append(self)
        original(self, *args, **kwargs)

    monkeypatch.setattr(OrchidPipelineRunner, "__init__", spy)

    async def child_handler(ctx: Any) -> dict[str, Any]:
        return {"value": "same-runner"}

    child = _child(child_handler, output_mapping={"value": "jobs.load.output.value"})
    result = _run(_parent(), {"child": child})

    assert result.status == "succeeded"
    assert len(instances) == 1
