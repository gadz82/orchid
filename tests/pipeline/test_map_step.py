"""Tests for map fan-out topology, item binding and failure policies."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest
from langgraph.errors import GraphBubbleUp

from orchid_ai.config.schema_pipeline import (
    OrchidPipelineDecisionJobConfig,
    OrchidPipelineDefinitionConfig,
    OrchidPipelineDeterministicJobConfig,
    OrchidPipelineMapJobConfig,
    OrchidPipelineRetryConfig,
    OrchidPipelineShapeConfig,
)
from orchid_ai.core.pipeline import (
    OrchidPipelineConfigError,
    OrchidPipelineRunContext,
    OrchidPipelineStep,
    OrchidPipelineStepError,
    OrchidPipelineStepResult,
)
from orchid_ai.pipeline.builder import build_pipeline_graph
from orchid_ai.pipeline.definition import CompiledJob, CompiledPipeline, PipelineLimits
from orchid_ai.pipeline.mapping import build_mapping_context
from orchid_ai.pipeline.runner import OrchidPipelineRunner
from orchid_ai.pipeline.schema import compile_shapes
from orchid_ai.pipeline.steps.map import _validate_item_write_targets


def _shapes() -> Any:
    return compile_shapes(
        {
            "Account": OrchidPipelineShapeConfig(
                fields={"name": {"type": "string", "required": True}},
            ),
        }
    )


def _template(
    handler: Any,
    *,
    writes: list[dict[str, Any]] | None = None,
    when: str | None = None,
) -> CompiledJob:
    config = OrchidPipelineDeterministicJobConfig.model_validate(
        {"kind": "deterministic", "uses": "pkg.item", "writes": writes or [], "when": when}
    )
    return CompiledJob(
        id="item",
        kind="deterministic",
        config=config,
        handler=handler,
        writes=tuple(config.writes),
        when=when,
    )


def _map(
    *,
    over: str = "input.accounts",
    binding: str = "account",
    concurrency: int = 1,
    on_item_error: str = "fail",
    when: str | None = None,
    depends_on: list[str] | None = None,
    job_id: str = "fan",
    on_error: str = "fail",
    timeout_seconds: float | None = None,
    writes: list[dict[str, Any]] | None = None,
) -> CompiledJob:
    config = OrchidPipelineMapJobConfig.model_validate(
        {
            "kind": "map",
            "over": over,
            "as": binding,
            "job": "item",
            "concurrency": concurrency,
            "on_item_error": on_item_error,
            "when": when,
            "depends_on": depends_on or [],
            "on_error": on_error,
            "timeout_seconds": timeout_seconds,
            "writes": writes or [],
        }
    )
    return CompiledJob(
        id=job_id,
        kind="map",
        config=config,
        map_template="item",
        when=config.when,
        depends_on=tuple(config.depends_on),
        on_error=on_error,
        timeout_seconds=timeout_seconds,
        writes=tuple(config.writes),
    )


def _pipeline(*jobs: CompiledJob, limits: PipelineLimits | None = None) -> CompiledPipeline:
    return CompiledPipeline(
        name="demo",
        version="1",
        description="",
        definition=OrchidPipelineDefinitionConfig(),
        jobs={job.id: job for job in jobs},
        limits=limits or PipelineLimits.from_config(),
        shapes=_shapes(),
    )


def _run(pipeline: CompiledPipeline, payload: dict[str, Any], **kwargs: Any) -> Any:
    return asyncio.run(OrchidPipelineRunner(pipeline).run(payload, user_id="u1", **kwargs))


# ── Fan-out and binding ────────────────────────────────────


def test_map_fans_out_and_binds_item_index_and_parent() -> None:
    seen: list[dict[str, Any]] = []

    async def item(ctx: Any) -> dict[str, Any]:
        seen.append(
            {
                "item": ctx.job.item,
                "index": ctx.job.index,
                "parent_job": ctx.job.parent_job,
                "binding": ctx.mapping_context["account"],
                "context_item": ctx.mapping_context["item"],
                "context_index": ctx.mapping_context["index"],
            }
        )
        return {"name": ctx.job.item["name"]}

    result = _run(
        _pipeline(_map(concurrency=2), _template(item)),
        {"accounts": [{"name": "a"}, {"name": "b"}, {"name": "c"}]},
    )

    assert result.status == "succeeded"
    assert set(result.jobs) == {"fan"}  # the template is not a normal job
    assert set(result.jobs["fan"]["output"]["items"]) == {"fan#0", "fan#1", "fan#2"}
    assert result.jobs["fan"]["output"]["failures"] == []
    assert result.jobs["fan"]["output"]["skipped"] == []
    assert sorted(entry["index"] for entry in seen) == [0, 1, 2]
    by_index = {entry["index"]: entry for entry in seen}
    assert by_index[1]["parent_job"] == "fan"
    assert by_index[1]["item"] == {"name": "b"}
    assert by_index[1]["binding"] == {"name": "b"}
    assert by_index[1]["context_item"] == {"name": "b"}
    assert by_index[1]["context_index"] == 1


def test_map_concurrency_is_capped() -> None:
    active = 0
    maximum = 0
    lock = asyncio.Lock()

    async def item(ctx: Any) -> dict[str, Any]:
        nonlocal active, maximum
        async with lock:
            active += 1
            maximum = max(maximum, active)
        await asyncio.sleep(0.02)
        async with lock:
            active -= 1
        return {"name": ctx.job.item["name"]}

    result = _run(
        _pipeline(_map(concurrency=2), _template(item)),
        {"accounts": [{"name": str(index)} for index in range(6)]},
    )

    assert result.status == "succeeded"
    assert maximum == 2


def test_empty_collection_succeeds_with_no_items() -> None:
    called: list[Any] = []

    async def item(ctx: Any) -> dict[str, Any]:
        called.append(ctx.job.index)
        return {"name": "unused"}

    result = _run(_pipeline(_map(), _template(item)), {"accounts": []})

    assert result.status == "succeeded"
    assert result.jobs["fan"]["output"] == {"items": {}, "failures": [], "skipped": [], "cancelled": []}
    assert called == []


def test_map_when_false_skips_the_template() -> None:
    called: list[Any] = []

    async def item(ctx: Any) -> dict[str, Any]:
        called.append(ctx.job.index)
        return {"name": "unused"}

    result = _run(
        _pipeline(_map(when="input.skip"), _template(item)),
        {"accounts": [{"name": "a"}], "skip": False},
    )

    assert result.status == "succeeded"
    assert result.jobs["fan"]["status"] == "skipped"
    assert "when expression" in result.jobs["fan"]["reason"]
    assert called == []


# ── Item failure policies ─────────────────────────────────


def test_item_failure_skip_lists_the_failure_and_keeps_the_run_green() -> None:
    async def item(ctx: Any) -> dict[str, Any]:
        if ctx.job.index == 1:
            raise OrchidPipelineStepError("item exploded", job_id="item", kind="deterministic")
        return {"name": ctx.job.item["name"]}

    result = _run(
        _pipeline(_map(on_item_error="skip"), _template(item)),
        {"accounts": [{"name": "a"}, {"name": "b"}, {"name": "c"}]},
    )

    output = result.jobs["fan"]["output"]
    assert result.status == "succeeded"
    assert result.jobs["fan"]["status"] == "succeeded"
    assert set(output["items"]) == {"fan#0", "fan#2"}
    assert output["failures"] == [
        {
            "slot": "fan#1",
            "index": 1,
            "error": "item exploded",
            "error_code": "pipeline_step_error",
            "retryable": False,
        }
    ]
    assert result.jobs["fan"]["metadata"]["item_failures"] == 1


def test_item_failure_fail_fails_the_map_job() -> None:
    async def item(ctx: Any) -> dict[str, Any]:
        if ctx.job.index == 1:
            raise OrchidPipelineStepError("item exploded", job_id="item", kind="deterministic")
        return {"name": ctx.job.item["name"]}

    result = _run(
        _pipeline(_map(on_item_error="fail"), _template(item)),
        {"accounts": [{"name": "a"}, {"name": "b"}]},
    )

    assert result.status == "failed"
    assert result.jobs["fan"]["status"] == "failed"
    assert "failed for 1 item" in result.jobs["fan"]["error"]
    assert result.jobs["fan"]["error_code"] == "pipeline_step_error"


def test_item_failure_with_generic_exception_is_sanitized() -> None:
    async def item(ctx: Any) -> dict[str, Any]:
        if ctx.job.index == 0:
            raise RuntimeError("secret sk-test")
        return {"name": ctx.job.item["name"]}

    result = _run(
        _pipeline(_map(on_item_error="skip"), _template(item)),
        {"accounts": [{"name": "a"}, {"name": "b"}]},
    )

    failure = result.jobs["fan"]["output"]["failures"][0]
    assert "secret" not in failure["error"]
    assert failure["error_code"] == "pipeline_step_error"


def test_map_depends_on_a_normal_job_and_feeds_a_dependent() -> None:
    async def before(ctx: Any) -> dict[str, Any]:
        return {"accounts": [{"name": "a"}, {"name": "b"}]}

    async def item(ctx: Any) -> dict[str, Any]:
        return {"name": ctx.job.item["name"]}

    def after(ctx: Any) -> dict[str, Any]:
        return {"count": len(ctx.mapping_context.jobs["fan"]["output"]["items"])}

    before_config = OrchidPipelineDeterministicJobConfig.model_validate({"kind": "deterministic", "uses": "pkg.before"})
    before_job = CompiledJob(id="before", kind="deterministic", config=before_config, handler=before)
    after_config = OrchidPipelineDeterministicJobConfig.model_validate(
        {"kind": "deterministic", "uses": "pkg.after", "depends_on": ["fan"]}
    )
    after_job = CompiledJob(
        id="after",
        kind="deterministic",
        config=after_config,
        handler=after,
        depends_on=("fan",),
    )

    result = _run(
        _pipeline(
            before_job,
            _map(over="jobs.before.output.accounts", depends_on=["before"]),
            _template(item),
            after_job,
        ),
        {},
    )

    assert result.status == "succeeded"
    assert set(result.jobs["fan"]["output"]["items"]) == {"fan#0", "fan#1"}
    assert result.jobs["after"]["output"] == {"count": 2}


class _UsageStep(OrchidPipelineStep):
    def __init__(self, fail_index: int) -> None:
        self._fail_index = fail_index

    async def execute(self, ctx: Any) -> OrchidPipelineStepResult:
        if ctx.job.index == self._fail_index:
            raise OrchidPipelineStepError("item exploded", job_id="item", kind="deterministic")
        return OrchidPipelineStepResult(
            output={"name": ctx.job.item["name"]},
            usage={"llm_calls": 5, "prompt_tokens": 10},
        )


def _after(join: str = "all") -> CompiledJob:
    config = OrchidPipelineDeterministicJobConfig.model_validate(
        {"kind": "deterministic", "uses": "pkg.after", "depends_on": ["fan"], "join": join}
    )
    return CompiledJob(
        id="after",
        kind="deterministic",
        config=config,
        handler=lambda ctx: {"ok": True},
        depends_on=("fan",),
        join=join,
    )


def test_join_any_dependent_runs_after_a_skipped_map() -> None:
    async def item(ctx: Any) -> dict[str, Any]:
        return {"name": "unused"}

    result = _run(
        _pipeline(_map(when="input.skip"), _template(item), _after(join="any")),
        {"accounts": [{"name": "a"}], "skip": False},
    )

    assert result.status == "succeeded"
    assert result.jobs["fan"]["status"] == "skipped"
    assert result.jobs["after"]["status"] == "succeeded"


def test_join_any_dependent_runs_after_a_failed_map_with_skip_policy() -> None:
    async def item(ctx: Any) -> dict[str, Any]:
        return {"name": "unused"}

    result = _run(
        _pipeline(
            _map(on_error="skip", over="input.accounts"),
            _template(item),
            _after(join="any"),
            limits=PipelineLimits(max_map_items=1),
        ),
        {"accounts": [{"name": "a"}, {"name": "b"}]},
    )

    assert result.status == "partial"
    assert result.jobs["fan"]["status"] == "failed"
    assert result.jobs["after"]["status"] == "succeeded"


def test_token_shaped_item_error_is_redacted() -> None:
    token = "sk-live-ABCDEF0123456789SECRET"

    async def item(ctx: Any) -> dict[str, Any]:
        raise OrchidPipelineStepError(f"upstream 401: Authorization: Bearer {token}")

    result = _run(
        _pipeline(_map(on_item_error="skip"), _template(item)),
        {"accounts": [{"name": "a"}]},
    )

    failure = result.jobs["fan"]["output"]["failures"][0]
    assert token not in failure["error"]
    assert "[redacted]" in failure["error"]


def test_failure_policy_preserves_items_failures_and_usage() -> None:
    result = _run(
        _pipeline(
            _map(on_item_error="fail"),
            _template(
                _UsageStep(fail_index=1),
                writes=[{"id": "account:{{ account.name }}", "shape": "Account"}],
            ),
        ),
        {"accounts": [{"name": "a"}, {"name": "b"}]},
    )

    job = result.jobs["fan"]
    assert result.status == "failed"
    assert job["status"] == "failed"
    assert job["output"]["items"] == {"fan#0": {"name": "a"}}
    assert job["output"]["failures"][0]["slot"] == "fan#1"
    assert job["llm_calls"] == 5
    assert job["prompt_tokens"] == 10
    assert job["objects_written"] == 1
    assert result.graph_stats["objects_written"] == 1


def test_map_timeout_records_a_timeout_failure() -> None:
    async def item(ctx: Any) -> dict[str, Any]:
        await asyncio.sleep(0.2)
        return {"name": ctx.job.item["name"]}

    result = _run(
        _pipeline(_map(timeout_seconds=0.01), _template(item)),
        {"accounts": [{"name": "a"}]},
    )

    assert result.status == "failed"
    failure = result.jobs["fan"]["output"]["failures"][0]
    assert failure["error_code"] == "pipeline_timeout"
    assert "timed out" in failure["error"]


def test_run_deadline_cancels_the_map() -> None:
    started: list[int] = []

    async def item(ctx: Any) -> dict[str, Any]:
        started.append(ctx.job.index)
        await asyncio.sleep(0.05)
        return {"name": ctx.job.item["name"]}

    result = _run(
        _pipeline(_map(concurrency=1), _template(item)),
        {"accounts": [{"name": str(index)} for index in range(5)]},
        timeout=0.06,
    )

    assert result.status == "cancelled"
    assert result.jobs["fan"]["status"] == "cancelled"
    assert result.jobs["fan"]["output"]["cancelled"]
    assert len(started) < 5


def test_item_context_exposes_the_normal_attribute_api() -> None:
    seen: list[Any] = []

    async def item(ctx: Any) -> dict[str, Any]:
        seen.append((ctx.mapping_context.item, ctx.mapping_context.index, ctx.mapping_context.jobs))
        return {"name": ctx.job.item["name"]}

    result = _run(_pipeline(_map(), _template(item)), {"accounts": [{"name": "a"}]})

    assert result.status == "succeeded"
    assert seen[0][0] == {"name": "a"}
    assert seen[0][1] == 0
    assert seen[0][2] == {}


def test_map_concurrency_is_capped_by_max_parallel() -> None:
    active = 0
    maximum = 0
    lock = asyncio.Lock()

    async def item(ctx: Any) -> dict[str, Any]:
        nonlocal active, maximum
        async with lock:
            active += 1
            maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        async with lock:
            active -= 1
        return {"name": ctx.job.item["name"]}

    result = _run(
        _pipeline(
            _map(concurrency=5),
            _template(item),
            limits=PipelineLimits(max_parallel=2),
        ),
        {"accounts": [{"name": str(index)} for index in range(6)]},
    )

    assert result.status == "succeeded"
    assert maximum == 2


# ── Limits and writes ────────────────────────────────────


def test_max_map_items_fails_before_any_item_runs() -> None:
    called: list[Any] = []

    async def item(ctx: Any) -> dict[str, Any]:
        called.append(ctx.job.index)
        return {"name": "unused"}

    result = _run(
        _pipeline(_map(), _template(item), limits=PipelineLimits(max_map_items=2)),
        {"accounts": [{"name": str(index)} for index in range(3)]},
    )

    assert result.status == "failed"
    assert result.jobs["fan"]["status"] == "failed"
    assert "max_map_items=2" in result.jobs["fan"]["error"]
    assert called == []


def test_item_writes_use_the_item_context_and_distinct_ids() -> None:
    async def item(ctx: Any) -> dict[str, Any]:
        return {"name": ctx.job.item["name"]}

    result = _run(
        _pipeline(
            _map(concurrency=2),
            _template(item, writes=[{"id": "account:{{ account.name }}", "shape": "Account"}]),
        ),
        {"accounts": [{"name": "a"}, {"name": "b"}, {"name": "c"}]},
    )

    assert result.status == "succeeded"
    assert result.graph_stats["objects_written"] == 3
    assert set(result.output["objects"]) >= {"account:a", "account:b", "account:c"}
    assert result.jobs["fan"]["objects_written"] == 3


def test_template_retry_applies_per_item() -> None:
    calls: list[int] = []

    async def item(ctx: Any) -> dict[str, Any]:
        calls.append(ctx.job.index)
        if len(calls) == 1:
            raise OrchidPipelineStepError("transient", retryable=True)
        return {"name": ctx.job.item["name"]}

    retry = OrchidPipelineRetryConfig(max=1, backoff="fixed", initial_delay_seconds=0.001, jitter=False)
    template = _template(item)
    template = CompiledJob(
        id=template.id,
        kind=template.kind,
        config=template.config,
        handler=template.handler,
        retry=retry,
        writes=template.writes,
    )
    result = _run(_pipeline(_map(), template), {"accounts": [{"name": "a"}]})

    assert result.status == "succeeded"
    assert calls == [0, 0]


def test_map_can_be_a_decision_target() -> None:
    class _DecisionStep(OrchidPipelineStep):
        async def execute(self, ctx: Any) -> OrchidPipelineStepResult:
            return OrchidPipelineStepResult(output={"label": "go", "confidence": 1.0})

    decision_config = OrchidPipelineDecisionJobConfig.model_validate(
        {
            "kind": "decision",
            "decision_model": "stub",
            "question": "Go?",
            "labels": ["go"],
            "branches": {"go": "fan"},
        }
    )
    decision = CompiledJob(
        id="choose",
        kind="decision",
        config=decision_config,
        handler=_DecisionStep(),
        branches={"go": "fan"},
    )

    async def item(ctx: Any) -> dict[str, Any]:
        return {"name": ctx.job.item["name"]}

    result = _run(
        _pipeline(decision, _map(), _template(item)),
        {"accounts": [{"name": "a"}]},
    )

    assert result.status == "succeeded"
    assert result.jobs["choose"]["output"]["label"] == "go"
    assert set(result.jobs["fan"]["output"]["items"]) == {"fan#0"}


def test_item_write_mapping_error_is_recorded_on_the_map_job() -> None:
    async def item(ctx: Any) -> dict[str, Any]:
        return {"name": ctx.job.item.get("name", "x")}

    result = _run(
        _pipeline(
            _map(on_error="skip"),
            _template(item, writes=[{"id": "account:{{ account.name }}", "shape": "Account"}]),
        ),
        {"accounts": [{"name": "a"}, {"other": "b"}]},
    )

    assert result.jobs["fan"]["status"] == "failed"
    assert "resolved to no value" in result.jobs["fan"]["error"]
    assert result.status == "partial"


def test_item_write_size_error_is_recorded_on_the_map_job() -> None:
    async def item(ctx: Any) -> dict[str, Any]:
        return {"name": "x"}

    target = "account:" + "x" * 80
    result = _run(
        _pipeline(
            _map(),
            _template(item, writes=[{"id": target, "shape": "Account"}]),
            limits=PipelineLimits(max_prompt_chars=20),
        ),
        {"accounts": [{"name": "a"}]},
    )

    assert result.jobs["fan"]["status"] == "failed"
    assert "exceeding the 20-character cap" in result.jobs["fan"]["error"]


def test_dynamic_duplicate_item_targets_are_rejected() -> None:
    async def item(ctx: Any) -> dict[str, Any]:
        return {"name": ctx.job.item["name"]}

    result = _run(
        _pipeline(
            _map(),
            _template(item, writes=[{"id": "account:{{ account.name }}", "shape": "Account"}]),
        ),
        {"accounts": [{"name": "same"}, {"name": "same"}]},
    )

    assert result.jobs["fan"]["status"] == "failed"
    assert "not item-specific" in result.jobs["fan"]["error"]


def test_map_own_write_cannot_overwrite_an_item_target() -> None:
    async def item(ctx: Any) -> dict[str, Any]:
        return {"name": ctx.job.item["name"]}

    result = _run(
        _pipeline(
            _map(writes=[{"id": "account:a", "shape": "Account", "from": "input.overwrite"}]),
            _template(item, writes=[{"id": "account:{{ account.name }}", "shape": "Account"}]),
        ),
        {"accounts": [{"name": "a"}], "overwrite": {"name": "map"}},
    )

    assert result.jobs["fan"]["status"] == "failed"
    assert "not item-specific" in result.jobs["fan"]["error"]


def test_template_when_skips_only_the_matching_items() -> None:
    async def item(ctx: Any) -> dict[str, Any]:
        return {"name": ctx.job.item["name"]}

    result = _run(
        _pipeline(
            _map(),
            _template(item, when="account.name == 'a'"),
        ),
        {"accounts": [{"name": "a"}, {"name": "b"}]},
    )

    output = result.jobs["fan"]["output"]
    assert result.status == "succeeded"
    assert set(output["items"]) == {"fan#0"}
    assert output["skipped"] == [{"slot": "fan#1", "index": 1, "reason": "when expression evaluated to false"}]


def test_item_graph_bubble_up_is_re_raised() -> None:
    async def item(ctx: Any) -> dict[str, Any]:
        raise GraphBubbleUp("control flow")

    result = _run(_pipeline(_map(), _template(item)), {"accounts": [{"name": "a"}]})

    assert result.status == "failed"


def test_template_when_excludes_skipped_items_from_target_validation() -> None:
    async def item(ctx: Any) -> dict[str, Any]:
        return {"name": ctx.job.item["name"]}

    result = _run(
        _pipeline(
            _map(),
            _template(
                item,
                writes=[{"id": "account:fixed", "shape": "Account"}],
                when="index == `0`",
            ),
        ),
        {"accounts": [{"name": "a"}, {"name": "b"}]},
    )

    assert result.status == "succeeded"
    assert set(result.jobs["fan"]["output"]["items"]) == {"fan#0"}
    assert result.jobs["fan"]["output"]["skipped"][0]["slot"] == "fan#1"


def test_duplicate_write_within_one_item_is_rejected() -> None:
    async def item(ctx: Any) -> dict[str, Any]:
        return {"name": ctx.job.item["name"]}

    result = _run(
        _pipeline(
            _map(),
            _template(
                item,
                writes=[
                    {"id": "account:fixed", "shape": "Account"},
                    {"id": "account:fixed", "shape": "Account"},
                ],
            ),
        ),
        {"accounts": [{"name": "a"}]},
    )

    assert result.jobs["fan"]["status"] == "failed"
    assert "declared twice in item 0" in result.jobs["fan"]["error"]


def test_object_and_relation_namespaces_do_not_collide() -> None:
    config = OrchidPipelineDeterministicJobConfig.model_validate(
        {
            "kind": "deterministic",
            "uses": "pkg.item",
            "writes": [{"id": "account:123:456", "shape": "Account"}],
            "links": [{"type": "account", "from": "123", "to": "456"}],
        }
    )
    template = CompiledJob(
        id="item",
        kind="deterministic",
        config=config,
        writes=tuple(config.writes),
        links=tuple(config.links),
    )
    run = OrchidPipelineRunContext(
        run_id="run-1",
        pipeline="demo",
        pipeline_version="1",
        mode="direct",
        tenant_key="t",
        user_id="u",
        started_at=datetime.now(UTC),
    )
    context = build_mapping_context(input={}, run=run)

    _validate_item_write_targets(template, _map(), [{"name": "a"}], context, "account", "fan", 1000)


def test_map_own_write_mapping_error_is_recorded() -> None:
    async def item(ctx: Any) -> dict[str, Any]:
        return {"name": ctx.job.item["name"]}

    result = _run(
        _pipeline(
            _map(on_error="skip", writes=[{"id": "summary:{{ input.missing }}", "shape": "Account"}]),
            _template(item),
        ),
        {"accounts": [{"name": "a"}]},
    )

    assert result.jobs["fan"]["status"] == "failed"
    assert "resolved to no value" in result.jobs["fan"]["error"]


def test_prevalidation_failure_records_a_join_any_dependent() -> None:
    async def item(ctx: Any) -> dict[str, Any]:
        return {"name": ctx.job.item["name"]}

    result = _run(
        _pipeline(
            _map(on_error="skip"),
            _template(item, writes=[{"id": "account:fixed", "shape": "Account"}]),
            _after(join="any"),
        ),
        {"accounts": [{"name": "a"}, {"name": "b"}]},
    )

    assert result.status == "partial"
    assert result.jobs["fan"]["status"] == "failed"
    assert result.jobs["after"]["status"] == "succeeded"


# ── Topology guards ──────────────────────────────────────


def test_nested_map_template_is_rejected() -> None:
    inner = _map(job_id="inner")
    outer = _map(job_id="outer")
    inner_template = CompiledJob(
        id="item",
        kind="map",
        config=inner.config,
        map_template="inner",
    )
    pipeline = _pipeline(outer, inner_template)

    with pytest.raises(OrchidPipelineConfigError, match="nested maps"):
        build_pipeline_graph(pipeline)


def test_static_item_write_targets_are_rejected_before_any_item_runs() -> None:
    called: list[Any] = []

    async def item(ctx: Any) -> dict[str, Any]:
        called.append(ctx.job.index)
        return {"name": ctx.job.item["name"]}

    result = _run(
        _pipeline(
            _map(),
            _template(item, writes=[{"id": "account:fixed", "shape": "Account"}]),
        ),
        {"accounts": [{"name": "a"}, {"name": "b"}]},
    )

    assert result.status == "failed"
    assert "not item-specific" in result.jobs["fan"]["error"]
    assert called == []


def test_template_aliases_are_rejected_at_build_time() -> None:
    template = _template(
        lambda ctx: {"name": "x"},
        writes=[{"id": "account:{{ account.name }}", "shape": "Account", "as": "account"}],
    )

    with pytest.raises(OrchidPipelineConfigError, match="declares aliases"):
        build_pipeline_graph(_pipeline(_map(), template))


def test_generated_node_name_collision_is_rejected() -> None:
    collision_config = OrchidPipelineDeterministicJobConfig.model_validate(
        {"kind": "deterministic", "uses": "pkg.collision"}
    )
    collision = CompiledJob(
        id="fan__item",
        kind="deterministic",
        config=collision_config,
        handler=lambda ctx: {},
    )

    with pytest.raises(OrchidPipelineConfigError, match="generated node name"):
        build_pipeline_graph(_pipeline(_map(), _template(lambda ctx: {"name": "x"}), collision))


def test_template_dependencies_are_rejected() -> None:
    before_config = OrchidPipelineDeterministicJobConfig.model_validate({"kind": "deterministic", "uses": "pkg.before"})
    before = CompiledJob(id="before", kind="deterministic", config=before_config, handler=lambda ctx: {})
    template_config = OrchidPipelineDeterministicJobConfig.model_validate(
        {"kind": "deterministic", "uses": "pkg.item", "depends_on": ["before"]}
    )
    template = CompiledJob(
        id="item",
        kind="deterministic",
        config=template_config,
        handler=lambda ctx: {"name": "x"},
        depends_on=("before",),
    )

    with pytest.raises(OrchidPipelineConfigError, match="must not declare dependencies"):
        build_pipeline_graph(_pipeline(_map(), template, before))


def test_dependency_on_a_map_template_is_rejected() -> None:
    after_config = OrchidPipelineDeterministicJobConfig.model_validate(
        {"kind": "deterministic", "uses": "pkg.after", "depends_on": ["item"]}
    )
    after = CompiledJob(
        id="after",
        kind="deterministic",
        config=after_config,
        handler=lambda ctx: {"ok": True},
        depends_on=("item",),
    )

    with pytest.raises(OrchidPipelineConfigError, match="depends on map template"):
        build_pipeline_graph(_pipeline(_map(), _template(lambda ctx: {"name": "x"}), after))


def test_conditional_target_that_is_a_map_template_is_rejected() -> None:
    from orchid_ai.config.schema_pipeline import OrchidPipelineDecisionJobConfig

    decision_config = OrchidPipelineDecisionJobConfig.model_validate(
        {
            "kind": "decision",
            "decision_model": "stub",
            "question": "Choose",
            "labels": ["go"],
            "branches": {"go": "item"},
        }
    )
    decision = CompiledJob(
        id="choose",
        kind="decision",
        config=decision_config,
        branches={"go": "item"},
    )

    with pytest.raises(OrchidPipelineConfigError, match="conditional target"):
        build_pipeline_graph(_pipeline(decision, _map(), _template(lambda ctx: {"name": "x"})))
