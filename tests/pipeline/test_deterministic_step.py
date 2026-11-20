"""Tests for the deterministic pipeline step."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from orchid_ai.config.schema_pipeline import OrchidPipelineDeterministicJobConfig
from orchid_ai.core.pipeline import (
    OrchidGraphMutation,
    OrchidPipelineStep,
    OrchidPipelineStepError,
    OrchidPipelineStepResult,
    OrchidPipelineTimeoutError,
)
from orchid_ai.pipeline.definition import CompiledJob
from orchid_ai.pipeline.steps.deterministic import DeterministicStep, normalize_result


def _job(**overrides: Any) -> CompiledJob:
    values: dict[str, Any] = {"kind": "deterministic", "uses": "pkg.step"}
    values.update(overrides)
    config = OrchidPipelineDeterministicJobConfig.model_validate(values)
    return CompiledJob(
        id="run",
        kind="deterministic",
        config=config,
        timeout_seconds=config.timeout_seconds,
    )


def _ctx(**overrides: Any) -> Any:
    values: dict[str, Any] = {"run": SimpleNamespace(run_id="run-1"), "job": SimpleNamespace(job_id="run")}
    values.update(overrides)
    return SimpleNamespace(**values)


def _mutation() -> OrchidGraphMutation:
    return OrchidGraphMutation(op="create", object_id="account:1", object_type="Account", data={"id": "1"})


# ── Normalization ───────────────────────────────────────────


def test_normalize_none_yields_an_empty_result() -> None:
    result = normalize_result(None, job_id="run", kind="deterministic")

    assert result.output is None
    assert result.mutations == ()


def test_normalize_mapping_is_the_output() -> None:
    result = normalize_result({"account": {"id": "1"}}, job_id="run", kind="deterministic")

    assert result.output == {"account": {"id": "1"}}


def test_normalize_step_result_is_returned_as_is() -> None:
    original = OrchidPipelineStepResult(output={"ok": True})

    assert normalize_result(original, job_id="run", kind="deterministic") is original


def test_normalize_mutation_list() -> None:
    mutation = _mutation()

    result = normalize_result([mutation], job_id="run", kind="deterministic")

    assert result.mutations == (mutation,)
    assert result.output == {}


def test_normalize_empty_list_is_an_empty_result() -> None:
    result = normalize_result([], job_id="run", kind="deterministic")

    assert result.output is None
    assert result.mutations == ()


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("text", id="string"),
        pytest.param(3, id="int"),
        pytest.param([1, 2], id="list-of-ints"),
    ],
)
def test_normalize_rejects_unsupported_values(value: Any) -> None:
    with pytest.raises(OrchidPipelineStepError, match="must return None, a dict"):
        normalize_result(value, job_id="run", kind="deterministic")


# ── Execution ───────────────────────────────────────────────


def test_async_handler_receives_context_and_params() -> None:
    seen: dict[str, Any] = {}

    async def handler(ctx: Any, **params: Any) -> dict[str, Any]:
        seen["ctx"] = ctx
        seen["params"] = params
        return {"ok": True}

    ctx = _ctx()
    step = DeterministicStep(_job(params={"source": "crm"}), handler=handler)

    result = asyncio.run(step.execute(ctx))

    assert result.output == {"ok": True}
    assert seen["ctx"] is ctx
    assert seen["params"] == {"source": "crm"}


def test_sync_handler_runs_off_the_event_loop() -> None:
    main_thread = threading.get_ident()
    seen: dict[str, int] = {}

    def handler(ctx: Any, **params: Any) -> dict[str, Any]:
        seen["thread"] = threading.get_ident()
        return {"ok": True}

    step = DeterministicStep(_job(), handler=handler)

    asyncio.run(step.execute(_ctx()))

    assert seen["thread"] != main_thread


def test_sync_handler_returning_an_awaitable_is_awaited() -> None:
    def handler(ctx: Any, **params: Any) -> Any:
        async def inner() -> dict[str, Any]:
            return {"ok": True}

        return inner()

    step = DeterministicStep(_job(), handler=handler)

    assert asyncio.run(step.execute(_ctx())).output == {"ok": True}


def test_handler_exception_propagates() -> None:
    async def handler(ctx: Any, **params: Any) -> dict[str, Any]:
        raise RuntimeError("boom")

    step = DeterministicStep(_job(), handler=handler)

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(step.execute(_ctx()))


def test_timeout_is_enforced() -> None:
    async def handler(ctx: Any, **params: Any) -> dict[str, Any]:
        await asyncio.sleep(0.5)
        return {}

    step = DeterministicStep(_job(timeout_seconds=0.01), handler=handler)

    with pytest.raises(OrchidPipelineTimeoutError, match="timed out after 0.01 seconds"):
        asyncio.run(step.execute(_ctx()))


def test_no_timeout_waits_for_the_handler() -> None:
    async def handler(ctx: Any, **params: Any) -> dict[str, Any]:
        await asyncio.sleep(0.01)
        return {"done": True}

    step = DeterministicStep(_job(), handler=handler)

    assert asyncio.run(step.execute(_ctx())).output == {"done": True}


def test_missing_concrete_job_is_rejected() -> None:
    with pytest.raises(TypeError, match="CompiledJob"):
        DeterministicStep("not-a-job")  # type: ignore[arg-type]


def test_missing_handler_is_a_step_error() -> None:
    job = CompiledJob(
        id="run",
        kind="deterministic",
        config=OrchidPipelineDeterministicJobConfig.model_validate({"kind": "deterministic", "uses": "pkg.step"}),
    )

    with pytest.raises(OrchidPipelineStepError, match="no callable 'uses' handler"):
        DeterministicStep(job)


def test_step_declares_its_kind() -> None:
    assert DeterministicStep.kind == "deterministic"
    assert issubclass(DeterministicStep, OrchidPipelineStep)


def test_slow_sync_handler_does_not_block_the_loop() -> None:
    ticks: list[float] = []

    def handler(ctx: Any, **params: Any) -> dict[str, Any]:
        time.sleep(0.1)
        return {}

    async def run() -> None:
        step = DeterministicStep(_job(), handler=handler)
        task = asyncio.create_task(step.execute(_ctx()))
        for _ in range(3):
            await asyncio.sleep(0.01)
            ticks.append(time.monotonic())
        await task

    asyncio.run(run())

    # The loop kept running while the worker thread slept.
    assert len(ticks) == 3
    assert ticks[-1] - ticks[0] < 0.09
