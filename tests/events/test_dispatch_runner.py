"""DispatchJobRunner routes by ``JobSpec.target`` and fails unknown targets hard."""

from __future__ import annotations

import datetime as _dt
import uuid as _uuid

import pytest

from orchid_ai.core.events.errors import JobRunnerError
from orchid_ai.core.events.job import JobRun, JobSpec, JobStatus
from orchid_ai.events.bootstrap import _build_runner
from orchid_ai.events.runners.dispatch_runner import DispatchJobRunner


class _RecordingRunner:
    def __init__(self, status: JobStatus = JobStatus.SUCCEEDED) -> None:
        self.calls: list[object] = []
        self._status = status

    async def run(self, run: JobRun, *, auth) -> None:
        self.calls.append(auth)
        run.status = self._status
        run.finished_at = _dt.datetime.now(tz=_dt.UTC)


def _run(*, target: str = "agent", pipeline_name: str | None = None) -> JobRun:
    spec = JobSpec(
        trigger_id="t1",
        signal_id=_uuid.uuid4(),
        agent_name="notifications" if target == "agent" else "",
        prompt="hello" if target == "agent" else "",
        identity_claim={"mode": "service_account", "name": "bot"},
        correlation_id=None,
        parallelism_key="sa:t:bot",
        target=target,
        pipeline_name=pipeline_name,
        pipeline_input={} if target == "pipeline" else None,
    )
    return JobRun(
        run_id=_uuid.uuid4(),
        spec=spec,
        attempt_number=1,
        status=JobStatus.RUNNING,
        queued_at=_dt.datetime.now(tz=_dt.UTC),
        started_at=_dt.datetime.now(tz=_dt.UTC),
    )


async def test_dispatch_routes_pipeline_target_to_pipeline_runner() -> None:
    agent = _RecordingRunner()
    pipeline = _RecordingRunner()
    dispatcher = DispatchJobRunner(runners={"agent": agent, "pipeline": pipeline})

    run = _run(target="pipeline", pipeline_name="weekly_digest")
    auth = object()
    await dispatcher.run(run, auth=auth)

    assert pipeline.calls == [auth]
    assert agent.calls == []
    assert run.status == JobStatus.SUCCEEDED


async def test_dispatch_routes_agent_target_to_agent_runner() -> None:
    agent = _RecordingRunner()
    pipeline = _RecordingRunner()
    dispatcher = DispatchJobRunner(runners={"agent": agent, "pipeline": pipeline})

    await dispatcher.run(_run(target="agent"), auth=object())

    assert len(agent.calls) == 1
    assert pipeline.calls == []


async def test_dispatch_unknown_target_fails_without_retry() -> None:
    dispatcher = DispatchJobRunner(runners={"agent": _RecordingRunner()})
    run = _run(target="pipeline", pipeline_name="weekly_digest")

    with pytest.raises(JobRunnerError) as exc_info:
        await dispatcher.run(run, auth=object())

    assert exc_info.value.retryable is False
    assert "unknown job target 'pipeline'" in str(exc_info.value)
    assert run.status == JobStatus.FAILED
    assert run.finished_at is not None


async def test_dispatch_requires_at_least_one_runner() -> None:
    with pytest.raises(ValueError, match="at least one target runner"):
        DispatchJobRunner(runners={})


class _StubPipelineRunner:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def run(self, name, input, **kwargs) -> dict:
        self.calls.append({"name": name, "input": input, **kwargs})
        return {
            "final_response": "Digest ready",
            "pipeline": name,
            "pipeline_run_id": "pr-1",
            "status": "succeeded",
            "output": None,
            "graph_stats": {},
        }


async def test_build_runner_dispatches_pipeline_target_when_runner_present() -> None:
    pipeline = _StubPipelineRunner()
    runner = _build_runner(chat_storage=None, pipeline_runner=pipeline)
    assert isinstance(runner, DispatchJobRunner)

    run = _run(target="pipeline", pipeline_name="weekly_digest")
    await runner.run(run, auth=object())

    assert run.status == JobStatus.SUCCEEDED
    assert run.result is not None
    assert run.result["pipeline_run_id"] == "pr-1"
    assert pipeline.calls[0]["mode"] == "bloom"
