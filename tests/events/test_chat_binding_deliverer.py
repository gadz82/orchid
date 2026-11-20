"""Shared chat-binding deliverer and the pipeline runner's delivery contract."""

from __future__ import annotations

import datetime as _dt
import json
import uuid as _uuid

import pytest

from orchid_ai.core.events.errors import (
    ChatBindingTargetNotFoundError,
    JobRunnerError,
)
from orchid_ai.core.events.job import JobRun, JobSpec, JobStatus
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.events.runners.pipeline_runner import PipelineJobRunner
from orchid_ai.persistence.models import OrchidChatSession

_NOW = _dt.datetime(2026, 11, 19, 12, 0, tzinfo=_dt.UTC)


class _FakeChatStorage:
    def __init__(self) -> None:
        self.chats: dict[str, OrchidChatSession] = {}
        self.messages: list[dict] = []
        self.fail_append = False
        self._next_id = 1

    async def create_chat(self, tenant_id: str, user_id: str, title: str = "") -> OrchidChatSession:
        chat_id = f"chat-{self._next_id}"
        self._next_id += 1
        session = OrchidChatSession(
            id=chat_id,
            tenant_id=tenant_id,
            user_id=user_id,
            title=title,
            created_at=_NOW,
            updated_at=_NOW,
        )
        self.chats[chat_id] = session
        return session

    async def get_chat_metadata(self, chat_id: str) -> OrchidChatSession | None:
        return self.chats.get(chat_id)

    async def can_write(self, chat: OrchidChatSession, auth: OrchidAuthContext) -> bool:
        return chat.tenant_id == auth.tenant_key and chat.user_id == auth.user_id

    async def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        agents_used: list[str] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        if self.fail_append:
            raise RuntimeError("simulated chat append failure")
        record = {
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "agents_used": list(agents_used or []),
            "metadata": dict(metadata or {}),
        }
        self.messages.append(record)
        return record


def _auth() -> OrchidAuthContext:
    return OrchidAuthContext(access_token="secret-token", tenant_key="t-1", user_id="u-7")


def _spec(*, target: str = "pipeline", chat_binding: dict | None = None, **overrides) -> JobSpec:
    values = {
        "trigger_id": "t1",
        "signal_id": _uuid.uuid4(),
        "agent_name": "notifications" if target == "agent" else "",
        "prompt": "Run the digest" if target == "agent" else "",
        "identity_claim": {"mode": "act_as_user", "user_id_from": "signal.user_id"},
        "correlation_id": "corr",
        "parallelism_key": "user:t-1:u-7",
        "visibility": "actor",
        "visibility_user_id": "u-7",
        "chat_binding": chat_binding,
        "target": target,
        "pipeline_name": "weekly_digest" if target == "pipeline" else None,
        "pipeline_input": {"topic": "release notes"} if target == "pipeline" else None,
    }
    values.update(overrides)
    return JobSpec(**values)


def _run(*, target: str = "pipeline", chat_binding: dict | None = None, **overrides) -> JobRun:
    return JobRun(
        run_id=_uuid.uuid4(),
        spec=_spec(target=target, chat_binding=chat_binding, **overrides),
        attempt_number=1,
        status=JobStatus.RUNNING,
        queued_at=_NOW,
        started_at=_NOW,
    )


def _chat(storage: _FakeChatStorage, chat_id: str = "C-1") -> None:
    storage.chats[chat_id] = OrchidChatSession(
        id=chat_id,
        tenant_id="t-1",
        user_id="u-7",
        title="My chat",
        created_at=_NOW,
        updated_at=_NOW,
    )


class _StubPipelineRunner:
    def __init__(self, func=None, *, result=None, error=None) -> None:
        self._func = func
        self._result = result
        self._error = error
        self.calls: list[dict] = []

    async def run(self, name, input, **kwargs):
        self.calls.append({"name": name, "input": input, **kwargs})
        if self._error is not None:
            raise self._error
        if self._func is not None:
            return await self._func(name, input, **kwargs)
        return self._result


class _FakePipelineResult:
    def __init__(
        self,
        *,
        run_id: str = "pr-1",
        pipeline: str = "weekly_digest",
        status: str = "succeeded",
        summary: str = "The weekly digest is ready.",
        output: dict | None = None,
        graph: dict | None = None,
        error: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.pipeline = pipeline
        self.status = status
        self.summary = summary
        self.output = output
        self.graph = graph or {}
        self.graph_stats = {"objects": 4}
        self.error = error

    def to_dict(self, *, include_graph: bool = False) -> dict:
        payload = {
            "run_id": self.run_id,
            "pipeline": self.pipeline,
            "status": self.status,
            "output": self.output,
            "summary": self.summary,
            "summary_source": "template",
            "jobs": {"load": {"status": "succeeded"}},
            "graph_stats": self.graph_stats,
            "error": self.error,
        }
        if include_graph:
            payload["graph"] = self.graph
        return payload


async def _pipeline_ok(name, input, **kwargs) -> dict:
    return {
        "final_response": "The weekly digest is ready.",
        "pipeline": name,
        "pipeline_run_id": "pr-1",
        "status": "succeeded",
        "output": {"items": 3},
        "graph_stats": {"objects": 4},
        "graph": {"objects": [{"id": "o1"}]},
    }


# ── Pipeline runner ────────────────────────────────────────


async def test_pipeline_runner_stores_result_and_drops_graph_by_default() -> None:
    storage = _FakeChatStorage()
    _chat(storage)
    stub = _StubPipelineRunner(_pipeline_ok)
    runner = PipelineJobRunner(pipeline_runner=stub, chat_storage=storage, clock=lambda: _NOW)
    run = _run(chat_binding={"chat_id": "C-1"})

    await runner.run(run, auth=_auth())

    assert run.status == JobStatus.SUCCEEDED
    assert run.result is not None
    assert run.result["final_response"] == "The weekly digest is ready."
    assert run.result["pipeline"] == "weekly_digest"
    assert run.result["pipeline_run_id"] == "pr-1"
    assert run.result["status"] == "succeeded"
    assert run.result["output"] == {"items": 3}
    assert run.result["graph_stats"] == {"objects": 4}
    assert "graph" not in run.result
    assert run.finished_at == _NOW
    [call] = stub.calls
    assert call["name"] == "weekly_digest"
    assert call["input"] == {"topic": "release notes"}
    assert call["mode"] == "bloom"
    assert call["correlation_id"] == "corr"
    assert call["chat_id"] == "C-1"
    assert call["include_graph"] is False
    assert call["auth"] is not None
    [message] = storage.messages
    assert "weekly digest is ready" in message["content"]
    assert message["metadata"]["pipeline_run_id"] == "pr-1"


async def test_pipeline_runner_persists_graph_when_requested() -> None:
    stub = _StubPipelineRunner(_pipeline_ok)
    runner = PipelineJobRunner(pipeline_runner=stub, clock=lambda: _NOW)
    run = _run(persist_graph=True)

    await runner.run(run, auth=_auth())

    assert run.result is not None
    assert run.result["graph"] == {"objects": [{"id": "o1"}]}
    assert stub.calls[0]["include_graph"] is True


async def test_pipeline_runner_normalizes_to_dict_result_and_drops_extra_keys() -> None:
    stub = _StubPipelineRunner(result=_FakePipelineResult(output={"items": 3}, graph={"objects": [{"id": "o1"}]}))
    runner = PipelineJobRunner(pipeline_runner=stub, clock=lambda: _NOW)
    run = _run(persist_graph=True)

    await runner.run(run, auth=_auth())

    assert run.status == JobStatus.SUCCEEDED
    assert run.result is not None
    assert run.result["final_response"] == "The weekly digest is ready."
    assert run.result["pipeline_run_id"] == "pr-1"
    assert run.result["graph"] == {"objects": [{"id": "o1"}]}
    assert set(run.result) == {
        "final_response",
        "pipeline",
        "pipeline_run_id",
        "status",
        "output",
        "graph_stats",
        "graph",
    }


async def test_pipeline_runner_uses_generic_summary_when_none_is_configured() -> None:
    storage = _FakeChatStorage()
    _chat(storage)
    stub = _StubPipelineRunner(result={"pipeline": "weekly_digest", "pipeline_run_id": "pr-9", "status": "succeeded"})
    runner = PipelineJobRunner(pipeline_runner=stub, chat_storage=storage, clock=lambda: _NOW)
    run = _run(chat_binding={"chat_id": "C-1"})

    await runner.run(run, auth=_auth())

    assert run.result is not None
    assert run.result["final_response"] == "Pipeline 'weekly_digest' completed successfully."
    [message] = storage.messages
    assert message["content"] == "Pipeline 'weekly_digest' completed successfully."
    assert "{'final_response'" not in message["content"]


async def test_pipeline_runner_fails_before_invoking_when_binding_rejected() -> None:
    storage = _FakeChatStorage()
    stub = _StubPipelineRunner(_pipeline_ok)
    runner = PipelineJobRunner(pipeline_runner=stub, chat_storage=storage, clock=lambda: _NOW)
    run = _run(chat_binding={"chat_id": "missing"})

    with pytest.raises(JobRunnerError) as exc_info:
        await runner.run(run, auth=_auth())

    assert exc_info.value.retryable is False
    assert isinstance(exc_info.value.__cause__, ChatBindingTargetNotFoundError)
    assert stub.calls == []
    assert storage.messages == []
    assert run.status == JobStatus.FAILED


async def test_pipeline_runner_posts_failure_when_pipeline_reports_failure() -> None:
    storage = _FakeChatStorage()
    _chat(storage)

    async def failing(name, input, **kwargs) -> dict:
        return {
            "final_response": "Pipeline failed while fetching data.",
            "pipeline": name,
            "pipeline_run_id": "pr-2",
            "status": "failed",
            "error": "upstream 503",
        }

    stub = _StubPipelineRunner(failing)
    runner = PipelineJobRunner(pipeline_runner=stub, chat_storage=storage, clock=lambda: _NOW)
    run = _run(chat_binding={"chat_id": "C-1", "on_failure": "post_error"})

    await runner.run(run, auth=_auth())

    assert run.status == JobStatus.FAILED
    assert run.error == "upstream 503"
    assert run.result is not None
    assert run.result["final_response"] == "Pipeline failed while fetching data."
    [message] = storage.messages
    assert str(run.run_id) in message["content"]
    assert "upstream 503" not in message["content"]
    assert message["metadata"]["status"] == "failed"
    assert message["metadata"]["pipeline"] == "weekly_digest"


async def test_pipeline_runner_caps_persisted_graph() -> None:
    async def oversized(name, input, **kwargs) -> dict:
        return await _pipeline_ok(name, input, **kwargs) | {"graph": {"blob": "x" * 200}}

    stub = _StubPipelineRunner(oversized)
    runner = PipelineJobRunner(
        pipeline_runner=stub,
        clock=lambda: _NOW,
        max_graph_bytes=100,
    )
    run = _run(persist_graph=True)

    await runner.run(run, auth=_auth())

    assert run.status == JobStatus.FAILED
    assert run.error is not None
    assert "persistence cap" in run.error
    assert run.result is not None
    # Decision A: the Bloom run fails, but the result keeps the pipeline's
    # factual status and summary for audit.
    assert run.result["status"] == "succeeded"
    assert run.result["final_response"] == "The weekly digest is ready."
    assert "graph" not in run.result


async def test_pipeline_runner_propagates_retryable_runner_error() -> None:
    stub = _StubPipelineRunner(error=JobRunnerError("temporary", retryable=True))
    runner = PipelineJobRunner(pipeline_runner=stub, clock=lambda: _NOW)
    run = _run()

    with pytest.raises(JobRunnerError) as exc_info:
        await runner.run(run, auth=_auth())

    assert exc_info.value.retryable is True


async def test_pipeline_runner_fails_when_status_is_missing() -> None:
    stub = _StubPipelineRunner(result={"final_response": "looks fine"})
    runner = PipelineJobRunner(pipeline_runner=stub, clock=lambda: _NOW)
    run = _run()

    await runner.run(run, auth=_auth())

    assert run.status == JobStatus.FAILED
    assert run.error == "pipeline runner returned no status"


async def test_pipeline_runner_does_not_leak_bearer_token_into_result() -> None:
    stub = _StubPipelineRunner(_pipeline_ok)
    runner = PipelineJobRunner(pipeline_runner=stub, clock=lambda: _NOW)
    run = _run()

    await runner.run(run, auth=_auth())

    assert run.result is not None
    encoded = json.dumps(run.result)
    assert "secret-token" not in encoded


async def test_pipeline_runner_fails_when_output_is_not_json_safe() -> None:
    for bad_value in (float("nan"), float("inf"), "a\x00b"):
        stub = _StubPipelineRunner(
            result={
                "pipeline": "weekly_digest",
                "pipeline_run_id": "pr-1",
                "status": "succeeded",
                "output": {"value": bad_value},
            }
        )
        runner = PipelineJobRunner(pipeline_runner=stub, clock=lambda: _NOW)
        run = _run()

        await runner.run(run, auth=_auth())

        assert run.status == JobStatus.FAILED
        assert run.result is not None
        assert run.result["status"] == "failed"
        assert "JSON-serializable" in (run.error or "")


async def test_pipeline_runner_fails_when_persisted_graph_is_not_json_safe() -> None:
    stub = _StubPipelineRunner(
        result={
            "pipeline": "weekly_digest",
            "pipeline_run_id": "pr-1",
            "status": "succeeded",
            "final_response": "done",
            "graph": {"score": float("nan")},
        }
    )
    runner = PipelineJobRunner(pipeline_runner=stub, clock=lambda: _NOW)
    run = _run(persist_graph=True)

    await runner.run(run, auth=_auth())

    assert run.status == JobStatus.FAILED
    assert "not JSON-serializable" in (run.error or "")
    assert run.result is not None
    assert "graph" not in run.result


async def test_pipeline_runner_treats_none_summary_as_empty() -> None:
    storage = _FakeChatStorage()
    _chat(storage)
    stub = _StubPipelineRunner(
        result={
            "pipeline": "weekly_digest",
            "pipeline_run_id": "pr-1",
            "status": "succeeded",
            "summary": None,
        }
    )
    runner = PipelineJobRunner(pipeline_runner=stub, chat_storage=storage, clock=lambda: _NOW)
    run = _run(chat_binding={"chat_id": "C-1"})

    await runner.run(run, auth=_auth())

    assert run.result is not None
    assert run.result["final_response"] == "Pipeline 'weekly_digest' completed successfully."
    [message] = storage.messages
    assert message["content"] != "None"


async def test_pipeline_runner_posts_failure_for_malformed_result() -> None:
    storage = _FakeChatStorage()
    _chat(storage)
    stub = _StubPipelineRunner(result=object())
    runner = PipelineJobRunner(pipeline_runner=stub, chat_storage=storage, clock=lambda: _NOW)
    run = _run(chat_binding={"chat_id": "C-1", "on_failure": "post_error"})

    with pytest.raises(JobRunnerError):
        await runner.run(run, auth=_auth())

    [message] = storage.messages
    assert str(run.run_id) in message["content"]
    assert message["metadata"]["status"] == "failed"


async def test_pipeline_runner_preserves_retryable_pipeline_exception() -> None:
    class _RetryablePipelineError(Exception):
        retryable = True

    stub = _StubPipelineRunner(error=_RetryablePipelineError("temporary"))
    runner = PipelineJobRunner(pipeline_runner=stub, clock=lambda: _NOW)
    run = _run()

    with pytest.raises(JobRunnerError) as exc_info:
        await runner.run(run, auth=_auth())

    assert exc_info.value.retryable is True


async def test_pipeline_runner_silent_failure_posts_nothing() -> None:
    storage = _FakeChatStorage()
    _chat(storage)
    stub = _StubPipelineRunner(result={"pipeline": "weekly_digest", "pipeline_run_id": "pr-1", "status": "failed"})
    runner = PipelineJobRunner(pipeline_runner=stub, chat_storage=storage, clock=lambda: _NOW)
    run = _run(chat_binding={"chat_id": "C-1", "on_failure": "silent"})

    await runner.run(run, auth=_auth())

    assert run.status == JobStatus.FAILED
    assert storage.messages == []


async def test_pipeline_runner_proactive_chat_receives_result_and_chat_id() -> None:
    storage = _FakeChatStorage()
    stub = _StubPipelineRunner(_pipeline_ok)
    runner = PipelineJobRunner(pipeline_runner=stub, chat_storage=storage, clock=lambda: _NOW)
    run = _run(proactive_chat=True)

    await runner.run(run, auth=_auth())

    assert run.status == JobStatus.SUCCEEDED
    [chat] = storage.chats.values()
    assert chat.title == "pipeline:weekly_digest"
    [message] = storage.messages
    assert message["chat_id"] == chat.id
    assert stub.calls[0]["chat_id"] == chat.id


async def test_pipeline_runner_sanitizes_nul_in_failure_text() -> None:
    stub = _StubPipelineRunner(
        result={
            "pipeline": "weekly_digest",
            "pipeline_run_id": "pr-1",
            "status": "failed",
            "final_response": "bad\x00summary",
            "error": "bad\x00error",
        }
    )
    runner = PipelineJobRunner(pipeline_runner=stub, clock=lambda: _NOW)
    run = _run()

    await runner.run(run, auth=_auth())

    assert run.status == JobStatus.FAILED
    assert run.result is not None
    assert "\x00" not in run.result["final_response"]
    assert "\x00" not in (run.error or "")
    assert "\ufffd" in run.result["final_response"]


async def test_pipeline_runner_over_cap_empty_summary_keeps_success_text() -> None:
    async def oversized(name, input, **kwargs) -> dict:
        return {
            "pipeline": name,
            "pipeline_run_id": "pr-1",
            "status": "succeeded",
            "graph": {"blob": "x" * 200},
        }

    stub = _StubPipelineRunner(oversized)
    runner = PipelineJobRunner(pipeline_runner=stub, clock=lambda: _NOW, max_graph_bytes=100)
    run = _run(persist_graph=True)

    await runner.run(run, auth=_auth())

    assert run.status == JobStatus.FAILED
    assert run.result is not None
    assert run.result["status"] == "succeeded"
    assert run.result["final_response"] == "Pipeline 'weekly_digest' completed successfully."


async def test_pipeline_runner_failed_pipeline_is_not_masked_by_graph_cap() -> None:
    async def failing_with_graph(name, input, **kwargs) -> dict:
        return {
            "pipeline": name,
            "pipeline_run_id": "pr-1",
            "status": "failed",
            "final_response": "upstream failed",
            "error": "upstream 503",
            "graph": {"blob": "x" * 200},
        }

    stub = _StubPipelineRunner(failing_with_graph)
    runner = PipelineJobRunner(pipeline_runner=stub, clock=lambda: _NOW, max_graph_bytes=100)
    run = _run(persist_graph=True)

    await runner.run(run, auth=_auth())

    assert run.status == JobStatus.FAILED
    assert run.error == "upstream 503"
    assert run.result is not None
    assert run.result["status"] == "failed"
    assert "graph" not in run.result
