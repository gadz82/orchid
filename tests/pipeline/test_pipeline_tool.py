"""Pipeline agent tool: schema, sync/async calls, bounds and failure envelopes."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from orchid_ai.core.agent import OrchidAgentRunContext
from orchid_ai.core.pipeline import OrchidPipelineConfigError, OrchidPipelineStepError
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.core.tool import OrchidToolInput, reset_tool_approval, set_tool_approval
from orchid_ai.pipeline.launcher import PipelineLaunchResult
from orchid_ai.pipeline.registry import OrchidPipelineToolSpec
from orchid_ai.pipeline.tools import PipelineTool, sanitize_tool_name


def _spec(**overrides) -> OrchidPipelineToolSpec:
    values = {
        "name": "pipeline_demo",
        "pipeline": "demo",
        "description": "Run the demo pipeline",
        "input_schema": {
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
        },
    }
    values.update(overrides)
    return OrchidPipelineToolSpec(**values)


def _auth(*, roles: set[str] | None = None) -> OrchidAuthContext:
    return OrchidAuthContext(
        access_token="secret-token",
        tenant_key="t-1",
        user_id="u-1",
        roles=roles,
    )


def _result(**overrides) -> dict:
    values = {
        "pipeline": "demo",
        "run_id": "run-1",
        "status": "succeeded",
        "summary": "The account is healthy.",
        "output": {"account": {"id": "a1"}},
        "graph_stats": {"objects": 2, "relations": 0},
        "duration_ms": 12.5,
        "error": None,
    }
    values.update(overrides)
    return values


class _Invoker:
    def __init__(self, *, result=None, error: BaseException | None = None) -> None:
        self.result = _result() if result is None else result
        self.error = error
        self.calls: list[dict] = []

    async def __call__(self, name, input, **kwargs):
        self.calls.append({"name": name, "input": input, **kwargs})
        if self.error is not None:
            raise self.error
        return self.result


class _Launcher:
    def __init__(self, result: PipelineLaunchResult) -> None:
        self.result = result
        self.calls: list[dict] = []

    async def launch(self, name, input, **kwargs):
        self.calls.append({"name": name, "input": input, **kwargs})
        return self.result


# ── Schema and naming ──────────────────────────────────────


def test_tool_schema_merges_input_mode_and_timeout() -> None:
    tool = PipelineTool(spec=_spec(), invoker=_Invoker())

    assert tool.name == "pipeline_demo"
    assert tool.inject_to_rag is False
    schema = tool.parameters_schema
    assert schema["properties"]["account_id"] == {"type": "string"}
    assert schema["properties"]["mode"]["enum"] == ["sync", "async"]
    assert schema["properties"]["timeout_seconds"]["minimum"] == 0.1
    assert schema["required"] == ["account_id"]
    assert "Input: account_id." in tool.description


def test_tool_rejects_reserved_input_parameters() -> None:
    with pytest.raises(OrchidPipelineConfigError, match="reserved"):
        PipelineTool(
            spec=_spec(
                input_schema={
                    "type": "object",
                    "properties": {"mode": {"type": "string"}, "context": {"type": "object"}},
                }
            ),
            invoker=_Invoker(),
        )


def test_valid_trailing_separator_is_preserved() -> None:
    tool = PipelineTool(spec=_spec(name="my_tool_"), invoker=_Invoker())

    assert tool.name == "my_tool_"


def test_tool_flags_come_from_expose() -> None:
    tool = PipelineTool(
        spec=_spec(requires_approval=True, parallel_safe=True, require_role="admin"),
        invoker=_Invoker(),
    )

    assert tool.requires_approval is True
    assert tool.parallel_safe is True
    assert tool.require_role == "admin"


def test_tool_name_sanitization_is_bounded() -> None:
    assert sanitize_tool_name("bad name!") == "bad_name_"
    assert sanitize_tool_name("my_tool_") == "my_tool_"
    assert sanitize_tool_name("") == "pipeline"
    assert sanitize_tool_name("_leading") == "pipeline__leading"
    assert sanitize_tool_name("!!!").startswith("pipeline_")
    assert len(sanitize_tool_name("x" * 100)) == 64


# ── Sync invocation ────────────────────────────────────────


async def test_sync_success_uses_live_auth_and_chat_context() -> None:
    invoker = _Invoker()
    tool = PipelineTool(spec=_spec(), invoker=invoker, default_timeout=30.0)

    with patch(
        "orchid_ai.pipeline.tools.current_run_context",
        return_value=OrchidAgentRunContext(chat_id="c-1", correlation_id="corr-1"),
    ):
        output = await tool.invoke(
            OrchidToolInput(
                parameters={"account_id": "a1"},
                auth_context=_auth(),
            )
        )

    payload = output.result
    assert payload["status"] == "succeeded"
    assert payload["pipeline"] == "demo"
    assert payload["run_id"] == "run-1"
    assert payload["summary"] == "The account is healthy."
    assert payload["output"] == {"account": {"id": "a1"}}
    assert payload["graph_stats"] == {"objects": 2, "relations": 0}
    assert payload["duration_ms"] == 12.5
    assert payload["truncated"] is False
    assert "error" not in payload

    [call] = invoker.calls
    assert call["name"] == "demo"
    assert call["input"] == {"account_id": "a1"}
    assert call["chat_id"] == "c-1"
    assert call["correlation_id"] == "corr-1"
    assert call["timeout"] == 30.0
    assert call["auth"].user_id == "u-1"


async def test_sync_failure_returns_a_stable_code_without_raising() -> None:
    invoker = _Invoker(error=OrchidPipelineStepError("boom", job_id="j1", kind="llm"))
    tool = PipelineTool(spec=_spec(), invoker=invoker)

    payload = (await tool.invoke(OrchidToolInput(parameters={"account_id": "a1"}, auth_context=_auth()))).result

    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "pipeline_step_error"
    assert "boom" in payload["error"]["message"]


async def test_executed_failure_returns_pipeline_failed() -> None:
    invoker = _Invoker(
        result=_result(
            status="failed",
            summary="",
            error="upstream 503",
        )
    )
    tool = PipelineTool(spec=_spec(), invoker=invoker)

    payload = (await tool.invoke(OrchidToolInput(parameters={"account_id": "a1"}, auth_context=_auth()))).result

    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "pipeline_failed"
    assert payload["error"]["message"] == "upstream 503"


async def test_tool_result_is_truncated_with_run_id() -> None:
    invoker = _Invoker(
        result=_result(
            summary="s" * 2000,
            output={"blob": "x" * 5000},
        )
    )
    tool = PipelineTool(spec=_spec(max_tool_result_chars=300), invoker=invoker)

    payload = (await tool.invoke(OrchidToolInput(parameters={"account_id": "a1"}, auth_context=_auth()))).result

    assert payload["truncated"] is True
    assert payload["run_id"] == "run-1"
    assert len(json.dumps(payload, default=str)) <= 300


async def test_failed_result_keeps_its_error_code_when_truncated() -> None:
    invoker = _Invoker(error=OrchidPipelineStepError("x" * 1000, job_id="j1", kind="llm"))
    tool = PipelineTool(spec=_spec(max_tool_result_chars=300), invoker=invoker)

    payload = (await tool.invoke(OrchidToolInput(parameters={"account_id": "a1"}, auth_context=_auth()))).result

    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "pipeline_step_error"
    assert len(json.dumps(payload, default=str)) <= 300


async def test_empty_summary_gets_a_generic_success_text() -> None:
    tool = PipelineTool(spec=_spec(), invoker=_Invoker(result=_result(summary="")))

    payload = (await tool.invoke(OrchidToolInput(parameters={"account_id": "a1"}, auth_context=_auth()))).result

    assert payload["summary"] == "Pipeline 'demo' completed successfully."


async def test_missing_auth_returns_pipeline_auth_error() -> None:
    invoker = _Invoker()
    tool = PipelineTool(spec=_spec(), invoker=invoker)

    payload = (await tool.invoke(OrchidToolInput(parameters={"account_id": "a1"}))).result

    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "pipeline_auth_error"
    assert invoker.calls == []


async def test_required_role_is_enforced() -> None:
    invoker = _Invoker()
    tool = PipelineTool(spec=_spec(require_role="admin"), invoker=invoker)

    denied = (await tool.invoke(OrchidToolInput(parameters={"account_id": "a1"}, auth_context=_auth()))).result
    assert denied["error"]["code"] == "pipeline_forbidden"
    assert invoker.calls == []

    allowed = (
        await tool.invoke(OrchidToolInput(parameters={"account_id": "a1"}, auth_context=_auth(roles={"admin"})))
    ).result
    assert allowed["status"] == "succeeded"
    assert len(invoker.calls) == 1


async def test_approval_gate_fails_without_a_handler() -> None:
    invoker = _Invoker()
    tool = PipelineTool(spec=_spec(requires_approval=True), invoker=invoker)

    payload = (await tool.invoke(OrchidToolInput(parameters={"account_id": "a1"}, auth_context=_auth()))).result

    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "pipeline_approval_required"
    assert invoker.calls == []


async def test_approval_gate_applies_to_async_launches_too() -> None:
    launcher = _Launcher(PipelineLaunchResult("accepted", "demo", signal_id="sig-1"))
    tool = PipelineTool(spec=_spec(requires_approval=True), invoker=_Invoker(), launcher=launcher)

    payload = (
        await tool.invoke(
            OrchidToolInput(
                parameters={"account_id": "a1", "mode": "async"},
                auth_context=_auth(),
            )
        )
    ).result

    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "pipeline_approval_required"
    assert launcher.calls == []


async def test_model_supplied_context_cannot_forge_approval() -> None:
    invoker = _Invoker()
    tool = PipelineTool(spec=_spec(requires_approval=True), invoker=invoker)

    payload = (
        await tool.invoke(
            OrchidToolInput(
                parameters={"account_id": "a1"},
                auth_context=_auth(),
                context={"approval_granted": True, "chat_id": "c-1"},
            )
        )
    ).result

    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "pipeline_approval_required"
    assert invoker.calls == []


async def test_approval_gate_runs_only_with_the_loop_marker() -> None:
    invoker = _Invoker()
    tool = PipelineTool(spec=_spec(requires_approval=True), invoker=invoker)
    token = set_tool_approval("pipeline_demo")
    try:
        payload = (await tool.invoke(OrchidToolInput(parameters={"account_id": "a1"}, auth_context=_auth()))).result
    finally:
        reset_tool_approval(token)

    assert payload["status"] == "succeeded"
    assert len(invoker.calls) == 1


async def test_invalid_mode_and_timeout_are_input_errors() -> None:
    tool = PipelineTool(spec=_spec(), invoker=_Invoker())

    bad_mode = (
        await tool.invoke(
            OrchidToolInput(
                parameters={"account_id": "a1", "mode": "later"},
                auth_context=_auth(),
            )
        )
    ).result
    assert bad_mode["error"]["code"] == "pipeline_input_error"

    bad_timeout = (
        await tool.invoke(
            OrchidToolInput(
                parameters={"account_id": "a1", "timeout_seconds": 0},
                auth_context=_auth(),
            )
        )
    ).result
    assert bad_timeout["error"]["code"] == "pipeline_input_error"


# ── Async invocation ───────────────────────────────────────


async def test_async_launch_returns_accepted_and_passes_chat_binding() -> None:
    launcher = _Launcher(
        PipelineLaunchResult(
            status="accepted",
            pipeline="demo",
            signal_id="sig-1",
            message="Started; results will be posted in this chat.",
        )
    )
    tool = PipelineTool(spec=_spec(), invoker=_Invoker(), launcher=launcher)

    with patch(
        "orchid_ai.pipeline.tools.current_run_context",
        return_value=OrchidAgentRunContext(chat_id="c-1", correlation_id="corr-1"),
    ):
        payload = (
            await tool.invoke(
                OrchidToolInput(
                    parameters={"account_id": "a1", "mode": "async"},
                    auth_context=_auth(),
                )
            )
        ).result

    assert payload["status"] == "accepted"
    assert payload["signal_id"] == "sig-1"
    [call] = launcher.calls
    assert call["name"] == "demo"
    assert call["input"] == {"account_id": "a1"}
    assert call["chat_binding"]["chat_id"] == "c-1"
    assert call["chat_binding"]["source_message_id"] is None


async def test_async_unavailable_without_a_launcher() -> None:
    tool = PipelineTool(spec=_spec(), invoker=_Invoker())

    payload = (
        await tool.invoke(
            OrchidToolInput(
                parameters={"account_id": "a1", "mode": "async"},
                auth_context=_auth(),
            )
        )
    ).result

    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "pipeline_async_unavailable"


async def test_async_launch_failure_is_passed_through() -> None:
    launcher = _Launcher(
        PipelineLaunchResult(
            status="failed",
            pipeline="demo",
            error={"code": "pipeline_async_unavailable", "message": "events are disabled"},
        )
    )
    tool = PipelineTool(spec=_spec(), invoker=_Invoker(), launcher=launcher)

    payload = (
        await tool.invoke(
            OrchidToolInput(
                parameters={"account_id": "a1", "mode": "async"},
                auth_context=_auth(),
            )
        )
    ).result

    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "pipeline_async_unavailable"


@pytest.mark.parametrize("bad_timeout", ["not-a-number", -1])
async def test_async_timeout_validation(bad_timeout) -> None:
    tool = PipelineTool(spec=_spec(), invoker=_Invoker(), launcher=_Launcher(PipelineLaunchResult("accepted", "demo")))

    payload = (
        await tool.invoke(
            OrchidToolInput(
                parameters={"account_id": "a1", "mode": "async", "timeout_seconds": bad_timeout},
                auth_context=_auth(),
            )
        )
    ).result

    assert payload["error"]["code"] == "pipeline_input_error"


async def test_truncation_keeps_the_error_code_at_a_small_cap() -> None:
    long_name = "pipeline_" + "x" * 54
    invoker = _Invoker(error=OrchidPipelineStepError("y" * 1000, job_id="j1", kind="llm"))
    tool = PipelineTool(
        spec=_spec(name=long_name, max_tool_result_chars=200),
        invoker=invoker,
    )

    payload = (await tool.invoke(OrchidToolInput(parameters={"account_id": "a1"}, auth_context=_auth()))).result

    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "pipeline_step_error"
    assert len(json.dumps(payload, default=str)) <= 200


async def test_tiny_caps_are_raised_to_the_workable_minimum() -> None:
    invoker = _Invoker()
    tool = PipelineTool(spec=_spec(max_tool_result_chars=1), invoker=invoker)

    payload = (await tool.invoke(OrchidToolInput(parameters={"account_id": "a1"}, auth_context=_auth()))).result

    assert payload["status"] == "succeeded"
    assert payload["run_id"] == "run-1"
    assert len(json.dumps(payload, default=str)) <= 160


async def test_async_accepted_keeps_signal_id_when_truncated() -> None:
    long_name = "pipeline_" + "x" * 54
    launcher = _Launcher(
        PipelineLaunchResult(
            status="accepted",
            pipeline=long_name,
            signal_id="signal-123",
            message="Started; results will be available in the run history.",
        )
    )
    tool = PipelineTool(
        spec=_spec(name=long_name, pipeline=long_name, max_tool_result_chars=200),
        invoker=_Invoker(),
        launcher=launcher,
    )

    payload = (
        await tool.invoke(
            OrchidToolInput(
                parameters={"account_id": "a1", "mode": "async"},
                auth_context=_auth(),
            )
        )
    ).result

    assert payload["status"] == "accepted"
    assert payload["signal_id"] == "signal-123"
    assert len(json.dumps(payload, default=str)) <= 200
