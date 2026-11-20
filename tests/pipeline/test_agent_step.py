"""Tests for the agent pipeline step."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.errors import GraphBubbleUp

from orchid_ai.config.schema import OrchidAgentConfig, OrchidMiniAgentConfig
from orchid_ai.config.schema_pipeline import (
    OrchidPipelineAgentJobConfig,
    OrchidPipelineDefinitionConfig,
    OrchidPipelineDeterministicJobConfig,
    OrchidPipelineShapeConfig,
)
from orchid_ai.core.agent import OrchidAgent
from orchid_ai.core.guardrails import (
    OrchidGuardrail,
    OrchidGuardrailAction,
    OrchidGuardrailChain,
    OrchidGuardrailContext,
    OrchidGuardrailResult,
)
from orchid_ai.core.pipeline import (
    OrchidPipelineAuthError,
    OrchidPipelineConfigError,
    OrchidPipelineJobContext,
    OrchidPipelineLimitError,
    OrchidPipelineRunContext,
    OrchidPipelineShapeError,
    OrchidPipelineStepError,
    OrchidPipelineTimeoutError,
)
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.graph.agent_executor import OrchidAgentExecutor
from orchid_ai.pipeline.definition import CompiledJob, CompiledPipeline, PipelineDependencies, PipelineLimits
from orchid_ai.pipeline.graph import DataGraph
from orchid_ai.pipeline.mapping import build_mapping_context
from orchid_ai.pipeline.runner import OrchidPipelineRunner
from orchid_ai.pipeline.schema import compile_shapes
from orchid_ai.pipeline.session import PipelineStepContext
from orchid_ai.pipeline.steps.agent import AgentStep

START = datetime(2026, 11, 18, 11, 0, tzinfo=UTC)


class _Agent(OrchidAgent):
    name = "helper"
    description = "helper"

    def __init__(
        self,
        response: Any = "agent answer",
        *,
        chat_model: Any | None = None,
        probe: Any | None = None,
    ) -> None:
        super().__init__(reader=object(), chat_model=chat_model)
        self.response = response
        self.states: list[dict[str, Any]] = []
        self._probe = probe
        self.probed: list[Any] = []

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        self.states.append(dict(state))
        if self._probe is not None:
            self.probed.append(self._probe())
        return {
            "messages": [AIMessage(content=self.response)],
            "mcp_context": {"server": {"tool": "result"}},
            "rag_context": {"server": ["chunk"]},
            "skills_metadata": {"detected": ["lookup"]},
        }


class _SpyExecutor:
    def __init__(self, result: dict[str, Any] | None = None, *, error: Exception | None = None) -> None:
        self.result = result or {"messages": [AIMessage(content="spy answer")]}
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def run(self, state: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"state": state, **kwargs})
        if self.error is not None:
            raise self.error
        return self.result


class _StaticRail(OrchidGuardrail):
    def __init__(self, name: str, result: OrchidGuardrailResult) -> None:
        self._name = name
        self._result = result

    @property
    def name(self) -> str:
        return self._name

    async def check(self, content: str, context: OrchidGuardrailContext) -> OrchidGuardrailResult:
        return self._result


def _blocking_rail(message: str = "blocked") -> OrchidGuardrail:
    return _StaticRail(
        "block",
        OrchidGuardrailResult(
            triggered=True,
            action=OrchidGuardrailAction.BLOCK,
            guardrail_name="block",
            message=message,
        ),
    )


def _redacting_rail(replacement: str = "[REDACTED]") -> OrchidGuardrail:
    return _StaticRail(
        "redact",
        OrchidGuardrailResult(
            triggered=True,
            action=OrchidGuardrailAction.REDACT,
            guardrail_name="redact",
            redacted_content=replacement,
        ),
    )


class _RecordingRouter:
    point = "model_routing"
    default_model = "default-model"

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.active_model: str | None = None

    @property
    def current_model(self) -> str:
        return self.active_model or self.default_model

    @asynccontextmanager
    async def bound(self, query: str, *, budget: Any | None = None):
        self.queries.append(query)
        self.active_model = "routed-model"
        try:
            yield "routed-model"
        finally:
            self.active_model = None


class _StubChatModel:
    def __init__(self, content: str = "aggregated answer") -> None:
        self.content = content
        self.calls: list[Any] = []

    async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> AIMessage:
        self.calls.append(messages)
        return AIMessage(content=self.content)


def _config(**overrides: Any) -> OrchidPipelineAgentJobConfig:
    values: dict[str, Any] = {
        "kind": "agent",
        "agent": "helper",
        "prompt": "Research {{ input.account_id }}",
    }
    values.update(overrides)
    return OrchidPipelineAgentJobConfig.model_validate(values)


def _mini_config() -> OrchidAgentConfig:
    return OrchidAgentConfig(
        name="helper",
        description="helper",
        prompt="prompt",
        mini_agent=OrchidMiniAgentConfig(enabled=True, max_count=2),
    )


def _job(config: OrchidPipelineAgentJobConfig | None = None, *, job_id: str = "research") -> CompiledJob:
    resolved = config or _config()
    return CompiledJob(
        id=job_id,
        kind="agent",
        config=resolved,
        depends_on=tuple(resolved.depends_on),
        when=resolved.when,
        timeout_seconds=resolved.timeout_seconds,
    )


def _shapes() -> Any:
    return compile_shapes(
        {"Brief": OrchidPipelineShapeConfig(fields={"headline": {"type": "string", "required": True}})}
    )


def _pipeline(*, shapes: Any = None, limits: PipelineLimits | None = None) -> CompiledPipeline:
    return CompiledPipeline(
        name="demo",
        version="1",
        description="",
        definition=OrchidPipelineDefinitionConfig(),
        jobs={},
        limits=limits or PipelineLimits.from_config(),
        shapes=shapes if shapes is not None else _shapes(),
        defaults={},
    )


def _ctx(
    *,
    job: CompiledJob | None = None,
    pipeline: CompiledPipeline | None = None,
    limits: PipelineLimits | None = None,
    input: dict[str, Any] | None = None,
    auth: Any = "default",
    mode: str = "direct",
    chat_id: str | None = None,
) -> PipelineStepContext:
    job = job or _job()
    pipeline = pipeline or _pipeline(limits=limits)
    if limits is not None:
        pipeline.limits = limits
    run = OrchidPipelineRunContext(
        run_id="run-1",
        pipeline="demo",
        pipeline_version="1",
        mode="chat" if mode == "chat" else "direct",
        tenant_key="t1",
        user_id="u1",
        started_at=START,
        chat_id=chat_id,
    )
    inputs = input or {"account_id": "a1"}
    return PipelineStepContext(
        run=run,
        job=OrchidPipelineJobContext(job_id=job.id, kind="agent"),
        input=inputs,
        graph=DataGraph().view(),
        auth=OrchidAuthContext(access_token="token", tenant_key="t1", user_id="u1") if auth == "default" else auth,
        chat_model=None,
        reader=None,
        logger=logging.getLogger("agent-step-test"),
        idempotency_key="key",
        pipeline=pipeline,
        mapping_context=build_mapping_context(input=inputs, run=run),
    )


def _step(
    *,
    job: CompiledJob | None = None,
    agent: Any = None,
    executor: Any = None,
    agent_configs: dict[str, Any] | None = None,
    agent_executors: dict[str, Any] | None = None,
) -> AgentStep:
    agent = agent or _Agent()
    factory = (lambda a, c: executor) if executor is not None else None
    return AgentStep(
        job or _job(),
        agents={"helper": agent},
        agent_configs=agent_configs or {},
        agent_executors=agent_executors or {},
        executor_factory=factory,
    )


def _run(step: AgentStep, ctx: Any) -> Any:
    return asyncio.run(step.execute(ctx))


# ── Execution ──────────────────────────────────────────────


def test_agent_runs_through_the_shared_executor_and_returns_text() -> None:
    agent = _Agent("the answer")
    step = _step(agent=agent)

    result = _run(step, _ctx())

    assert result.output == {"text": "the answer"}
    assert agent.states[0]["messages"][0].content == "Research a1"


def test_persist_memory_is_false_by_default() -> None:
    executor = _SpyExecutor()
    _run(_step(executor=executor), _ctx())

    assert executor.calls[0]["persist_memory"] is False


def test_persist_memory_is_honoured_when_enabled() -> None:
    executor = _SpyExecutor()
    job = _job(_config(persist_memory=True))

    _run(_step(job=job, executor=executor), _ctx(job=job))

    assert executor.calls[0]["persist_memory"] is True


def test_executor_receives_pipeline_auth_and_correlation() -> None:
    executor = _SpyExecutor()
    ctx = _ctx()

    _run(_step(executor=executor), ctx)

    call = executor.calls[0]
    assert call["auth"] is ctx.auth
    assert call["correlation_id"] == "run-1"
    assert call["message_id"] is None


def test_chat_mode_passes_the_chat_id_and_state() -> None:
    executor = _SpyExecutor()
    ctx = _ctx(mode="chat", chat_id="c-1")

    _run(_step(executor=executor), ctx)

    assert executor.calls[0]["chat_id"] == "c-1"
    assert executor.calls[0]["state"]["chat_id"] == "c-1"


def test_direct_mode_has_no_chat_scope() -> None:
    executor = _SpyExecutor()
    _run(_step(executor=executor), _ctx(mode="direct", chat_id="c-1"))

    assert executor.calls[0]["chat_id"] is None
    assert executor.calls[0]["state"]["chat_id"] == ""


def test_input_mapping_is_rendered_into_the_prompt() -> None:
    executor = _SpyExecutor()
    job = _job(_config(input={"account": "{{ input.account_id }}", "facts": ["{{ input.account_id }}"]}))

    _run(_step(job=job, executor=executor), _ctx(job=job))

    content = executor.calls[0]["state"]["messages"][0].content
    assert "Structured input" in content
    assert '"account": "a1"' in content


def test_agent_config_is_passed_to_the_executor_factory() -> None:
    seen: list[Any] = []
    agent = _Agent()
    agent_config = SimpleNamespace(children=None)

    def factory(a: Any, config: Any) -> Any:
        seen.append((a, config))
        return _SpyExecutor()

    step = AgentStep(
        _job(),
        agents={"helper": agent},
        agent_configs={"helper": agent_config},
        executor_factory=factory,
    )
    _run(step, _ctx())

    assert seen == [(agent, agent_config)]


def test_default_executor_enables_inline_mini_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    from orchid_ai.pipeline.steps import agent as agent_module

    captured: dict[str, Any] = {}

    class _FactoryExecutor:
        def __init__(self, agent: Any, **kwargs: Any) -> None:
            captured["agent"] = agent
            captured.update(kwargs)

        async def run(self, state: Any, **kwargs: Any) -> dict[str, Any]:
            return {"messages": [AIMessage(content="factory answer")]}

    monkeypatch.setattr(agent_module, "OrchidAgentExecutor", _FactoryExecutor)
    agent = _Agent()
    config = SimpleNamespace(children=None)
    step = AgentStep(_job(), agents={"helper": agent}, agent_configs={"helper": config})

    result = _run(step, _ctx())

    assert result.output == {"text": "factory answer"}
    assert captured["agent"] is agent
    assert captured["agent_config"] is config
    assert captured["inline_mini_agents"] is True


def test_a_graph_built_executor_is_preferred_over_the_factory() -> None:
    injected = _SpyExecutor()
    fallback = _SpyExecutor({"messages": [AIMessage(content="fallback")]})

    result = _run(
        _step(executor=fallback, agent_executors={"helper": injected}),
        _ctx(),
    )

    assert result.output == {"text": "spy answer"}
    assert injected.calls and not fallback.calls


def test_mini_agent_outcomes_are_captured_as_bounded_metadata() -> None:
    executor = _SpyExecutor(
        {
            "messages": [AIMessage(content="aggregated")],
            "mini_agent_outcomes": {"helper#1": {"output": "x" * 5000}},
        }
    )

    result = _run(_step(executor=executor), _ctx())

    assert result.output == {"text": "aggregated"}
    assert result.metadata["mini_agent_outcomes"]["truncated"] is True


def test_inline_mini_fork_runs_before_the_job_output(monkeypatch: pytest.MonkeyPatch) -> None:
    from orchid_ai.agents import mini_agent_node

    async def fake_maybe_decompose(**kwargs: Any) -> dict[str, Any]:
        return {
            "mini_agent_decisions": {
                "helper": {
                    "should_fork": True,
                    "sub_tasks": [
                        {
                            "id": "mini_0",
                            "description": "task",
                            "instruction": "do it",
                            "allowed_tools": [],
                            "rationale": "independent",
                            "resolved_tool_subset": [],
                        }
                    ],
                    "reasoning": "independent",
                }
            },
            "messages": [],
        }

    def fake_mini_factory(*, parent_config: Any, chat_model: Any, mcp_clients: Any):
        from orchid_ai.agents.mini_agent_node import _wrap_state_update

        async def node(state: dict[str, Any], config: Any = None) -> dict[str, Any]:
            mini_id = state["_active_mini_id"]
            outcome = {
                "mini_id": mini_id,
                "sub_task_description": "task",
                "status": "ok",
                "summary": "mini summary",
                "error": None,
                "duration_ms": 1,
                "tool_results": {},
            }
            return _wrap_state_update(parent_config.name, f"{parent_config.name}#{mini_id}", outcome)

        return node

    monkeypatch.setattr("orchid_ai.agents.mini_agent_decomposer.maybe_decompose", fake_maybe_decompose)
    monkeypatch.setattr(mini_agent_node, "mini_agent_node_factory", fake_mini_factory)

    agent = _Agent(chat_model=_StubChatModel())
    config = _mini_config()
    executor = OrchidAgentExecutor(agent, agent_config=config, inline_mini_agents=True)
    result = _run(_step(agent=agent, agent_executors={"helper": executor}), _ctx())

    assert "aggregated answer" in result.output["text"]
    assert set(result.metadata["mini_agent_outcomes"]) == {"helper#mini_0"}
    assert result.usage["llm_calls"] == 3  # decomposer + one mini + aggregator


def test_inline_decomposer_rejection_falls_back_to_the_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_maybe_decompose(**kwargs: Any) -> dict[str, Any]:
        return {
            "messages": [AIMessage(content="[Helper Agent] I couldn't break this request into sub-tasks")],
            "mcp_context": {"helper": {"summary": "rejected"}},
        }

    monkeypatch.setattr("orchid_ai.agents.mini_agent_decomposer.maybe_decompose", fake_maybe_decompose)

    agent = _Agent("normal answer", chat_model=_StubChatModel())
    executor = OrchidAgentExecutor(agent, agent_config=_mini_config(), inline_mini_agents=True)
    result = _run(_step(agent=agent, agent_executors={"helper": executor}), _ctx())

    assert len(agent.states) == 1
    assert result.output == {"text": "normal answer"}
    assert result.usage["llm_calls"] == 2  # decomposer attempt + agent


def test_inline_decomposer_no_fork_counts_the_decomposer(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_maybe_decompose(**kwargs: Any) -> None:
        return None

    monkeypatch.setattr("orchid_ai.agents.mini_agent_decomposer.maybe_decompose", fake_maybe_decompose)

    agent = _Agent("normal answer", chat_model=_StubChatModel())
    executor = OrchidAgentExecutor(agent, agent_config=_mini_config(), inline_mini_agents=True)
    result = _run(_step(agent=agent, agent_executors={"helper": executor}), _ctx())

    assert result.output == {"text": "normal answer"}
    assert result.usage["llm_calls"] == 2  # decomposer + agent


# ── Guardrails and routing parity ─────────────────────────


def test_graph_built_executor_applies_input_guardrails() -> None:
    agent = _Agent()
    chain = OrchidGuardrailChain([_blocking_rail("blocked input")])
    executor = OrchidAgentExecutor(agent, input_guardrails=chain)

    result = _run(_step(agent=agent, agent_executors={"helper": executor}), _ctx())

    assert agent.states == []
    assert result.output == {"text": "[Helper Agent] blocked input"}
    assert result.usage["llm_calls"] == 0


def test_graph_built_executor_applies_input_redaction() -> None:
    agent = _Agent()
    chain = OrchidGuardrailChain([_redacting_rail()])
    executor = OrchidAgentExecutor(agent, input_guardrails=chain)

    result = _run(_step(agent=agent, agent_executors={"helper": executor}), _ctx())

    assert agent.states[0]["messages"][0].content == "[REDACTED]"
    assert result.output == {"text": "agent answer"}
    assert result.usage["llm_calls"] == 1


def test_graph_built_executor_applies_output_guardrails() -> None:
    agent = _Agent("raw answer")
    chain = OrchidGuardrailChain([_blocking_rail("blocked output")])
    executor = OrchidAgentExecutor(agent, output_guardrails=chain)

    result = _run(_step(agent=agent, agent_executors={"helper": executor}), _ctx())

    assert agent.states
    assert result.output == {"text": "[Helper Agent] blocked output"}


def test_graph_built_executor_reports_the_routing_decision() -> None:
    router = _RecordingRouter()
    agent = _Agent(probe=lambda: router.current_model)
    executor = OrchidAgentExecutor(agent, model_router=router)

    result = _run(_step(agent=agent, agent_executors={"helper": executor}), _ctx())

    assert router.queries == ["Research a1"]
    assert agent.probed == ["routed-model"]
    assert result.metadata["model_routing"] == {
        "point": "model_routing",
        "model": "routed-model",
        "default_model": "default-model",
        "routed": True,
    }


# ── Resolution and failure paths ──────────────────────────


def test_missing_agent_is_a_config_error() -> None:
    step = AgentStep(_job(), agents={})

    with pytest.raises(OrchidPipelineConfigError, match="unknown agent 'helper'"):
        _run(step, _ctx())


def test_parent_agent_is_rejected() -> None:
    agent = _Agent()
    config = SimpleNamespace(children={"child": object()})
    step = AgentStep(_job(), agents={"helper": agent}, agent_configs={"helper": config})

    with pytest.raises(OrchidPipelineConfigError, match="top-level leaf agents"):
        _run(step, _ctx())


def test_missing_auth_is_an_auth_error() -> None:
    step = _step(executor=_SpyExecutor())

    with pytest.raises(OrchidPipelineAuthError, match="authenticated caller"):
        _run(step, _ctx(auth=None))


def test_agent_auth_error_message_is_detected() -> None:
    executor = _SpyExecutor({"messages": [AIMessage(content="[helper] Error: no auth context")]})

    with pytest.raises(OrchidPipelineAuthError, match="no auth context"):
        _run(_step(executor=executor), _ctx())


def test_executor_error_is_sanitized() -> None:
    executor = _SpyExecutor(error=RuntimeError("secret sk-agent"))

    with pytest.raises(OrchidPipelineStepError) as excinfo:
        _run(_step(executor=executor), _ctx())

    message = str(excinfo.value)
    assert "failed while running agent 'helper'" in message
    assert "secret" not in message


def test_factory_error_is_sanitized() -> None:
    def factory(agent: Any, config: Any) -> Any:
        raise RuntimeError("secret sk-agent")

    step = AgentStep(_job(), agents={"helper": _Agent()}, executor_factory=factory)

    with pytest.raises(OrchidPipelineStepError) as excinfo:
        _run(step, _ctx())

    assert "failed while running agent 'helper'" in str(excinfo.value)
    assert "secret" not in str(excinfo.value)


def test_executor_result_error_is_sanitized() -> None:
    executor = _SpyExecutor({"messages": [AIMessage(content="x")], "error": "RuntimeError: secret sk-agent"})

    with pytest.raises(OrchidPipelineStepError) as excinfo:
        _run(_step(executor=executor), _ctx())

    assert "secret" not in str(excinfo.value)


def test_no_assistant_message_fails() -> None:
    executor = _SpyExecutor({"messages": []})

    with pytest.raises(OrchidPipelineStepError, match="returned no assistant message"):
        _run(_step(executor=executor), _ctx())


def test_graph_bubble_up_is_re_raised() -> None:
    executor = _SpyExecutor(error=GraphBubbleUp())

    with pytest.raises(GraphBubbleUp):
        _run(_step(executor=executor), _ctx())


def test_malformed_usage_metadata_is_a_sanitized_step_error() -> None:
    executor = _SpyExecutor(
        {
            "messages": [AIMessage(content="ok")],
            "mini_agent_decisions": {"helper": {}},
            "mini_agent_outcomes": 5,
        }
    )

    with pytest.raises(OrchidPipelineStepError, match="invalid result") as excinfo:
        _run(_step(executor=executor), _ctx())

    assert "object of type 'int'" not in str(excinfo.value)


def test_non_mapping_result_is_a_sanitized_step_error() -> None:
    class _BadResultExecutor:
        async def run(self, state: Any, **kwargs: Any) -> Any:
            return None

    with pytest.raises(OrchidPipelineStepError, match="invalid result"):
        _run(_step(executor=_BadResultExecutor()), _ctx())


def test_executor_without_trace_support_still_runs() -> None:
    class _LegacyExecutor:
        async def run(
            self,
            state: Any,
            *,
            auth: Any,
            chat_id: str | None = None,
            message_id: str | None = None,
            correlation_id: str | None = None,
            persist_memory: bool = True,
        ) -> dict[str, Any]:
            return {"messages": [AIMessage(content="legacy answer")]}

    result = _run(_step(executor=_LegacyExecutor()), _ctx())

    assert result.output == {"text": "legacy answer"}
    assert result.usage["llm_calls"] == 1


def test_job_timeout_is_enforced() -> None:
    class _SlowExecutor:
        async def run(self, state: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(0.3)
            return {"messages": [AIMessage(content="late")]}

    job = replace(_job(), timeout_seconds=0.01)

    with pytest.raises(OrchidPipelineTimeoutError, match="timed out after 0.010 seconds"):
        _run(_step(job=job, executor=_SlowExecutor()), _ctx(job=job))


def test_run_deadline_caps_the_job_timeout() -> None:
    class _SlowExecutor:
        async def run(self, state: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(0.3)
            return {"messages": [AIMessage(content="late")]}

    job = replace(_job(), timeout_seconds=10.0)
    ctx = _ctx(job=job)
    ctx.run = replace(ctx.run, deadline=datetime.now(UTC) + timedelta(seconds=0.05))

    with pytest.raises(OrchidPipelineTimeoutError):
        _run(_step(job=job, executor=_SlowExecutor()), ctx)


def test_expired_deadline_fails_before_the_executor() -> None:
    executor = _SpyExecutor()
    ctx = _ctx()
    ctx.run = replace(ctx.run, deadline=datetime.now(UTC) - timedelta(seconds=1))

    with pytest.raises(OrchidPipelineTimeoutError, match="deadline has passed"):
        _run(_step(executor=executor), ctx)

    assert executor.calls == []


# ── Output shapes ─────────────────────────────────────────


def test_multi_part_message_content_is_joined() -> None:
    agent = _Agent([{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}])

    result = _run(_step(agent=agent), _ctx())

    assert result.output == {"text": "hello world"}


def test_non_sequence_message_content_falls_back_to_text() -> None:
    from orchid_ai.pipeline.steps.agent import _text

    assert _text(123) == "123"


def test_json_output_is_shape_validated() -> None:
    agent = _Agent('{"headline": "growth"}')
    job = _job(_config(output={"shape": "Brief"}))

    result = _run(_step(job=job, agent=agent), _ctx(job=job))

    assert result.output == {"headline": "growth"}


def test_fenced_json_output_is_parsed() -> None:
    agent = _Agent('```json\n{"headline": "growth"}\n```')
    job = _job(_config(output={"shape": "Brief"}))
    result = _run(_step(job=job, agent=agent), _ctx(job=job))

    assert result.output == {"headline": "growth"}


def test_invalid_json_output_is_bounded() -> None:
    agent = _Agent("not json at all")
    job = _job(_config(output={"shape": "Brief"}))

    with pytest.raises(OrchidPipelineShapeError) as excinfo:
        _run(_step(job=job, agent=agent), _ctx(job=job))

    message = str(excinfo.value)
    assert "shape 'Brief'" in message
    assert "not json at all" not in message


def test_non_object_json_output_is_rejected() -> None:
    agent = _Agent("[1, 2]")
    job = _job(_config(output={"shape": "Brief"}))

    with pytest.raises(OrchidPipelineShapeError, match="shape 'Brief'"):
        _run(_step(job=job, agent=agent), _ctx(job=job))


def test_malformed_json_output_is_rejected() -> None:
    agent = _Agent('{"headline": }')
    job = _job(_config(output={"shape": "Brief"}))

    with pytest.raises(OrchidPipelineShapeError, match="not valid JSON"):
        _run(_step(job=job, agent=agent), _ctx(job=job))


def test_shape_violation_keeps_the_structured_error_context() -> None:
    agent = _Agent('{"headline": 3}')
    job = _job(_config(output={"shape": "Brief"}))

    with pytest.raises(OrchidPipelineShapeError) as excinfo:
        _run(_step(job=job, agent=agent), _ctx(job=job))

    error = excinfo.value
    assert "shape 'Brief'" in str(error)
    assert error.shape == "Brief"
    assert error.object_id is None
    assert error.field_path == "$.headline"
    assert error.errors


def test_unknown_shape_is_a_config_error() -> None:
    agent = _Agent('{"headline": "x"}')
    job = _job(_config(output={"shape": "Missing"}))

    with pytest.raises(OrchidPipelineConfigError, match="unknown shape 'Missing'"):
        _run(_step(job=job, agent=agent), _ctx(job=job))


def test_shape_without_declared_shapes_is_a_config_error() -> None:
    agent = _Agent('{"headline": "x"}')
    job = _job(_config(output={"shape": "Brief"}))
    pipeline = _pipeline()
    pipeline.shapes = None

    with pytest.raises(OrchidPipelineConfigError, match="declares no shapes"):
        _run(_step(job=job, agent=agent), _ctx(job=job, pipeline=pipeline))


# ── Bounded metadata ──────────────────────────────────────


def test_metadata_is_bounded() -> None:
    executor = _SpyExecutor(
        {
            "messages": [AIMessage(content="ok")],
            "mcp_context": {"server": "x" * 5000},
            "rag_context": {"server": ["y" * 5000]},
            "skills_metadata": {"detected": ["lookup"]},
        }
    )

    result = _run(_step(executor=executor), _ctx())

    assert result.metadata["mcp_context"]["truncated"] is True
    assert result.metadata["rag_context"]["truncated"] is True
    assert result.metadata["skills_metadata"] == {"detected": ["lookup"]}
    assert len(json.dumps(result.metadata)) < 4000


@pytest.mark.parametrize(
    "key",
    ["mcp_context", "rag_context", "skills_metadata", "mini_agent_outcomes"],
)
def test_non_finite_metadata_is_replaced_with_a_preview(key: str) -> None:
    executor = _SpyExecutor(
        {
            "messages": [AIMessage(content="ok")],
            key: {"score": float("nan")},
        }
    )

    result = _run(_step(executor=executor), _ctx())

    assert result.metadata[key]["truncated"] is True
    assert isinstance(result.metadata[key]["preview"], str)
    assert json.dumps(result.metadata)


def test_circular_metadata_is_replaced_with_a_preview() -> None:
    circular: list[Any] = []
    circular.append(circular)
    executor = _SpyExecutor({"messages": [AIMessage(content="ok")], "mcp_context": circular})

    result = _run(_step(executor=executor), _ctx())

    assert result.metadata["mcp_context"]["truncated"] is True


def test_non_finite_metadata_does_not_break_the_run_result() -> None:
    job = _job()
    pipeline = _pipeline()
    pipeline.jobs = {job.id: job}
    executor = _SpyExecutor({"messages": [AIMessage(content="ok")], "mcp_context": {"score": float("inf")}})

    result = asyncio.run(
        OrchidPipelineRunner(
            pipeline,
            deps=PipelineDependencies(
                agents={"helper": _Agent()},
                agent_executors={"helper": executor},
            ),
        ).run({"account_id": "a1"}, user_id="u1")
    )

    assert result.status == "succeeded"
    assert json.dumps(result.to_dict())


# ── Input bounds ──────────────────────────────────────────


def test_structured_input_over_the_context_cap_fails() -> None:
    job = _job(_config(input={"account": "x" * 100}))
    limits = PipelineLimits(max_context_chars=10)

    with pytest.raises(OrchidPipelineLimitError, match="max_context_chars=10"):
        _run(_step(job=job, executor=_SpyExecutor()), _ctx(job=job, limits=limits))


def test_prompt_plus_input_over_the_prompt_cap_fails() -> None:
    job = _job(_config(prompt="{{ input.account_id }}", input={"account": "x" * 100}))
    limits = PipelineLimits(max_prompt_chars=10, max_context_chars=1000)

    with pytest.raises(OrchidPipelineLimitError, match="max_prompt_chars=10"):
        _run(_step(job=job, executor=_SpyExecutor()), _ctx(job=job, limits=limits))


# ── Registry / runner integration ─────────────────────────


def test_pipeline_dependencies_keep_the_positional_prefix() -> None:
    model = object()
    tool = object()
    emitter = object()
    config = object()
    executor = object()

    deps = PipelineDependencies(
        None, {"model": model}, {"tool": tool}, emitter, {"helper": config}, {"helper": executor}
    )

    assert deps.agents is None
    assert deps.decision_models == {"model": model}
    assert deps.tools == {"tool": tool}
    assert deps.signal_emitter is emitter
    assert deps.agent_configs == {"helper": config}
    assert deps.agent_executors == {"helper": executor}


def test_agent_kind_is_registered() -> None:
    from orchid_ai.pipeline.registry import PIPELINE_STEP_REGISTRY, resolve_pipeline_step
    from orchid_ai.pipeline.steps import register_builtin_steps

    before = dict(PIPELINE_STEP_REGISTRY)
    try:
        PIPELINE_STEP_REGISTRY.pop("agent", None)
        register_builtin_steps()
        step = resolve_pipeline_step(
            "agent",
            _job(),
            PipelineDependencies(
                agents={"helper": _Agent()},
                agent_configs={},
                agent_executors={},
            ),
        )
        assert isinstance(step, AgentStep)
    finally:
        PIPELINE_STEP_REGISTRY.clear()
        PIPELINE_STEP_REGISTRY.update(before)


def test_agent_job_runs_through_the_runner() -> None:
    agent = _Agent("runner answer")
    job = _job()
    pipeline = _pipeline()
    pipeline.jobs = {job.id: job}

    result = asyncio.run(
        OrchidPipelineRunner(
            pipeline,
            deps=PipelineDependencies(agents={"helper": agent}, agent_configs={}, agent_executors={}),
        ).run({"account_id": "a1"}, user_id="u1")
    )

    assert result.status == "succeeded"
    assert result.jobs["research"]["output"] == {"text": "runner answer"}
    assert result.jobs["research"]["metadata"]["skills_metadata"] == {"detected": ["lookup"]}
    assert result.jobs["research"]["llm_calls"] == 1


def test_non_agent_job_is_rejected() -> None:
    job = CompiledJob(
        id="x",
        kind="deterministic",
        config=OrchidPipelineDeterministicJobConfig.model_validate({"kind": "deterministic", "uses": "pkg.step"}),
    )
    with pytest.raises(TypeError, match="OrchidPipelineAgentJobConfig"):
        AgentStep(job, agents={})
