"""Tests for decision-backed external-agent delegation.

The ``delegation`` point may choose ``local`` or one *configured and
allowlisted* external CLI agent. It never invents an agent, never sees a
command/argument/secret, and every uncertain outcome keeps local handling. A
local outcome suppresses external-agent tools for the run; an accepted external
outcome narrows the agentic loop to that one tool, which still dispatches through
the existing ``ExternalAgentCLITool``/``CLIRunner`` path (including HITL).

All tests are offline: decision providers are in-process fakes, CLI execution
uses a fake runner, and no real subprocess is allowed to start.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

import orchid_ai.agents.delegation as delegation_module
from orchid_ai.agents.delegation import (
    LOCAL_TARGET,
    MAX_DESCRIPTION_CHARS,
    MAX_QUERY_CHARS,
    DelegationCatalog,
    DelegationRequest,
    decide_delegation,
)
from orchid_ai.agents.generic_agent import GenericAgent
from orchid_ai.agents.mcp_dispatcher import MCPCapabilities
from orchid_ai.config import registry as agent_registry
from orchid_ai.config.schema import OrchidAgentConfig, OrchidAgentsConfig
from orchid_ai.config.schema_decision import OrchidDecisionPointConfig
from orchid_ai.config.schema_external_agent import OrchidExternalAgentConfig
from orchid_ai.config.tool_registry import TOOL_REGISTRY
from orchid_ai.core.agent import OrchidAgent, OrchidAgentRunContext
from orchid_ai.core.decision import (
    OrchidDecisionAnswer,
    OrchidDecisionError,
    OrchidDecisionQuestion,
    OrchidDecisionResult,
    OrchidDecisionState,
)
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.core.tool import OrchidTool, OrchidToolInput, OrchidToolOutput
from orchid_ai.decision.registry import get_decision_point
from orchid_ai.decision.stub import StubDecisionModel
from orchid_ai.graph.decision_bindings import DecisionBinding, resolve_decision_binding
from orchid_ai.graph.graph import build_graph
from orchid_ai.rag.backends.null import NullVectorReader
from orchid_ai.runtime import OrchidRuntime
from orchid_ai.tools.cli_runner import AsyncSubprocessCLIRunner, CLIResult, CLIRunner
from orchid_ai.tools.external_cli import ExternalAgentCLITool

SECRET = "sk-CLI-SECRET-DO-NOT-LEAK"
QUERY = "SENTINEL-USER-REQUEST-DO-NOT-LOG"
LOOP_TOOL = "lookup"


# ── Test doubles ────────────────────────────────────────────


class _ChoiceProvider:
    """Return a fixed choice and record exactly what the gate sent."""

    def __init__(self, value: str, *, confidence: float = 0.95) -> None:
        self.value = value
        self.confidence = confidence
        self.calls = 0
        self.questions: list[OrchidDecisionQuestion] = []
        self.states: list[OrchidDecisionState] = []
        self.timeouts: list[float | None] = []

    async def decide(self, state, questions, *, timeout=None):
        self.calls += 1
        self.states.append(state)
        self.questions.extend(questions)
        self.timeouts.append(timeout)
        question = questions[0]
        return OrchidDecisionResult(
            answers={
                question.name: OrchidDecisionAnswer(
                    name=question.name,
                    kind="choice",
                    value=self.value,
                    confidence=self.confidence,
                )
            },
            provider="recorder",
            model="recorder",
        )


class _BooleanProvider:
    async def decide(self, state, questions, *, timeout=None):
        question = questions[0]
        return OrchidDecisionResult(
            answers={
                question.name: OrchidDecisionAnswer(
                    name=question.name,
                    kind="boolean",
                    value=True,
                    confidence=0.95,
                )
            },
            provider="boolean",
            model="boolean",
        )


class _RefusalProvider:
    async def decide(self, state, questions, *, timeout=None):
        question = questions[0]
        return OrchidDecisionResult(
            answers={
                question.name: OrchidDecisionAnswer(
                    name=question.name,
                    kind="choice",
                    value=None,
                    confidence=0.9,
                    refusal="not enough context",
                )
            },
            provider="refusal",
            model="refusal",
        )


class _EmptyProvider:
    async def decide(self, state, questions, *, timeout=None):
        return OrchidDecisionResult(answers={}, provider="empty", model="empty")


class _TimeoutProvider:
    async def decide(self, state, questions, *, timeout=None):
        raise TimeoutError("deadline exceeded")


class _ErrorProvider:
    def __init__(self, code: str) -> None:
        self.code = code

    async def decide(self, state, questions, *, timeout=None):
        raise OrchidDecisionError("provider failed", code=self.code, provider="error-provider")


class _RaisingProvider:
    async def decide(self, state, questions, *, timeout=None):
        raise RuntimeError(f"unexpected failure {SECRET}")


class _MalformedProvider:
    async def decide(self, state, questions, *, timeout=None):
        return object()


class _ScriptedChatModel:
    """Offline chat model returning one scripted response per call."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.tool_defs: list[dict[str, Any]] = []

    def bind_tools(self, tool_defs: list[dict[str, Any]]) -> _ScriptedChatModel:
        self.tool_defs = list(tool_defs)
        return self

    async def ainvoke(self, messages: list, **kwargs: Any) -> AIMessage:
        index = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[index]


class _FakeCLIRunner(CLIRunner):
    """Records the exact CLI request without starting a process."""

    def __init__(self, result: CLIResult | None = None) -> None:
        self.result = result or CLIResult(exit_code=0, stdout="external answer", stderr="")
        self.calls: list[dict[str, Any]] = []

    async def run(
        self,
        *,
        command: list[str],
        prompt: str,
        cwd: str | None,
        timeout: float,
        env: dict[str, str] | None,
        stdin_mode: str,
    ) -> CLIResult:
        self.calls.append(
            {
                "command": command,
                "prompt": prompt,
                "cwd": cwd,
                "timeout": timeout,
                "env": env,
                "stdin_mode": stdin_mode,
            }
        )
        return self.result


class _RecordingExternalTool(ExternalAgentCLITool):
    """External CLI tool that records the auth context it received."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tool_inputs: list[OrchidToolInput] = []

    async def invoke(self, tool_input: OrchidToolInput) -> OrchidToolOutput:
        self.tool_inputs.append(tool_input)
        return await super().invoke(tool_input)


class _EchoTool(OrchidTool):
    name = LOOP_TOOL
    description = "local lookup tool"
    parameters_schema: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

    async def invoke(self, tool_input: OrchidToolInput) -> OrchidToolOutput:
        return OrchidToolOutput(result={"ok": True})


class _FakeMCPDispatcher:
    """Capabilities stub that advertises one MCP tool."""

    def __init__(self, tool_name: str = "mcp_lookup") -> None:
        self.tool_name = tool_name

    async def render_capabilities(self, auth: OrchidAuthContext, *, agent_name: str = "") -> MCPCapabilities:
        return MCPCapabilities(
            raw_tools=[{"name": self.tool_name, "description": "mcp tool", "schema": {"type": "object"}}],
            tool_client_map={},
        )


class _StrictLegacyAgent(OrchidAgent):
    """Custom agent with a strict constructor: no ``**kwargs`` catch-all."""

    def __init__(
        self,
        *,
        model_id: str = "",
        reader: Any,
        mcp_clients: list[Any] | None = None,
        chat_model: Any | None = None,
        config: Any = None,
        upload_namespace: str = "uploads",
        graph_store: Any | None = None,
        summary_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            reader=reader,
            mcp_clients=mcp_clients,
            chat_model=chat_model,
            config=config,
        )
        self._config = config

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def description(self) -> str:
        return self._config.description

    def needs_peer_wiring(self) -> bool:
        return False

    async def run(self, state: Any) -> Any:
        return {"messages": [AIMessage(content="strict answer")]}


# ── Fixtures and helpers ────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_tool_registry() -> Any:
    """Restore the process-wide tool registry after every test."""
    before = dict(TOOL_REGISTRY.get_all())
    yield
    TOOL_REGISTRY.clear()
    for tool in before.values():
        TOOL_REGISTRY.register(tool)


@pytest.fixture(autouse=True)
def _forbid_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fail(*args: Any, **kwargs: Any) -> CLIResult:
        raise AssertionError("delegation tests must not start a real subprocess")

    monkeypatch.setattr(AsyncSubprocessCLIRunner, "run", _fail)


def _auth() -> OrchidAuthContext:
    return OrchidAuthContext(access_token="test-token", tenant_key="t", user_id="u")


def _binding(
    provider: Any,
    *,
    threshold: float = 0.75,
    options: dict[str, Any] | None = None,
) -> DecisionBinding:
    return DecisionBinding(
        spec=get_decision_point("delegation"),
        config=OrchidDecisionPointConfig(
            model="scripted",
            confidence_threshold=threshold,
            options=options or {},
        ),
        model=provider,
    )


def _external_configs(*names: str, requires_approval: bool = False) -> dict[str, OrchidExternalAgentConfig]:
    return {
        name: OrchidExternalAgentConfig(
            command=["fake-cli"],
            args=["--token", SECRET],
            env={"API_TOKEN": SECRET},
            description=f"{name} helper",
            requires_approval=requires_approval,
        )
        for name in names
    }


def _catalog(*names: str, allowed_tools: list[str] | None = None) -> DelegationCatalog:
    return DelegationCatalog.from_config(
        _external_configs(*names),
        allowed_tools=list(names) if allowed_tools is None else allowed_tools,
    )


def _request(query: str = QUERY) -> DelegationRequest:
    return DelegationRequest.from_query(query, agent_name="a")


def _decision_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == "orchid.decision"]


def _register_local_tool() -> None:
    TOOL_REGISTRY.unregister(LOOP_TOOL)
    TOOL_REGISTRY.register(_EchoTool())


def _install_external(name: str, runner: _FakeCLIRunner, *, requires_approval: bool = False) -> _RecordingExternalTool:
    tool = _RecordingExternalTool(
        name=name,
        description=f"{name} helper",
        command=["fake-cli"],
        runner=runner,
    )
    tool.requires_approval = requires_approval
    TOOL_REGISTRY.unregister(name)
    TOOL_REGISTRY.register(tool)
    return tool


def _agent_entry(tools: list[str]) -> dict[str, Any]:
    return {
        "description": "test agent",
        "prompt": "you are a test agent",
        "llm": {"model": "test/model"},
        "rag": {"enabled": False},
        "tools": tools,
    }


def _graph_config(
    *,
    tools: tuple[str, ...] = ("ask", "lookup"),
    external: tuple[str, ...] = ("ask",),
    bind: bool = True,
    agent_level: bool = True,
    threshold: float = 0.75,
    requires_approval: bool = False,
    skill_tool: str | None = None,
    safe_skill: bool = False,
    agent_class: str | None = None,
) -> OrchidAgentsConfig:
    agent = _agent_entry(list(tools))
    if agent_class is not None:
        agent["class"] = agent_class
    skills: dict[str, Any] = {}
    if skill_tool is not None:
        skills["external_skill"] = {
            "description": "calls an external agent",
            "steps": [{"tool": skill_tool}],
        }
    if safe_skill:
        skills["safe_skill"] = {
            "description": "safe local skill",
            "steps": [{"tool": LOOP_TOOL}],
        }
    if skills:
        agent["skills"] = skills
    external_agents: dict[str, Any] = {
        name: {
            "command": ["fake-cli"],
            "args": ["--token", SECRET],
            "env": {"API_TOKEN": SECRET},
            "description": f"{name} helper",
            "requires_approval": requires_approval,
        }
        for name in external
    }
    data: dict[str, Any] = {
        "version": "1",
        "agents": {"a": agent},
        "external_agents": external_agents,
        "decision_models": {"stub": {"protocol": "stub"}},
    }
    if bind:
        point: dict[str, Any] = {"model": "stub", "confidence_threshold": threshold}
        if agent_level:
            agent["decisions"] = {"points": {"delegation": point}}
        else:
            data["decisions"] = {"points": {"delegation": point}}
    return OrchidAgentsConfig.model_validate(data)


def _build_agent(
    config: OrchidAgentsConfig,
    provider: Any,
    chat_model: Any,
    *,
    include_provider: bool = True,
) -> GenericAgent:
    runtime = OrchidRuntime(
        default_model="test/model",
        chat_model=chat_model,
        decision_models={"stub": provider} if include_provider else {},
    )
    agents: dict[str, Any] = {}
    build_graph(config=config, runtime=runtime, agents_out=agents)
    return agents["a"]


async def _run_agent(agent: GenericAgent, query: str = QUERY) -> dict[str, Any]:
    auth = _auth()
    token = agent.set_run_context(OrchidAgentRunContext(auth=auth))
    try:
        return await agent.run({"messages": [HumanMessage(content=query)]})
    finally:
        agent.reset_run_context(token)


def _tool_names(chat_model: _ScriptedChatModel) -> list[str]:
    return [tool_def["function"]["name"] for tool_def in chat_model.tool_defs]


# ── Catalog and request bounds ──────────────────────────────


def test_catalog_exposes_only_names_and_descriptions() -> None:
    catalog = DelegationCatalog.from_config(_external_configs("ask"), allowed_tools=["ask"])

    assert catalog.names == ("ask",)
    assert catalog.agents[0].to_payload() == {"name": "ask", "description": "ask helper"}
    serialised = repr(catalog)
    assert SECRET not in serialised
    assert "fake-cli" not in serialised
    assert "--token" not in serialised


def test_catalog_narrows_to_the_agents_tool_allowlist() -> None:
    catalog = DelegationCatalog.from_config(_external_configs("ask", "other"), allowed_tools=["ask"])

    assert catalog.names == ("ask",)


def test_catalog_without_an_allowlist_keeps_every_configured_external_agent() -> None:
    catalog = DelegationCatalog.from_config(_external_configs("ask", "other"))

    assert catalog.names == ("ask", "other")


def test_catalog_description_is_collapsed_and_bounded() -> None:
    config = _external_configs("ask")
    config["ask"].description = "line one\nline two " + ("x" * (MAX_DESCRIPTION_CHARS * 2))

    catalog = DelegationCatalog.from_config(config, allowed_tools=["ask"])

    assert "\n" not in catalog.agents[0].description
    assert len(catalog.agents[0].description) == MAX_DESCRIPTION_CHARS


def test_request_truncates_the_query_to_the_hard_bound() -> None:
    request = DelegationRequest.from_query(QUERY + ("x" * MAX_QUERY_CHARS) + "TAIL-SENTINEL")

    assert len(request.query) == MAX_QUERY_CHARS
    assert "TAIL-SENTINEL" not in request.query


def test_oversized_external_agent_name_is_skipped_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    oversized = "x" * (delegation_module.MAX_AGENT_NAME_CHARS + 1)

    with caplog.at_level(logging.WARNING, logger="orchid_ai.agents.delegation"):
        catalog = DelegationCatalog.from_config(
            _external_configs(oversized, "ask"),
            allowed_tools=[oversized, "ask"],
        )

    assert catalog.names == ("ask",)
    assert "longer than" in caplog.text


# ── Gate outcomes ───────────────────────────────────────────


async def test_accepted_external_agent_is_selected(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ChoiceProvider("ask")

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_delegation(_binding(provider), _request(), _catalog("ask", "other"))

    assert decision.delegated
    assert decision.target == "ask"
    assert decision.source == "decision"
    assert decision.confidence == pytest.approx(0.95)
    assert provider.questions[0].name == "target"
    assert provider.questions[0].options == (LOCAL_TARGET, "ask", "other")
    assert provider.states[0].payload["external_agents"] == [
        {"name": "ask", "description": "ask helper"},
        {"name": "other", "description": "other helper"},
    ]
    lines = _decision_lines(caplog)
    assert len(lines) == 1
    for fragment in (
        "decision point=delegation",
        "path=decision",
        "provider=recorder",
        "outcome=accepted",
        "fallback_reason=null",
    ):
        assert fragment in lines[0]


async def test_accepted_local_target_keeps_local_handling(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ChoiceProvider(LOCAL_TARGET)

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_delegation(_binding(provider), _request(), _catalog("ask"))

    assert decision.accepted
    assert not decision.delegated
    assert decision.target == LOCAL_TARGET
    assert "outcome=accepted" in _decision_lines(caplog)[0]


async def test_unknown_agent_falls_back_to_local(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ChoiceProvider("telepathy")

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_delegation(_binding(provider), _request(), _catalog("ask"))

    assert decision.source == "fallback"
    assert decision.target == LOCAL_TARGET
    assert decision.fallback_reason == "unknown_agent"
    assert "fallback_reason=unknown_agent" in _decision_lines(caplog)[0]


async def test_agent_outside_the_agent_tool_allowlist_is_never_a_candidate() -> None:
    catalog = DelegationCatalog.from_config(_external_configs("ask", "other"), allowed_tools=["ask"])
    provider = _ChoiceProvider("other")

    decision = await decide_delegation(_binding(provider), _request(), catalog)

    assert decision.target == LOCAL_TARGET
    assert decision.fallback_reason == "unknown_agent"
    assert provider.questions[0].options == (LOCAL_TARGET, "ask")


async def test_options_allowlist_narrows_candidates_and_rejects_an_outside_pick(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _ChoiceProvider("ask")

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_delegation(
            _binding(provider, options={"agents": ["other"]}),
            _request(),
            _catalog("ask", "other"),
        )

    assert decision.target == LOCAL_TARGET
    assert decision.fallback_reason == "out_of_allowlist"
    assert provider.questions[0].options == (LOCAL_TARGET, "other")
    assert "fallback_reason=out_of_allowlist" in _decision_lines(caplog)[0]


async def test_invalid_options_fall_back_without_a_provider_call(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ChoiceProvider("ask")

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_delegation(
            _binding(provider, options={"agents": ["ghost"]}),
            _request(),
            _catalog("ask"),
        )

    assert decision.target == LOCAL_TARGET
    assert decision.fallback_reason == "invalid_options"
    assert provider.calls == 0
    line = _decision_lines(caplog)[0]
    assert "fallback_reason=invalid_options" in line
    assert "latency_ms=null" in line


async def test_reserved_local_agent_name_falls_back_without_a_call() -> None:
    provider = _ChoiceProvider(LOCAL_TARGET)
    catalog = DelegationCatalog.from_config(_external_configs("local"), allowed_tools=["local"])

    decision = await decide_delegation(_binding(provider), _request(), catalog)

    assert decision.target == LOCAL_TARGET
    assert decision.fallback_reason == "invalid_options"
    assert provider.calls == 0


async def test_too_many_candidates_falls_back_without_a_call(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(delegation_module, "MAX_CANDIDATES", 1)
    provider = _ChoiceProvider("ask")

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_delegation(_binding(provider), _request(), _catalog("ask", "other"))

    assert decision.fallback_reason == "too_many_candidates"
    assert provider.calls == 0
    assert "latency_ms=null" in _decision_lines(caplog)[0]


async def test_low_confidence_falls_back_to_local(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ChoiceProvider("ask", confidence=0.2)

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_delegation(_binding(provider, threshold=0.8), _request(), _catalog("ask"))

    assert decision.target == LOCAL_TARGET
    assert decision.fallback_reason == "low_confidence"
    assert "fallback_reason=low_confidence" in _decision_lines(caplog)[0]


async def test_refusal_falls_back_to_local(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_delegation(_binding(_RefusalProvider()), _request(), _catalog("ask"))

    assert decision.target == LOCAL_TARGET
    assert decision.fallback_reason == "refusal"
    assert "fallback_reason=refusal" in _decision_lines(caplog)[0]


async def test_timeout_falls_back_to_local(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_delegation(_binding(_TimeoutProvider()), _request(), _catalog("ask"))

    assert decision.target == LOCAL_TARGET
    assert decision.fallback_reason == "timeout"
    assert "fallback_reason=timeout" in _decision_lines(caplog)[0]


async def test_provider_error_falls_back_to_local(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_delegation(
            _binding(_ErrorProvider("provider_unavailable")),
            _request(),
            _catalog("ask"),
        )

    assert decision.target == LOCAL_TARGET
    assert decision.fallback_reason == "provider_unavailable"
    assert "fallback_reason=provider_unavailable" in _decision_lines(caplog)[0]


async def test_unexpected_provider_error_falls_back_and_does_not_log_the_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG):
        decision = await decide_delegation(_binding(_RaisingProvider()), _request(), _catalog("ask"))

    assert decision.target == LOCAL_TARGET
    assert decision.fallback_reason == "provider_error"
    assert SECRET not in caplog.text
    assert QUERY not in caplog.text


async def test_missing_answer_falls_back_to_local(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_delegation(_binding(_EmptyProvider()), _request(), _catalog("ask"))

    assert decision.target == LOCAL_TARGET
    assert decision.fallback_reason == "invalid_response"


async def test_non_choice_answer_falls_back_to_local() -> None:
    decision = await decide_delegation(_binding(_BooleanProvider()), _request(), _catalog("ask"))

    assert decision.target == LOCAL_TARGET
    assert decision.fallback_reason == "invalid_response"


async def test_malformed_provider_result_falls_back_to_local() -> None:
    decision = await decide_delegation(_binding(_MalformedProvider()), _request(), _catalog("ask"))

    assert decision.target == LOCAL_TARGET
    assert decision.fallback_reason == "invalid_response"


async def test_no_external_agents_skips_the_provider(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ChoiceProvider("ask")

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_delegation(_binding(provider), _request(), DelegationCatalog())

    assert decision.source == "disabled"
    assert decision.target == LOCAL_TARGET
    assert provider.calls == 0
    assert _decision_lines(caplog) == []


async def test_disabled_point_skips_the_provider(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        decision = await decide_delegation(None, _request(), _catalog("ask"))

    assert decision.source == "disabled"
    assert decision.target == LOCAL_TARGET
    assert _decision_lines(caplog) == []


async def test_decision_state_and_logs_never_carry_cli_secrets(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ChoiceProvider("ask")

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        await decide_delegation(_binding(provider), _request(), _catalog("ask"))

    sent = repr(provider.states[0].payload) + provider.states[0].text
    assert SECRET not in sent
    assert "fake-cli" not in sent
    assert "--token" not in sent
    assert QUERY not in caplog.text
    assert SECRET not in caplog.text


# ── Graph wiring ────────────────────────────────────────────


def _graph_runtime(provider: Any | None = None) -> OrchidRuntime:
    chat = MagicMock(spec=BaseChatModel)
    chat.bind_tools = MagicMock(return_value=chat)
    return OrchidRuntime(
        default_model="test/model",
        chat_model=chat,
        decision_models={"stub": provider or StubDecisionModel()},
    )


def test_graph_resolves_the_agent_scoped_delegation_binding() -> None:
    config = _graph_config(agent_level=True)
    agents: dict[str, Any] = {}

    build_graph(config=config, runtime=_graph_runtime(), agents_out=agents)

    binding = agents["a"]._delegation_binding
    assert binding is not None
    assert binding.point == "delegation"
    assert binding.provider == "stub"
    assert agents["a"]._delegation_catalog.names == ("ask",)


def test_graph_resolves_the_global_delegation_binding() -> None:
    config = _graph_config(agent_level=False)
    agents: dict[str, Any] = {}

    build_graph(config=config, runtime=_graph_runtime(), agents_out=agents)

    binding = agents["a"]._delegation_binding
    assert binding is not None
    assert binding.point == "delegation"


def test_graph_without_a_delegation_binding_keeps_the_legacy_path() -> None:
    config = _graph_config(bind=False)
    agents: dict[str, Any] = {}

    build_graph(config=config, runtime=_graph_runtime(), agents_out=agents)

    assert agents["a"]._delegation_binding is None
    assert agents["a"]._delegation_fail_closed is False


def test_graph_catalog_is_empty_when_the_agent_lists_no_external_tools() -> None:
    config = _graph_config(tools=("lookup",))
    agents: dict[str, Any] = {}

    build_graph(config=config, runtime=_graph_runtime(), agents_out=agents)

    assert agents["a"]._delegation_binding is not None
    assert agents["a"]._delegation_catalog.names == ()
    assert agents["a"]._delegation_external_agents == frozenset({"ask"})


def test_missing_runtime_provider_yields_no_delegation_binding(caplog: pytest.LogCaptureFixture) -> None:
    config = _graph_config()
    runtime = OrchidRuntime(default_model="test/model")

    with caplog.at_level(logging.WARNING, logger="orchid_ai.graph.decision_bindings"):
        binding = resolve_decision_binding(config, runtime, "delegation", agent_name="a")

    assert binding is None
    assert "missing from the runtime" in caplog.text


def test_child_agents_receive_their_own_delegation_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    import orchid_ai.graph.graph as graph_module

    child = _agent_entry(["ask", "lookup"])
    child["decisions"] = {"points": {"delegation": {"model": "stub"}}}
    parent = _agent_entry(["lookup"])
    parent["children"] = {"kid": child, "sibling": _agent_entry(["lookup"])}
    config = OrchidAgentsConfig.model_validate(
        {
            "version": "1",
            "agents": {"parent": parent},
            "external_agents": {"ask": {"command": ["fake-cli"], "description": "ask helper"}},
            "decision_models": {"stub": {"protocol": "stub"}},
        }
    )
    created: dict[str, Any] = {}
    real_instantiate = graph_module._instantiate_agent

    def _capture(name, agent_config, *args, **kwargs):
        agent = real_instantiate(name, agent_config, *args, **kwargs)
        created[name] = agent
        return agent

    monkeypatch.setattr(graph_module, "_instantiate_agent", _capture)
    build_graph(config=config, runtime=_graph_runtime())

    assert created["kid"]._delegation_binding is not None
    assert created["kid"]._delegation_catalog.names == ("ask",)
    assert created["sibling"]._delegation_binding is None
    assert created["sibling"]._delegation_catalog.names == ()


async def test_generic_agent_fails_closed_when_the_gate_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr("orchid_ai.agents.generic_agent.decide_delegation", _boom)
    agent = GenericAgent(
        config=OrchidAgentConfig(description="d", prompt="p", name="a"),
        reader=NullVectorReader(),
        delegation_binding=_binding(_ChoiceProvider("ask")),
        delegation_catalog=_catalog("ask"),
    )

    decision = await agent._resolve_delegation(QUERY)

    assert decision is not None
    assert decision.target == LOCAL_TARGET
    assert decision.source == "fallback"
    assert not decision.delegated


# ── GenericAgent tool policy (full run, fake CLI) ───────────


async def test_accepted_delegation_uses_the_existing_external_tool_path() -> None:
    provider = _ChoiceProvider("ask")
    runner = _FakeCLIRunner()
    chat = _ScriptedChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "ask", "args": {"prompt": QUERY}, "id": "call-1"}],
            ),
            AIMessage(content="external answer"),
        ]
    )
    agent = _build_agent(_graph_config(tools=("ask", "lookup")), provider, chat)
    _register_local_tool()
    tool = _install_external("ask", runner)

    result = await _run_agent(agent)

    assert _tool_names(chat) == ["ask"]
    assert [call["prompt"] for call in runner.calls] == [QUERY]
    assert tool.tool_inputs[0].auth_context is not None
    assert tool.tool_inputs[0].auth_context.access_token == "test-token"
    assert "external answer" in result["messages"][-1].content
    assert provider.calls == 1


async def test_local_decision_suppresses_every_external_agent_tool() -> None:
    provider = _ChoiceProvider(LOCAL_TARGET)
    runner = _FakeCLIRunner()
    other_runner = _FakeCLIRunner()
    chat = _ScriptedChatModel([AIMessage(content="local answer")])
    agent = _build_agent(_graph_config(tools=("ask", "other", "lookup"), external=("ask", "other")), provider, chat)
    _register_local_tool()
    _install_external("ask", runner)
    _install_external("other", other_runner)

    result = await _run_agent(agent)

    assert _tool_names(chat) == ["lookup"]
    assert runner.calls == []
    assert other_runner.calls == []
    assert "local answer" in result["messages"][-1].content


async def test_local_decision_skips_a_skill_that_calls_an_external_agent() -> None:
    provider = _ChoiceProvider(LOCAL_TARGET)
    runner = _FakeCLIRunner()
    chat = _ScriptedChatModel([AIMessage(content="local answer")])
    agent = _build_agent(_graph_config(tools=("ask", "lookup"), skill_tool="ask"), provider, chat)
    _register_local_tool()
    _install_external("ask", runner)

    async def _detect(_query: str) -> str:
        return "external_skill"

    async def _must_not_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("a local delegation outcome must not run an external-agent skill step")

    agent._detect_skill = _detect  # type: ignore[method-assign]
    agent._skill_executor.run_skill = _must_not_run  # type: ignore[method-assign]

    await _run_agent(agent)

    assert _tool_names(chat) == ["lookup"]
    assert runner.calls == []


async def test_low_confidence_fallback_suppresses_external_agent_tools() -> None:
    provider = _ChoiceProvider("ask", confidence=0.2)
    runner = _FakeCLIRunner()
    chat = _ScriptedChatModel([AIMessage(content="local answer")])
    agent = _build_agent(
        _graph_config(tools=("ask", "lookup"), threshold=0.8),
        provider,
        chat,
    )
    _register_local_tool()
    _install_external("ask", runner)

    await _run_agent(agent)

    assert _tool_names(chat) == ["lookup"]
    assert runner.calls == []


async def test_agent_outside_the_tool_allowlist_is_not_delegated() -> None:
    provider = _ChoiceProvider("other")
    runner = _FakeCLIRunner()
    other_runner = _FakeCLIRunner()
    chat = _ScriptedChatModel([AIMessage(content="local answer")])
    agent = _build_agent(
        _graph_config(tools=("ask", "lookup"), external=("ask", "other")),
        provider,
        chat,
    )
    _register_local_tool()
    _install_external("ask", runner)
    _install_external("other", other_runner)

    await _run_agent(agent)

    assert provider.questions[0].options == (LOCAL_TARGET, "ask")
    assert _tool_names(chat) == ["lookup"]
    assert runner.calls == []
    assert other_runner.calls == []


async def test_disabled_point_keeps_the_legacy_tool_inventory() -> None:
    runner = _FakeCLIRunner()
    chat = _ScriptedChatModel([AIMessage(content="legacy answer")])
    agent = _build_agent(_graph_config(tools=("ask", "lookup"), bind=False), StubDecisionModel(), chat)
    _register_local_tool()
    _install_external("ask", runner)

    await _run_agent(agent)

    assert set(_tool_names(chat)) == {"ask", "lookup"}
    assert runner.calls == []


async def test_no_external_agents_makes_zero_provider_calls() -> None:
    provider = _ChoiceProvider("ask")
    chat = _ScriptedChatModel([AIMessage(content="local answer")])
    agent = _build_agent(_graph_config(tools=("lookup",), external=()), provider, chat)
    _register_local_tool()

    await _run_agent(agent)

    assert provider.calls == 0
    assert _tool_names(chat) == ["lookup"]


async def test_accepted_delegation_preserves_the_hitl_approval_path(monkeypatch: pytest.MonkeyPatch) -> None:
    interrupts: list[dict[str, Any]] = []

    def _approve(payload: dict[str, Any]) -> dict[str, Any]:
        interrupts.append(payload)
        return {"approved": True}

    monkeypatch.setattr("langgraph.types.interrupt", _approve)
    provider = _ChoiceProvider("ask")
    runner = _FakeCLIRunner()
    chat = _ScriptedChatModel(
        [
            AIMessage(content="", tool_calls=[{"name": "ask", "args": {"prompt": QUERY}, "id": "call-1"}]),
            AIMessage(content="external answer"),
        ]
    )
    agent = _build_agent(
        _graph_config(tools=("ask",), external=("ask",), requires_approval=True),
        provider,
        chat,
    )
    _install_external("ask", runner, requires_approval=True)

    await _run_agent(agent)

    assert interrupts == [{"type": "tool_approval", "tool": "ask", "args": {"prompt": QUERY}, "agent": "a"}]
    assert [call["prompt"] for call in runner.calls] == [QUERY]


async def test_hitl_denial_never_starts_the_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    interrupts: list[dict[str, Any]] = []

    def _deny(payload: dict[str, Any]) -> dict[str, Any]:
        interrupts.append(payload)
        return {"approved": False}

    monkeypatch.setattr("langgraph.types.interrupt", _deny)
    provider = _ChoiceProvider("ask")
    runner = _FakeCLIRunner()
    chat = _ScriptedChatModel(
        [
            AIMessage(content="", tool_calls=[{"name": "ask", "args": {"prompt": QUERY}, "id": "call-1"}]),
            AIMessage(content="local answer"),
        ]
    )
    agent = _build_agent(
        _graph_config(tools=("ask",), external=("ask",), requires_approval=True),
        provider,
        chat,
    )
    _install_external("ask", runner, requires_approval=True)

    result = await _run_agent(agent)

    assert interrupts == [{"type": "tool_approval", "tool": "ask", "args": {"prompt": QUERY}, "agent": "a"}]
    assert runner.calls == []
    assert "cancelled" in result["mcp_context"]["a"]["ask"].lower()


async def test_missing_provider_fails_closed_on_the_full_graph() -> None:
    """F1: enabled point + missing provider must not fall back to legacy tools."""
    provider = _ChoiceProvider("ask")
    runner = _FakeCLIRunner()
    chat = _ScriptedChatModel(
        [
            AIMessage(content="", tool_calls=[{"name": "ask", "args": {"prompt": QUERY}, "id": "call-1"}]),
            AIMessage(content="local answer"),
        ]
    )
    agent = _build_agent(
        _graph_config(tools=("ask", "lookup")),
        provider,
        chat,
        include_provider=False,
    )
    _register_local_tool()
    _install_external("ask", runner)

    await _run_agent(agent)

    assert agent._delegation_binding is None
    assert agent._delegation_fail_closed is True
    assert agent._delegation_external_agents == frozenset({"ask"})
    assert _tool_names(chat) == ["lookup"]
    assert runner.calls == []
    assert provider.calls == 0


@pytest.mark.parametrize(
    "provider",
    [_ErrorProvider("provider_unavailable"), _TimeoutProvider()],
    ids=["provider-error", "timeout"],
)
async def test_provider_failure_suppresses_external_tools_on_the_full_graph(provider: Any) -> None:
    """F4: a decision failure at agent runtime must suppress the CLI."""
    runner = _FakeCLIRunner()
    chat = _ScriptedChatModel(
        [
            AIMessage(content="", tool_calls=[{"name": "ask", "args": {"prompt": QUERY}, "id": "call-1"}]),
            AIMessage(content="local answer"),
        ]
    )
    agent = _build_agent(_graph_config(tools=("ask", "lookup")), provider, chat)
    _register_local_tool()
    _install_external("ask", runner)

    await _run_agent(agent)

    assert _tool_names(chat) == ["lookup"]
    assert runner.calls == []


async def test_accepted_external_agent_suppresses_mcp_tools() -> None:
    """F4: delegation narrows the loop to the selected external builtin only."""
    provider = _ChoiceProvider("ask")
    runner = _FakeCLIRunner()
    chat = _ScriptedChatModel(
        [
            AIMessage(content="", tool_calls=[{"name": "ask", "args": {"prompt": QUERY}, "id": "call-1"}]),
            AIMessage(content="external answer"),
        ]
    )
    agent = _build_agent(_graph_config(tools=("ask", "lookup")), provider, chat)
    agent._mcp_dispatcher = _FakeMCPDispatcher("mcp_lookup")  # type: ignore[assignment]
    _register_local_tool()
    _install_external("ask", runner)

    await _run_agent(agent)

    assert _tool_names(chat) == ["ask"]
    assert "mcp_lookup" not in _tool_names(chat)
    assert [call["prompt"] for call in runner.calls] == [QUERY]


async def test_empty_catalog_still_blocks_a_skill_step_naming_an_external_agent() -> None:
    """F2: the full configured set, not the narrowed catalog, guards skills."""
    provider = _ChoiceProvider("ask")
    runner = _FakeCLIRunner()
    chat = _ScriptedChatModel([AIMessage(content="local answer")])
    agent = _build_agent(
        _graph_config(tools=("lookup",), external=("ask",), skill_tool="ask"),
        provider,
        chat,
    )
    _register_local_tool()
    _install_external("ask", runner)

    async def _detect(_query: str) -> str:
        return "external_skill"

    async def _must_not_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("empty catalog must not let a skill bypass the delegation gate")

    agent._detect_skill = _detect  # type: ignore[method-assign]
    agent._skill_executor.run_skill = _must_not_run  # type: ignore[method-assign]

    result = await _run_agent(agent)

    assert provider.calls == 0
    assert _tool_names(chat) == ["lookup"]
    assert runner.calls == []
    assert result["skills_metadata"]["a"] == {"outcome": "skipped", "reason": "external_agent_skill"}


async def test_non_allowlisted_skill_step_is_skipped() -> None:
    """F2: a skill cannot reach a configured external agent outside ``tools``."""
    provider = _ChoiceProvider(LOCAL_TARGET)
    ask_runner = _FakeCLIRunner()
    other_runner = _FakeCLIRunner()
    chat = _ScriptedChatModel([AIMessage(content="local answer")])
    agent = _build_agent(
        _graph_config(tools=("ask", "lookup"), external=("ask", "other"), skill_tool="other"),
        provider,
        chat,
    )
    _register_local_tool()
    _install_external("ask", ask_runner)
    _install_external("other", other_runner)

    async def _detect(_query: str) -> str:
        return "external_skill"

    async def _must_not_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("a non-allowlisted skill step must not invoke an external agent")

    agent._detect_skill = _detect  # type: ignore[method-assign]
    agent._skill_executor.run_skill = _must_not_run  # type: ignore[method-assign]

    result = await _run_agent(agent)

    assert provider.calls == 1
    assert _tool_names(chat) == ["lookup"]
    assert ask_runner.calls == []
    assert other_runner.calls == []
    assert result["skills_metadata"]["a"] == {"outcome": "skipped", "reason": "external_agent_skill"}


async def test_skipped_skill_detection_does_not_report_the_previous_run() -> None:
    """F6: a delegated run must not surface the prior run's detected skill."""
    provider = _ChoiceProvider(LOCAL_TARGET)
    runner = _FakeCLIRunner()
    chat = _ScriptedChatModel(
        [
            AIMessage(content="safe_skill"),
            AIMessage(content="first answer"),
            AIMessage(content="", tool_calls=[{"name": "ask", "args": {"prompt": QUERY}, "id": "call-1"}]),
            AIMessage(content="second answer"),
        ]
    )
    agent = _build_agent(
        _graph_config(tools=("ask", "lookup"), safe_skill=True),
        provider,
        chat,
    )
    _register_local_tool()
    _install_external("ask", runner)

    first = await _run_agent(agent)
    assert first["skills_metadata"]["a"]["skill"] == "safe_skill"

    provider.value = "ask"
    second = await _run_agent(agent)

    assert second["skills_metadata"]["a"] == {"outcome": "skipped", "reason": "delegation"}
    assert [call["prompt"] for call in runner.calls] == [QUERY]


async def test_oversized_external_name_does_not_abort_graph_build() -> None:
    """F7: an oversized operator key is skipped and fails closed, not fatal."""
    oversized = "x" * (delegation_module.MAX_AGENT_NAME_CHARS + 1)
    provider = _ChoiceProvider("ask")
    runner = _FakeCLIRunner()
    chat = _ScriptedChatModel([AIMessage(content="local answer")])
    config = _graph_config(tools=(oversized, "lookup"), external=(oversized,))
    agent = _build_agent(config, provider, chat)
    _register_local_tool()
    _install_external(oversized, runner)

    await _run_agent(agent)

    assert agent._delegation_catalog.names == ()
    assert agent._delegation_external_agents == frozenset({oversized})
    assert provider.calls == 0
    assert _tool_names(chat) == ["lookup"]
    assert runner.calls == []


def test_legacy_custom_agent_constructor_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """F4: a disabled point injects no new kwargs into a strict custom agent."""
    monkeypatch.setitem(agent_registry._REGISTRY, "strict_legacy", _StrictLegacyAgent)
    config = _graph_config(bind=False, agent_class="strict_legacy")
    agents: dict[str, Any] = {}

    build_graph(config=config, runtime=_graph_runtime(), agents_out=agents)

    assert type(agents["a"]) is _StrictLegacyAgent
    assert not hasattr(agents["a"], "_delegation_fail_closed")


def test_fail_closed_delegation_does_not_break_strict_custom_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """R2: enabled/missing-provider point must not inject kwargs into custom agents."""
    monkeypatch.setitem(agent_registry._REGISTRY, "strict_legacy", _StrictLegacyAgent)
    config = _graph_config(agent_class="strict_legacy")
    chat = MagicMock(spec=BaseChatModel)
    chat.bind_tools = MagicMock(return_value=chat)
    runtime = OrchidRuntime(default_model="test/model", chat_model=chat, decision_models={})
    agents: dict[str, Any] = {}

    build_graph(config=config, runtime=runtime, agents_out=agents)

    assert type(agents["a"]) is _StrictLegacyAgent
    assert not hasattr(agents["a"], "_delegation_fail_closed")
