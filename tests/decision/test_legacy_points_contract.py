"""Contract tests for the four legacy decision points.

The registry and the additive ``decisions.points`` map must not change how
``routing``, ``skills``, ``tools`` and ``rag`` parse, resolve, build or fall
back. These tests pin that contract at the seam the rest of the framework uses:
configuration resolution, the narrow binding, the runtime accessor, and the
existing routing strategy.
"""

from __future__ import annotations

import logging

import pytest
from langchain_core.messages import HumanMessage

from orchid_ai.config.schema import OrchidAgentsConfig
from orchid_ai.config.schema_decision import OrchidDecisionPointConfig
from orchid_ai.core.decision import (
    OrchidDecisionPointBinding,
    OrchidDecisionPointSpec,
    OrchidDecisionQuestion,
    OrchidDecisionState,
)
from orchid_ai.decision.factory import build_decision_models
from orchid_ai.decision.observability import DECISION_LOG_FIELDS, log_decision
from orchid_ai.decision.registry import DECISION_POINT_REGISTRY, get_decision_point, register_decision_point
from orchid_ai.decision.stub import StubDecisionModel
from orchid_ai.graph.decision_bindings import (
    DecisionBinding,
    bind_decision_points,
    resolve_decision_binding,
)
from orchid_ai.graph.routing import build_routing_strategy
from orchid_ai.runtime import OrchidRuntime

PROVIDERS = {"stub": {"protocol": "stub"}}


def _agent(decisions: dict | None = None) -> dict:
    agent: dict = {"description": "test agent", "prompt": "you are a test agent"}
    if decisions is not None:
        agent["decisions"] = decisions
    return agent


def _config(
    *,
    decisions: dict | None = None,
    agents: dict | None = None,
    providers: dict | None = None,
    allowed: list[str] | None = None,
) -> OrchidAgentsConfig:
    data: dict = {"agents": agents if agents is not None else {"a": _agent()}}
    if decisions is not None:
        data["decisions"] = decisions
    if providers is not None:
        data["decision_models"] = providers
    if allowed is not None:
        data["allowed_decision_hosts"] = allowed
    return OrchidAgentsConfig.model_validate(data)


def _runtime(provider: StubDecisionModel | None = None) -> OrchidRuntime:
    return OrchidRuntime(default_model="ollama/llama3.2", decision_models={"stub": provider or StubDecisionModel()})


@pytest.fixture
def registry_state():
    saved = dict(DECISION_POINT_REGISTRY)
    yield
    DECISION_POINT_REGISTRY.clear()
    DECISION_POINT_REGISTRY.update(saved)


# ── Configuration contract ───────────────────────────────────


def test_legacy_yaml_parses_and_resolves_the_four_points() -> None:
    config = _config(
        decisions={
            "routing": {"model": "stub", "confidence_threshold": 0.8},
            "skills": {"model": "stub"},
            "tools": {"model": "stub", "max_candidates": 4},
            "rag": {"model": "stub", "mode": "filter"},
        },
        providers=PROVIDERS,
    )
    assert config.decisions.points == {}
    for point in ("routing", "skills", "tools", "rag"):
        resolved = config.decisions.resolve(point)
        assert resolved is not None, point
        assert resolved.model == "stub"
    assert config.decisions.resolve("tools").max_candidates == 4  # type: ignore[union-attr]
    assert config.decisions.resolve("rag").mode == "filter"  # type: ignore[union-attr]


def test_legacy_resolution_chain_is_unchanged() -> None:
    config = _config(
        decisions={
            "default": {"confidence_threshold": 0.9},
            "tools": {"model": "stub", "confidence_threshold": 0.5, "max_candidates": 4},
        },
        agents={
            "a": _agent({"tools": {"confidence_threshold": 0.95}}),
            "b": _agent(),
        },
        providers=PROVIDERS,
    )
    overridden = config.resolve_decision("tools", "a")
    assert overridden is not None
    assert overridden.confidence_threshold == 0.95
    assert overridden.max_candidates == 4

    inherited = config.resolve_decision("tools", "b")
    assert inherited is not None
    assert inherited.confidence_threshold == 0.5


def test_legacy_agent_can_disable_an_inherited_point() -> None:
    config = _config(
        decisions={"tools": {"model": "stub"}},
        agents={"a": _agent({"tools": {"enabled": False}})},
        providers=PROVIDERS,
    )
    assert config.decisions.resolve("tools") is not None
    assert config.resolve_decision("tools", "a") is None


def test_legacy_routing_ignores_agent_context() -> None:
    config = _config(
        decisions={"routing": {"model": "stub"}},
        agents={"a": _agent()},
        providers=PROVIDERS,
    )
    resolved = config.resolve_decision("routing", "a")
    assert resolved is not None
    assert resolved.model == "stub"


def test_unknown_decision_point_lists_registered_names() -> None:
    config = _config()
    with pytest.raises(ValueError) as excinfo:
        config.decisions.resolve("not_a_point")
    message = str(excinfo.value)
    assert "registered points" in message
    assert "routing" in message


# ── Narrow binding contract ─────────────────────────────────


def test_binding_wraps_a_legacy_provider_and_keeps_its_limit() -> None:
    binding = DecisionBinding(
        spec=get_decision_point("tools"),
        config=OrchidDecisionPointConfig(model="stub"),
        model=StubDecisionModel(),
    )
    assert isinstance(binding, OrchidDecisionPointBinding)
    assert binding.point == "tools"
    assert binding.provider == "stub"
    assert binding.confidence_threshold == 0.75
    assert binding.max_candidates == 64
    assert binding.fallback_policy == "all_tools"
    assert dict(binding.metadata()) == {
        "decision_point": "tools",
        "provider": "stub",
        "confidence_threshold": 0.75,
        "max_candidates": 64,
        "on_low_confidence": "llm",
        "fallback": "all_tools",
    }


def test_binding_configuration_narrows_the_point_ceiling() -> None:
    binding = DecisionBinding(
        spec=get_decision_point("tools"),
        config=OrchidDecisionPointConfig(model="stub", max_candidates=4),
        model=StubDecisionModel(),
    )
    assert binding.max_candidates == 4


async def test_binding_delegates_to_the_provider_unchanged() -> None:
    binding = DecisionBinding(
        spec=get_decision_point("rag"),
        config=OrchidDecisionPointConfig(model="stub"),
        model=StubDecisionModel(answers={"relevant": {"probability": 0.9}}),
    )
    result = await binding.decide(
        OrchidDecisionState(text="query"),
        [OrchidDecisionQuestion(name="relevant", kind="boolean")],
    )
    assert result.answers["relevant"].value is True
    assert result.provider == "stub"


def test_resolve_legacy_binding_from_yaml() -> None:
    config = _config(
        decisions={"tools": {"model": "stub", "confidence_threshold": 0.8}},
        providers=PROVIDERS,
    )
    binding = resolve_decision_binding(config, _runtime(), "tools")
    assert binding is not None
    assert binding.provider == "stub"
    assert binding.confidence_threshold == 0.8


def test_disabled_point_has_no_binding() -> None:
    config = _config(
        decisions={"tools": {"enabled": False, "model": "stub"}},
        providers=PROVIDERS,
    )
    assert resolve_decision_binding(config, _runtime(), "tools") is None


def test_missing_provider_produces_no_binding(caplog: pytest.LogCaptureFixture) -> None:
    config = _config(decisions={"tools": {"model": "stub"}}, providers=PROVIDERS)
    with caplog.at_level(logging.WARNING, logger="orchid_ai.graph.decision_bindings"):
        binding = resolve_decision_binding(config, OrchidRuntime(default_model="ollama/llama3.2"), "tools")
    assert binding is None
    assert "missing from the runtime" in caplog.text


def test_fleet_point_rejects_an_agent_binding() -> None:
    config = _config(decisions={"routing": {"model": "stub"}}, providers=PROVIDERS)
    with pytest.raises(ValueError, match="fleet-scoped"):
        resolve_decision_binding(config, _runtime(), "routing", agent_name="a")


def test_bind_decision_points_exposes_only_enabled_points() -> None:
    config = _config(
        decisions={"routing": {"model": "stub"}, "skills": {"model": "stub"}},
        providers=PROVIDERS,
    )
    runtime = _runtime()
    bindings = bind_decision_points(config, runtime)

    assert set(bindings) == {"routing", "skills"}
    assert runtime.get_decision_binding("routing") is bindings["routing"]
    assert runtime.get_decision_binding("skills") is bindings["skills"]
    assert runtime.get_decision_binding("tools") is None
    assert runtime.get_decision_binding("rag") is None


def test_rebinding_removes_a_stale_disabled_point() -> None:
    runtime = _runtime()
    enabled = _config(decisions={"tools": {"model": "stub"}}, providers=PROVIDERS)
    assert bind_decision_points(enabled, runtime)["tools"] is not None
    assert runtime.get_decision_binding("tools") is not None

    disabled = _config(decisions={"tools": {"enabled": False}}, providers=PROVIDERS)
    assert bind_decision_points(disabled, runtime) == {}
    assert runtime.get_decision_binding("tools") is None


# ── Existing routing strategy contract ──────────────────────


def _state(text: str = "I was charged twice") -> dict:
    return {"messages": [HumanMessage(content=text)], "chat_id": "c-1"}


async def test_legacy_routing_strategy_still_accepts_a_decision() -> None:
    provider = StubDecisionModel(
        answers={
            "agent:billing": {"probability": 0.95},
            "agent:technical": {"probability": 0.1},
            "execution": {"choice": "parallel"},
        }
    )
    config = _config(decisions={"routing": {"model": "stub"}}, providers=PROVIDERS)
    strategy = build_routing_strategy(config, _runtime(provider))
    assert strategy is not None

    decision = await strategy.route(
        state=_state(),
        agent_descriptions={"billing": "Handles invoices.", "technical": "Handles bugs."},
    )
    assert decision is not None
    assert decision.agents == ["billing"]
    assert decision.execution == "parallel"
    assert strategy.metadata["outcome"] == "accepted"


async def test_legacy_routing_low_confidence_still_falls_back() -> None:
    provider = StubDecisionModel(mode="low_confidence")
    config = _config(decisions={"routing": {"model": "stub", "confidence_threshold": 0.8}}, providers=PROVIDERS)
    strategy = build_routing_strategy(config, _runtime(provider))
    assert strategy is not None

    decision = await strategy.route(
        state=_state(),
        agent_descriptions={"billing": "Handles invoices.", "technical": "Handles bugs."},
    )
    assert decision is None
    assert strategy.metadata["outcome"] == "fallback"
    assert strategy.metadata["fallback_reason"] == "no_agent_selected"


def test_legacy_routing_without_a_built_provider_returns_no_strategy(caplog: pytest.LogCaptureFixture) -> None:
    config = _config(decisions={"routing": {"model": "stub"}}, providers=PROVIDERS)
    with caplog.at_level(logging.WARNING, logger="orchid_ai.graph.routing"):
        strategy = build_routing_strategy(config, OrchidRuntime(default_model="ollama/llama3.2"))
    assert strategy is None
    assert "missing from the runtime" in caplog.text


# ── Default-off and observability ───────────────────────────


def test_old_yaml_does_not_build_or_bind_an_unconfigured_consumer_point(registry_state) -> None:
    register_decision_point(OrchidDecisionPointSpec(name="tool_choice", scope="agent", fallback="all_tools"))
    config = _config(
        decisions={"routing": {"model": "stub"}},
        providers={"stub": {"protocol": "stub"}, "unused": {"protocol": "stub"}},
    )
    built = build_decision_models(config)
    assert set(built) == {"stub"}

    runtime = _runtime(built["stub"])  # type: ignore[arg-type]
    assert resolve_decision_binding(config, runtime, "tool_choice") is None
    assert "tool_choice" not in bind_decision_points(config, runtime)


def test_legacy_routing_fallback_uses_the_structured_observability_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("INFO", logger="orchid.decision"):
        record = log_decision(
            point="routing",
            path="llm_fallback",
            provider="stub",
            latency_ms=3.25,
            confidence=None,
            outcome="fallback",
            fallback_reason="no_agent_selected",
        )
    assert tuple(record) == DECISION_LOG_FIELDS
    assert record["point"] == "routing"
    assert record["fallback_reason"] == "no_agent_selected"
    assert any("decision point=routing" in entry.getMessage() for entry in caplog.records)


def test_routing_binding_reports_the_builtin_candidate_ceiling() -> None:
    binding = DecisionBinding(
        spec=get_decision_point("routing"),
        config=OrchidDecisionPointConfig(model="stub"),
        model=StubDecisionModel(),
    )
    assert binding.max_candidates == 32
    assert binding.metadata()["max_candidates"] == 32


# ── Flow-level R-011 observability (M1) ─────────────────────


async def test_routing_strategy_emits_one_structured_accepted_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = StubDecisionModel(
        answers={
            "agent:billing": {"probability": 0.95},
            "agent:technical": {"probability": 0.1},
            "execution": {"choice": "parallel"},
        }
    )
    config = _config(decisions={"routing": {"model": "stub"}}, providers=PROVIDERS)
    strategy = build_routing_strategy(config, _runtime(provider))
    assert strategy is not None

    with caplog.at_level("INFO", logger="orchid.decision"):
        decision = await strategy.route(
            state=_state(),
            agent_descriptions={"billing": "Handles invoices.", "technical": "Handles bugs."},
        )

    assert decision is not None
    lines = [record.getMessage() for record in caplog.records if record.name == "orchid.decision"]
    assert len(lines) == 1
    line = lines[0]
    for fragment in (
        "decision point=routing",
        "path=decision",
        "provider=stub",
        "latency_ms=",
        "confidence=",
        "outcome=accepted",
        "fallback_reason=null",
    ):
        assert fragment in line


async def test_routing_strategy_emits_one_structured_fallback_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = StubDecisionModel(mode="low_confidence")
    config = _config(decisions={"routing": {"model": "stub", "confidence_threshold": 0.8}}, providers=PROVIDERS)
    strategy = build_routing_strategy(config, _runtime(provider))
    assert strategy is not None

    with caplog.at_level("INFO", logger="orchid.decision"):
        decision = await strategy.route(
            state=_state(),
            agent_descriptions={"billing": "Handles invoices.", "technical": "Handles bugs."},
        )

    assert decision is None
    lines = [record.getMessage() for record in caplog.records if record.name == "orchid.decision"]
    assert len(lines) == 1
    line = lines[0]
    for fragment in (
        "decision point=routing",
        "path=llm_fallback",
        "provider=stub",
        "outcome=fallback",
        "fallback_reason=no_agent_selected",
    ):
        assert fragment in line


async def test_supervisor_emits_a_not_configured_line_without_a_strategy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from orchid_ai.graph.supervisor import OrchidRoutingDecision, _route

    class FakeStructured:
        async def ainvoke(self, messages, **kwargs):
            return OrchidRoutingDecision(reasoning="only one agent", execution="parallel", agents=["helper"])

    class FakeChat:
        def with_structured_output(self, schema):
            return FakeStructured()

    with caplog.at_level("INFO", logger="orchid.decision"):
        result = await _route(
            {"messages": [HumanMessage(content="hello")]},  # type: ignore[arg-type]
            model="stub",
            agent_descriptions={"helper": "Answers questions."},
            chat_model=FakeChat(),  # type: ignore[arg-type]
        )

    assert result["active_agents"] == ["helper"]
    lines = [record.getMessage() for record in caplog.records if record.name == "orchid.decision"]
    assert len(lines) == 1
    assert "decision point=routing" in lines[0]
    assert "path=llm_fallback" in lines[0]
    assert "outcome=fallback" in lines[0]
    assert "fallback_reason=not_configured" in lines[0]
