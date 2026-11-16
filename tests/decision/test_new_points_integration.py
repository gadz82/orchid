"""End-to-end proof that a new decision point is enabled by YAML alone.

The fixture is an ordinary configuration with one additive
``decisions.points.retrieval_strategy`` block. These tests load it through the
real loader, composition factory, binding resolver, and graph builder; exercise
both the accepted and fallback branches against the deterministic stub; and pin
that the legacy points and the default-off baseline are unchanged.

``tests/decision/conftest.py`` guards live ``httpx.AsyncClient`` construction, so
the whole module is network-free by construction. The graph is built but not
executed end-to-end: the pipeline and binding are reached through private
white-box seams, while the focused retrieval-strategy tests cover adapter
behaviour.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import yaml
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from orchid_ai.config.loader import load_config
from orchid_ai.config.schema import OrchidAgentsConfig, OrchidDecisionsConfig
from orchid_ai.core.decision import OrchidDecisionQuestion, OrchidDecisionState
from orchid_ai.decision.factory import build_decision_models
from orchid_ai.decision.registry import get_decision_point
from orchid_ai.decision.stub import StubDecisionModel
from orchid_ai.graph.decision_bindings import bind_decision_points, resolve_decision_binding
from orchid_ai.graph.graph import build_graph
from orchid_ai.rag.scopes import OrchidRAGScope
from orchid_ai.runtime import OrchidRuntime

FIXTURE = Path(__file__).parent / "fixtures" / "new_points.yaml"

NEW_POINTS = (
    "retrieval_strategy",
    "model_routing",
    "tool_choice",
    "guardrails",
    "memory",
    "events",
    "ingestion",
    "delegation",
    "agentic_loop",
)

# name -> (scope, candidate ceiling, fallback mode); the pre-existing contract.
LEGACY_POINTS: dict[str, tuple[str, int | None, str]] = {
    "routing": ("fleet", 32, "llm"),
    "skills": ("agent", None, "llm"),
    "tools": ("agent", 64, "all_tools"),
    "rag": ("agent", None, "keep_original"),
}


def _config() -> OrchidAgentsConfig:
    return load_config(FIXTURE)


def _runtime(config: OrchidAgentsConfig) -> OrchidRuntime:
    # A deterministic in-process chat model: only the configured hyde fallback
    # uses it, and no provider SDK or network client is constructed.
    chat = FakeListChatModel(responses=["A plausible hypothetical answer."] * 16)
    return OrchidRuntime(
        default_model="test/model",
        chat_model=chat,
        decision_models=build_decision_models(config),
    )


def _scope(agent_id: str) -> OrchidRAGScope:
    return OrchidRAGScope(tenant_id="tenant", user_id="user", chat_id="chat", agent_id=agent_id)


def _decision_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == "orchid.decision"]


def test_fixture_enables_the_new_point_through_yaml_only() -> None:
    config = _config()

    assert set(config.decisions.points) == {"retrieval_strategy"}
    resolved = config.decisions.resolve("retrieval_strategy")
    assert resolved is not None
    assert resolved.model == "router"
    assert resolved.confidence_threshold == 0.8
    assert resolved.options == {"strategies": ["simple", "hyde"]}
    # The configured strategy that the fallback preserves.
    assert config.agents["support"].rag.retrieval.strategy == "hyde"


def test_legacy_points_and_framework_defaults_are_unchanged() -> None:
    config = _config()

    for name, (scope, max_candidates, fallback) in LEGACY_POINTS.items():
        assert config.decisions.resolve(name) is None
        spec = get_decision_point(name)
        assert spec.builtin is True
        assert spec.scope == scope
        assert spec.max_candidates == max_candidates
        assert spec.fallback.mode == fallback

    for name in NEW_POINTS:
        if name != "retrieval_strategy":
            assert config.decisions.resolve(name) is None

    # A decisions block that only carries defaults never activates a point.
    default = OrchidDecisionsConfig()
    for name in (*NEW_POINTS, *LEGACY_POINTS):
        assert default.resolve(name) is None


def test_fixture_without_any_points_block_stays_default_off(tmp_path: Path) -> None:
    data = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    data.pop("decisions")
    data["agents"]["fallback"].pop("decisions")
    config_path = tmp_path / "agents.yaml"
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    config = load_config(config_path)

    # Every provider is unreferenced, so the composition root builds nothing.
    assert build_decision_models(config) == {}
    for name in (*NEW_POINTS, *LEGACY_POINTS):
        assert config.decisions.resolve(name) is None


def test_runtime_builds_only_the_referenced_providers(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="orchid_ai.decision.factory"):
        providers = build_decision_models(_config())

    assert set(providers) == {"router", "cautious"}
    assert isinstance(providers["router"], StubDecisionModel)
    assert isinstance(providers["cautious"], StubDecisionModel)
    assert "unused" in caplog.text


async def test_binding_resolves_from_yaml_and_answers_a_real_question() -> None:
    config = _config()
    runtime = _runtime(config)

    bindings = bind_decision_points(config, runtime)
    assert set(bindings) == {"retrieval_strategy"}

    binding = runtime.get_decision_binding("retrieval_strategy")
    assert binding is not None
    assert binding.point == "retrieval_strategy"
    assert binding.provider == "router"
    assert binding.confidence_threshold == 0.8
    assert binding.options == {"strategies": ["simple", "hyde"]}
    assert binding.fallback_policy == "configured_strategy"
    assert dict(binding.metadata()) == {
        "decision_point": "retrieval_strategy",
        "provider": "router",
        "confidence_threshold": 0.8,
        "max_candidates": None,
        "on_low_confidence": "llm",
        "fallback": "configured_strategy",
    }
    # Enabling one point does not expose any other new point.
    assert runtime.get_decision_binding("tool_choice") is None

    result = await binding.decide(
        OrchidDecisionState(text="How do I reset my password?"),
        [OrchidDecisionQuestion(name="strategy", kind="choice", options=("hyde", "simple"))],
    )
    answer = result.answers["strategy"]
    assert answer.value == "simple"
    assert answer.confidence == pytest.approx(0.95)
    assert result.provider == "router"


async def test_graph_wires_the_point_and_logs_an_accepted_decision(caplog: pytest.LogCaptureFixture) -> None:
    config = _config()
    agents: dict[str, Any] = {}
    build_graph(config=config, runtime=_runtime(config), agents_out=agents)

    pipeline = agents["support"]._rag_pipeline
    binding = pipeline._retrieval_strategy_binding
    assert binding is not None
    assert binding.provider == "router"

    # Only the configured point is wired; the other new points stay disabled.
    assert agents["support"]._tool_choice_binding is None
    assert agents["support"]._ingestion_binding is None
    assert agents["support"]._agentic_loop_binding is None

    with caplog.at_level(logging.INFO):
        passages = await pipeline.retrieve(
            "How do I reset my password?",
            _scope("support"),
            rag_namespace="support",
            k=3,
            enabled=True,
            retrieval_strategy="hyde",
        )

    assert passages == []
    lines = _decision_lines(caplog)
    assert len(lines) == 1
    for fragment in (
        "decision point=retrieval_strategy",
        "path=decision",
        "provider=router",
        "outcome=accepted",
        "fallback_reason=null",
    ):
        assert fragment in lines[0]
    # The YAML answer selected the simple strategy instead of the configured hyde.
    assert any("retrieval strategy selected 'simple'" in record.getMessage() for record in caplog.records)


async def test_agent_override_falls_back_to_the_configured_strategy(caplog: pytest.LogCaptureFixture) -> None:
    config = _config()
    runtime = _runtime(config)
    bind_decision_points(config, runtime)
    agents: dict[str, Any] = {}
    build_graph(config=config, runtime=runtime, agents_out=agents)

    # The global binding stays on the confident router; the per-agent override
    # is resolved only for ``fallback``.
    global_binding = runtime.get_decision_binding("retrieval_strategy")
    assert global_binding is not None
    assert global_binding.provider == "router"
    overridden = resolve_decision_binding(config, runtime, "retrieval_strategy", agent_name="fallback")
    assert overridden is not None
    assert overridden.provider == "cautious"

    pipeline = agents["fallback"]._rag_pipeline
    binding = pipeline._retrieval_strategy_binding
    assert binding is not None
    assert binding.provider == "cautious"

    with caplog.at_level(logging.INFO):
        passages = await pipeline.retrieve(
            "How do I reset my password?",
            _scope("fallback"),
            rag_namespace="support",
            k=3,
            enabled=True,
            retrieval_strategy="hyde",
        )

    assert passages == []
    lines = _decision_lines(caplog)
    assert len(lines) == 1
    for fragment in (
        "decision point=retrieval_strategy",
        "path=llm_fallback",
        "provider=cautious",
        "outcome=fallback",
        "fallback_reason=low_confidence",
    ):
        assert fragment in lines[0]
    # The configured strategy is kept, so the default behaviour is preserved.
    assert any("retrieval strategy decision fell back to 'hyde'" in record.getMessage() for record in caplog.records)
