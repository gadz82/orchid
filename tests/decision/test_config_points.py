"""Tests for the additive ``decisions.points`` schema and registry validation."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from orchid_ai import Orchid
from orchid_ai.config.schema import OrchidAgentsConfig
from orchid_ai.config.schema_decision import OrchidDecisionPointConfig
from orchid_ai.core.decision import (
    OrchidDecisionPointBinding,
    OrchidDecisionPointSpec,
    OrchidDecisionQuestion,
    OrchidDecisionState,
)
from orchid_ai.decision.factory import build_decision_models
from orchid_ai.decision.registry import DECISION_POINT_REGISTRY, get_decision_point, register_decision_point
from orchid_ai.decision.stub import StubDecisionModel
from orchid_ai.graph.decision_bindings import DecisionBinding, bind_decision_points, resolve_decision_binding
from orchid_ai.orchid.lifecycle import OrchidLifecycle
from orchid_ai.runtime import OrchidRuntime

PROVIDERS = {"stub": {"protocol": "stub"}, "other": {"protocol": "stub"}}
HTTP_PROVIDER = {
    "protocol": "state-questions",
    "model": "decision-model-v1",
    "base_url": "https://api.example.com",
}


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
    data: dict[str, Any] = {"agents": agents if agents is not None else {"a": _agent()}}
    if decisions is not None:
        data["decisions"] = decisions
    if providers is not None:
        data["decision_models"] = providers
    if allowed is not None:
        data["allowed_decision_hosts"] = allowed
    return OrchidAgentsConfig.model_validate(data)


@pytest.fixture
def registered():
    """Restore the process-wide registry after a test mutates it."""
    saved = dict(DECISION_POINT_REGISTRY)

    def add(spec: OrchidDecisionPointSpec) -> OrchidDecisionPointSpec:
        return register_decision_point(spec)

    yield add
    DECISION_POINT_REGISTRY.clear()
    DECISION_POINT_REGISTRY.update(saved)


@pytest.fixture
def points(registered):
    registered(OrchidDecisionPointSpec(name="custom_point", scope="agent", max_candidates=8, fallback="all_tools"))
    registered(OrchidDecisionPointSpec(name="custom_fleet", scope="fleet", fallback="default_model"))
    return registered


def test_points_map_defaults_to_empty(points) -> None:
    config = _config(providers=PROVIDERS)
    assert config.decisions.points == {}
    assert config.decisions.resolve("custom_point") is None


def test_registered_point_parses_and_resolves(points) -> None:
    config = _config(
        decisions={"points": {"custom_point": {"enabled": True, "model": "stub"}}},
        providers=PROVIDERS,
    )
    resolved = config.decisions.resolve("custom_point")
    assert resolved is not None
    assert resolved.model == "stub"


def test_one_configured_point_does_not_activate_another(points) -> None:
    config = _config(
        decisions={"points": {"custom_point": {"model": "stub"}}},
        providers=PROVIDERS,
    )
    assert config.decisions.resolve("custom_point") is not None
    assert config.decisions.resolve("custom_fleet") is None


def test_unknown_point_fails_and_lists_registered_names(points) -> None:
    with pytest.raises(ValidationError) as excinfo:
        _config(decisions={"points": {"does_not_exist": {"model": "stub"}}}, providers=PROVIDERS)
    message = str(excinfo.value)
    assert "unknown decision point" in message
    assert "registered points" in message
    assert "routing" in message
    assert "custom_point" in message


def test_legacy_point_can_be_configured_through_the_points_map() -> None:
    config = _config(
        decisions={"points": {"routing": {"model": "stub"}}},
        providers=PROVIDERS,
    )
    resolved = config.decisions.resolve("routing")
    assert resolved is not None
    assert resolved.model == "stub"


def test_legacy_point_cannot_be_configured_twice() -> None:
    with pytest.raises(ValidationError, match="configured twice"):
        _config(
            decisions={"routing": {"model": "stub"}, "points": {"routing": {"model": "other"}}},
            providers=PROVIDERS,
        )


def test_fleet_point_rejects_a_per_agent_override(points) -> None:
    with pytest.raises(ValidationError, match="fleet-level"):
        _config(
            agents={"a": _agent({"points": {"custom_fleet": {"model": "stub"}}})},
            providers=PROVIDERS,
        )


def test_fleet_point_ignores_agent_context_when_configured_globally(points) -> None:
    config = _config(
        decisions={"points": {"custom_fleet": {"model": "stub"}}},
        providers=PROVIDERS,
    )
    assert config.resolve_decision("custom_fleet", "a") is not None


def test_agent_point_resolves_agent_then_global_then_default(points) -> None:
    config = _config(
        decisions={
            "default": {"confidence_threshold": 0.9},
            "points": {
                "custom_point": {
                    "model": "stub",
                    "confidence_threshold": 0.6,
                    "max_candidates": 4,
                    "options": {"mode": "global"},
                }
            },
        },
        agents={
            "a": _agent({"points": {"custom_point": {"confidence_threshold": 0.8, "options": {"mode": "agent"}}}}),
            "b": _agent(),
        },
        providers=PROVIDERS,
    )

    overridden = config.resolve_decision("custom_point", "a")
    assert overridden is not None
    assert overridden.model == "stub"
    assert overridden.confidence_threshold == 0.8
    assert overridden.max_candidates == 4
    assert overridden.options == {"mode": "agent"}

    inherited = config.resolve_decision("custom_point", "b")
    assert inherited is not None
    assert inherited.confidence_threshold == 0.6
    assert inherited.options == {"mode": "global"}

    global_binding = config.decisions.resolve("custom_point")
    assert global_binding is not None
    assert global_binding.confidence_threshold == 0.6


def test_agent_can_disable_an_inherited_point(points) -> None:
    config = _config(
        decisions={"points": {"custom_point": {"model": "stub"}}},
        agents={"a": _agent({"points": {"custom_point": {"enabled": False}}})},
        providers=PROVIDERS,
    )
    assert config.decisions.resolve("custom_point") is not None
    assert config.resolve_decision("custom_point", "a") is None


def test_agent_block_alone_enables_a_point(points) -> None:
    config = _config(
        agents={"a": _agent({"points": {"custom_point": {"model": "stub"}}})},
        providers=PROVIDERS,
    )
    assert config.decisions.resolve("custom_point") is None
    assert config.resolve_decision("custom_point", "a") is not None


def test_agent_point_cannot_be_configured_twice(points) -> None:
    with pytest.raises(ValidationError, match="configured twice"):
        _config(
            agents={"a": _agent({"tools": {"model": "stub"}, "points": {"tools": {"model": "other"}}})},
            providers=PROVIDERS,
        )


def test_point_validator_receives_explicit_global_config(registered) -> None:
    seen: list[dict[str, Any]] = []

    def validate(payload: dict[str, Any]) -> None:
        seen.append(dict(payload))

    registered(
        OrchidDecisionPointSpec(
            name="custom_point",
            scope="agent",
            validate_config=validate,
        )
    )
    _config(
        decisions={"points": {"custom_point": {"enabled": True, "model": "stub", "options": {"mode": "fast"}}}},
        providers=PROVIDERS,
    )
    assert seen
    assert seen[0]["model"] == "stub"
    assert seen[0]["options"] == {"mode": "fast"}
    assert seen[0]["enabled"] is True


def test_point_validator_failure_is_reported_with_the_point_label(registered) -> None:
    def validate(payload: dict[str, Any]) -> None:
        if payload.get("options", {}).get("mode") == "bad":
            raise ValueError("unsupported mode")

    registered(OrchidDecisionPointSpec(name="custom_point", scope="agent", validate_config=validate))
    with pytest.raises(ValidationError, match="decisions.points.custom_point: unsupported mode"):
        _config(
            decisions={"points": {"custom_point": {"model": "stub", "options": {"mode": "bad"}}}},
            providers=PROVIDERS,
        )


def test_agent_point_validator_failure_is_reported_with_the_agent_label(registered) -> None:
    def validate(payload: dict[str, Any]) -> None:
        if payload.get("options", {}).get("mode") == "bad":
            raise ValueError("unsupported mode")

    registered(OrchidDecisionPointSpec(name="custom_point", scope="agent", validate_config=validate))
    with pytest.raises(ValidationError, match=r"agents.<name>.decisions.points.custom_point: unsupported mode"):
        _config(
            agents={"a": _agent({"points": {"custom_point": {"model": "stub", "options": {"mode": "bad"}}}})},
            providers=PROVIDERS,
        )


def test_registered_point_rejects_an_unknown_provider(points) -> None:
    with pytest.raises(ValidationError, match="unknown decision provider 'missing'"):
        _config(decisions={"points": {"custom_point": {"model": "missing"}}}, providers=PROVIDERS)


def test_registered_http_point_requires_the_egress_allowlist(points) -> None:
    with pytest.raises(ValidationError, match="allowed_decision_hosts"):
        _config(
            decisions={"points": {"custom_point": {"model": "fast"}}},
            providers={"fast": HTTP_PROVIDER},
        )

    with pytest.raises(ValidationError, match="not listed in allowed_decision_hosts"):
        _config(
            decisions={"points": {"custom_point": {"model": "fast"}}},
            providers={"fast": HTTP_PROVIDER},
            allowed=["other.example.com"],
        )

    config = _config(
        decisions={"points": {"custom_point": {"model": "fast"}}},
        providers={"fast": HTTP_PROVIDER},
        allowed=["api.example.com"],
    )
    assert config.decisions.resolve("custom_point") is not None


def test_factory_builds_only_the_referenced_registered_point(points, caplog) -> None:
    config = _config(
        decisions={"points": {"custom_point": {"model": "stub"}}},
        providers=PROVIDERS,
    )
    with caplog.at_level(logging.WARNING, logger="orchid_ai.decision.factory"):
        built = build_decision_models(config)
    assert set(built) == {"stub"}
    assert "no decision point references it" in caplog.text


def test_factory_builds_nothing_when_a_registered_point_is_not_configured(points) -> None:
    config = _config(providers=PROVIDERS)
    assert build_decision_models(config) == {}


def test_registered_point_with_options_does_not_require_legacy_point_fields(points) -> None:
    config = _config(
        decisions={
            "points": {
                "custom_point": {
                    "model": "stub",
                    "max_candidates": 3,
                    "on_low_confidence": "default",
                    "options": {"allow_parallel": True},
                }
            }
        },
        providers=PROVIDERS,
    )
    resolved = config.decisions.resolve("custom_point")
    assert resolved is not None
    assert resolved.max_candidates == 3
    assert resolved.on_low_confidence == "default"
    assert resolved.options == {"allow_parallel": True}


# ── Candidate limits (H1) ───────────────────────────────────


def test_registered_point_narrows_the_declared_candidate_limit(points) -> None:
    config = _config(
        decisions={"points": {"custom_point": {"model": "stub", "max_candidates": 4}}},
        providers=PROVIDERS,
    )
    binding = resolve_decision_binding(config, _runtime(), "custom_point")
    assert binding is not None
    assert binding.max_candidates == 4


def test_registered_point_cannot_raise_the_declared_candidate_limit(points) -> None:
    with pytest.raises(ValidationError, match="exceeds the point limit 8"):
        _config(
            decisions={"points": {"custom_point": {"model": "stub", "max_candidates": 16}}},
            providers=PROVIDERS,
        )


def test_binding_defensively_clamps_an_oversized_direct_config(points) -> None:
    binding = DecisionBinding(
        spec=get_decision_point("custom_point"),
        config=OrchidDecisionPointConfig(model="stub", max_candidates=16),
        model=StubDecisionModel(),
    )
    assert binding.max_candidates == 8


def test_registered_point_without_a_declared_limit_rejects_max_candidates(points) -> None:
    with pytest.raises(ValidationError, match="requires a point that declares a candidate limit"):
        _config(
            decisions={"points": {"custom_fleet": {"model": "stub", "max_candidates": 4}}},
            providers=PROVIDERS,
        )


# ── Point-specific field rules (M2) ─────────────────────────


@pytest.mark.parametrize("point", ["routing", "skills", "tools", "rag"])
def test_legacy_points_reject_options(point: str) -> None:
    with pytest.raises(ValidationError, match="options is not supported"):
        _config(
            decisions={point: {"model": "stub", "options": {"mode": "fast"}}},
            providers=PROVIDERS,
        )


def test_legacy_point_via_points_map_rejects_options() -> None:
    with pytest.raises(ValidationError, match="options is not supported"):
        _config(
            decisions={"points": {"tools": {"model": "stub", "options": {"mode": "fast"}}}},
            providers=PROVIDERS,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mode", "filter", "mode is only valid for rag"),
        ("strategies", ["simple"], "strategies is only valid for rag"),
        ("history_max_turns", 5, "history_max_turns is only valid for routing"),
        ("history_max_chars", 500, "history_max_chars is only valid for routing"),
        ("on_low_confidence", "escalate", "'escalate' is only valid for routing"),
    ],
)
def test_non_legacy_points_reject_legacy_only_fields(points, field: str, value: Any, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _config(
            decisions={"points": {"custom_point": {"model": "stub", field: value}}},
            providers=PROVIDERS,
        )


def test_agent_non_legacy_point_rejects_legacy_only_fields(points) -> None:
    with pytest.raises(ValidationError, match="mode is only valid for rag"):
        _config(
            agents={"a": _agent({"points": {"custom_point": {"model": "stub", "mode": "filter"}}})},
            providers=PROVIDERS,
        )


# ── Provider-key safety (M4) ────────────────────────────────


@pytest.mark.parametrize("key", ["evil\nkey", "a=b", "bad key", "bad\tkey"])
def test_decision_provider_keys_must_be_safe_tokens(key: str) -> None:
    with pytest.raises(ValidationError, match="decision provider key"):
        _config(providers={key: {"protocol": "stub"}})


# ── Runtime binding production (H2) ─────────────────────────


def _runtime() -> OrchidRuntime:
    return OrchidRuntime(default_model="ollama/llama3.2", decision_models={"stub": StubDecisionModel()})


async def test_bootstrap_exposes_a_configured_consumer_point(tmp_path, registered) -> None:
    registered(OrchidDecisionPointSpec(name="custom_fleet", scope="fleet", fallback="default_model"))
    agents = tmp_path / "agents.yaml"
    agents.write_text(
        "agents:\n"
        "  helper:\n"
        "    description: Test agent\n"
        "    prompt: Be helpful.\n"
        "decision_models:\n"
        "  stub:\n"
        "    protocol: stub\n"
        "decisions:\n"
        "  points:\n"
        "    custom_fleet:\n"
        "      model: stub\n",
        encoding="utf-8",
    )
    orchid_config = tmp_path / "orchid.yml"
    orchid_config.write_text(f"agents:\n  config_path: {agents}\n", encoding="utf-8")

    orchid = await Orchid.from_config_path(
        str(orchid_config),
        vector_backend="null",
        skip_yaml_sections={"rag", "storage"},
    )
    try:
        binding = orchid.runtime.get_decision_binding("custom_fleet")
        assert isinstance(binding, OrchidDecisionPointBinding)

        resolved = resolve_decision_binding(orchid.config, orchid.runtime, "custom_fleet")
        assert resolved is not None
        result = await resolved.decide(
            OrchidDecisionState(text="pick a model"),
            [OrchidDecisionQuestion(name="use_default", kind="boolean")],
        )
        assert result.answers["use_default"].value is True
    finally:
        await orchid.close()


async def test_hot_reload_refreshes_decision_bindings(registered) -> None:
    registered(OrchidDecisionPointSpec(name="custom_fleet", scope="fleet", fallback="default_model"))
    initial = _config(
        decisions={"points": {"custom_fleet": {"enabled": False}}},
        providers=PROVIDERS,
    )
    enabled = _config(
        decisions={"points": {"custom_fleet": {"model": "stub"}}},
        providers=PROVIDERS,
    )
    disabled = _config(
        decisions={"points": {"custom_fleet": {"enabled": False}}},
        providers=PROVIDERS,
    )
    runtime = _runtime()
    watcher = MagicMock()
    watcher.reload_if_changed.side_effect = [
        SimpleNamespace(config=enabled),
        SimpleNamespace(config=disabled),
    ]

    with (
        patch("orchid_ai.orchid.lifecycle.build_graph", return_value=MagicMock()),
        patch(
            "orchid_ai.decision.factory.build_decision_models",
            return_value={"stub": StubDecisionModel()},
        ),
    ):
        lifecycle = OrchidLifecycle(config=initial, runtime=runtime, config_watcher=watcher)
        assert await lifecycle.reload_config() is True
        assert isinstance(runtime.get_decision_binding("custom_fleet"), OrchidDecisionPointBinding)

        assert await lifecycle.reload_config() is True
        assert runtime.get_decision_binding("custom_fleet") is None


def test_bind_decision_points_is_publicly_usable(points) -> None:
    config = _config(
        decisions={"points": {"custom_point": {"model": "stub", "max_candidates": 4}}},
        providers=PROVIDERS,
    )
    runtime = _runtime()
    bindings = bind_decision_points(config, runtime)
    assert isinstance(bindings["custom_point"], OrchidDecisionPointBinding)
    assert runtime.get_decision_binding("custom_point") is bindings["custom_point"]
