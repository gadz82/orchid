"""Contract tests for the nine extensible decision-point specs.

The specs are registered in production (not only in tests), so a real YAML can
reference them through ``decisions.points``. These tests pin the registry
metadata, scope rules, candidate ceilings, fallback contracts, validators, and
the default-off behaviour that keeps existing configurations unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from orchid_ai.config.loader import load_config
from orchid_ai.config.schema import OrchidAgentsConfig
from orchid_ai.core.decision import OrchidDecisionPointSpec
from orchid_ai.decision.factory import build_decision_models
from orchid_ai.decision.registry import get_decision_point, iter_decision_points, list_decision_points
from orchid_ai.graph.decision_bindings import bind_decision_points
from orchid_ai.runtime import OrchidRuntime

PROVIDERS = {"stub": {"protocol": "stub"}}

# name -> (scope, max_candidates, fallback mode)
NEW_POINTS: dict[str, tuple[str, int | None, str]] = {
    "retrieval_strategy": ("agent", None, "configured_strategy"),
    "model_routing": ("agent", None, "default_model"),
    "tool_choice": ("agent", 32, "all_tools"),
    "guardrails": ("agent", None, "configured_guardrails"),
    "memory": ("fleet", None, "configured_memory"),
    "events": ("fleet", None, "deterministic_processors"),
    "ingestion": ("agent", None, "registry_default"),
    "delegation": ("agent", None, "no_delegation"),
    "agentic_loop": ("agent", None, "max_tool_rounds"),
}

LEGACY_POINTS: dict[str, tuple[str, int | None, str]] = {
    "routing": ("fleet", 32, "llm"),
    "skills": ("agent", None, "llm"),
    "tools": ("agent", 64, "all_tools"),
    "rag": ("agent", None, "keep_original"),
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
) -> OrchidAgentsConfig:
    data: dict = {"agents": agents if agents is not None else {"a": _agent()}}
    if decisions is not None:
        data["decisions"] = decisions
    if providers is not None:
        data["decision_models"] = providers
    return OrchidAgentsConfig.model_validate(data)


def _runtime(provider=None) -> OrchidRuntime:
    return OrchidRuntime(
        default_model="test/model",
        decision_models={"stub": provider} if provider is not None else {},
    )


@pytest.mark.parametrize("name", sorted(NEW_POINTS))
def test_new_points_are_registered_as_builtins(name: str) -> None:
    spec = get_decision_point(name)
    assert isinstance(spec, OrchidDecisionPointSpec)
    assert spec.name == name
    assert spec.builtin is True


@pytest.mark.parametrize(("name", "expected"), NEW_POINTS.items())
def test_new_point_scope_cap_and_fallback(name: str, expected: tuple[str, int | None, str]) -> None:
    scope, max_candidates, fallback = expected
    spec = get_decision_point(name)
    assert spec.scope == scope
    assert spec.max_candidates == max_candidates
    assert spec.fallback.mode == fallback
    assert spec.fallback.description, f"{name} must document its fallback"


@pytest.mark.parametrize("name", sorted(NEW_POINTS))
def test_new_points_have_a_config_validator(name: str) -> None:
    assert callable(get_decision_point(name).validate_config)


@pytest.mark.parametrize("name", sorted(LEGACY_POINTS))
def test_legacy_points_are_unchanged(name: str) -> None:
    scope, max_candidates, fallback = LEGACY_POINTS[name]
    spec = get_decision_point(name)
    assert spec.builtin is True
    assert spec.scope == scope
    assert spec.max_candidates == max_candidates
    assert spec.fallback.mode == fallback


def test_registry_lists_every_builtin_in_sorted_order() -> None:
    assert set(list_decision_points()) >= set(NEW_POINTS) | set(LEGACY_POINTS)
    assert [spec.name for spec in iter_decision_points()] == list(list_decision_points())


@pytest.mark.parametrize("name", sorted(NEW_POINTS))
def test_decisions_points_reference_validates(name: str) -> None:
    config = _config(
        decisions={"points": {name: {"enabled": True, "model": "stub"}}},
        providers=PROVIDERS,
    )
    resolved = config.decisions.resolve(name)
    assert resolved is not None
    assert resolved.model == "stub"


def test_real_yaml_can_reference_a_new_point(tmp_path: Path) -> None:
    config_path = tmp_path / "agents.yaml"
    config_path.write_text(
        """
version: "1"
agents:
  a:
    description: test agent
    prompt: you are a test agent
decision_models:
  stub:
    protocol: stub
decisions:
  points:
    retrieval_strategy:
      model: stub
      confidence_threshold: 0.8
""",
        encoding="utf-8",
    )

    config = load_config(config_path)
    resolved = config.decisions.resolve("retrieval_strategy")
    assert resolved is not None
    assert resolved.model == "stub"
    assert resolved.confidence_threshold == 0.8


def test_every_new_point_can_be_configured_together() -> None:
    config = _config(
        decisions={"points": {name: {"model": "stub"} for name in NEW_POINTS}},
        providers=PROVIDERS,
    )
    for name in NEW_POINTS:
        assert config.decisions.resolve(name) is not None, name


@pytest.mark.parametrize("name", sorted(name for name, meta in NEW_POINTS.items() if meta[0] == "agent"))
def test_agent_points_accept_per_agent_overrides(name: str) -> None:
    config = _config(
        agents={"a": _agent({"points": {name: {"model": "stub"}}})},
        providers=PROVIDERS,
    )
    assert config.decisions.resolve(name) is None
    resolved = config.resolve_decision(name, "a")
    assert resolved is not None
    assert resolved.model == "stub"


@pytest.mark.parametrize("name", sorted(name for name, meta in NEW_POINTS.items() if meta[0] == "fleet"))
def test_fleet_points_reject_per_agent_overrides(name: str) -> None:
    with pytest.raises(ValidationError, match="fleet-level"):
        _config(
            agents={"a": _agent({"points": {name: {"model": "stub"}}})},
            providers=PROVIDERS,
        )


def test_unknown_point_fails_and_lists_the_new_names() -> None:
    with pytest.raises(ValidationError) as excinfo:
        _config(
            decisions={"points": {"does_not_exist": {"model": "stub"}}},
            providers=PROVIDERS,
        )
    message = str(excinfo.value)
    assert "unknown decision point" in message
    assert "registered points" in message
    for name in NEW_POINTS:
        assert name in message


def test_candidate_cap_is_enforced_for_tool_choice() -> None:
    config = _config(
        decisions={"points": {"tool_choice": {"model": "stub", "max_candidates": 32}}},
        providers=PROVIDERS,
    )
    assert config.decisions.resolve("tool_choice").max_candidates == 32  # type: ignore[union-attr]

    with pytest.raises(ValidationError, match="exceeds the point limit"):
        _config(
            decisions={"points": {"tool_choice": {"model": "stub", "max_candidates": 33}}},
            providers=PROVIDERS,
        )


@pytest.mark.parametrize("name", sorted(name for name, meta in NEW_POINTS.items() if meta[1] is None))
def test_uncapped_points_reject_max_candidates(name: str) -> None:
    with pytest.raises(ValidationError, match="candidate limit"):
        _config(
            decisions={"points": {name: {"model": "stub", "max_candidates": 4}}},
            providers=PROVIDERS,
        )


def test_validator_rejects_non_json_safe_options() -> None:
    spec = get_decision_point("retrieval_strategy")
    assert spec.validate_config is not None
    spec.validate_config({"options": {"names": ["simple", 1, True, None], "nested": {"k": 0.5}}})
    with pytest.raises(ValueError, match="non-JSON"):
        spec.validate_config({"options": {"bad": {1, 2}}})
    with pytest.raises(ValueError, match="non-empty strings"):
        spec.validate_config({"options": {1: "value"}})
    with pytest.raises(ValueError, match="must be a mapping"):
        spec.validate_config({"options": ["not", "a", "mapping"]})
    with pytest.raises(ValueError, match="nesting depth"):
        nested: dict = {}
        cursor = nested
        for _ in range(12):
            cursor["next"] = {}
            cursor = cursor["next"]
        spec.validate_config({"options": nested})
    with pytest.raises(ValueError, match="finite numbers"):
        spec.validate_config({"options": {"bad": float("inf")}})


@pytest.mark.parametrize("name", sorted(NEW_POINTS))
@pytest.mark.parametrize("value", [[], "", 0, False])
def test_option_validators_reject_falsy_non_mappings(name: str, value: object) -> None:
    spec = get_decision_point(name)
    assert spec.validate_config is not None
    with pytest.raises(ValueError, match="must be a mapping"):
        spec.validate_config({"options": value})


def test_retrieval_strategy_options_validate_the_allowlist() -> None:
    spec = get_decision_point("retrieval_strategy")
    assert spec.validate_config is not None
    spec.validate_config({"options": {"strategies": ["simple", "hyde"]}})
    with pytest.raises(ValueError, match="non-empty list"):
        spec.validate_config({"options": {"strategies": []}})
    with pytest.raises(ValueError, match="non-empty list"):
        spec.validate_config({"options": {"strategies": "simple"}})
    with pytest.raises(ValueError, match="unknown retrieval strategies"):
        spec.validate_config({"options": {"strategies": ["telepathy"]}})

    config = _config(
        decisions={
            "points": {
                "retrieval_strategy": {
                    "model": "stub",
                    "options": {"strategies": ["simple", "hyde"]},
                }
            }
        },
        providers=PROVIDERS,
    )
    assert config.decisions.resolve("retrieval_strategy").options == {"strategies": ["simple", "hyde"]}  # type: ignore[union-attr]

    with pytest.raises(ValidationError, match="unknown retrieval strategies"):
        _config(
            decisions={
                "points": {
                    "retrieval_strategy": {
                        "model": "stub",
                        "options": {"strategies": ["telepathy"]},
                    }
                }
            },
            providers=PROVIDERS,
        )


def test_new_points_are_off_without_a_binding() -> None:
    config = _config(providers={"stub": {"protocol": "stub"}, "unused": {"protocol": "stub"}})
    runtime = _runtime()

    assert build_decision_models(config) == {}
    for name in NEW_POINTS:
        assert config.resolve_decision(name, "a") is None
        assert runtime.get_decision_binding(name) is None
    assert bind_decision_points(config, runtime) == {}


def test_default_block_never_activates_a_new_point() -> None:
    config = _config(
        decisions={"default": {"confidence_threshold": 0.9}},
        providers=PROVIDERS,
    )
    for name in NEW_POINTS:
        assert config.decisions.resolve(name) is None
