"""Tests for the decision-model configuration schema and resolution chain."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orchid_ai.config.schema import OrchidAgentsConfig

PROVIDER = {
    "protocol": "state-questions",
    "model": "decision-model-v1",
    "base_url": "https://api.example.com",
    "api_key_env": "DECISION_API_KEY",
}


def _agent(decisions: dict | None = None) -> dict:
    agent: dict = {"description": "test agent", "prompt": "you are a test agent"}
    if decisions is not None:
        agent["decisions"] = decisions
    return agent


def _config(**overrides) -> OrchidAgentsConfig:
    base: dict = {"agents": {"a": _agent()}}
    base.update(overrides)
    return OrchidAgentsConfig.model_validate(base)


def test_minimal_config_has_no_decisions() -> None:
    config = _config()
    assert config.decision_models == {}
    assert config.allowed_decision_hosts == []
    assert config.decisions.resolve("routing") is None


def test_provider_parses() -> None:
    config = _config(decision_models={"fast": PROVIDER})
    provider = config.decision_models["fast"]
    assert provider.protocol == "state-questions"
    assert provider.model == "decision-model-v1"
    assert provider.base_url == "https://api.example.com"
    assert provider.api_key_env == "DECISION_API_KEY"


def test_systemone_is_an_alias_for_state_questions() -> None:
    config = _config(decision_models={"fast": {**PROVIDER, "protocol": "systemone"}})
    assert config.decision_models["fast"].protocol == "state-questions"


def test_openai_compatible_protocol_parses() -> None:
    config = _config(decision_models={"fast": {**PROVIDER, "protocol": "openai-compatible"}})
    assert config.decision_models["fast"].protocol == "openai-compatible"


def test_llm_protocol_requires_model_and_rejects_base_url() -> None:
    config = _config(decision_models={"local": {"protocol": "llm", "model": "gemini/gemini-2.5-flash"}})
    assert config.decision_models["local"].protocol == "llm"
    with pytest.raises(ValidationError):
        _config(decision_models={"local": {"protocol": "llm", "model": "x", "base_url": "https://api.example.com"}})


def test_stub_protocol_rejects_model_and_base_url() -> None:
    config = _config(decision_models={"stub": {"protocol": "stub"}})
    assert config.decision_models["stub"].protocol == "stub"
    with pytest.raises(ValidationError):
        _config(decision_models={"stub": {"protocol": "stub", "model": "x"}})


@pytest.mark.parametrize(
    "base_url",
    [
        "http://api.example.com",
        "file:///etc/passwd",
        "https://user:pass@api.example.com",
        "https://api.example.com?x=1",
        "https://api.example.com#frag",
        "https://127.0.0.1",
        "https://localhost",
        "https://169.254.169.254",
        "https://[::1]",
    ],
)
def test_provider_rejects_unsafe_base_url(base_url: str) -> None:
    with pytest.raises(ValidationError):
        _config(decision_models={"fast": {**PROVIDER, "base_url": base_url}})


def test_http_protocol_requires_base_url_and_model() -> None:
    with pytest.raises(ValidationError):
        _config(decision_models={"fast": {**PROVIDER, "base_url": None}})
    with pytest.raises(ValidationError):
        _config(decision_models={"fast": {**PROVIDER, "model": ""}})


def test_api_key_env_must_be_non_empty() -> None:
    with pytest.raises(ValidationError):
        _config(decision_models={"fast": {**PROVIDER, "api_key_env": "  "}})


def test_point_rejects_unknown_provider() -> None:
    with pytest.raises(ValidationError, match="unknown decision provider 'missing'"):
        _config(
            decision_models={"fast": PROVIDER},
            decisions={"routing": {"model": "missing"}},
            allowed_decision_hosts=["api.example.com"],
        )


def test_point_enabled_without_model_fails() -> None:
    with pytest.raises(ValidationError, match="no provider is selected"):
        _config(
            decision_models={"fast": PROVIDER},
            decisions={"routing": {"confidence_threshold": 0.9}},
            allowed_decision_hosts=["api.example.com"],
        )


def test_enabled_point_requires_allowlist() -> None:
    with pytest.raises(ValidationError, match="allowed_decision_hosts"):
        _config(decision_models={"fast": PROVIDER}, decisions={"routing": {"model": "fast"}})


def test_disabled_points_do_not_require_allowlist() -> None:
    config = _config(decision_models={"fast": PROVIDER})
    assert config.allowed_decision_hosts == []


def test_allowed_decision_hosts_normalise_and_validate() -> None:
    config = _config(allowed_decision_hosts=["API.Example.com"])
    assert config.allowed_decision_hosts == ["api.example.com"]
    for bad in ["https://api.example.com", "api.example.com/path", "api.example.com:443", "user@host"]:
        with pytest.raises(ValidationError):
            _config(allowed_decision_hosts=[bad])


def test_default_alone_does_not_enable_a_point() -> None:
    config = _config(
        decision_models={"fast": PROVIDER},
        decisions={"default": {"model": "fast", "confidence_threshold": 0.9}},
        allowed_decision_hosts=["api.example.com"],
    )
    assert config.decisions.resolve("routing") is None


def test_default_parameters_apply_to_an_enabled_point() -> None:
    config = _config(
        decision_models={"fast": PROVIDER},
        decisions={"default": {"confidence_threshold": 0.9}, "routing": {"model": "fast"}},
        allowed_decision_hosts=["api.example.com"],
    )
    resolved = config.decisions.resolve("routing")
    assert resolved is not None
    assert resolved.model == "fast"
    assert resolved.confidence_threshold == 0.9


def test_agent_override_overlays_global() -> None:
    config = _config(
        agents={"a": _agent({"tools": {"model": "fast", "confidence_threshold": 0.95}})},
        decision_models={"fast": PROVIDER, "other": {**PROVIDER, "model": "other-model"}},
        decisions={"tools": {"model": "fast", "confidence_threshold": 0.5, "max_candidates": 4}},
        allowed_decision_hosts=["api.example.com"],
    )
    resolved = config.resolve_decision("tools", "a")
    assert resolved is not None
    assert resolved.confidence_threshold == 0.95
    assert resolved.max_candidates == 4

    other = _config(
        agents={"a": _agent({"tools": {"model": "other"}})},
        decision_models={"fast": PROVIDER, "other": {**PROVIDER, "model": "other-model"}},
        decisions={"tools": {"model": "fast"}},
        allowed_decision_hosts=["api.example.com"],
    )
    assert other.resolve_decision("tools", "a").model == "other"  # type: ignore[union-attr]


def test_agent_block_alone_enables_the_point() -> None:
    config = _config(
        agents={"a": _agent({"tools": {"model": "fast"}})},
        decision_models={"fast": PROVIDER},
        allowed_decision_hosts=["api.example.com"],
    )
    assert config.decisions.resolve("tools") is None
    assert config.resolve_decision("tools", "a") is not None


def test_agent_routing_override_is_rejected_with_explanation() -> None:
    with pytest.raises(ValidationError, match="routing is a fleet-level"):
        _config(agents={"a": _agent({"routing": {"model": "fast"}})})


def test_point_specific_options_are_validated() -> None:
    with pytest.raises(ValidationError, match="max_candidates is only valid for tools"):
        _config(
            decision_models={"fast": PROVIDER},
            decisions={"routing": {"model": "fast", "max_candidates": 3}},
            allowed_decision_hosts=["api.example.com"],
        )
    with pytest.raises(ValidationError, match="mode is only valid for rag"):
        _config(
            decision_models={"fast": PROVIDER},
            decisions={"skills": {"model": "fast", "mode": "filter"}},
            allowed_decision_hosts=["api.example.com"],
        )
    with pytest.raises(ValidationError, match="only valid for routing"):
        _config(
            decision_models={"fast": PROVIDER},
            decisions={"tools": {"model": "fast", "on_low_confidence": "escalate"}},
            allowed_decision_hosts=["api.example.com"],
        )


def test_extra_fields_are_forbidden() -> None:
    with pytest.raises(ValidationError):
        _config(decision_models={"fast": {**PROVIDER, "typo": True}})
    with pytest.raises(ValidationError):
        _config(decision_models={"fast": PROVIDER}, decisions={"routing": {"model": "fast", "typo": 1}})


def test_merge_from_db_revalidates_bindings() -> None:
    config = _config(
        decision_models={"fast": PROVIDER},
        decisions={"routing": {"model": "fast"}},
        allowed_decision_hosts=["api.example.com"],
    )
    with pytest.raises(ValueError, match="unknown decision provider"):
        config.merge_from_db(
            [
                {
                    "name": "b",
                    "config": _agent({"tools": {"model": "missing"}}),
                }
            ]
        )
