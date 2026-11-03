"""Default-off parity: no decision configuration constructs no provider."""

from __future__ import annotations

import logging

from orchid_ai.config.schema import OrchidAgentsConfig
from orchid_ai.decision.factory import build_decision_models
from orchid_ai.decision.stub import StubDecisionModel

PROVIDER = {
    "protocol": "state-questions",
    "model": "decision-model-v1",
    "base_url": "https://api.example.com",
    "api_key_env": "DECISION_API_KEY",
}


class FakeClient:
    host = "api.example.com"

    async def request_json(self, path, payload, *, timeout=None):
        return {"answers": {}}

    async def aclose(self):
        return None


def _config() -> OrchidAgentsConfig:
    return OrchidAgentsConfig.model_validate({"agents": {"a": {"description": "d", "prompt": "p"}}})


def test_no_decision_blocks_builds_nothing() -> None:
    assert build_decision_models(_config()) == {}


def test_no_decision_blocks_parse_to_empty() -> None:
    config = _config()
    assert config.decision_models == {}
    assert config.decisions.resolve("routing") is None


def test_unused_provider_logs_a_warning(caplog) -> None:
    config = OrchidAgentsConfig.model_validate(
        {
            "agents": {"a": {"description": "d", "prompt": "p"}},
            "decision_models": {"fast": PROVIDER},
            "allowed_decision_hosts": ["api.example.com"],
        }
    )
    with caplog.at_level(logging.WARNING, logger="orchid_ai.decision.factory"):
        built = build_decision_models(
            config,
            client_factory=lambda cfg, key: FakeClient(),
            api_key_resolver=lambda name: "key",
        )
    assert isinstance(built["fast"], StubDecisionModel) is False
    assert "no decision point references it" in caplog.text


def test_stub_protocol_builds_without_any_key() -> None:
    config = OrchidAgentsConfig.model_validate(
        {
            "agents": {"a": {"description": "d", "prompt": "p"}},
            "decision_models": {"stub": {"protocol": "stub"}},
        }
    )
    built = build_decision_models(config)
    assert isinstance(built["stub"], StubDecisionModel)
