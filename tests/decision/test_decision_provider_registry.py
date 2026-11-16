"""Tests for the decision provider factory and protocol registry."""

from __future__ import annotations

from typing import Any

import pytest

from orchid_ai.config.errors import OrchidConfigError
from orchid_ai.config.schema import OrchidAgentsConfig
from orchid_ai.core.decision import OrchidDecisionError, OrchidDecisionQuestion, OrchidDecisionState
from orchid_ai.decision import factory
from orchid_ai.decision.llm import LLMDecisionModel
from orchid_ai.decision.openai_compatible import OpenAICompatibleDecisionModel
from orchid_ai.decision.state_questions import StateQuestionsDecisionModel
from orchid_ai.decision.stub import StubDecisionModel

PROVIDER = {
    "protocol": "state-questions",
    "model": "decision-model-v1",
    "base_url": "https://api.example.com",
    "api_key_env": "DECISION_API_KEY",
}
HOSTS = ["api.example.com"]


class FakeClient:
    host = "api.example.com"

    def __init__(self) -> None:
        self.closed = False

    async def request_json(self, path: str, payload: dict[str, Any], *, timeout: float | None = None) -> dict:
        return {"answers": {}}

    async def aclose(self) -> None:
        self.closed = True


class FakeChat:
    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages, **kwargs):
        from orchid_ai.decision.llm import LLMAnswerSchema, LLMResultSchema

        return LLMResultSchema(answers=[LLMAnswerSchema(name="q", probability=0.9)])


def _config(providers: dict, decisions: dict | None = None) -> OrchidAgentsConfig:
    if decisions is None and providers:
        decisions = {"routing": {"model": next(iter(providers))}}
    return OrchidAgentsConfig.model_validate(
        {
            "agents": {"a": {"description": "test", "prompt": "test"}},
            "decision_models": providers,
            "decisions": decisions or {},
            "allowed_decision_hosts": HOSTS,
        }
    )


def test_empty_config_builds_nothing() -> None:
    config = OrchidAgentsConfig.model_validate({"agents": {"a": {"description": "d", "prompt": "p"}}})
    assert factory.build_decision_models(config) == {}


def test_state_questions_provider_is_built_with_injected_client() -> None:
    client = FakeClient()
    config = _config({"fast": PROVIDER})
    providers = factory.build_decision_models(
        config, client_factory=lambda cfg, key: client, api_key_resolver=lambda name: "key"
    )
    assert isinstance(providers["fast"], StateQuestionsDecisionModel)


def test_openai_compatible_provider_is_built() -> None:
    client = FakeClient()
    config = _config({"fast": {**PROVIDER, "protocol": "openai-compatible"}})
    providers = factory.build_decision_models(
        config, client_factory=lambda cfg, key: client, api_key_resolver=lambda name: "key"
    )
    assert isinstance(providers["fast"], OpenAICompatibleDecisionModel)


def test_llm_provider_requires_a_chat_model_factory() -> None:
    config = _config({"local": {"protocol": "llm", "model": "gemini/gemini-2.5-flash"}})
    with pytest.raises(OrchidConfigError):
        factory.build_decision_models(config)
    providers = factory.build_decision_models(config, chat_model_factory=lambda model: FakeChat())
    assert isinstance(providers["local"], LLMDecisionModel)


def test_stub_provider_is_built() -> None:
    config = _config({"stub": {"protocol": "stub"}})
    providers = factory.build_decision_models(config)
    assert isinstance(providers["stub"], StubDecisionModel)


def test_api_key_is_resolved_from_the_configured_environment_variable() -> None:
    seen: dict[str, str] = {}

    def client_factory(cfg, api_key):
        seen["key"] = api_key
        return FakeClient()

    config = _config({"fast": PROVIDER})
    factory.build_decision_models(config, client_factory=client_factory, api_key_resolver=lambda name: f"value:{name}")
    assert seen["key"] == "value:DECISION_API_KEY"


def test_unknown_protocol_fails_at_build_time() -> None:
    config = _config({"weird": {"protocol": "acme", "model": "m", "base_url": "https://api.example.com"}})
    with pytest.raises(OrchidConfigError, match="unknown protocol"):
        factory.build_decision_models(config)


def test_custom_protocol_can_be_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    def builder(name, cfg, context):
        return StubDecisionModel(provider=name)

    monkeypatch.setitem(factory._PROTOCOL_BUILDERS, "recorder", builder)
    config = _config({"custom": {"protocol": "recorder", "model": "m", "base_url": "https://api.example.com"}})
    providers = factory.build_decision_models(config)
    assert isinstance(providers["custom"], StubDecisionModel)


def _redact(state: OrchidDecisionState) -> OrchidDecisionState:
    return OrchidDecisionState(text="redacted", payload=state.payload)


def _boom(state: OrchidDecisionState) -> OrchidDecisionState:
    raise RuntimeError("boom")


def _wrong(state: OrchidDecisionState) -> str:
    return "not a state"


def test_state_transform_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("orchid_ai.decision.factory.import_class", lambda path: _redact)
    config = _config({"stub": {"protocol": "stub", "state_transform": "acme.transform"}})
    providers = factory.build_decision_models(
        config,
        extras={"stub_answers": {"redacted:q": {"probability": 0.9}}},
    )
    import asyncio

    result = asyncio.run(
        providers["stub"].decide(
            OrchidDecisionState(text="secret"),
            [OrchidDecisionQuestion(name="q", kind="boolean")],
        )
    )
    assert result.answers["q"].value is True
    assert result.answers["q"].confidence == pytest.approx(0.9)


def test_missing_state_transform_fails_at_build_time(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(path):
        raise ImportError("nope")

    monkeypatch.setattr("orchid_ai.decision.factory.import_class", missing)
    config = _config({"stub": {"protocol": "stub", "state_transform": "acme.missing"}})
    with pytest.raises(OrchidConfigError, match="state_transform"):
        factory.build_decision_models(config)


def test_raising_state_transform_aborts_the_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("orchid_ai.decision.factory.import_class", lambda path: _boom)
    config = _config({"stub": {"protocol": "stub", "state_transform": "acme.boom"}})
    providers = factory.build_decision_models(config)
    import asyncio

    with pytest.raises(OrchidDecisionError) as excinfo:
        asyncio.run(
            providers["stub"].decide(
                OrchidDecisionState(text="secret"),
                [OrchidDecisionQuestion(name="q", kind="boolean")],
            )
        )
    assert excinfo.value.code == "invalid_response"


def test_transform_returning_a_non_state_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("orchid_ai.decision.factory.import_class", lambda path: _wrong)
    config = _config({"stub": {"protocol": "stub", "state_transform": "acme.wrong"}})
    providers = factory.build_decision_models(config)
    import asyncio

    with pytest.raises(OrchidDecisionError):
        asyncio.run(
            providers["stub"].decide(
                OrchidDecisionState(text="secret"),
                [OrchidDecisionQuestion(name="q", kind="boolean")],
            )
        )


def test_missing_api_key_fails_at_build_time() -> None:
    config = _config({"fast": PROVIDER})
    with pytest.raises(OrchidConfigError, match="DECISION_API_KEY"):
        factory.build_decision_models(config, api_key_resolver=lambda name: None)


def test_llm_default_timeout_is_ten_seconds() -> None:
    config = _config({"local": {"protocol": "llm", "model": "gemini/gemini-2.5-flash"}})
    providers = factory.build_decision_models(config, chat_model_factory=lambda model: FakeChat())
    assert providers["local"]._timeout == 10.0


def test_llm_explicit_timeout_wins() -> None:
    config = _config({"local": {"protocol": "llm", "model": "gemini/gemini-2.5-flash", "timeout_seconds": 3.0}})
    providers = factory.build_decision_models(config, chat_model_factory=lambda model: FakeChat())
    assert providers["local"]._timeout == 3.0


def test_llm_protocol_logs_an_egress_warning(caplog: pytest.LogCaptureFixture) -> None:
    config = _config({"local": {"protocol": "llm", "model": "gemini/gemini-2.5-flash"}})
    with caplog.at_level("WARNING", logger="orchid_ai.decision.factory"):
        factory.build_decision_models(config, chat_model_factory=lambda model: FakeChat())
    assert "not allowed_decision_hosts" in caplog.text


def test_default_client_construction_uses_config(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class RecordingClient:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

    monkeypatch.setattr(factory, "DecisionHttpClient", RecordingClient)
    config = _config(
        {
            "fast": {
                **PROVIDER,
                "timeout_seconds": 3.0,
                "max_retries": 2,
                "max_response_bytes": 4096,
            }
        },
        decisions={"routing": {"model": "fast"}},
    )
    providers = factory.build_decision_models(config, api_key_resolver=lambda name: "secret")
    assert isinstance(providers["fast"], StateQuestionsDecisionModel)
    assert seen["allowed_hosts"] == HOSTS
    assert seen["timeout"] == 3.0
    assert seen["max_retries"] == 2
    assert seen["max_response_bytes"] == 4096
    assert seen["failure_threshold"] == config.decisions.budget.failure_threshold
    assert seen["cooldown_s"] == config.decisions.budget.breaker_cooldown_s
    assert seen["api_key"] == "secret"


def test_register_protocol_validates_inputs() -> None:
    with pytest.raises(ValueError):
        factory.register_decision_protocol("", lambda name, cfg, ctx: StubDecisionModel())
    with pytest.raises(TypeError):
        factory.register_decision_protocol("bad", 42)  # type: ignore[arg-type]


def _slow(state: OrchidDecisionState) -> OrchidDecisionState:
    import asyncio

    async def inner() -> OrchidDecisionState:
        await asyncio.sleep(0.5)
        return state

    return asyncio.run(inner())


async def test_transform_timeout_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("orchid_ai.decision.factory.import_class", lambda path: _slow)
    config = _config({"stub": {"protocol": "stub", "state_transform": "acme.slow"}})
    providers = factory.build_decision_models(config)
    with pytest.raises(OrchidDecisionError) as excinfo:
        await providers["stub"].decide(
            OrchidDecisionState(text="secret"),
            [OrchidDecisionQuestion(name="q", kind="boolean")],
            timeout=0.05,
        )
    assert excinfo.value.code == "timeout"


def test_sync_transform_runs_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    import time

    def slow_sync(state: OrchidDecisionState) -> OrchidDecisionState:
        time.sleep(0.01)
        return OrchidDecisionState(text="redacted")

    monkeypatch.setattr("orchid_ai.decision.factory.import_class", lambda path: slow_sync)
    config = _config({"stub": {"protocol": "stub", "state_transform": "acme.sync"}})
    providers = factory.build_decision_models(config, extras={"stub_answers": {"redacted:q": {"probability": 0.9}}})
    result = asyncio.run(
        providers["stub"].decide(
            OrchidDecisionState(text="secret"),
            [OrchidDecisionQuestion(name="q", kind="boolean")],
        )
    )
    assert result.answers["q"].value is True


async def test_stub_config_answers_reach_the_provider() -> None:

    from orchid_ai.core.decision import OrchidDecisionQuestion, OrchidDecisionState

    config = _config(
        {
            "stub": {
                "protocol": "stub",
                "stub": {"answers": {"q": {"probability": 0.2}}, "confidence": 0.8},
            }
        }
    )
    providers = factory.build_decision_models(config)
    result = await providers["stub"].decide(
        OrchidDecisionState(text="x"),
        [OrchidDecisionQuestion(name="q", kind="boolean")],
    )
    assert result.answers["q"].value is False
    assert result.answers["q"].confidence == pytest.approx(0.8)


def test_decide_path_is_forwarded_to_the_adapter() -> None:
    client = FakeClient()
    config = _config({"fast": {**PROVIDER, "decide_path": "/v1/systemone"}})
    providers = factory.build_decision_models(
        config, client_factory=lambda cfg, key: client, api_key_resolver=lambda name: "key"
    )
    assert providers["fast"]._decide_path == "/v1/systemone"
