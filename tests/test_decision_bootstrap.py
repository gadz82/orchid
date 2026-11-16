"""Tests for decision-provider injection through the facade and runtime."""

from __future__ import annotations

from orchid_ai import Orchid
from orchid_ai.config.schema import OrchidAgentConfig, OrchidAgentsConfig
from orchid_ai.decision.stub import StubDecisionModel
from orchid_ai.runtime import OrchidRuntime


def _config() -> OrchidAgentsConfig:
    return OrchidAgentsConfig(
        agents={
            "helper": OrchidAgentConfig(
                description="Answers questions.",
                prompt="You are a helper.",
            )
        }
    )


def test_runtime_exposes_configured_providers() -> None:
    providers = {"stub": StubDecisionModel()}
    runtime = OrchidRuntime(default_model="ollama/llama3.2", decision_models=providers)
    assert runtime.get_decision_model("stub") is providers["stub"]
    assert runtime.get_decision_model("missing") is None


def test_runtime_defaults_to_no_providers() -> None:
    runtime = OrchidRuntime(default_model="ollama/llama3.2")
    assert runtime.decision_models == {}
    assert runtime.get_decision_model("anything") is None


async def test_facade_exposes_injected_providers() -> None:
    providers = {"stub": StubDecisionModel()}
    runtime = OrchidRuntime(default_model="ollama/llama3.2", decision_models=providers)
    client = Orchid(config=_config(), runtime=runtime)
    try:
        assert client.runtime.get_decision_model("stub") is providers["stub"]
    finally:
        await client.close()


async def test_hot_reload_rebuilds_and_closes_providers(monkeypatch) -> None:
    import types
    from unittest.mock import AsyncMock, MagicMock, patch

    from orchid_ai.orchid.lifecycle import OrchidLifecycle

    old = MagicMock()
    old.aclose = AsyncMock()
    new = MagicMock()
    new.aclose = AsyncMock()
    runtime = OrchidRuntime(default_model="ollama/llama3.2", decision_models={"old": old})
    changed_config = OrchidAgentsConfig.model_validate(
        {
            "agents": {"helper": {"description": "Answers questions.", "prompt": "You are a helper."}},
            "decision_models": {"stub": {"protocol": "stub"}},
        }
    )
    watcher = MagicMock()
    watcher.reload_if_changed.return_value = types.SimpleNamespace(config=changed_config)

    with (
        patch("orchid_ai.orchid.lifecycle.build_graph", return_value=MagicMock()),
        patch("orchid_ai.decision.factory.build_decision_models", return_value={"new": new}),
    ):
        lifecycle = OrchidLifecycle(config=_config(), runtime=runtime, config_watcher=watcher)
        assert await lifecycle.reload_config() is True

    old.aclose.assert_awaited_once()
    assert runtime.decision_models == {"new": new}


def test_invoker_refreshes_graph_after_reload() -> None:
    from orchid_ai.orchid.invoker import OrchidInvoker

    old = object()
    new = object()
    invoker = OrchidInvoker(graph=old)
    invoker.refresh_graph(new)
    assert invoker._graph is new
