"""Facade-level tests for pipelines: run, stream, registry and reload."""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from orchid_ai import Orchid
from orchid_ai.config.schema import OrchidAgentsConfig
from orchid_ai.core.pipeline import OrchidPipelineNotFoundError, OrchidPipelineResult
from orchid_ai.runtime import OrchidRuntime


@pytest.fixture
def facade_steps(monkeypatch: pytest.MonkeyPatch) -> str:
    module = types.ModuleType("facade_steps")

    def load(ctx: Any) -> dict[str, Any]:
        return {"id": ctx.input["account_id"], "name": "Acme"}

    def assess(ctx: Any) -> dict[str, Any]:
        return {"label": "retain"}

    module.load = load  # type: ignore[attr-defined]
    module.assess = assess  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "facade_steps", module)
    return "facade_steps"


def _pipelines(names: tuple[str, ...] = ("demo",)) -> dict[str, Any]:
    shapes = {
        "Input": {"fields": {"account_id": {"type": "string", "required": True}}},
        "Account": {"fields": {"id": {"type": "string", "required": True}, "name": {"type": "string"}}},
        "Assessment": {"fields": {"label": {"type": "string"}}},
    }
    definitions: dict[str, Any] = {}
    for name in names:
        definitions[name] = {
            "description": f"{name} pipeline",
            "input": {"shape": "Input"},
            "jobs": {
                "load": {
                    "kind": "deterministic",
                    "uses": "facade_steps.load",
                    "writes": [
                        {
                            "id": "account:{{ input.account_id }}",
                            "shape": "Account",
                            "from": "result",
                            "as": "account",
                        }
                    ],
                },
                "assess": {
                    "kind": "deterministic",
                    "uses": "facade_steps.assess",
                    "depends_on": ["load"],
                    "writes": [{"id": "assessment:{{ input.account_id }}", "shape": "Assessment"}],
                },
            },
        }
    return {"shapes": shapes, "definitions": definitions}


def _config(*, pipelines: dict[str, Any] | None = None) -> OrchidAgentsConfig:
    payload: dict[str, Any] = {
        "agents": {
            "helper": {
                "description": "Answers questions.",
                "prompt": "You are a helper.",
            }
        }
    }
    if pipelines is not None:
        payload["pipelines"] = pipelines
    return OrchidAgentsConfig.model_validate(payload)


def _client(config: OrchidAgentsConfig, *, chat_repo: Any = None) -> Orchid:
    runtime = OrchidRuntime(default_model="ollama/llama3.2")
    with patch("orchid_ai.orchid.lifecycle.build_graph", return_value=MagicMock()):
        return Orchid(config=config, runtime=runtime, chat_repo=chat_repo)


# ── Registry surface ───────────────────────────────────────


def test_pipelines_is_none_without_a_block() -> None:
    client = _client(_config())

    assert client.pipelines is None


def test_pipelines_lists_and_resolves_configured_pipelines(facade_steps: str) -> None:
    client = _client(_config(pipelines=_pipelines(("alpha", "beta"))))

    registry = client.pipelines
    assert registry is not None
    infos = registry.list()
    assert {info.name for info in infos} == {"alpha", "beta"}
    assert registry.get("alpha") is not None
    assert registry.get("missing") is None


def test_unknown_pipeline_is_not_found(facade_steps: str) -> None:
    client = _client(_config(pipelines=_pipelines()))

    with pytest.raises(OrchidPipelineNotFoundError, match="missing"):
        __import__("asyncio").run(client.run_pipeline("missing", {"account_id": "a1"}, user_id="u1"))


def test_unknown_pipeline_is_not_found_without_a_registry() -> None:
    client = _client(_config())

    with pytest.raises(OrchidPipelineNotFoundError, match="missing"):
        __import__("asyncio").run(client.run_pipeline("missing", {}, user_id="u1"))


# ── Run and stream ─────────────────────────────────────────


def test_run_pipeline_returns_a_result(facade_steps: str) -> None:
    client = _client(_config(pipelines=_pipelines()))

    result = __import__("asyncio").run(client.run_pipeline("demo", {"account_id": "a1"}, user_id="u1", tenant_id="t1"))

    assert isinstance(result, OrchidPipelineResult)
    assert result.status == "succeeded"
    assert result.jobs["load"]["status"] == "succeeded"
    assert result.graph["objects"]
    assert result.graph_stats["objects_written"] == 2


def test_run_pipeline_can_omit_the_graph(facade_steps: str) -> None:
    client = _client(_config(pipelines=_pipelines()))

    result = __import__("asyncio").run(
        client.run_pipeline("demo", {"account_id": "a1"}, user_id="u1", include_graph=False)
    )

    assert result.graph == {}
    assert result.graph_stats["objects"] == 3


def test_run_pipeline_uses_an_explicit_auth(facade_steps: str) -> None:
    from orchid_ai.core.state import OrchidAuthContext

    auth = OrchidAuthContext(access_token="token", tenant_key="explicit", user_id="owner")
    client = _client(_config(pipelines=_pipelines()))

    result = __import__("asyncio").run(client.run_pipeline("demo", {"account_id": "a1"}, auth=auth))

    assert result.status == "succeeded"


def test_chat_storage_is_never_called(facade_steps: str) -> None:
    spy = MagicMock()
    client = _client(_config(pipelines=_pipelines()), chat_repo=spy)

    __import__("asyncio").run(client.run_pipeline("demo", {"account_id": "a1"}, user_id="u1", chat_id="c-1"))

    spy.create_chat.assert_not_called()
    spy.add_message.assert_not_called()
    spy.get_messages.assert_not_called()


def test_stream_pipeline_yields_events(facade_steps: str) -> None:
    client = _client(_config(pipelines=_pipelines()))

    async def collect() -> list[Any]:
        return [event async for event in client.stream_pipeline("demo", {"account_id": "a1"}, user_id="u1")]

    events = __import__("asyncio").run(collect())

    assert events[0].type == "pipeline.started"
    assert events[-1].type == "pipeline.finished"
    assert events[-1].data["result"]["status"] == "succeeded"


def test_closed_client_rejects_pipeline_calls(facade_steps: str) -> None:
    import asyncio

    client = _client(_config(pipelines=_pipelines()))
    asyncio.run(client.close())

    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(client.run_pipeline("demo", {"account_id": "a1"}, user_id="u1"))


# ── Reload ─────────────────────────────────────────────────


def _watcher(config: OrchidAgentsConfig) -> MagicMock:
    watcher = MagicMock()
    watcher.reload_if_changed.return_value = types.SimpleNamespace(config=config)
    return watcher


def test_reload_swaps_the_pipeline_registry(facade_steps: str) -> None:
    import asyncio

    runtime = OrchidRuntime(default_model="ollama/llama3.2")
    with patch("orchid_ai.orchid.lifecycle.build_graph", return_value=MagicMock()):
        from orchid_ai.orchid.lifecycle import OrchidLifecycle

        lifecycle = OrchidLifecycle(
            config=_config(pipelines=_pipelines(("alpha",))),
            runtime=runtime,
            config_watcher=_watcher(_config(pipelines=_pipelines(("alpha", "beta")))),
        )
        assert lifecycle.pipeline_registry is not None
        assert lifecycle.pipeline_registry.names() == ("alpha",)

        assert asyncio.run(lifecycle.reload_config()) is True

    assert lifecycle.pipeline_registry is not None
    assert lifecycle.pipeline_registry.names() == ("alpha", "beta")
    assert runtime.pipeline_registry is lifecycle.pipeline_registry


def test_failed_reload_keeps_the_previous_registry(facade_steps: str) -> None:
    import asyncio

    broken = _pipelines(("alpha",))
    broken["definitions"]["alpha"]["jobs"]["load"]["uses"] = "facade_steps.missing"
    runtime = OrchidRuntime(default_model="ollama/llama3.2")
    with patch("orchid_ai.orchid.lifecycle.build_graph", return_value=MagicMock()):
        from orchid_ai.orchid.lifecycle import OrchidLifecycle

        lifecycle = OrchidLifecycle(
            config=_config(pipelines=_pipelines(("alpha",))),
            runtime=runtime,
            config_watcher=_watcher(_config(pipelines=broken)),
        )
        previous = lifecycle.pipeline_registry

        assert asyncio.run(lifecycle.reload_config()) is False

    assert lifecycle.pipeline_registry is previous
    assert runtime.pipeline_registry is previous


# ── Programmatic construction and extension ────────────────


def test_a_pipeline_can_be_built_programmatically(facade_steps: str) -> None:
    from orchid_ai.config.schema_pipeline import (
        OrchidPipelineDefinitionConfig,
        OrchidPipelineDeterministicJobConfig,
    )
    from orchid_ai.pipeline.definition import compile_pipeline
    from orchid_ai.pipeline.registry import PipelineRegistry
    from orchid_ai.pipeline.runner import OrchidPipelineRunner

    definition = OrchidPipelineDefinitionConfig(
        jobs={
            "load": OrchidPipelineDeterministicJobConfig.model_validate(
                {"kind": "deterministic", "uses": "facade_steps.load"}
            )
        }
    )
    compiled = compile_pipeline("inline", definition)
    registry = PipelineRegistry([compiled])

    result = __import__("asyncio").run(
        OrchidPipelineRunner(registry.get("inline")).run({"account_id": "a1"}, user_id="u1")
    )

    assert result.status == "succeeded"
    assert result.jobs["load"]["output"] == {"id": "a1", "name": "Acme"}


def test_curated_pipeline_exports_resolve() -> None:
    import orchid_ai

    for name in (
        "OrchidGraphMutation",
        "OrchidGraphObject",
        "OrchidGraphRelation",
        "OrchidPipelineEvent",
        "OrchidPipelineObserver",
        "OrchidPipelineResult",
        "OrchidPipelineStep",
        "OrchidPipelineStepContext",
        "OrchidPipelineStepResult",
        "register_pipeline_step",
    ):
        assert hasattr(orchid_ai, name), name


def _fake_build_graph(
    *,
    config: Any,
    runtime: Any,
    agents_out: dict[str, Any],
    executors_out: dict[str, Any] | None = None,
) -> MagicMock:
    agents_out["helper"] = MagicMock()
    if executors_out is not None:
        executors_out["helper"] = MagicMock()
    return MagicMock()


def test_lifecycle_exposes_pipeline_dependencies(facade_steps: str) -> None:
    runtime = OrchidRuntime(default_model="ollama/llama3.2")
    with patch("orchid_ai.orchid.lifecycle.build_graph", side_effect=_fake_build_graph):
        from orchid_ai.orchid.lifecycle import OrchidLifecycle

        lifecycle = OrchidLifecycle(config=_config(pipelines=_pipelines()), runtime=runtime)

    deps = lifecycle.pipeline_dependencies
    assert deps is not None
    assert set(deps.agents) == {"helper"}
    assert deps.decision_models is not None
    assert set(deps.agent_configs) == {"helper"}
    assert set(deps.agent_executors) == {"helper"}


def test_facade_passes_the_pipeline_dependencies(facade_steps: str) -> None:
    import asyncio

    from orchid_ai.core.pipeline import OrchidPipelineResult
    from orchid_ai.pipeline import runner as runner_module

    captured: dict[str, Any] = {}

    class _SpyRunner:
        def __init__(self, definition: Any, **kwargs: Any) -> None:
            captured["deps"] = kwargs.get("deps")
            captured["execution"] = kwargs.get("execution")

        async def run(self, *args: Any, **kwargs: Any) -> OrchidPipelineResult:
            from datetime import UTC, datetime

            return OrchidPipelineResult(
                run_id="run-1",
                pipeline="demo",
                status="succeeded",
                started_at=datetime.now(UTC),
            )

    runtime = OrchidRuntime(default_model="ollama/llama3.2")
    with patch("orchid_ai.orchid.lifecycle.build_graph", side_effect=_fake_build_graph):
        client = Orchid(config=_config(pipelines=_pipelines()), runtime=runtime)
    with patch.object(runner_module, "OrchidPipelineRunner", _SpyRunner):
        asyncio.run(client.run_pipeline("demo", {"account_id": "a1"}, user_id="u1"))

    assert captured["deps"] is client._lifecycle.pipeline_dependencies
    assert set(captured["deps"].agents) == {"helper"}
