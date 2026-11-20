"""Pipeline tool lifecycle: registration, collision, reload swap and ownership."""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from orchid_ai.config.schema import OrchidAgentsConfig
from orchid_ai.config.tool_registry import TOOL_REGISTRY, register_tool, unregister
from orchid_ai.core.pipeline import OrchidPipelineConfigError
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.core.tool import OrchidToolInput
from orchid_ai.pipeline.tools import PipelineTool
from orchid_ai.runtime import OrchidRuntime


@pytest.fixture
def lifecycle_steps(monkeypatch: pytest.MonkeyPatch) -> str:
    module = types.ModuleType("lifecycle_steps")

    def load(ctx):
        return {"ok": True}

    module.load = load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lifecycle_steps", module)
    return "lifecycle_steps"


def _pipelines(
    names: tuple[str, ...],
    *,
    requires_approval: bool = False,
    parallel_safe: bool = False,
) -> dict[str, Any]:
    definitions: dict[str, Any] = {}
    for name in names:
        definitions[name] = {
            "description": f"{name} pipeline",
            "input": {"fields": {"account_id": {"type": "string", "required": True}}},
            "jobs": {
                "load": {
                    "kind": "deterministic",
                    "uses": "lifecycle_steps.load",
                }
            },
            "expose": {
                "as_tool": True,
                "description": f"Run the {name} pipeline",
                "tool_name": f"pipeline_{name}",
                "requires_approval": requires_approval,
                "parallel_safe": parallel_safe,
            },
        }
    return {"enabled": True, "definitions": definitions}


def _config(
    *,
    pipelines: dict[str, Any] | None = None,
    agent_tools: list[str] | None = None,
) -> OrchidAgentsConfig:
    payload: dict[str, Any] = {
        "agents": {
            "helper": {
                "description": "Answers questions.",
                "prompt": "You are a helper.",
                "tools": list(agent_tools or []),
            }
        }
    }
    if pipelines is not None:
        payload["pipelines"] = pipelines
    return OrchidAgentsConfig.model_validate(payload)


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


def _watcher(config: OrchidAgentsConfig) -> MagicMock:
    watcher = MagicMock()
    watcher.reload_if_changed.return_value = types.SimpleNamespace(config=config)
    return watcher


def _lifecycle(config: OrchidAgentsConfig, *, watcher: Any = None):
    from orchid_ai.orchid.lifecycle import OrchidLifecycle

    return OrchidLifecycle(
        config=config,
        runtime=OrchidRuntime(default_model="ollama/llama3.2"),
        config_watcher=watcher,
    )


def _registered() -> set[str]:
    return set(TOOL_REGISTRY.get_all())


def _cleanup(*names: str) -> None:
    for name in names:
        unregister(name)


def test_exposed_pipeline_registers_a_tool_and_close_removes_it(lifecycle_steps: str) -> None:
    with patch("orchid_ai.orchid.lifecycle.build_graph", side_effect=_fake_build_graph):
        lifecycle = _lifecycle(_config(pipelines=_pipelines(("alpha",))))
    try:
        assert "pipeline_alpha" in _registered()
        [tool] = lifecycle.pipeline_tools
        assert isinstance(tool, PipelineTool)
        assert tool.pipeline == "alpha"
        assert tool.name == "pipeline_alpha"
    finally:
        asyncio.run(lifecycle.close())

    assert "pipeline_alpha" not in _registered()


def test_no_pipelines_means_no_tools() -> None:
    with patch("orchid_ai.orchid.lifecycle.build_graph", side_effect=_fake_build_graph):
        lifecycle = _lifecycle(_config())
    try:
        assert lifecycle.pipeline_registry is None
        assert lifecycle.pipeline_tools == []
    finally:
        asyncio.run(lifecycle.close())


def test_registration_alone_does_not_list_the_tool_on_an_agent(lifecycle_steps: str) -> None:
    with patch("orchid_ai.orchid.lifecycle.build_graph", side_effect=_fake_build_graph):
        lifecycle = _lifecycle(_config(pipelines=_pipelines(("alpha",))))
    try:
        assert "pipeline_alpha" in _registered()
        assert lifecycle.config.agents["helper"].tools == []
    finally:
        asyncio.run(lifecycle.close())


def test_collision_with_a_non_pipeline_tool_is_a_configuration_error(lifecycle_steps: str) -> None:
    register_tool("pipeline_alpha", lambda: "existing", description="Existing tool")
    try:
        with (
            patch("orchid_ai.orchid.lifecycle.build_graph", side_effect=_fake_build_graph),
            pytest.raises(OrchidPipelineConfigError, match="collide"),
        ):
            _lifecycle(_config(pipelines=_pipelines(("alpha",))))
        assert "pipeline_alpha" in _registered()
    finally:
        _cleanup("pipeline_alpha")


def test_reload_adds_and_swaps_tools(lifecycle_steps: str) -> None:
    watcher = _watcher(_config(pipelines=_pipelines(("alpha", "beta"))))
    with patch("orchid_ai.orchid.lifecycle.build_graph", side_effect=_fake_build_graph):
        lifecycle = _lifecycle(_config(pipelines=_pipelines(("alpha",))), watcher=watcher)
        try:
            assert _registered() & {"pipeline_alpha"} == {"pipeline_alpha"}
            assert asyncio.run(lifecycle.reload_config()) is True
            assert {"pipeline_alpha", "pipeline_beta"} <= _registered()
            assert {tool.name for tool in lifecycle.pipeline_tools} == {
                "pipeline_alpha",
                "pipeline_beta",
            }
        finally:
            asyncio.run(lifecycle.close())


def test_reload_removes_a_stale_tool(lifecycle_steps: str) -> None:
    watcher = _watcher(_config(pipelines=_pipelines(("alpha",))))
    with patch("orchid_ai.orchid.lifecycle.build_graph", side_effect=_fake_build_graph):
        lifecycle = _lifecycle(
            _config(pipelines=_pipelines(("alpha", "beta"))),
            watcher=watcher,
        )
        try:
            assert "pipeline_beta" in _registered()
            assert asyncio.run(lifecycle.reload_config()) is True
            assert "pipeline_alpha" in _registered()
            assert "pipeline_beta" not in _registered()
        finally:
            asyncio.run(lifecycle.close())


def test_reload_keeps_the_previous_registry_and_tools_on_build_failure(lifecycle_steps: str) -> None:
    broken = _pipelines(("alpha",))
    broken["definitions"]["alpha"]["jobs"]["load"]["uses"] = "lifecycle_steps.missing"
    watcher = _watcher(_config(pipelines=broken))
    with patch("orchid_ai.orchid.lifecycle.build_graph", side_effect=_fake_build_graph):
        lifecycle = _lifecycle(_config(pipelines=_pipelines(("alpha",))), watcher=watcher)
        try:
            previous = lifecycle.pipeline_registry
            assert asyncio.run(lifecycle.reload_config()) is False
            assert lifecycle.pipeline_registry is previous
            assert "pipeline_alpha" in _registered()
        finally:
            asyncio.run(lifecycle.close())


def test_reload_rejects_an_agent_referencing_a_removed_tool(lifecycle_steps: str) -> None:
    watcher = _watcher(
        _config(
            pipelines=_pipelines(("beta",)),
            agent_tools=["pipeline_alpha"],
        )
    )
    with patch("orchid_ai.orchid.lifecycle.build_graph", side_effect=_fake_build_graph):
        lifecycle = _lifecycle(
            _config(pipelines=_pipelines(("alpha",)), agent_tools=["pipeline_alpha"]),
            watcher=watcher,
        )
        try:
            assert asyncio.run(lifecycle.reload_config()) is False
            assert "pipeline_alpha" in _registered()
            assert lifecycle.pipeline_registry.names() == ("alpha",)
        finally:
            asyncio.run(lifecycle.close())


def test_reload_tool_swap_rolls_back_on_a_collision(lifecycle_steps: str) -> None:
    watcher = _watcher(_config(pipelines=_pipelines(("alpha", "beta"))))
    with patch("orchid_ai.orchid.lifecycle.build_graph", side_effect=_fake_build_graph):
        lifecycle = _lifecycle(_config(pipelines=_pipelines(("alpha",))), watcher=watcher)
        register_tool("pipeline_beta", lambda: "existing", description="Existing tool")
        try:
            assert asyncio.run(lifecycle.reload_config()) is False
            assert "pipeline_alpha" in _registered()
            assert "pipeline_beta" in _registered()
            assert {tool.name for tool in lifecycle.pipeline_tools} == {"pipeline_alpha"}
        finally:
            asyncio.run(lifecycle.close())
            _cleanup("pipeline_beta")


def test_two_instances_own_only_their_own_tools(lifecycle_steps: str) -> None:
    with patch("orchid_ai.orchid.lifecycle.build_graph", side_effect=_fake_build_graph):
        first = _lifecycle(_config(pipelines=_pipelines(("alpha",))))
        second = _lifecycle(_config(pipelines=_pipelines(("beta",))))
    try:
        assert {"pipeline_alpha", "pipeline_beta"} <= _registered()
        asyncio.run(first.close())
        assert "pipeline_alpha" not in _registered()
        assert "pipeline_beta" in _registered()
    finally:
        asyncio.run(second.close())

    assert "pipeline_beta" not in _registered()


def test_two_instances_cannot_share_a_tool_name(lifecycle_steps: str) -> None:
    with patch("orchid_ai.orchid.lifecycle.build_graph", side_effect=_fake_build_graph):
        first = _lifecycle(_config(pipelines=_pipelines(("alpha",))))
        try:
            with pytest.raises(OrchidPipelineConfigError, match="collide"):
                _lifecycle(_config(pipelines=_pipelines(("alpha",))))
            assert "pipeline_alpha" in _registered()
        finally:
            asyncio.run(first.close())


def test_registered_tool_runs_a_real_pipeline(lifecycle_steps: str) -> None:
    with patch("orchid_ai.orchid.lifecycle.build_graph", side_effect=_fake_build_graph):
        lifecycle = _lifecycle(_config(pipelines=_pipelines(("alpha",))))
    try:
        [tool] = lifecycle.pipeline_tools
        output = asyncio.run(
            tool.invoke(
                OrchidToolInput(
                    parameters={"account_id": "a1"},
                    auth_context=OrchidAuthContext(
                        access_token="secret-token",
                        tenant_key="t-1",
                        user_id="u-1",
                    ),
                )
            )
        )
        payload = output.result
        assert payload["status"] == "succeeded"
        assert payload["pipeline"] == "alpha"
        assert payload["run_id"]
        assert payload["summary"].startswith("Pipeline 'alpha'")
        assert "secret-token" not in str(payload)
    finally:
        asyncio.run(lifecycle.close())


def test_pipeline_tool_policies_are_annotated_before_graph_build(lifecycle_steps: str) -> None:
    config = _config(
        pipelines=_pipelines(("risk",), requires_approval=True, parallel_safe=True),
        agent_tools=["pipeline_risk"],
    )
    with patch("orchid_ai.orchid.lifecycle.build_graph", side_effect=_fake_build_graph):
        lifecycle = _lifecycle(config)
    try:
        agent_config = lifecycle.config.agents["helper"]
        assert "pipeline_risk" in agent_config.approval_tools
        assert "pipeline_risk" in agent_config.parallel_safe_builtin_tools
    finally:
        asyncio.run(lifecycle.close())


def test_close_does_not_remove_a_tool_replaced_by_another_registration(lifecycle_steps: str) -> None:
    with patch("orchid_ai.orchid.lifecycle.build_graph", side_effect=_fake_build_graph):
        lifecycle = _lifecycle(_config(pipelines=_pipelines(("alpha",))))
    register_tool("pipeline_alpha", lambda: "intruder", description="Intruder")
    try:
        asyncio.run(lifecycle.close())
        assert "pipeline_alpha" in _registered()
        assert (
            TOOL_REGISTRY.get_all()["pipeline_alpha"] is not lifecycle.pipeline_tools[0]
            if lifecycle.pipeline_tools
            else True
        )
    finally:
        _cleanup("pipeline_alpha")


def test_reload_restores_an_owned_tool_replaced_during_build(lifecycle_steps: str) -> None:
    watcher = _watcher(_config(pipelines=_pipelines(("alpha",))))
    with patch("orchid_ai.orchid.lifecycle.build_graph", side_effect=_fake_build_graph):
        lifecycle = _lifecycle(_config(pipelines=_pipelines(("alpha",))), watcher=watcher)
        original = lifecycle.pipeline_tools[0]
        register_tool("pipeline_alpha", lambda: "intruder", description="Intruder")
        try:
            assert asyncio.run(lifecycle.reload_config()) is False
            assert TOOL_REGISTRY.get_all()["pipeline_alpha"] is original
        finally:
            asyncio.run(lifecycle.close())


def test_annotated_pipeline_tool_is_wrapped_for_hitl(lifecycle_steps: str) -> None:
    from orchid_ai.agents.tools import build_langchain_tools

    config = _config(
        pipelines=_pipelines(("risk",), requires_approval=True),
        agent_tools=["pipeline_risk"],
    )
    with patch("orchid_ai.orchid.lifecycle.build_graph", side_effect=_fake_build_graph):
        lifecycle = _lifecycle(config)
    try:
        [pipeline_tool] = lifecycle.pipeline_tools
        wrappers = build_langchain_tools(
            builtin_names={"pipeline_risk"},
            builtin_tool_defs=[pipeline_tool.get_llm_function_schema()],
            mcp_tool_defs=[],
            mcp_tool_client_map={},
            auth=OrchidAuthContext(access_token="t"),
            approval_tools=lifecycle.config.agents["helper"].approval_tools,
        )
        assert wrappers[0].requires_approval is True
    finally:
        asyncio.run(lifecycle.close())


def test_reload_reannotates_pipeline_tool_policies(lifecycle_steps: str) -> None:
    watcher = _watcher(
        _config(
            pipelines=_pipelines(("risk",), requires_approval=True, parallel_safe=True),
            agent_tools=["pipeline_risk"],
        )
    )
    with patch("orchid_ai.orchid.lifecycle.build_graph", side_effect=_fake_build_graph):
        lifecycle = _lifecycle(
            _config(pipelines=_pipelines(("risk",)), agent_tools=["pipeline_risk"]),
            watcher=watcher,
        )
        try:
            assert asyncio.run(lifecycle.reload_config()) is True
            agent_config = lifecycle.config.agents["helper"]
            assert "pipeline_risk" in agent_config.approval_tools
            assert "pipeline_risk" in agent_config.parallel_safe_builtin_tools
        finally:
            asyncio.run(lifecycle.close())
