from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from orchid_ai.config.schema_external_agent import OrchidExternalAgentConfig
from orchid_ai.config.tool_registry import TOOL_REGISTRY, clear
from orchid_ai.tools.external_cli_config import load_external_agents_from_config


class _FakeChatModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, messages: list[dict[str, Any]], **kwargs: Any) -> MagicMock:
        self.calls.append({"messages": messages, **kwargs})
        result = MagicMock()
        result.content = "normalised"
        return result


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    clear()
    yield
    clear()


class TestLoadExternalAgentsFromConfig:
    def test_registers_tools_into_registry(self) -> None:
        config = {
            "ask_external": OrchidExternalAgentConfig.model_validate({"command": ["cmd"]}),
            "ask_other": OrchidExternalAgentConfig.model_validate({"command": ["other"]}),
        }
        load_external_agents_from_config(config)

        assert "ask_external" in TOOL_REGISTRY.get_all()
        assert "ask_other" in TOOL_REGISTRY.get_all()

    def test_idempotent_re_run(self) -> None:
        config = {"t1": OrchidExternalAgentConfig.model_validate({"command": ["cmd"]})}
        load_external_agents_from_config(config)
        load_external_agents_from_config(config)

        tools = TOOL_REGISTRY.get_all()
        assert len(tools) == 1
        assert "t1" in tools

    def test_tool_has_correct_name_and_description(self) -> None:
        config = {
            "my_tool": OrchidExternalAgentConfig.model_validate({"command": ["cmd"], "description": "does things"}),
        }
        load_external_agents_from_config(config)

        tool = TOOL_REGISTRY.get("my_tool")
        assert tool.name == "my_tool"
        assert tool.description == "does things"

    def test_description_fallback_to_name(self) -> None:
        config = {"my_tool": OrchidExternalAgentConfig.model_validate({"command": ["cmd"]})}
        load_external_agents_from_config(config)

        tool = TOOL_REGISTRY.get("my_tool")
        assert tool.description == "my_tool"

    def test_requires_approval_default_true(self) -> None:
        config = {"t": OrchidExternalAgentConfig.model_validate({"command": ["cmd"]})}
        load_external_agents_from_config(config)

        tool = TOOL_REGISTRY.get("t")
        assert tool.requires_approval is True

    def test_requires_approval_override(self) -> None:
        config = {
            "t": OrchidExternalAgentConfig.model_validate({"command": ["cmd"], "requires_approval": False}),
        }
        load_external_agents_from_config(config)

        tool = TOOL_REGISTRY.get("t")
        assert tool.requires_approval is False

    def test_parallel_safe_override(self) -> None:
        config = {
            "t": OrchidExternalAgentConfig.model_validate({"command": ["cmd"], "parallel_safe": True}),
        }
        load_external_agents_from_config(config)

        tool = TOOL_REGISTRY.get("t")
        assert tool.parallel_safe is True

    def test_inject_to_rag_and_rag_ttl_override(self) -> None:
        config = {
            "t": OrchidExternalAgentConfig.model_validate({"command": ["cmd"], "inject_to_rag": True, "rag_ttl": 3600}),
        }
        load_external_agents_from_config(config)

        tool = TOOL_REGISTRY.get("t")
        assert tool.inject_to_rag is True
        assert tool.rag_ttl == 3600

    def test_llm_normalizer_built_when_chat_model_provided(self) -> None:
        chat_model = _FakeChatModel()
        config = {
            "t": OrchidExternalAgentConfig.model_validate(
                {
                    "command": ["cmd"],
                    "normalizer": "llm",
                    "normalizer_instruction": "Simplify.",
                }
            ),
        }
        load_external_agents_from_config(config, chat_model=chat_model)

        tool = TOOL_REGISTRY.get("t")
        assert type(tool._normalizer).__name__ == "LLMNormalizer"

    def test_llm_normalizer_falls_back_to_passthrough_when_no_chat_model(self) -> None:
        config = {
            "t": OrchidExternalAgentConfig.model_validate({"command": ["cmd"], "normalizer": "llm"}),
        }
        load_external_agents_from_config(config, chat_model=None)

        tool = TOOL_REGISTRY.get("t")
        assert type(tool._normalizer).__name__ == "PassthroughNormalizer"

    def test_passthrough_normalizer_default(self) -> None:
        config = {"t": OrchidExternalAgentConfig.model_validate({"command": ["cmd"]})}
        load_external_agents_from_config(config)

        tool = TOOL_REGISTRY.get("t")
        assert type(tool._normalizer).__name__ == "PassthroughNormalizer"

    def test_empty_config_no_tools(self) -> None:
        load_external_agents_from_config({})

        assert len(TOOL_REGISTRY.get_all()) == 0
