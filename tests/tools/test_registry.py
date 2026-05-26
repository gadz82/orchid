from __future__ import annotations

from pathlib import Path
import sys

import pytest

from orchid_ai.config.loader import load_config
from orchid_ai.config.schema import OrchidBuiltinToolConfig
from orchid_ai.config.tool_registry import clear, get_tool, load_tools_from_config
from orchid_ai.core.tool import OrchidTool, OrchidToolInput, OrchidToolOutput
from orchid_ai.tools.registry import OrchidToolRegistry


class DummyTool(OrchidTool):
    name = "dummy"
    description = "Dummy"
    parameters_schema = {"type": "object", "properties": {}}

    async def invoke(self, tool_input: OrchidToolInput) -> OrchidToolOutput:
        return OrchidToolOutput(result=tool_input.parameters)


def test_register_and_get():
    registry = OrchidToolRegistry()
    tool = DummyTool()

    registry.register(tool)

    assert registry.get("dummy") is tool


def test_register_duplicate_raises():
    registry = OrchidToolRegistry()
    registry.register(DummyTool())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(DummyTool())


def test_get_missing_raises():
    registry = OrchidToolRegistry()

    with pytest.raises(KeyError, match="not registered"):
        registry.get("missing")


def test_unregister_and_clear():
    registry = OrchidToolRegistry()
    registry.register(DummyTool())

    registry.unregister("dummy")
    with pytest.raises(KeyError):
        registry.get("dummy")

    registry.register(DummyTool())
    registry.clear()
    assert registry.get_all() == {}


def test_get_all_returns_copy():
    registry = OrchidToolRegistry()
    registry.register(DummyTool())

    tools = registry.get_all()
    tools["other"] = DummyTool()

    assert set(registry.get_all()) == {"dummy"}


def test_load_tools_from_config_supports_class_path():
    clear()
    workspace_root = str(Path(__file__).resolve().parents[3])
    if workspace_root not in sys.path:
        sys.path.insert(0, workspace_root)
    try:
        load_tools_from_config(
            {
                "get_player_stats": OrchidBuiltinToolConfig(
                    class_="examples.basketball.tools.basketball.GetPlayerStatsTool",
                )
            }
        )
        tool = get_tool("get_player_stats")
        assert tool.name == "get_player_stats"
        assert tool.get_parameters_schema()["properties"]["player_name"]["type"] == "string"
    finally:
        clear()


@pytest.mark.parametrize(
    ("tool_name", "class_path", "field_name"),
    [
        (
            "lookup_artist",
            "examples.festival-producer.tools.booking.LookupArtistTool",
            "artist_name",
        ),
        (
            "analyze_structure",
            "examples.architecture_review.tools.structural.AnalyzeStructureTool",
            "building_type",
        ),
        (
            "lookup_glossary",
            "examples.wiki.hooks.tools.LookupGlossaryTool",
            "term",
        ),
        (
            "extract_concepts",
            "examples.education.tools.content.extract_concepts.ExtractConceptsTool",
            "source_text",
        ),
    ],
)
def test_load_tools_from_config_resolves_example_tool_classes(tool_name: str, class_path: str, field_name: str):
    clear()
    workspace_root = str(Path(__file__).resolve().parents[3])
    if workspace_root not in sys.path:
        sys.path.insert(0, workspace_root)
    try:
        load_tools_from_config({tool_name: OrchidBuiltinToolConfig(class_=class_path)})
        tool = get_tool(tool_name)
        assert tool.name == tool_name
        assert tool.get_parameters_schema()["properties"][field_name]
    finally:
        clear()


@pytest.mark.parametrize(
    "config_path",
    [
        "examples/education/agents.yaml",
        "examples/festival-producer/agents.yaml",
        "examples/architecture_review/agents.yaml",
        "examples/wiki/agents.yaml",
    ],
)
def test_load_tools_from_config_registers_all_tools_from_migrated_examples(config_path: str):
    clear()
    workspace_root = Path(__file__).resolve().parents[3]
    workspace_root_str = str(workspace_root)
    if workspace_root_str not in sys.path:
        sys.path.insert(0, workspace_root_str)
    try:
        config = load_config(workspace_root / config_path)
        load_tools_from_config(config.tools)
        assert set(config.tools).issubset(set(get_tool(name).name for name in config.tools))
    finally:
        clear()
