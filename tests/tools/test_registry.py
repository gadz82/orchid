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


def test_register_duplicate_overwrites_with_warning(caplog):
    registry = OrchidToolRegistry()
    registry.register(DummyTool())

    # Duplicate registration logs a warning and overwrites (does not raise).
    with caplog.at_level("WARNING"):
        registry.register(DummyTool())
    assert any("already registered" in m for m in caplog.messages)
    assert "dummy" in registry.get_all()


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


def test_get_all_returns_read_only_view():
    registry = OrchidToolRegistry()
    registry.register(DummyTool())

    tools = registry.get_all()
    with pytest.raises(TypeError):
        tools["other"] = DummyTool()

    assert set(registry.get_all()) == {"dummy"}


def _ensure_examples_importable() -> None:
    """Add workspace root to sys.path and skip if examples package is unavailable."""
    workspace_root = str(Path(__file__).resolve().parents[3])
    if workspace_root not in sys.path:
        sys.path.insert(0, workspace_root)
    try:
        import examples  # noqa: F401
    except ImportError:
        pytest.skip("examples package not available — only in monorepo workspace")


def test_load_tools_from_config_supports_class_path():
    clear()
    _ensure_examples_importable()
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
    _ensure_examples_importable()
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
    _ensure_examples_importable()
    workspace_root = Path(__file__).resolve().parents[3]
    try:
        config = load_config(workspace_root / config_path)
        load_tools_from_config(config.tools)
        assert set(config.tools).issubset(set(get_tool(name).name for name in config.tools))
    finally:
        clear()
