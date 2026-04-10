"""Tests for src.config.tool_registry — built-in tool registry."""

from __future__ import annotations

import asyncio

import pytest

from orchid_ai.config import tool_registry as treg
from orchid_ai.config.schema import BuiltinToolConfig


@pytest.fixture(autouse=True)
def _clean_registry():
    """Clear the tool registry before and after each test."""
    treg.clear()
    yield
    treg.clear()


class TestRegisterAndGet:
    def test_register_and_get(self):
        treg.register_tool("greet", lambda name: f"hi {name}", "Greets")
        entry = treg.get_tool("greet")
        assert entry.name == "greet"
        assert entry.description == "Greets"
        assert entry.handler("world") == "hi world"

    def test_get_unknown_raises_key_error(self):
        with pytest.raises(KeyError, match="not registered"):
            treg.get_tool("no_such_tool")


class TestCallTool:
    def test_call_sync_handler(self):
        treg.register_tool("add", lambda a, b: a + b)
        result = asyncio.get_event_loop().run_until_complete(treg.call_tool("add", a=1, b=2))
        assert result == 3

    def test_call_async_handler(self):
        async def async_add(a, b):
            return a + b

        treg.register_tool("async_add", async_add)
        result = asyncio.get_event_loop().run_until_complete(treg.call_tool("async_add", a=3, b=4))
        assert result == 7


class TestResolveHandler:
    def test_valid_dotted_path(self):
        handler = treg._resolve_handler("os.path.join")
        import os.path

        assert handler is os.path.join

    def test_invalid_path_raises(self):
        with pytest.raises(ValueError, match="Invalid handler path"):
            treg._resolve_handler("no_dots_here")


class TestListAndClear:
    def test_list_tools(self):
        treg.register_tool("a", lambda: None, "Tool A")
        treg.register_tool("b", lambda: None, "Tool B")
        names = {e.name for e in treg.list_tools()}
        assert names == {"a", "b"}

    def test_clear(self):
        treg.register_tool("temp", lambda: None)
        treg.clear()
        assert treg.list_tools() == []


class TestLoadToolsFromConfig:
    def test_loads_from_config(self):
        config = {
            "join_path": BuiltinToolConfig(handler="os.path.join", description="Join paths"),
        }
        treg.load_tools_from_config(config)
        entry = treg.get_tool("join_path")
        assert entry.description == "Join paths"
        assert callable(entry.handler)
