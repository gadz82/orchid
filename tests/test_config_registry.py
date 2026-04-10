"""Tests for src.config.registry — agent class registry."""

from __future__ import annotations

import pytest

from orchid_ai.config import registry as reg


@pytest.fixture(autouse=True)
def _clean_registry():
    """Clear the registry before and after each test."""
    saved = dict(reg._REGISTRY)
    reg._REGISTRY.clear()
    yield
    reg._REGISTRY.clear()
    reg._REGISTRY.update(saved)


class TestRegistry:
    def test_register_and_get_class(self):
        class FakeAgent:
            pass

        reg.register("fake", FakeAgent)
        assert reg.get_class("fake") is FakeAgent

    def test_get_class_none_returns_generic_agent(self):
        from orchid_ai.agents.generic_agent import GenericAgent

        assert reg.get_class(None) is GenericAgent

    def test_get_class_dotted_path(self):
        cls = reg.get_class("orchid_ai.config.schema.LLMConfig")
        from orchid_ai.config.schema import LLMConfig

        assert cls is LLMConfig

    def test_get_class_invalid_path_raises(self):
        with pytest.raises(ImportError, match="Cannot resolve agent class"):
            reg.get_class("nonexistent.module.ClassName")
