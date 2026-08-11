"""Tests for content source injection into agents and tool wiring."""

from __future__ import annotations

from orchid_ai.agents.generic_agent import GenericAgent
from orchid_ai.config.schema import OrchidAgentConfig
from orchid_ai.content.local import LocalFileContentSource
from orchid_ai.rag.backends.null import NullVectorReader


class TestAgentContentSources:
    def test_content_sources_injected(self, tmp_path):
        data_dir = tmp_path / "content"
        data_dir.mkdir()
        (data_dir / "f.txt").write_text("hello")
        source = LocalFileContentSource(path=str(data_dir))

        config = OrchidAgentConfig(name="test", description="test", prompt="test prompt")
        agent = GenericAgent(
            config=config,
            reader=NullVectorReader(),
            content_sources=[source],
        )
        assert agent._content_sources == [source]

    def test_content_sources_property(self, tmp_path):
        data_dir = tmp_path / "content"
        data_dir.mkdir()
        (data_dir / "f.txt").write_text("hello")
        source = LocalFileContentSource(path=str(data_dir))

        config = OrchidAgentConfig(name="test", description="test", prompt="test prompt")
        agent = GenericAgent(
            config=config,
            reader=NullVectorReader(),
            content_sources=[source],
        )
        assert agent.content_sources == [source]

    def test_no_content_sources_defaults_to_empty(self):
        config = OrchidAgentConfig(name="test", description="test", prompt="test prompt")
        agent = GenericAgent(
            config=config,
            reader=NullVectorReader(),
        )
        assert agent._content_sources == []
        assert agent.content_sources == []
