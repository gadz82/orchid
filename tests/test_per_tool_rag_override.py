"""ADR-024 — ``OrchidAgentConfig.effective_rag(tool_name)`` deep-merge."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from orchid_ai.config.schema import OrchidAgentsConfig


def _parse(yaml_text: str) -> OrchidAgentsConfig:
    return OrchidAgentsConfig.model_validate(yaml.safe_load(yaml_text))


class TestEffectiveRagDefaults:
    def test_no_override_returns_agent_rag(self):
        cfg = _parse(
            """
            version: '1'
            tools:
              search_kb:
                handler: tests.tools.search
            agents:
              search:
                description: KB search
                prompt: 'You search.'
                tools: [search_kb]
                rag:
                  namespace: kb-default
                  ingestion: { strategy: recursive, chunk_size: 1000 }
                  retrieval: { strategy: simple }
            """
        )
        agent = cfg.agents["search"]
        assert agent.effective_rag("search_kb") is agent.rag

    def test_unknown_tool_returns_agent_rag(self):
        cfg = _parse(
            """
            version: '1'
            agents:
              search:
                description: KB search
                prompt: 'You search.'
                rag: { namespace: kb }
            """
        )
        agent = cfg.agents["search"]
        assert agent.effective_rag("not_registered") is agent.rag


class TestEffectiveRagMcpToolOverride:
    def test_mcp_tool_namespace_override(self):
        cfg = _parse(
            """
            version: '1'
            agents:
              search:
                description: KB search
                prompt: 'You search.'
                rag: { namespace: kb-default, k: 5 }
                mcp_servers:
                  - name: kb_server
                    url: http://kb
                    tools:
                      - name: lookup_record
                        rag:
                          namespace: records-cache
            """
        )
        agent = cfg.agents["search"]
        eff = agent.effective_rag("lookup_record")
        assert eff.namespace == "records-cache"
        # k inherits from agent.rag.
        assert eff.k == 5

    def test_mcp_tool_overrides_only_set_fields(self):
        """A tool override must NOT clobber unrelated nested defaults."""
        cfg = _parse(
            """
            version: '1'
            agents:
              search:
                description: KB search
                prompt: 'You search.'
                rag:
                  namespace: kb-default
                  ingestion: { strategy: recursive, chunk_size: 1500, chunk_overlap: 250 }
                  retrieval: { strategy: hyde }
                mcp_servers:
                  - name: srv
                    url: http://srv
                    tools:
                      - name: tool_a
                        rag:
                          ingestion: { chunk_size: 400 }
            """
        )
        eff = cfg.agents["search"].effective_rag("tool_a")
        # Only ``chunk_size`` flips; ``strategy`` + ``chunk_overlap``
        # inherit from the agent block.
        assert eff.ingestion.strategy == "recursive"
        assert eff.ingestion.chunk_size == 400
        assert eff.ingestion.chunk_overlap == 250
        # Retrieval block untouched.
        assert eff.retrieval.strategy == "hyde"
        # Top-level namespace untouched.
        assert eff.namespace == "kb-default"


class TestEffectiveRagBuiltinToolOverride:
    def test_builtin_tool_resolves_through_cached_configs(self):
        """Built-in overrides flow through ``builtin_tool_configs``."""
        cfg = _parse(
            """
            version: '1'
            tools:
              format_date:
                handler: tests.tools.format_date
                rag:
                  namespace: dates-cache
                  ingestion: { chunk_size: 200 }
            agents:
              utility:
                description: utility
                prompt: 'p'
                tools: [format_date]
                rag:
                  namespace: utility-default
                  ingestion: { strategy: recursive, chunk_size: 1000 }
            """
        )
        agent = cfg.agents["utility"]
        # Cache populated.
        assert "format_date" in agent.builtin_tool_configs
        eff = agent.effective_rag("format_date")
        assert eff.namespace == "dates-cache"
        assert eff.ingestion.chunk_size == 200
        # Strategy still recursive (inherited).
        assert eff.ingestion.strategy == "recursive"

    def test_builtin_without_rag_block_returns_agent_rag(self):
        cfg = _parse(
            """
            version: '1'
            tools:
              format_date:
                handler: tests.tools.format_date
            agents:
              utility:
                description: utility
                prompt: 'p'
                tools: [format_date]
                rag: { namespace: utility-default }
            """
        )
        agent = cfg.agents["utility"]
        assert "format_date" in agent.builtin_tool_configs
        # Tool config exists but ``rag`` is None → effective is agent.rag.
        assert agent.effective_rag("format_date") is agent.rag


class TestEffectiveRagPrecedence:
    def test_mcp_tool_override_wins_over_builtin_with_same_name(self):
        """If a tool name lives in both spaces, the MCP variant wins.

        ``effective_rag`` looks up MCP tools first to match the
        agentic loop's dispatch order — MCP servers win when names
        collide.
        """
        cfg = _parse(
            """
            version: '1'
            tools:
              shared_name:
                handler: tests.tools.x
                rag: { namespace: builtin-ns }
            agents:
              hybrid:
                description: h
                prompt: 'p'
                tools: [shared_name]
                rag: { namespace: agent-ns }
                mcp_servers:
                  - name: srv
                    url: http://srv
                    tools:
                      - name: shared_name
                        rag: { namespace: mcp-ns }
            """
        )
        agent = cfg.agents["hybrid"]
        assert agent.effective_rag("shared_name").namespace == "mcp-ns"


class TestSchemaValidation:
    def test_unknown_field_inside_tool_rag_is_rejected(self):
        """``OrchidRAGConfig`` is ``extra='forbid'`` — typos surface
        at validation time, not at retrieval time."""
        with pytest.raises(ValidationError):
            _parse(
                """
                version: '1'
                tools:
                  bad_tool:
                    handler: tests.tools.x
                    rag:
                      namespacxe: typo-here
                agents:
                  agent_a:
                    description: x
                    prompt: 'p'
                    tools: [bad_tool]
                """
            )
