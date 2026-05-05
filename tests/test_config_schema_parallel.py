"""Schema tests for the parallel-tool YAML surface (`parallel_safe`, `parallel_tools`)."""

from __future__ import annotations

from orchid_ai.config.schema import (
    OrchidAgentConfig,
    OrchidAgentsConfig,
    OrchidBuiltinToolConfig,
    OrchidMCPServerConfig,
    OrchidRAGConfig,
    OrchidToolConfig,
)


# ── OrchidToolConfig.parallel_safe ─────────────────────────────


class TestOrchidToolConfigParallelSafe:
    def test_default_is_none(self):
        cfg = OrchidToolConfig(name="lookup_record")
        assert cfg.parallel_safe is None

    def test_explicit_true_persists(self):
        cfg = OrchidToolConfig(name="lookup_record", parallel_safe=True)
        assert cfg.parallel_safe is True

    def test_explicit_false_persists(self):
        cfg = OrchidToolConfig(name="lookup_record", parallel_safe=False)
        assert cfg.parallel_safe is False

    def test_round_trip_via_dict(self):
        cfg = OrchidToolConfig(**{"name": "x", "parallel_safe": True})
        assert cfg.parallel_safe is True


# ── OrchidBuiltinToolConfig.parallel_safe ──────────────────────


class TestOrchidBuiltinToolConfigParallelSafe:
    def test_default_is_none(self):
        cfg = OrchidBuiltinToolConfig(handler="myapp.tools.format_date")
        assert cfg.parallel_safe is None

    def test_explicit_true_persists(self):
        cfg = OrchidBuiltinToolConfig(
            handler="myapp.tools.format_date",
            parallel_safe=True,
        )
        assert cfg.parallel_safe is True


# ── OrchidAgentConfig.parallel_tools ───────────────────────────


class TestOrchidAgentConfigParallelTools:
    def test_default_is_false(self):
        cfg = OrchidAgentConfig(description="x", prompt="y")
        assert cfg.parallel_tools is False

    def test_explicit_true_persists(self):
        cfg = OrchidAgentConfig(description="x", prompt="y", parallel_tools=True)
        assert cfg.parallel_tools is True


# ── parallel_safe_builtin_tools precomputed at validation ─────


class TestParallelSafeBuiltinToolsComputed:
    """Mirrors the ``approval_tools`` precomputation pattern."""

    def test_collects_only_explicit_true(self):
        config = OrchidAgentsConfig(
            tools={
                "format_date": OrchidBuiltinToolConfig(
                    handler="myapp.tools.format_date",
                    parallel_safe=True,
                ),
                "delete_user": OrchidBuiltinToolConfig(
                    handler="myapp.tools.delete_user",
                    parallel_safe=False,
                ),
                "send_email": OrchidBuiltinToolConfig(
                    handler="myapp.tools.send_email",
                    # Default ``None`` — does not collect.
                ),
            },
            agents={
                "support": OrchidAgentConfig(
                    description="support",
                    prompt="support",
                    rag=OrchidRAGConfig(enabled=False),
                    tools=["format_date", "delete_user", "send_email"],
                ),
            },
        )
        agent = config.agents["support"]
        assert agent.parallel_safe_builtin_tools == {"format_date"}

    def test_empty_when_agent_has_no_builtins(self):
        config = OrchidAgentsConfig(
            tools={
                "format_date": OrchidBuiltinToolConfig(
                    handler="myapp.tools.format_date",
                    parallel_safe=True,
                ),
            },
            agents={
                "support": OrchidAgentConfig(
                    description="support",
                    prompt="support",
                    rag=OrchidRAGConfig(enabled=False),
                    # tools list does not reference format_date.
                ),
            },
        )
        agent = config.agents["support"]
        assert agent.parallel_safe_builtin_tools == set()


# ── End-to-end: full nested YAML round-trip ────────────────────


class TestPhaseAFullRoundTrip:
    def test_yaml_dict_round_trip(self):
        config = OrchidAgentsConfig(
            tools={
                "format_date": OrchidBuiltinToolConfig(
                    handler="myapp.tools.format_date",
                    parallel_safe=True,
                ),
            },
            agents={
                "support": OrchidAgentConfig(
                    description="support agent",
                    prompt="be helpful",
                    rag=OrchidRAGConfig(enabled=False),
                    parallel_tools=True,
                    tools=["format_date"],
                    mcp_servers=[
                        OrchidMCPServerConfig(
                            name="kb",
                            url="http://kb.example.com",
                            tools=[
                                OrchidToolConfig(name="search_kb", parallel_safe=True),
                                OrchidToolConfig(name="lookup_record", parallel_safe=False),
                            ],
                        ),
                    ],
                ),
            },
        )
        agent = config.agents["support"]
        assert agent.parallel_tools is True
        assert agent.parallel_safe_builtin_tools == {"format_date"}
        kb_tools = {t.name: t.parallel_safe for t in agent.mcp_servers[0].tools}
        assert kb_tools == {"search_kb": True, "lookup_record": False}
