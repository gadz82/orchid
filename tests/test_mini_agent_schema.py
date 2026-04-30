"""Schema tests for ``OrchidMiniAgentConfig`` and the nesting validator."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orchid_ai.config.schema import (
    OrchidAgentConfig,
    OrchidAgentsConfig,
    OrchidMiniAgentConfig,
    OrchidRAGConfig,
)


# ── Field defaults and bounds ──────────────────────────────────


class TestOrchidMiniAgentConfigDefaults:
    def test_default_disabled(self):
        cfg = OrchidMiniAgentConfig()
        assert cfg.enabled is False
        assert cfg.max_count == 3
        assert cfg.timeout_seconds == 60
        assert cfg.tool_allowlist_mode == "strict"
        assert cfg.stream_inner_tokens is False
        assert cfg.decomposer_model is None
        assert cfg.decomposer_prompt is None
        assert cfg.aggregator_prompt is None

    def test_max_count_lower_bound(self):
        # max_count=1 is rejected (forking 1 sub-task is pointless).
        with pytest.raises(ValidationError):
            OrchidMiniAgentConfig(max_count=1)

    def test_max_count_upper_bound(self):
        # Hard cap at 8 keeps the LangGraph fan-out bounded by config alone.
        with pytest.raises(ValidationError):
            OrchidMiniAgentConfig(max_count=9)

    def test_max_count_inside_range(self):
        for n in range(2, 9):
            cfg = OrchidMiniAgentConfig(max_count=n)
            assert cfg.max_count == n

    def test_timeout_bounds(self):
        with pytest.raises(ValidationError):
            OrchidMiniAgentConfig(timeout_seconds=4)
        with pytest.raises(ValidationError):
            OrchidMiniAgentConfig(timeout_seconds=601)
        # Boundary values are accepted.
        assert OrchidMiniAgentConfig(timeout_seconds=5).timeout_seconds == 5
        assert OrchidMiniAgentConfig(timeout_seconds=600).timeout_seconds == 600

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            OrchidMiniAgentConfig(unknown_field=True)


# ── OrchidAgentConfig integration ──────────────────────────────


class TestOrchidAgentConfigMiniAgent:
    def test_default_disabled_when_not_set(self):
        cfg = OrchidAgentConfig(description="x", prompt="y")
        assert isinstance(cfg.mini_agent, OrchidMiniAgentConfig)
        assert cfg.mini_agent.enabled is False

    def test_yaml_round_trip(self):
        cfg = OrchidAgentConfig(
            description="support",
            prompt="be helpful",
            mini_agent=OrchidMiniAgentConfig(
                enabled=True,
                max_count=5,
                timeout_seconds=30,
                tool_allowlist_mode="inferred",
            ),
        )
        assert cfg.mini_agent.enabled is True
        assert cfg.mini_agent.max_count == 5
        assert cfg.mini_agent.timeout_seconds == 30
        assert cfg.mini_agent.tool_allowlist_mode == "inferred"


# ── Nesting validator: child agents may not enable mini-agents ──


class TestNestingForbidden:
    def test_child_with_mini_agent_enabled_rejected(self):
        with pytest.raises(ValidationError) as excinfo:
            OrchidAgentsConfig(
                agents={
                    "parent": OrchidAgentConfig(
                        description="parent",
                        prompt="prompt",
                        rag=OrchidRAGConfig(enabled=False),
                        children={
                            "child": OrchidAgentConfig(
                                description="child",
                                prompt="prompt",
                                rag=OrchidRAGConfig(enabled=False),
                                mini_agent=OrchidMiniAgentConfig(enabled=True),
                            ),
                        },
                    ),
                },
            )
        # The validator's message is wrapped by Pydantic — assert the
        # essential "no nesting" wording survives.
        assert "no nesting" in str(excinfo.value)

    def test_top_level_with_mini_agent_enabled_accepted(self):
        config = OrchidAgentsConfig(
            agents={
                "support": OrchidAgentConfig(
                    description="support",
                    prompt="prompt",
                    rag=OrchidRAGConfig(enabled=False),
                    mini_agent=OrchidMiniAgentConfig(enabled=True),
                ),
            },
        )
        agent = config.agents["support"]
        assert agent.mini_agent.enabled is True

    def test_child_without_mini_agent_accepted(self):
        config = OrchidAgentsConfig(
            agents={
                "parent": OrchidAgentConfig(
                    description="parent",
                    prompt="prompt",
                    rag=OrchidRAGConfig(enabled=False),
                    mini_agent=OrchidMiniAgentConfig(enabled=True),  # parent IS allowed
                    children={
                        "child": OrchidAgentConfig(
                            description="child",
                            prompt="prompt",
                            rag=OrchidRAGConfig(enabled=False),
                            # child stays at default (enabled=False)
                        ),
                    },
                ),
            },
        )
        assert config.agents["parent"].mini_agent.enabled is True
        assert config.agents["parent"].children["child"].mini_agent.enabled is False
