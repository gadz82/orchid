"""Tests for ``orchid_ai.config.mcp_gateway`` — Pydantic models for the
MCP gateway exposure configuration (tool overrides + MCP Prompts)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from orchid_ai.config import (
    OrchidAgentsConfig,
    OrchidMCPGatewayConfig,
    OrchidMCPGatewayPrompt,
    OrchidMCPGatewayPromptArgument,
    OrchidMCPGatewayToolOverride,
    load_config,
)


# ── OrchidMCPGatewayToolOverride ─────────────────────────────────


class TestToolOverride:
    def test_defaults_are_all_none(self):
        ov = OrchidMCPGatewayToolOverride()
        assert ov.title is None
        assert ov.description is None

    def test_accepts_title_only(self):
        ov = OrchidMCPGatewayToolOverride(title="Hi")
        assert ov.title == "Hi"
        assert ov.description is None

    def test_accepts_description_only(self):
        ov = OrchidMCPGatewayToolOverride(description="Do something.")
        assert ov.description == "Do something."


# ── OrchidMCPGatewayPromptArgument ───────────────────────────────


class TestPromptArgument:
    def test_minimal(self):
        arg = OrchidMCPGatewayPromptArgument(name="department")
        assert arg.name == "department"
        assert arg.description is None
        assert arg.required is False

    def test_rejects_invalid_name(self):
        with pytest.raises(ValidationError):
            OrchidMCPGatewayPromptArgument(name="has space")
        with pytest.raises(ValidationError):
            OrchidMCPGatewayPromptArgument(name="1bad-start")
        with pytest.raises(ValidationError):
            OrchidMCPGatewayPromptArgument(name="")

    def test_accepts_dash_and_underscore(self):
        OrchidMCPGatewayPromptArgument(name="dept_name-v2")  # no error


# ── OrchidMCPGatewayPrompt ───────────────────────────────────────


class TestPrompt:
    def test_minimal(self):
        p = OrchidMCPGatewayPrompt(name="hello", template="Hi {{who}}.")
        assert p.name == "hello"
        assert p.title is None
        assert p.description is None
        assert p.arguments == []

    def test_rejects_invalid_name(self):
        with pytest.raises(ValidationError):
            OrchidMCPGatewayPrompt(name="bad name", template="x")

    def test_rejects_duplicate_argument_names(self):
        with pytest.raises(ValidationError, match="duplicate argument"):
            OrchidMCPGatewayPrompt(
                name="dupe",
                template="x",
                arguments=[{"name": "a"}, {"name": "a"}],
            )

    def test_argument_coercion_from_dicts(self):
        p = OrchidMCPGatewayPrompt(
            name="p",
            template="x",
            arguments=[{"name": "dept", "required": True}],
        )
        assert p.arguments[0].name == "dept"
        assert p.arguments[0].required is True


# ── OrchidMCPGatewayConfig ───────────────────────────────────────


class TestGatewayConfig:
    def test_empty_config(self):
        cfg = OrchidMCPGatewayConfig()
        assert cfg.tools == {}
        assert cfg.prompts == []

    def test_tool_dict_coercion(self):
        cfg = OrchidMCPGatewayConfig(
            tools={"orchid_ask": {"title": "Ask Acme"}},
        )
        assert isinstance(cfg.tools["orchid_ask"], OrchidMCPGatewayToolOverride)
        assert cfg.tools["orchid_ask"].title == "Ask Acme"

    def test_rejects_duplicate_prompt_names(self):
        with pytest.raises(ValidationError, match="Duplicate prompt name"):
            OrchidMCPGatewayConfig(
                prompts=[
                    {"name": "p", "template": "x"},
                    {"name": "p", "template": "y"},
                ]
            )

    def test_programmatic_with_model_instances(self):
        cfg = OrchidMCPGatewayConfig(
            tools={"orchid_ask": OrchidMCPGatewayToolOverride(title="Ask")},
            prompts=[
                OrchidMCPGatewayPrompt(
                    name="p1",
                    template="Say {{greeting}}.",
                    arguments=[OrchidMCPGatewayPromptArgument(name="greeting", required=True)],
                ),
            ],
        )
        assert cfg.tools["orchid_ask"].title == "Ask"
        assert cfg.prompts[0].arguments[0].required is True

    def test_accepts_arbitrary_tool_names(self):
        # The framework does not know which tools a gateway exposes.
        cfg = OrchidMCPGatewayConfig(
            tools={"not_a_real_tool": {"title": "fine"}},
        )
        assert "not_a_real_tool" in cfg.tools


# ── Integration with OrchidAgentsConfig + YAML loader ────────────


class TestAgentsConfigIntegration:
    def test_absent_mcp_gateway_block_parses_empty(self):
        # Loading a YAML without mcp_gateway: should still work and
        # leave an empty config in place.
        cfg = OrchidAgentsConfig()
        assert isinstance(cfg.mcp_gateway, OrchidMCPGatewayConfig)
        assert cfg.mcp_gateway.tools == {}
        assert cfg.mcp_gateway.prompts == []

    def test_inline_construction(self):
        cfg = OrchidAgentsConfig(
            mcp_gateway={
                "tools": {"orchid_ask": {"title": "Ask Acme"}},
                "prompts": [
                    {
                        "name": "compliance_report",
                        "title": "Compliance report",
                        "description": "Generate a compliance report.",
                        "arguments": [
                            {"name": "department", "required": True},
                        ],
                        "template": "Produce report for {{department}}.",
                    }
                ],
            }
        )
        assert cfg.mcp_gateway.tools["orchid_ask"].title == "Ask Acme"
        assert cfg.mcp_gateway.prompts[0].name == "compliance_report"
        assert cfg.mcp_gateway.prompts[0].arguments[0].required is True

    def test_round_trip_via_yaml_loader(self, tmp_path: Path):
        agents_yaml = {
            "version": "1",
            "mcp_gateway": {
                "tools": {
                    "orchid_ask": {
                        "title": "Ask the Acme Knowledge Base",
                        "description": "Route a question to the agents.",
                    },
                },
                "prompts": [
                    {
                        "name": "hello",
                        "description": "Say hello",
                        "template": "Hello, {{who}}.",
                        "arguments": [{"name": "who", "required": True}],
                    },
                ],
            },
        }
        path = tmp_path / "agents.yaml"
        path.write_text(yaml.safe_dump(agents_yaml))

        cfg = load_config(str(path))
        assert cfg.mcp_gateway.tools["orchid_ask"].title == "Ask the Acme Knowledge Base"
        assert cfg.mcp_gateway.prompts[0].template == "Hello, {{who}}."
        assert cfg.mcp_gateway.prompts[0].arguments[0].name == "who"
