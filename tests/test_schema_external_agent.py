from __future__ import annotations

import pytest
import yaml

from orchid_ai.config.schema_external_agent import OrchidExternalAgentConfig


class TestOrchidExternalAgentConfig:
    def test_minimal_config_parsed(self) -> None:
        cfg = OrchidExternalAgentConfig.model_validate({"command": ["mycli"]})
        assert cfg.command == ["mycli"]
        assert cfg.args == []
        assert cfg.cwd == ""
        assert cfg.timeout == 600.0
        assert cfg.env == {}
        assert cfg.stdin_mode == "arg"
        assert cfg.normalizer == "passthrough"
        assert cfg.normalizer_instruction == ""
        assert cfg.normalizer_model == ""
        assert cfg.description == ""
        assert cfg.requires_approval is True
        assert cfg.parallel_safe is False
        assert cfg.inject_to_rag is False
        assert cfg.rag_ttl is None

    def test_full_config_parsed(self) -> None:
        cfg = OrchidExternalAgentConfig.model_validate(
            {
                "command": ["mycli", "--print"],
                "args": ["--verbose"],
                "cwd": "/tmp",
                "timeout": 60.0,
                "env": {"FOO": "bar"},
                "stdin_mode": "stdin",
                "normalizer": "llm",
                "normalizer_instruction": "Rewrite.",
                "normalizer_model": "gpt-4",
                "description": "A test tool.",
                "requires_approval": False,
                "parallel_safe": True,
                "inject_to_rag": True,
                "rag_ttl": 3600,
            }
        )
        assert cfg.command == ["mycli", "--print"]
        assert cfg.args == ["--verbose"]
        assert cfg.cwd == "/tmp"
        assert cfg.timeout == 60.0
        assert cfg.env == {"FOO": "bar"}
        assert cfg.stdin_mode == "stdin"
        assert cfg.normalizer == "llm"
        assert cfg.normalizer_instruction == "Rewrite."
        assert cfg.normalizer_model == "gpt-4"
        assert cfg.description == "A test tool."
        assert cfg.requires_approval is False
        assert cfg.parallel_safe is True
        assert cfg.inject_to_rag is True
        assert cfg.rag_ttl == 3600

    def test_command_is_required(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            OrchidExternalAgentConfig.model_validate({})

    def test_yaml_round_trip(self) -> None:
        yaml_text = """
command: ["mycli"]
description: "delegate tasks"
requires_approval: false
"""
        data = yaml.safe_load(yaml_text)
        cfg = OrchidExternalAgentConfig.model_validate(data)
        assert cfg.command == ["mycli"]
        assert cfg.description == "delegate tasks"
        assert cfg.requires_approval is False
