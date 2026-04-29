"""
YAML-driven agent configuration — ADR-016.

Public API:
  load_config(path)  → OrchidAgentsConfig (validated Pydantic model)
  register / get_class  → agent class registry
"""

from __future__ import annotations

from .loader import load_config
from .mcp_gateway import (
    OrchidMCPGatewayConfig,
    OrchidMCPGatewayPrompt,
    OrchidMCPGatewayPromptArgument,
    OrchidMCPGatewayToolOverride,
)
from .registry import get_class, register
from .schema import (
    OrchidAgentConfig,
    OrchidAgentSkillConfig,
    OrchidAgentsConfig,
    OrchidBuiltinToolConfig,
    OrchidOrchestratorSkillConfig,
)
from .tool_registry import call_tool, load_tools_from_config, register_tool

__all__ = [
    "OrchidAgentsConfig",
    "OrchidAgentConfig",
    "OrchidAgentSkillConfig",
    "OrchidBuiltinToolConfig",
    "OrchidMCPGatewayConfig",
    "OrchidMCPGatewayPrompt",
    "OrchidMCPGatewayPromptArgument",
    "OrchidMCPGatewayToolOverride",
    "OrchidOrchestratorSkillConfig",
    "load_config",
    "register",
    "get_class",
    "register_tool",
    "call_tool",
    "load_tools_from_config",
]
