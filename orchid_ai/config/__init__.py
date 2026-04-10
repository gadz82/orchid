"""
YAML-driven agent configuration — ADR-016.

Public API:
  load_config(path)  → AgentsConfig (validated Pydantic model)
  register / get_class  → agent class registry
"""

from .loader import load_config
from .registry import get_class, register
from .schema import (
    AgentConfig,
    AgentSkillConfig,
    AgentsConfig,
    BuiltinToolConfig,
    OrchestratorSkillConfig,
)
from .tool_registry import call_tool, load_tools_from_config, register_tool

__all__ = [
    "AgentsConfig",
    "AgentConfig",
    "AgentSkillConfig",
    "BuiltinToolConfig",
    "OrchestratorSkillConfig",
    "load_config",
    "register",
    "get_class",
    "register_tool",
    "call_tool",
    "load_tools_from_config",
]
