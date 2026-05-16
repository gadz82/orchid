"""
YAML and Markdown-driven agent configuration.

Public API:
  load_config(path)        → OrchidAgentsConfig (YAML)
  load_md_config(root_dir) → OrchidAgentsConfig (Markdown)
  register / get_class     → agent class registry
"""

from __future__ import annotations

from .frontmatter import MarkdownFile, load_markdown_file, parse_frontmatter
from .loader import load_config
from .mcp_gateway import (
    OrchidMCPGatewayConfig,
    OrchidMCPGatewayPrompt,
    OrchidMCPGatewayPromptArgument,
    OrchidMCPGatewayToolOverride,
)
from .md_loader import load_md_config, md_infrastructure_to_env
from .registry import get_class, register
from .schema import (
    OrchidAgentConfig,
    OrchidAgentSkillConfig,
    OrchidAgentsConfig,
    OrchidBuiltinToolConfig,
    OrchidOrchestratorSkillConfig,
)
from .tool_registry import call_tool, load_tools_from_config, register_tool
from .watcher import (
    ConfigSnapshot,
    OrchidConfigSnapshot,
    OrchidConfigWatcher,
    OrchidConfigWatcherBase,
    OrchidYamlConfigWatcher,
    YamlConfigWatcher,
)

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
    "ConfigSnapshot",
    "MarkdownFile",
    "OrchidConfigSnapshot",
    "OrchidConfigWatcher",
    "OrchidConfigWatcherBase",
    "OrchidYamlConfigWatcher",
    "YamlConfigWatcher",
    "load_config",
    "load_markdown_file",
    "load_md_config",
    "md_infrastructure_to_env",
    "parse_frontmatter",
    "register",
    "get_class",
    "register_tool",
    "call_tool",
    "load_tools_from_config",
]
