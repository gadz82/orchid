"""
MCP client abstraction.

Concrete implementations live in agents/src/mcp/:
  - StreamableHttpMCPClient (production)
  - MockMCPClient (tests)

The agent never knows which transport is being used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .state import AuthContext


# ── Segregated Interfaces (ISP) ────────────────────────────


class MCPToolCaller(ABC):
    """Minimal interface for invoking MCP tools — most consumers only need this."""

    @abstractmethod
    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        auth: AuthContext,
    ) -> MCPToolResult:
        """Invoke a tool on the remote/local MCP server."""
        ...

    @property
    @abstractmethod
    def server_url(self) -> str:
        """Base URL of the MCP server (for logging/debugging)."""
        ...


class MCPDiscoverable(ABC):
    """Interface for discovering MCP server capabilities."""

    @abstractmethod
    async def list_tools(self, auth: AuthContext) -> list[dict[str, Any]]:
        """List available tools on the MCP server."""
        ...

    @abstractmethod
    async def list_prompts(self, auth: AuthContext) -> list[dict[str, Any]]:
        """List available prompts on the MCP server."""
        ...

    @abstractmethod
    async def list_resources(self, auth: AuthContext) -> list[dict[str, Any]]:
        """List available resources on the MCP server."""
        ...

    @abstractmethod
    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, str],
        auth: AuthContext,
    ) -> list[dict[str, Any]]:
        """Render a prompt template and return its messages."""
        ...

    @abstractmethod
    async def read_resource(self, uri: str, auth: AuthContext) -> str:
        """Read the content of a resource by URI."""
        ...


@dataclass
class MCPToolResult:
    """Normalised result from an MCP tool call."""

    content: list[dict[str, Any]] = field(default_factory=list)
    is_error: bool = False

    @property
    def text(self) -> str:
        """Convenience: concatenate all text content blocks."""
        return "\n".join(item.get("text", "") for item in self.content if item.get("type") == "text")


class MCPClient(MCPToolCaller, MCPDiscoverable, ABC):
    """
    Combined MCP client interface (backward compatible).

    Extends both ``MCPToolCaller`` (tool invocation) and ``MCPDiscoverable``
    (capability discovery).  New code that only needs tool calling should
    depend on ``MCPToolCaller`` instead for better interface segregation.
    """

    pass
