"""Abstract interfaces for MCP clients.

Split into segregated ABCs so a caller that only needs to invoke tools
(``OrchidMCPToolCaller``) doesn't depend on capability discovery
(``OrchidMCPDiscoverable``). The combined ``OrchidMCPClient`` is the
canonical type for concrete clients that implement both halves
(e.g. :class:`StreamableHttpMCPClient`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .mcp_result import OrchidMCPToolResult
from .state import OrchidAuthContext


class OrchidMCPToolCaller(ABC):
    """Minimal interface for invoking MCP tools — most consumers only need this."""

    @abstractmethod
    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        auth: OrchidAuthContext,
    ) -> OrchidMCPToolResult:
        """Invoke a tool on the remote/local MCP server."""
        ...

    @property
    @abstractmethod
    def server_url(self) -> str:
        """Base URL of the MCP server (for logging/debugging)."""
        ...


class OrchidMCPDiscoverable(ABC):
    """Interface for discovering MCP server capabilities."""

    @abstractmethod
    async def list_tools(self, auth: OrchidAuthContext) -> list[dict[str, Any]]:
        """List available tools on the MCP server."""
        ...

    @abstractmethod
    async def list_prompts(self, auth: OrchidAuthContext) -> list[dict[str, Any]]:
        """List available prompts on the MCP server."""
        ...

    @abstractmethod
    async def list_resources(self, auth: OrchidAuthContext) -> list[dict[str, Any]]:
        """List available resources on the MCP server."""
        ...

    @abstractmethod
    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, str],
        auth: OrchidAuthContext,
    ) -> list[dict[str, Any]]:
        """Render a prompt template and return its messages."""
        ...

    @abstractmethod
    async def read_resource(self, uri: str, auth: OrchidAuthContext) -> str:
        """Read the content of a resource by URI."""
        ...


class OrchidMCPClient(OrchidMCPToolCaller, OrchidMCPDiscoverable, ABC):
    """
    Combined MCP client interface.

    Extends both ``OrchidMCPToolCaller`` (tool invocation) and ``OrchidMCPDiscoverable``
    (capability discovery).  Code that only needs tool calling should
    depend on ``OrchidMCPToolCaller`` instead for better interface segregation.
    """

    pass
