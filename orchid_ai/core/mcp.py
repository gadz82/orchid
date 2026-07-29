"""Umbrella module that re-exports the MCP client surface.

The MCP types live in five themed sibling modules — tool-call results,
error types, segregated abstract interfaces, per-server DCR
registrations, and per-user OAuth tokens.  This file is the single
import path (``orchid_ai.core.mcp``) every consumer uses.

Themed modules:
  - :mod:`mcp_result` — :class:`OrchidMCPToolResult`
  - :mod:`mcp_errors` — :class:`OrchidMCPAuthRequiredError`,
                       :class:`OrchidMCPDiscoveryError`
  - :mod:`mcp_interfaces` — :class:`OrchidMCPToolCaller`,
                           :class:`OrchidMCPDiscoverable`,
                           :class:`OrchidMCPClient`
  - :mod:`mcp_registration` — per-server DCR record + store ABC
  - :mod:`mcp_tokens` — per-user token record + store ABC
"""

from __future__ import annotations

from .mcp_errors import OrchidMCPAuthRequiredError, OrchidMCPDiscoveryError
from .mcp_interfaces import OrchidCacheableMCPClient, OrchidMCPClient, OrchidMCPDiscoverable, OrchidMCPToolCaller
from .mcp_registration import OrchidMCPClientRegistration, OrchidMCPClientRegistrationStore
from .mcp_result import OrchidMCPToolResult
from .mcp_tokens import OrchidMCPTokenRecord, OrchidMCPTokenStore

__all__ = [
    "OrchidCacheableMCPClient",
    "OrchidMCPAuthRequiredError",
    "OrchidMCPClient",
    "OrchidMCPClientRegistration",
    "OrchidMCPClientRegistrationStore",
    "OrchidMCPDiscoverable",
    "OrchidMCPDiscoveryError",
    "OrchidMCPTokenRecord",
    "OrchidMCPTokenStore",
    "OrchidMCPToolCaller",
    "OrchidMCPToolResult",
]
