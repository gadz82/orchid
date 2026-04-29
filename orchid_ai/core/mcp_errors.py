"""MCP client error types.

Distinct from the abstract interfaces so error handling code can import
the exceptions without pulling in the full client surface.
"""

from __future__ import annotations


class OrchidMCPAuthRequiredError(Exception):
    """Raised when an MCP server requires OAuth but the user has not authorized.

    Caught at the fault-isolation boundaries in ``mcp_dispatcher.py``
    (the existing ``except Exception`` blocks) — one unauthorized server
    must not crash the entire agent.
    """

    def __init__(self, server_name: str) -> None:
        super().__init__(f"OAuth authorization required for MCP server '{server_name}'")
        self.server_name = server_name


class OrchidMCPDiscoveryError(Exception):
    """Raised when MCP authorization discovery (RFC 9728 / RFC 8414 / RFC 7591) fails.

    Signals a spec-compliance issue with the MCP server or its
    authorization server: missing ``WWW-Authenticate`` header on 401,
    malformed protected-resource metadata, auth-server metadata without
    a ``registration_endpoint``, DCR rejection, etc.  Distinct from
    :class:`OrchidMCPAuthRequiredError` which signals a completely
    normal "user hasn't authenticated yet" state.
    """

    def __init__(self, server_name: str, reason: str) -> None:
        super().__init__(f"MCP authorization discovery failed for '{server_name}': {reason}")
        self.server_name = server_name
        self.reason = reason
