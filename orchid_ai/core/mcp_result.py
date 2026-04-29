"""Normalised MCP tool-call result.

Kept in its own module so that callers needing only the value object can
depend on it without dragging in the abstract client interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OrchidMCPToolResult:
    """Normalised result from an MCP tool call."""

    content: list[dict[str, Any]] = field(default_factory=list)
    is_error: bool = False

    @property
    def text(self) -> str:
        """Convenience: concatenate all text content blocks."""
        return "\n".join(item.get("text", "") for item in self.content if item.get("type") == "text")
