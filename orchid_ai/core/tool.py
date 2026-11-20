from __future__ import annotations

import contextvars
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

# ORCHID-MEM: 01a1f39d-8002-7da9-8619-f10930c5b76b — Per-tool HITL approval marker; the agentic loop sets the invoked tool name only after an approved interrupt.
_APPROVAL_VAR: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "orchid.tool_approval_granted",
    default=frozenset(),
)


def tool_approval_granted(tool_name: str) -> bool:
    """True while the agentic loop is invoking this exact approved tool."""
    return tool_name in _APPROVAL_VAR.get()


def set_tool_approval(tool_name: str) -> contextvars.Token[frozenset[str]]:
    """Mark one tool name as approved for the current invocation."""
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError("tool_name must be a non-empty string")
    return _APPROVAL_VAR.set(_APPROVAL_VAR.get() | {tool_name})


def reset_tool_approval(token: contextvars.Token[frozenset[str]]) -> None:
    """Restore the approval marker captured by :func:`set_tool_approval`."""
    _APPROVAL_VAR.reset(token)


@dataclass(frozen=True)
class OrchidToolInput:
    """Input supplied to :meth:`OrchidTool.invoke`."""

    parameters: dict[str, Any] = field(default_factory=dict)
    query: str | None = None
    context: dict[str, Any] | None = None
    auth_context: Any = None
    content_sources: Any = None


@dataclass
class OrchidToolOutput:
    """Result returned by :meth:`OrchidTool.invoke`."""

    result: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class OrchidTool(ABC):
    """Unified contract for framework tools."""

    name: str = ""
    description: str = ""
    parameters_schema: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

    requires_approval: bool = False
    parallel_safe: bool = False
    inject_to_rag: bool = False
    rag_ttl: int | None = None
    rag_overrides: Any = None

    @abstractmethod
    async def invoke(self, tool_input: OrchidToolInput) -> OrchidToolOutput:
        """Execute the tool."""
        ...

    def get_parameters_schema(self) -> dict[str, Any]:
        """Return a copy of the JSON Schema for this tool's parameters."""
        schema = self.parameters_schema or {"type": "object", "properties": {}}
        return copy.deepcopy(schema)

    def get_llm_function_schema(self) -> dict[str, Any]:
        """Return the OpenAI function-calling schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_parameters_schema(),
            },
        }

    def to_litellm_tool_def(self) -> dict[str, Any]:
        """Alias for :meth:`get_llm_function_schema`."""
        return self.get_llm_function_schema()
