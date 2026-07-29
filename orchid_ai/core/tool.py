from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar


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
