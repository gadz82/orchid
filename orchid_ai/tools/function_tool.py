from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from ..core.tool import OrchidTool, OrchidToolInput, OrchidToolOutput
from .registry import OrchidToolRegistry, clone_schema, filter_to_signature, schema_to_parameters

_FRAMEWORK_PARAMS = frozenset({"query", "context", "auth_context", "content_sources"})


class FunctionTool(OrchidTool):
    """Adapter that exposes a plain Python callable through the OrchidTool API."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str,
        description: str = "",
        parameters_schema: dict[str, Any] | None = None,
    ) -> None:
        self._fn = fn
        self.handler = fn
        self.name = name
        self.description = description
        self.parameters_schema = (
            clone_schema(parameters_schema)
            if parameters_schema is not None
            else OrchidToolRegistry._auto_extract_schema(fn)
        )

    @property
    def parameters(self) -> dict[str, Any]:
        """Compatibility view for legacy callers that expect parameter metadata."""
        return schema_to_parameters(self.get_parameters_schema())

    async def invoke(self, tool_input: OrchidToolInput) -> OrchidToolOutput:
        kwargs = {
            **tool_input.parameters,
            "query": tool_input.query,
            "context": tool_input.context,
            "auth_context": tool_input.auth_context,
            "content_sources": tool_input.content_sources,
        }
        try:
            signature = inspect.signature(self._fn)
            accepted = filter_to_signature(kwargs, signature)
        except (TypeError, ValueError):
            accepted = dict(kwargs)

        # When the handler has **kwargs, filter_to_signature returns ALL
        # kwargs including framework params.  Strip framework params that
        # the handler does NOT explicitly declare — framework params should
        # only reach the handler when it asks for them by name.
        has_var_keyword = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
        if has_var_keyword:
            explicit_params = set(signature.parameters.keys())
            accepted = {k: v for k, v in accepted.items() if k not in _FRAMEWORK_PARAMS or k in explicit_params}

        if inspect.iscoroutinefunction(self._fn):
            result = await self._fn(**accepted)
        else:
            result = await asyncio.to_thread(self._fn, **accepted)

        if inspect.isawaitable(result):
            result = await result

        return OrchidToolOutput(result=result)
