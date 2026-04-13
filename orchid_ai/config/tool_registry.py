"""
Built-in tool registry — maps YAML ``tools`` declarations to Python callables (ADR-017).

Resolution:
  1. Short name in registry → pre-registered callable
  2. Dotted handler path → ``importlib.import_module`` + ``getattr``

Sync functions are automatically wrapped with ``asyncio.to_thread`` at call time.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuiltinToolEntry:
    """A registered built-in tool."""

    name: str
    handler: Callable[..., Any]
    description: str


_REGISTRY: dict[str, BuiltinToolEntry] = {}


def register_tool(name: str, handler: Callable[..., Any], description: str = "") -> None:
    """Register a built-in tool by short name."""
    _REGISTRY[name] = BuiltinToolEntry(name=name, handler=handler, description=description)
    logger.debug("[ToolRegistry] registered '%s'", name)


def get_tool(name: str) -> BuiltinToolEntry:
    """Look up a built-in tool by name. Raises ``KeyError`` if not found."""
    if name not in _REGISTRY:
        raise KeyError(f"Built-in tool '{name}' is not registered. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


async def call_tool(name: str, **kwargs: Any) -> Any:
    """
    Call a registered built-in tool by name.

    If the handler is a coroutine function, it is awaited directly.
    Otherwise, it is run in a thread via ``asyncio.to_thread``.
    """
    entry = get_tool(name)
    if asyncio.iscoroutinefunction(entry.handler):
        return await entry.handler(**kwargs)
    return await asyncio.to_thread(entry.handler, **kwargs)


def _resolve_handler(handler_path: str) -> Callable[..., Any]:
    """
    Resolve a dotted Python path to a callable.

    Example: ``"myproject.tools.dates.format_date"`` →
    ``importlib.import_module("myproject.tools.dates").format_date``
    """
    module_path, _, attr_name = handler_path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid handler path '{handler_path}': must be a dotted path like 'module.function'")
    module = importlib.import_module(module_path)
    handler = getattr(module, attr_name, None)
    if handler is None:
        raise AttributeError(f"Module '{module_path}' has no attribute '{attr_name}'")
    if not callable(handler):
        raise TypeError(f"'{handler_path}' resolved to a non-callable: {type(handler)}")
    return handler


def load_tools_from_config(tools_config: dict[str, Any]) -> None:
    """
    Resolve all tool declarations from the YAML ``tools:`` section and register them.

    Parameters
    ----------
    tools_config : dict
        Mapping of tool name → ``BuiltinToolConfig``-like object with ``handler`` and ``description``.
    """
    for tool_name, tool_cfg in tools_config.items():
        handler_path = tool_cfg.handler if hasattr(tool_cfg, "handler") else tool_cfg.get("handler", "")
        description = tool_cfg.description if hasattr(tool_cfg, "description") else tool_cfg.get("description", "")
        try:
            handler = _resolve_handler(handler_path)
            register_tool(tool_name, handler, description)
        except Exception as exc:
            logger.error("[ToolRegistry] Failed to resolve tool '%s' (handler=%s): %s", tool_name, handler_path, exc)
            raise


def list_tools() -> list[BuiltinToolEntry]:
    """Return all registered built-in tools."""
    return list(_REGISTRY.values())


def clear() -> None:
    """Clear the registry (useful for testing)."""
    _REGISTRY.clear()
