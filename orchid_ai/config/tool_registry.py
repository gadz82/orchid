"""
Built-in tool registry — maps YAML ``tools`` declarations to Python callables.

Resolution:
  1. Short name in registry → pre-registered callable
  2. Dotted handler path → ``importlib.import_module`` + ``getattr``

Sync functions are automatically wrapped with ``asyncio.to_thread`` at call time.

Parameters:
  When a ``OrchidBuiltinToolConfig`` declares ``parameters``, they are stored
  alongside the handler.  When omitted, parameters are auto-extracted
  from the Python function signature via ``inspect``.  This metadata is
  used by the CLI skill generator to produce accurate Claude Code
  skill documentation and CLI wrappers.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolParameter:
    """Metadata for a single tool parameter."""

    name: str
    type: str = "string"  # string, int, float, bool
    description: str = ""
    required: bool = True
    default: Any = None


@dataclass(frozen=True)
class BuiltinToolEntry:
    """A registered built-in tool with optional parameter metadata."""

    name: str
    handler: Callable[..., Any]
    description: str
    parameters: dict[str, ToolParameter] = field(default_factory=dict)


_REGISTRY: dict[str, BuiltinToolEntry] = {}


def register_tool(
    name: str,
    handler: Callable[..., Any],
    description: str = "",
    parameters: dict[str, ToolParameter] | None = None,
) -> None:
    """Register a built-in tool by short name.

    When ``parameters`` is ``None``, they are auto-extracted from the
    handler's function signature via ``inspect``.
    """
    params = parameters if parameters is not None else _extract_parameters_from_handler(handler)
    _REGISTRY[name] = BuiltinToolEntry(name=name, handler=handler, description=description, parameters=params)
    logger.debug("[ToolRegistry] registered '%s' (%d params)", name, len(params))


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


_ANNOTATION_TO_TYPE: dict[Any, str] = {
    str: "string",
    int: "int",
    float: "float",
    bool: "bool",
    "str": "string",
    "int": "int",
    "float": "float",
    "bool": "bool",
}

# Parameters that GenericAgent injects automatically — skip these
# when auto-extracting from function signatures.
_FRAMEWORK_PARAMS = frozenset({"kwargs", "self", "cls", "query", "context", "auth_context", "_kwargs"})


def _extract_parameters_from_handler(handler: Callable[..., Any]) -> dict[str, ToolParameter]:
    """Auto-extract parameter metadata from a function's signature and docstring.

    Skips framework-injected parameters (``query``, ``context``,
    ``auth_context``, ``**kwargs``, etc.) so only the tool's
    "business" parameters are returned.
    """
    try:
        sig = inspect.signature(handler)
    except (ValueError, TypeError):
        return {}

    docstring = inspect.getdoc(handler) or ""
    params: dict[str, ToolParameter] = {}

    for param_name, param in sig.parameters.items():
        # Skip framework-injected and catch-all params
        if param_name in _FRAMEWORK_PARAMS:
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        # Determine type
        ann = param.annotation
        param_type = _ANNOTATION_TO_TYPE.get(ann, "string")

        # Determine required + default
        has_default = param.default is not inspect.Parameter.empty
        required = not has_default
        default = param.default if has_default else None

        # Try to find description in docstring
        desc = find_param_doc(docstring, param_name) or param_type

        params[param_name] = ToolParameter(
            name=param_name,
            type=param_type,
            description=desc,
            required=required,
            default=default,
        )

    return params


def find_param_doc(docstring: str, param_name: str) -> str:
    """Extract a parameter description from a NumPy/Google-style docstring.

    Handles both formats:
    - NumPy: ``param : str`` followed by an indented description line.
    - Google: ``param: description`` on the same line.

    Returns an empty string when the parameter is not documented.
    """
    lines = docstring.splitlines()
    in_params = False
    found_param = False
    for line in lines:
        stripped = line.strip()
        if stripped in ("Parameters", "Parameters:", "Args:", "Arguments:"):
            in_params = True
            continue
        if in_params and stripped.startswith("---"):
            continue
        if in_params and stripped in ("Returns", "Returns:", "Raises", "Raises:", "Yields", "Yields:"):
            break
        if in_params:
            # Look for "param_name : type" or "param_name:" line
            if stripped.startswith(param_name) and (":" in stripped or stripped == param_name):
                # Google-style: "param_name: description" on same line
                if ":" in stripped:
                    after_colon = stripped.split(":", 1)[1].strip()
                    # NumPy-style has type after colon (e.g. "param : str"), desc on next line
                    # Google-style has description (e.g. "param: Some description")
                    # Heuristic: if it contains spaces, it's a description, not a type
                    if after_colon and " " in after_colon:
                        return after_colon
                found_param = True
                continue
            if found_param:
                if stripped and (not stripped[0].isalpha() or line.startswith("    ")):
                    if stripped:
                        return stripped
                    break
                else:
                    break
    return ""


def load_tools_from_config(tools_config: dict[str, Any]) -> None:
    """
    Resolve all tool declarations from the YAML ``tools:`` section and register them.

    When a tool declares ``parameters`` in YAML, they take precedence
    over auto-extraction from the function signature.

    Parameters
    ----------
    tools_config : dict
        Mapping of tool name → ``OrchidBuiltinToolConfig``-like object with ``handler``, ``description``,
        and optional ``parameters``.
    """
    for tool_name, tool_cfg in tools_config.items():
        handler_path = tool_cfg.handler if hasattr(tool_cfg, "handler") else tool_cfg.get("handler", "")
        description = tool_cfg.description if hasattr(tool_cfg, "description") else tool_cfg.get("description", "")
        try:
            handler = _resolve_handler(handler_path)

            # Convert YAML-declared parameters to ToolParameter instances
            yaml_params = _get_yaml_parameters(tool_cfg)

            register_tool(tool_name, handler, description, parameters=yaml_params)
        except Exception as exc:
            logger.error("[ToolRegistry] Failed to resolve tool '%s' (handler=%s): %s", tool_name, handler_path, exc)
            raise


def _get_yaml_parameters(tool_cfg: Any) -> dict[str, ToolParameter] | None:
    """Extract YAML-declared parameters from a tool config, if any.

    Returns ``None`` when no parameters are declared (triggering
    auto-extraction from the function signature).
    """
    raw_params = getattr(tool_cfg, "parameters", None) or {}
    if not raw_params:
        return None  # trigger auto-extraction

    params: dict[str, ToolParameter] = {}
    for param_name, param_cfg in raw_params.items():
        # Support both Pydantic model and plain dict
        if hasattr(param_cfg, "type"):
            params[param_name] = ToolParameter(
                name=param_name,
                type=param_cfg.type,
                description=param_cfg.description,
                required=param_cfg.required,
                default=param_cfg.default,
            )
        elif isinstance(param_cfg, dict):
            params[param_name] = ToolParameter(
                name=param_name,
                type=param_cfg.get("type", "string"),
                description=param_cfg.get("description", ""),
                required=param_cfg.get("required", True),
                default=param_cfg.get("default"),
            )
    return params if params else None


def list_tools() -> list[BuiltinToolEntry]:
    """Return all registered built-in tools."""
    return list(_REGISTRY.values())


def clear() -> None:
    """Clear the registry (useful for testing)."""
    _REGISTRY.clear()
