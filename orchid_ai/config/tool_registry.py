"""Built-in tool registry backed by the OrchidTool abstraction."""

from __future__ import annotations

import importlib
import inspect
import logging
from typing import Any, Callable

from ..core.tool import OrchidTool, OrchidToolInput
from ..tools.function_tool import FunctionTool
from ..tools.registry import (
    ToolParameter,
    OrchidToolRegistry,
    clone_schema,
    coerce_parameters_from_schema,
    filter_to_schema,
    find_param_doc,
    parameters_to_json_schema,
    schema_to_parameters,
)

logger = logging.getLogger(__name__)

TOOL_REGISTRY = OrchidToolRegistry()

_FRAMEWORK_PARAMS = frozenset(
    {"kwargs", "self", "cls", "query", "context", "auth_context", "_kwargs", "content_sources"}
)


def register_tool(
    name: str | OrchidTool,
    handler: Callable[..., Any] | OrchidTool | None = None,
    description: str = "",
    parameters: dict[str, ToolParameter] | None = None,
) -> None:
    """Register a built-in tool by name or tool instance."""
    tool = _coerce_registered_tool(name=name, handler=handler, description=description, parameters=parameters)
    TOOL_REGISTRY.register(tool)
    logger.debug("[ToolRegistry] registered '%s'", tool.name)


def get_tool(name: str) -> OrchidTool:
    """Look up a built-in tool by name."""
    return TOOL_REGISTRY.get(name)


def list_tools() -> list[OrchidTool]:
    """Return all registered built-in tools."""
    return list(TOOL_REGISTRY.get_all().values())


def clear() -> None:
    """Clear the registry (useful for testing)."""
    TOOL_REGISTRY.clear()


def unregister(name: str) -> None:
    """Remove one tool from the registry."""
    TOOL_REGISTRY.unregister(name)


async def call_tool(name: str, **kwargs: Any) -> Any:
    """Call a registered built-in tool by name and return its primary result."""
    tool = get_tool(name)
    tool_input = build_tool_input(tool, **kwargs)
    output = await tool.invoke(tool_input)
    return output.result


def build_tool_input(tool: OrchidTool, **kwargs: Any) -> OrchidToolInput:
    """Split framework kwargs from business parameters and coerce by schema."""
    framework: dict[str, Any] = {}
    parameters: dict[str, Any] = {}
    schema_properties = set(tool.get_parameters_schema().get("properties", {}).keys())

    for key, value in kwargs.items():
        if key in _FRAMEWORK_PARAMS:
            framework[key] = value
            if key in schema_properties:
                parameters[key] = value
        else:
            parameters[key] = value

    coerced = coerce_parameters_from_schema(parameters, tool.get_parameters_schema())
    return OrchidToolInput(
        parameters=coerced,
        query=framework.get("query"),
        context=framework.get("context"),
        auth_context=framework.get("auth_context"),
        content_sources=framework.get("content_sources"),
    )


def _coerce_registered_tool(
    *,
    name: str | OrchidTool,
    handler: Callable[..., Any] | OrchidTool | None,
    description: str,
    parameters: dict[str, ToolParameter] | None,
) -> OrchidTool:
    if isinstance(name, OrchidTool):
        if handler is not None:
            raise ValueError("When registering a tool instance, do not also pass 'handler'")
        tool = name
        # Instance passed directly — do NOT mutate it.
        # Use the instance's own name/description/schema as-is.
        if not tool.name:
            raise ValueError("Tool instance must have a non-empty name")
    elif isinstance(handler, OrchidTool):
        tool = handler
        if tool.name and tool.name != str(name):
            logger.error(
                "[ToolRegistry] Tool instance already named '%s'; re-registering as '%s' — "
                "the second registration will overwrite the first",
                tool.name,
                name,
            )
        # Instance passed as handler — do NOT mutate it.
        # The registry will map it under the requested ``name`` key.
    else:
        if handler is None:
            raise ValueError("register_tool() requires either a tool instance or a callable handler")
        schema = parameters_to_json_schema(parameters) if parameters is not None else None
        tool = FunctionTool(
            handler,
            name=str(name),
            description=description,
            parameters_schema=schema,
        )
    return tool


def _resolve_handler(handler_path: str) -> Callable[..., Any]:
    """Resolve a dotted Python path to a callable."""
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


def _resolve_tool_class(class_path: str) -> type[OrchidTool]:
    """Resolve a dotted Python path to an OrchidTool subclass."""
    module_path, _, attr_name = class_path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid class path '{class_path}': must be a dotted path like 'module.ToolClass'")
    module = importlib.import_module(module_path)
    tool_cls = getattr(module, attr_name, None)
    if tool_cls is None:
        raise AttributeError(f"Module '{module_path}' has no attribute '{attr_name}'")
    if not inspect.isclass(tool_cls):
        raise TypeError(f"'{class_path}' resolved to a non-class: {type(tool_cls)}")
    if not issubclass(tool_cls, OrchidTool):
        raise TypeError(f"'{class_path}' must be a subclass of OrchidTool")
    return tool_cls


def _extract_parameters_from_handler(handler: Callable[..., Any]) -> dict[str, ToolParameter]:
    """Compatibility wrapper that exposes signature-derived parameters."""
    return schema_to_parameters(OrchidToolRegistry._auto_extract_schema(handler))


def load_tools_from_config(tools_config: dict[str, Any]) -> None:
    """Resolve and register all tools declared in the config ``tools:`` block."""
    for tool_name, tool_cfg in tools_config.items():
        class_path = _config_value(tool_cfg, "class_") or _config_value(tool_cfg, "class")
        handler_path = _config_value(tool_cfg, "handler")

        if tool_name in TOOL_REGISTRY.get_all():
            TOOL_REGISTRY.unregister(tool_name)

        try:
            tool = _tool_from_config(tool_name, tool_cfg, class_path=class_path, handler_path=handler_path)
            TOOL_REGISTRY.register(tool)
        except Exception as exc:
            logger.error(
                "[ToolRegistry] Failed to resolve tool '%s' (class=%s, handler=%s): %s",
                tool_name,
                class_path,
                handler_path,
                exc,
            )
            raise


def _tool_from_config(
    tool_name: str,
    tool_cfg: Any,
    *,
    class_path: str | None,
    handler_path: str | None,
) -> OrchidTool:
    description = _config_value(tool_cfg, "description")
    explicit = _config_explicit_fields(tool_cfg)
    yaml_params = _get_yaml_parameters(tool_cfg)
    schema_override = parameters_to_json_schema(yaml_params)

    if class_path:
        tool_cls = _resolve_tool_class(class_path)
        tool = tool_cls()
    elif handler_path:
        handler = _resolve_handler(handler_path)
        tool = FunctionTool(handler, name=tool_name, description=description or "", parameters_schema=schema_override)
    else:
        raise ValueError("Tool must declare either 'class' or 'handler'")

    tool.name = tool_name
    if "description" in explicit and description:
        tool.description = description
    if schema_override is not None:
        tool.parameters_schema = clone_schema(schema_override)

    _apply_runtime_overrides(tool, tool_cfg, explicit)
    return tool


def _apply_runtime_overrides(tool: OrchidTool, tool_cfg: Any, explicit: set[str]) -> None:
    if "requires_approval" in explicit:
        tool.requires_approval = bool(_config_value(tool_cfg, "requires_approval"))
    if "parallel_safe" in explicit:
        tool.parallel_safe = bool(_config_value(tool_cfg, "parallel_safe"))
    if "inject_to_rag" in explicit:
        tool.inject_to_rag = bool(_config_value(tool_cfg, "inject_to_rag"))
    if "rag_ttl" in explicit:
        tool.rag_ttl = _config_value(tool_cfg, "rag_ttl")
    if "rag" in explicit:
        tool.rag_overrides = _config_value(tool_cfg, "rag")


def _config_value(tool_cfg: Any, key: str) -> Any:
    if hasattr(tool_cfg, key):
        return getattr(tool_cfg, key)
    if isinstance(tool_cfg, dict):
        return tool_cfg.get(key)
    return None


def _config_explicit_fields(tool_cfg: Any) -> set[str]:
    if isinstance(tool_cfg, dict):
        return set(tool_cfg.keys())
    return set(getattr(tool_cfg, "model_fields_set", set()))


def _get_yaml_parameters(tool_cfg: Any) -> dict[str, ToolParameter] | None:
    """Extract YAML-declared parameters from a tool config, if any."""
    raw_params = getattr(tool_cfg, "parameters", None)
    if raw_params is None and isinstance(tool_cfg, dict):
        raw_params = tool_cfg.get("parameters")
    raw_params = raw_params or {}
    if not raw_params:
        return None

    params: dict[str, ToolParameter] = {}
    for param_name, param_cfg in raw_params.items():
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
    return params or None


__all__ = [
    "TOOL_REGISTRY",
    "ToolParameter",
    "build_tool_input",
    "call_tool",
    "clear",
    "filter_to_schema",
    "find_param_doc",
    "get_tool",
    "list_tools",
    "load_tools_from_config",
    "register_tool",
    "schema_to_parameters",
    "unregister",
]
