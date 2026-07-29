from __future__ import annotations

import copy
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType, NoneType, UnionType
from typing import Any, Union, get_args, get_origin

from ..core.tool import OrchidTool

logger = logging.getLogger(__name__)

_FRAMEWORK_PARAMS = frozenset(
    {"kwargs", "self", "cls", "query", "context", "auth_context", "_kwargs", "content_sources"}
)

_ANNOTATION_TO_SCHEMA_TYPE: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    bytes: "string",
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "bytes": "string",
}

_PARAMETER_TYPE_TO_SCHEMA_TYPE: dict[str, str] = {
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
}

_SCHEMA_TYPE_TO_PARAMETER_TYPE: dict[str, str] = {
    "string": "string",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
}

_TYPE_COERCE: dict[str, Callable[[Any], Any]] = {
    "int": int,
    "integer": int,
    "float": float,
    "number": float,
    "bool": lambda v: v if isinstance(v, bool) else str(v).lower() in ("true", "1", "yes"),
    "boolean": lambda v: v if isinstance(v, bool) else str(v).lower() in ("true", "1", "yes"),
}

_TYPE_CHECK: dict[str, type] = {
    "int": int,
    "integer": int,
    "float": float,
    "number": float,
    "bool": bool,
    "boolean": bool,
    "string": str,
}


@dataclass(frozen=True)
class ToolParameter:
    """Metadata for one tool parameter."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None


class OrchidToolRegistry:
    """Framework-level registry of OrchidTool instances."""

    def __init__(self) -> None:
        self._tools: dict[str, OrchidTool] = {}

    def register(self, tool: OrchidTool) -> None:
        """Register a tool instance by ``tool.name``.

        If a tool with the same name is already registered, the new instance
        overwrites it and a warning is logged.
        """
        if not tool.name:
            raise ValueError("Tool must define a non-empty name before registration")
        if tool.name in self._tools:
            logger.warning(
                "[ToolRegistry] Tool '%s' already registered; overwriting with new instance",
                tool.name,
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> OrchidTool:
        """Look up a tool by name."""
        if name not in self._tools:
            raise KeyError(f"Built-in tool '{name}' is not registered. Available: {list(self._tools.keys())}")
        return self._tools[name]

    def get_all(self) -> MappingProxyType:
        """Return a read-only view of the registry."""
        return MappingProxyType(dict(self._tools))

    def unregister(self, name: str) -> None:
        """Remove a tool if present."""
        self._tools.pop(name, None)

    def clear(self) -> None:
        """Remove all tools."""
        self._tools.clear()

    @staticmethod
    def _auto_extract_schema(fn: Callable[..., Any]) -> dict[str, Any]:
        """Build a JSON Schema object from a callable signature."""
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return {"type": "object", "properties": {}}

        docstring = inspect.getdoc(fn) or ""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in _FRAMEWORK_PARAMS:
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue

            schema_type = _annotation_to_schema_type(param.annotation)
            prop: dict[str, Any] = {"type": schema_type}

            description = find_param_doc(docstring, param_name)
            if description:
                prop["description"] = description

            if param.default is inspect.Parameter.empty:
                required.append(param_name)
            else:
                prop["default"] = param.default

            properties[param_name] = prop

        schema: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema


def _annotation_to_schema_type(annotation: Any) -> str:
    if annotation in _ANNOTATION_TO_SCHEMA_TYPE:
        return _ANNOTATION_TO_SCHEMA_TYPE[annotation]

    origin = get_origin(annotation)
    if origin in (list, tuple, set, frozenset):
        return "array"
    if origin is dict:
        return "object"
    if origin in (UnionType, Union):
        non_none = [arg for arg in get_args(annotation) if arg is not NoneType]
        if len(non_none) == 1:
            return _annotation_to_schema_type(non_none[0])

    return "string"


def parameters_to_json_schema(parameters: dict[str, ToolParameter] | None) -> dict[str, Any] | None:
    """Convert compatibility ``ToolParameter`` metadata into JSON Schema."""
    if parameters is None:
        return None

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in parameters.items():
        prop: dict[str, Any] = {
            "type": _PARAMETER_TYPE_TO_SCHEMA_TYPE.get(param.type.lower(), param.type.lower()),
        }
        if param.description:
            prop["description"] = param.description
        if param.default is not None:
            prop["default"] = param.default
        properties[name] = prop
        if param.required:
            required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def schema_to_parameters(schema: dict[str, Any] | None) -> dict[str, ToolParameter]:
    """Convert a JSON Schema object into compatibility ``ToolParameter`` metadata."""
    if not schema:
        return {}

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    parameters: dict[str, ToolParameter] = {}

    for name, prop in properties.items():
        if not isinstance(prop, dict):
            continue
        schema_type = str(prop.get("type", "string")).lower()
        parameters[name] = ToolParameter(
            name=name,
            type=_SCHEMA_TYPE_TO_PARAMETER_TYPE.get(schema_type, schema_type),
            description=str(prop.get("description", "")),
            required=name in required,
            default=prop.get("default"),
        )

    return parameters


def coerce_parameters_from_schema(parameters: dict[str, Any], schema: dict[str, Any] | None) -> dict[str, Any]:
    """Coerce declared parameters to their schema types and preserve extras."""
    if not schema:
        return dict(parameters)

    properties = schema.get("properties", {})
    if not properties:
        return dict(parameters)

    required = set(schema.get("required", []))
    coerced: dict[str, Any] = {}

    for name, prop in properties.items():
        if not isinstance(prop, dict):
            continue
        if name not in parameters:
            if name in required:
                continue
            if "default" in prop:
                coerced[name] = prop["default"]
            continue
        coerced[name] = _coerce_param(name, parameters[name], str(prop.get("type", "string")))

    for name, value in parameters.items():
        if name not in coerced:
            coerced[name] = value

    return coerced


def filter_to_schema(parameters: dict[str, Any], schema: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only arguments declared in ``schema``; empty schema means unrestricted."""
    if not schema:
        return dict(parameters)
    properties = schema.get("properties", {})
    if not properties:
        return dict(parameters)
    allowed = set(properties.keys())
    return {name: value for name, value in parameters.items() if name in allowed}


def filter_to_signature(values: dict[str, Any], signature: inspect.Signature) -> dict[str, Any]:
    """Filter a kwargs dict to the parameters accepted by ``signature``."""
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return dict(values)

    accepted: dict[str, Any] = {}
    for name, param in signature.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name in values:
            accepted[name] = values[name]
    return accepted


def find_param_doc(docstring: str, param_name: str) -> str:
    """Extract a parameter description from a NumPy- or Google-style docstring."""
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
        if not in_params:
            continue
        if stripped.startswith(param_name) and (":" in stripped or stripped == param_name):
            if ":" in stripped:
                after_colon = stripped.split(":", 1)[1].strip()
                if after_colon and " " in after_colon:
                    return after_colon
            found_param = True
            continue
        if found_param:
            if stripped and (not stripped[0].isalpha() or line.startswith("    ")):
                return stripped
            break
    return ""


def _coerce_param(name: str, value: Any, declared_type: str) -> Any:
    if value is None:
        return None

    declared = declared_type.lower()
    expected = _TYPE_CHECK.get(declared)
    if expected is not None and isinstance(value, expected):
        return value

    coercer = _TYPE_COERCE.get(declared)
    if coercer is None:
        return value

    try:
        return coercer(value)
    except (TypeError, ValueError):
        return value


def clone_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    """Return a defensive copy of a tool schema."""
    if not schema:
        return {"type": "object", "properties": {}}
    return copy.deepcopy(schema)
