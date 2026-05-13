"""
Shared tool utilities used by both ``GenericAgent`` and ``mini_agent_node``.

Extracted from duplicate implementations to honour DIP — high-level code
depends on shared abstractions, not copied low-level logic.
"""

from __future__ import annotations

from typing import Any


def _tool_param_type_to_json_schema(t: str) -> str:
    """Map tool parameter types to JSON Schema types."""
    return {"int": "integer", "float": "number", "bool": "boolean"}.get(t, "string")


def tools_to_litellm_format(
    tool_names: list[str],
    *,
    skip_tools: set[str] | None = None,
) -> tuple[set[str], list[dict[str, Any]]]:
    """Convert registered built-in tools to litellm function-calling format.

    Returns ``(builtin_tool_names, litellm_tool_defs)``.
    """
    from ..config.tool_registry import get_tool

    names: set[str] = set()
    defs: list[dict[str, Any]] = []

    for tool_name in tool_names:
        if skip_tools and (tool_name in skip_tools or f"builtin_{tool_name}" in skip_tools):
            continue
        try:
            entry = get_tool(tool_name)
        except KeyError:
            continue

        names.add(tool_name)

        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in entry.parameters.values():
            prop: dict[str, str] = {
                "type": _tool_param_type_to_json_schema(p.type),
                "description": p.description,
            }
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        schema: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required

        defs.append(
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": entry.description,
                    "parameters": schema,
                },
            }
        )

    return names, defs


def resolve_parallel_safety(
    *,
    tool_map: dict[str, Any],
    builtin_tool_names: set[str],
    caps: Any,  # MCPCapabilities — typed loosely to avoid circular import
    parallel_tools_enabled: bool,
    approval_tools: set[str] | None = None,
    parallel_safe_builtin_tools: set[str] | None = None,
    mcp_parallel_overrides: dict[str, bool] | None = None,
) -> dict[str, bool] | None:
    """Resolve which tools may run in parallel within one round.

    Returns ``None`` when ``parallel_tools_enabled`` is ``False``.
    When ``True``, returns a per-tool-name bool computed via this precedence:

    1. ``requires_approval=True`` → ``False`` (HITL must serialise).
    2. Built-in tool → ``True`` iff it is in ``parallel_safe_builtin_tools``.
    3. MCP tool with explicit override → use it.
    4. MCP tool without override → ``True`` iff the server advertised
       ``readOnlyHint=True`` for that tool.
    """
    if not parallel_tools_enabled:
        return None

    approval_set = approval_tools or set()
    builtin_safe = parallel_safe_builtin_tools or set()
    overrides = mcp_parallel_overrides or {}

    safety: dict[str, bool] = {}
    for tool_name in tool_map:
        if tool_name in approval_set:
            safety[tool_name] = False
            continue
        if tool_name in builtin_tool_names:
            safety[tool_name] = tool_name in builtin_safe
            continue

        override = overrides.get(tool_name)
        if override is not None:
            safety[tool_name] = override
            continue

        annotations = caps.tool_annotations.get(tool_name) if caps else None
        safety[tool_name] = bool(annotations and annotations.read_only_hint is True)

    return safety
