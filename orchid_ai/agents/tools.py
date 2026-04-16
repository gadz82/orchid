"""
LangChain ``BaseTool`` wrappers for MCP and built-in tools.

These wrappers allow the agentic loop to use LangGraph's ``ToolNode``
for uniform tool dispatch, improving code clarity and LangSmith tracing.

Auth is baked into each wrapper at construction time (per-request
creation).  This avoids threading auth through the LangChain Runnable
config, which would be fragile and non-standard.

Example::

    from orchid_ai.agents.tools import build_langchain_tools

    tools = build_langchain_tools(
        builtin_names={"search_players"},
        mcp_caps=caps,
        auth=auth_context,
        agent_name="basketball",
    )
    tool_node = ToolNode(tools)
    result = await tool_node.ainvoke({"messages": [ai_msg]})
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import BaseTool

from ..core.mcp import MCPToolCaller
from ..core.state import AuthContext

logger = logging.getLogger(__name__)


class MCPToolWrapper(BaseTool):
    """Wraps an MCP tool as a LangChain ``BaseTool``.

    Auth is captured at construction time so ``ToolNode`` can dispatch
    without custom config wiring.
    """

    mcp_client: Any  # MCPToolCaller — Any to avoid Pydantic ABC issues
    auth: Any  # AuthContext
    agent_name: str = ""
    requires_approval: bool = False  # HITL: pause and ask user before executing

    class Config:
        arbitrary_types_allowed = True

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError("MCPToolWrapper is async-only; use ainvoke()")

    async def _arun(self, **kwargs: Any) -> str:
        try:
            result = await self.mcp_client.call_tool(self.name, kwargs, self.auth)
            text = result.text
            if result.is_error:
                text = f"[Tool error] {text}"
                logger.warning("[%s] MCP tool '%s' returned error: %s", self.agent_name, self.name, text[:300])
            return text
        except Exception as exc:
            logger.error(
                "[%s] MCP tool '%s' exception: %s",
                self.agent_name,
                self.name,
                exc,
                exc_info=True,
            )
            return f"[Tool error] {exc}"


class BuiltinToolWrapper(BaseTool):
    """Wraps a registered built-in Python tool as a LangChain ``BaseTool``.

    Auth is captured at construction time.  The tool handler is called
    via the tool registry's ``call_tool()`` function.
    """

    auth: Any  # AuthContext
    agent_name: str = ""
    requires_approval: bool = False  # HITL: pause and ask user before executing

    class Config:
        arbitrary_types_allowed = True

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError("BuiltinToolWrapper is async-only; use ainvoke()")

    async def _arun(self, **kwargs: Any) -> str:
        from ..config.tool_registry import call_tool

        try:
            result = await call_tool(self.name, auth_context=self.auth, **kwargs)
            return json.dumps(result, default=str) if not isinstance(result, str) else result
        except Exception as exc:
            logger.error(
                "[%s] Built-in tool '%s' exception: %s",
                self.agent_name,
                self.name,
                exc,
                exc_info=True,
            )
            return f"[Tool error] {exc}"


def build_langchain_tools(
    *,
    builtin_names: set[str],
    builtin_tool_defs: list[dict[str, Any]],
    mcp_tool_defs: list[dict[str, Any]],
    mcp_tool_client_map: dict[str, tuple[MCPToolCaller, Any]],
    auth: AuthContext,
    agent_name: str = "",
    approval_tools: set[str] | None = None,
) -> list[BaseTool]:
    """Build a list of LangChain ``BaseTool`` instances for ToolNode dispatch.

    Parameters
    ----------
    builtin_names : set[str]
        Names of built-in tools available to this agent.
    builtin_tool_defs : list[dict]
        OpenAI-format tool definitions for built-in tools.
    mcp_tool_defs : list[dict]
        OpenAI-format tool definitions for MCP tools.
    mcp_tool_client_map : dict
        Maps tool name -> (MCPToolCaller, MCPServerConfig).
    auth : AuthContext
        Auth context baked into each wrapper.
    agent_name : str
        Agent name for logging.
    approval_tools : set[str] | None
        Tool names that require human approval before execution (HITL).

    Returns
    -------
    list[BaseTool]
        Ready-to-use tool instances for ``ToolNode(tools)``.
    """
    _approval = approval_tools or set()
    tools: list[BaseTool] = []

    # Built-in tools
    for tool_def in builtin_tool_defs:
        fn = tool_def.get("function", {})
        name = fn.get("name", "")
        desc = fn.get("description", name)
        schema = fn.get("parameters", {"type": "object", "properties": {}})

        tool = BuiltinToolWrapper(
            name=name,
            description=desc,
            args_schema=None,
            auth=auth,
            agent_name=agent_name,
            requires_approval=name in _approval,
        )
        # Set the JSON schema directly for tool binding
        tool.args_schema = None
        tool.metadata = {"json_schema": schema}
        tools.append(tool)

    # MCP tools
    for tool_def in mcp_tool_defs:
        fn = tool_def.get("function", {})
        name = fn.get("name", "")
        desc = fn.get("description", name)

        client_info = mcp_tool_client_map.get(name)
        if not client_info:
            continue

        client, _server_cfg = client_info
        tool = MCPToolWrapper(
            name=name,
            description=desc,
            args_schema=None,
            mcp_client=client,
            auth=auth,
            agent_name=agent_name,
            requires_approval=name in _approval,
        )
        tools.append(tool)

    return tools
