"""
Tool call strategies for MCP server interaction — Strategy pattern (OCP).

Each strategy defines *how* tools on a single MCP server are invoked:
  - ``all``          — call every whitelisted tool, collect all results
  - ``sequential``   — call tools in order, chaining context forward
  - ``llm_decides``  — ask the LLM which tools to call and with what args

Adding a new strategy:
  1. Subclass ``OrchidToolCallStrategy``
  2. Register it in ``STRATEGY_REGISTRY``
  3. Reference by name in ``agents.yaml`` → ``tool_call_strategy: my_strategy``

.. note:: When strategies are used

   These strategies are executed by :meth:`MCPDispatcher.fetch`, which
   is called during **skill execution** (:class:`SkillExecutor`).  For
   regular (non-skill) queries, :class:`GenericAgent` uses its own
   agentic tool-calling loop (:meth:`GenericAgent._agentic_tool_loop`)
   that always uses "LLM decides" semantics via native ``tool_calls``.
   In that path, the ``tool_call_strategy`` YAML field is not
   consulted — the LLM directly selects which tools to call.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from ..config.schema import OrchidMCPServerConfig, OrchidToolConfig
from ..core.mcp import OrchidMCPToolCaller
from ..core.state import OrchidAuthContext

logger = logging.getLogger(__name__)


class OrchidToolCallStrategy(ABC):
    """Strategy for calling MCP tools on a single server."""

    @abstractmethod
    async def execute(
        self,
        client: OrchidMCPToolCaller,
        tools: list[OrchidToolConfig],
        query: str,
        auth: OrchidAuthContext,
        *,
        agent_name: str = "",
        server_config: OrchidMCPServerConfig | None = None,
        llm_model: str | None = None,
        chat_model: Any | None = None,
    ) -> dict[str, Any]:
        """
        Invoke tools according to this strategy's rules.

        Parameters
        ----------
        client : OrchidMCPClient
            The MCP client to call tools on.
        tools : list[OrchidToolConfig]
            Tools available for this invocation.
        query : str
            The user's query.
        auth : OrchidAuthContext
            Authentication context for tool calls.
        agent_name : str
            Name of the calling agent (for logging).
        server_config : OrchidMCPServerConfig | None
            Full server configuration (for strategies that need it).
        llm_model : str | None
            LLM model identifier (for strategies that use LLM).
        chat_model : Any | None
            LLM provider for strategies that need LLM (e.g. llm_decides).
        """
        ...


class CallAllStrategy(OrchidToolCallStrategy):
    """Call every tool in the list concurrently and collect results."""

    async def execute(
        self,
        client,
        tools,
        query,
        auth,
        *,
        agent_name="",
        **kwargs,
    ) -> dict[str, Any]:
        async def _call_one(tool):
            try:
                args = {"query": query, **tool.arguments}
                result = await client.call_tool(tool.name, args, auth)
                return tool.name, result.text
            except Exception as exc:
                # Fault-isolation boundary: MCP tool calls cross a
                # network/process boundary and can raise arbitrary
                # exceptions (McpError, httpx.HTTPStatusError,
                # AttributeError from buggy implementations, etc.).
                # We catch broadly here to prevent one failing tool
                # from crashing the entire agent.
                logger.error("[%s] Tool '%s' failed: %s", agent_name, tool.name, exc)
                return f"{tool.name}_error", str(exc)

        pairs = await asyncio.gather(*(_call_one(t) for t in tools))
        return dict(pairs)


class SequentialStrategy(OrchidToolCallStrategy):
    """Call tools in order, passing previous results as context."""

    async def execute(
        self,
        client,
        tools,
        query,
        auth,
        *,
        agent_name="",
        **kwargs,
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        previous_results: dict[str, Any] = {}

        for tool in tools:
            try:
                args: dict[str, Any] = {"query": query, **tool.arguments}
                if previous_results:
                    args["previous_results"] = json.dumps(previous_results, default=str)
                result = await client.call_tool(tool.name, args, auth)
                results[tool.name] = result.text
                previous_results[tool.name] = result.text
            except Exception as exc:
                logger.error("[%s] Sequential tool '%s' failed: %s", agent_name, tool.name, exc)
                results[f"{tool.name}_error"] = str(exc)

        return results


class LLMDecidesStrategy(OrchidToolCallStrategy):
    """Ask the LLM which tools to call and with what arguments."""

    async def execute(
        self,
        client,
        tools,
        query,
        auth,
        *,
        agent_name="",
        server_config=None,
        llm_model=None,
        chat_model=None,
        **kwargs,
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}

        # Discover available tool descriptions from the server
        try:
            available_tools = await client.list_tools(auth)
        except Exception as exc:
            logger.warning("[%s] Could not list tools: %s", agent_name, exc)
            available_tools = []

        # Filter to whitelisted tools (skip filter when discover_all / wildcard)
        discover_all = server_config.discover_all if server_config else False
        if discover_all:
            relevant_tools = available_tools
        else:
            whitelist = {t.name for t in tools}
            relevant_tools = [t for t in available_tools if t["name"] in whitelist]

        if not relevant_tools:
            return results

        # Ask the LLM which tools to call
        tool_descriptions = "\n".join(
            f"- {t['name']}: {t.get('description', 'No description')}" for t in relevant_tools
        )

        decision_prompt = (
            f"User query: {query}\n\n"
            f"Available tools:\n{tool_descriptions}\n\n"
            "Decide which tools to call and with what arguments. "
            "Respond with a JSON array of objects: "
            '[{"tool": "tool_name", "arguments": {...}}, ...]\n'
            "Only include tools that are relevant to the query. "
            "Respond ONLY with the JSON array."
        )

        _model = llm_model or "ollama/llama3.2"

        try:
            raw = await self._llm_complete(
                chat_model,
                _model,
                [{"role": "user", "content": decision_prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
        except Exception:
            logger.exception("[%s] LLM API error during tool decision", agent_name)
            # Fallback: call all tools
            return await get_strategy("all").execute(client, tools, query, auth, agent_name=agent_name)

        try:
            decisions = json.loads(raw)
            if isinstance(decisions, dict):
                decisions = decisions.get("tools", [])
        except json.JSONDecodeError:
            logger.warning("[%s] LLM tool decision was not valid JSON: %s", agent_name, raw[:200])
            return await get_strategy("all").execute(client, tools, query, auth, agent_name=agent_name)

        # Execute decided tools
        allowed = {t["name"] for t in relevant_tools}
        for decision in decisions:
            tool_name = decision.get("tool", "")
            tool_args = decision.get("arguments", {})
            if tool_name not in allowed:
                continue
            try:
                result = await client.call_tool(tool_name, tool_args, auth)
                results[tool_name] = result.text
            except Exception as exc:
                logger.error("[%s] LLM-decided tool '%s' failed: %s", agent_name, tool_name, exc)
                results[f"{tool_name}_error"] = str(exc)

        return results

    @staticmethod
    async def _llm_complete(
        chat_model: Any | None,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        response_format: dict[str, str] | None = None,
    ) -> str:
        """Call LLM via the injected BaseChatModel."""
        if not chat_model:
            raise RuntimeError("LLMDecidesStrategy requires a BaseChatModel. Pass chat_model= when building the graph.")
        kwargs: dict[str, Any] = {"temperature": temperature}
        if response_format:
            kwargs["response_format"] = response_format
        result = await chat_model.ainvoke(messages, **kwargs)
        return result.content or ""


# ── Strategy Registry ──────────────────────────────────────────

STRATEGY_REGISTRY: dict[str, type[OrchidToolCallStrategy]] = {
    "all": CallAllStrategy,
    "sequential": SequentialStrategy,
    "llm_decides": LLMDecidesStrategy,
}


def register_strategy(name: str, cls: type[OrchidToolCallStrategy]) -> None:
    """Register a custom tool call strategy by name."""
    STRATEGY_REGISTRY[name] = cls
    logger.info("[Strategies] Registered '%s' → %s", name, cls.__name__)


def clear_strategies() -> None:
    """Reset to built-in strategies (useful for test isolation)."""
    STRATEGY_REGISTRY.clear()
    STRATEGY_REGISTRY.update(
        {
            "all": CallAllStrategy,
            "sequential": SequentialStrategy,
            "llm_decides": LLMDecidesStrategy,
        }
    )


def get_strategy(name: str) -> OrchidToolCallStrategy:
    """Look up and instantiate a tool call strategy by name.

    Falls back to whatever is registered under ``"all"`` when the name
    is unknown — this lets integrators override the default by
    re-registering ``"all"`` without editing this function.
    """
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        logger.warning("Unknown tool call strategy '%s', falling back to 'all'", name)
        cls = STRATEGY_REGISTRY.get("all", CallAllStrategy)
    return cls()
