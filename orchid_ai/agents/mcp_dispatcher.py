"""MCP tool dispatch — orchestrates tool calls across MCP servers."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ..config.schema import MCPServerConfig, ToolConfig
from ..core.mcp import MCPClient
from ..core.state import AuthContext

logger = logging.getLogger(__name__)


@dataclass
class MCPCapabilities:
    """Aggregated MCP capabilities across all configured servers.

    Returned by :meth:`MCPDispatcher.render_capabilities` so that the
    calling agent can build a rich system prompt and a unified tool list.
    """

    #: Tool definitions in the raw MCP format (name, description, schema).
    raw_tools: list[dict[str, Any]] = field(default_factory=list)
    #: Mapping from tool name → (MCPClient, MCPServerConfig) for routing calls.
    tool_client_map: dict[str, tuple[MCPClient, MCPServerConfig]] = field(default_factory=dict)
    #: Zero-arg prompts rendered to text: [{"name": ..., "text": ...}].
    rendered_prompts: list[dict[str, str]] = field(default_factory=list)
    #: Resource contents: {name: content_str}.
    resource_contents: dict[str, str] = field(default_factory=dict)
    #: Prompts that require arguments (listed but not rendered).
    skipped_prompts: list[dict[str, Any]] = field(default_factory=list)


class MCPDispatcher:
    """Dispatches tool calls to MCP servers using configured strategies."""

    def __init__(self, mcp_clients: list[MCPClient], server_configs: list[MCPServerConfig]):
        self._clients = mcp_clients
        self._configs = server_configs

    async def fetch(
        self,
        query: str,
        auth: AuthContext,
        *,
        agent_name: str = "",
        llm_model: str | None = None,
        llm_service: Any = None,
        skip_tools: set[str] | None = None,
    ) -> dict[str, Any]:
        """Call MCP tools concurrently across servers, per configured strategy.

        This method honours the ``tool_call_strategy`` YAML setting
        (``all``, ``sequential``, ``llm_decides``).  It is used by
        :class:`SkillExecutor` during multi-step skill execution.

        For regular (non-skill) queries, :class:`GenericAgent` uses
        :meth:`render_capabilities` + its own agentic loop instead,
        which always uses "LLM decides" semantics via native
        ``tool_calls``.  In that path, ``tool_call_strategy`` is not
        consulted.
        """
        from .strategies import get_strategy

        if not self._clients or not self._configs:
            return {}

        async def _fetch_server(i: int, server_config: MCPServerConfig) -> dict[str, Any]:
            if i >= len(self._clients):
                logger.warning("[%s] No MCP client for server '%s' (index %d)", agent_name, server_config.name, i)
                return {}

            client = self._clients[i]
            server_results: dict[str, Any] = {}
            try:
                effective_tools = server_config.tools
                if (
                    server_config.discover_all_tools
                    or server_config.prompts
                    or server_config.resources
                    or server_config.discover_all_prompts
                    or server_config.discover_all_resources
                ):
                    discovered_tools, mcp_meta = await self._discover_capabilities(
                        client,
                        server_config,
                        auth,
                        agent_name,
                    )
                    if server_config.discover_all_tools:
                        effective_tools = discovered_tools
                    server_results.update(mcp_meta)

                # Filter out tools with cache hits
                if skip_tools:
                    effective_tools = [t for t in effective_tools if t.name not in skip_tools]
                    if not effective_tools:
                        logger.info("[%s] All tools for '%s' skipped (cache hits)", agent_name, server_config.name)
                        return server_results

                strategy = get_strategy(server_config.tool_call_strategy)
                tool_results = await strategy.execute(
                    client,
                    effective_tools,
                    query,
                    auth,
                    agent_name=agent_name,
                    server_config=server_config,
                    llm_model=llm_model,
                    llm_service=llm_service,
                )
                server_results.update(tool_results)
            except (ConnectionError, TimeoutError, ValueError, RuntimeError, OSError) as exc:
                logger.error("[%s] MCP server '%s' failed: %s", agent_name, server_config.name, exc)
                server_results[f"{server_config.name}_error"] = str(exc)

            return server_results

        per_server = await asyncio.gather(*(_fetch_server(i, cfg) for i, cfg in enumerate(self._configs)))

        merged: dict[str, Any] = {}
        for server_result in per_server:
            merged.update(server_result)
        return merged

    async def call_tool_by_source(
        self,
        source_name: str,
        tool_name: str,
        query: str,
        auth: AuthContext,
        extra_args: dict[str, Any],
        previous_results: dict[str, Any],
    ) -> str:
        """Call an MCP tool by server name (used in skill steps)."""
        for i, server_config in enumerate(self._configs):
            if server_config.name == source_name and i < len(self._clients):
                client = self._clients[i]
                args: dict[str, Any] = {"query": query, **extra_args}
                if previous_results:
                    args["previous_results"] = json.dumps(previous_results, default=str)
                result = await client.call_tool(tool_name, args, auth)
                return result.text
        raise ValueError(f"MCP server '{source_name}' not found")

    async def _discover_capabilities(
        self,
        client: MCPClient,
        server_config: MCPServerConfig,
        auth: AuthContext,
        agent_name: str,
    ) -> tuple[list[ToolConfig], dict[str, Any]]:
        """Discover tools, prompts, and resources from an MCP server."""
        server_name = server_config.name
        meta: dict[str, Any] = {}
        discovered_tools: list[ToolConfig] = []

        # Discover tools, prompts, and resources concurrently
        async def _discover_tools():
            if not server_config.discover_all_tools:
                return []
            try:
                raw_tools = await client.list_tools(auth)
                tools = [ToolConfig(name=t["name"]) for t in raw_tools]
                logger.info(
                    "[%s] Discovered %d tools from '%s': %s",
                    agent_name,
                    len(tools),
                    server_name,
                    [t.name for t in tools],
                )
                return tools
            except (ConnectionError, TimeoutError, ValueError, RuntimeError, OSError) as exc:
                logger.warning("[%s] Could not discover tools from '%s': %s", agent_name, server_name, exc)
                return []

        async def _discover_prompts():
            if not (server_config.discover_all_prompts or server_config.prompts):
                return None
            try:
                prompts = await client.list_prompts(auth)
                if not server_config.discover_all_prompts and server_config.prompts:
                    allowed = set(server_config.prompts)
                    prompts = [p for p in prompts if p["name"] in allowed]
                if prompts:
                    logger.info("[%s] Loaded %d prompts from '%s'", agent_name, len(prompts), server_name)
                    return prompts
            except (ConnectionError, TimeoutError, ValueError, RuntimeError, OSError) as exc:
                logger.warning("[%s] Could not load prompts from '%s': %s", agent_name, server_name, exc)
            return None

        async def _discover_resources():
            if not (server_config.discover_all_resources or server_config.resources):
                return None
            try:
                resources = await client.list_resources(auth)
                if not server_config.discover_all_resources and server_config.resources:
                    allowed = set(server_config.resources)
                    resources = [r for r in resources if r["name"] in allowed or r.get("uri") in allowed]
                if resources:
                    logger.info("[%s] Loaded %d resources from '%s'", agent_name, len(resources), server_name)
                    return resources
            except (ConnectionError, TimeoutError, ValueError, RuntimeError, OSError) as exc:
                logger.warning("[%s] Could not load resources from '%s': %s", agent_name, server_name, exc)
            return None

        tools_result, prompts_result, resources_result = await asyncio.gather(
            _discover_tools(),
            _discover_prompts(),
            _discover_resources(),
        )

        discovered_tools = tools_result or []
        if prompts_result:
            meta[f"{server_name}_prompts"] = prompts_result
        if resources_result:
            meta[f"{server_name}_resources"] = resources_result

        return discovered_tools, meta

    # ── Capability rendering (for agentic loop) ────────────────

    async def render_capabilities(
        self,
        auth: AuthContext,
        *,
        agent_name: str = "",
    ) -> MCPCapabilities:
        """Discover and render MCP capabilities across all servers.

        Unlike :meth:`fetch` (which executes tools via a configured
        strategy and is used by :class:`SkillExecutor` during skill
        execution), this method only *discovers* capabilities and
        returns them in a format ready for injection into the
        :meth:`GenericAgent._agentic_tool_loop`:

        * Raw tool definitions (for conversion to litellm format)
        * Tool → client routing map (for calling tools during the loop)
        * Rendered prompt text (zero-arg prompts evaluated, arg-prompts listed)
        * Resource contents (pre-read text resources)

        The agentic loop always uses "LLM decides" semantics — the
        ``tool_call_strategy`` YAML setting is not consulted here.
        """
        caps = MCPCapabilities()

        if not self._clients or not self._configs:
            return caps

        async def _render_server(i: int, server_config: MCPServerConfig) -> None:
            if i >= len(self._clients):
                return

            client = self._clients[i]
            server_name = server_config.name

            # ── Tools ──────────────────────────────────────────
            if server_config.discover_all_tools or server_config.tools:
                try:
                    raw_tools = await client.list_tools(auth)
                    if not server_config.discover_all_tools:
                        whitelist = {t.name for t in server_config.tools}
                        raw_tools = [t for t in raw_tools if t["name"] in whitelist]
                    for t in raw_tools:
                        caps.raw_tools.append(t)
                        caps.tool_client_map[t["name"]] = (client, server_config)
                    logger.info(
                        "[%s] Discovered %d tools from '%s'",
                        agent_name,
                        len(raw_tools),
                        server_name,
                    )
                except (ConnectionError, TimeoutError, ValueError, RuntimeError, OSError) as exc:
                    logger.warning("[%s] Could not discover tools from '%s': %s", agent_name, server_name, exc)

            # ── Prompts ────────────────────────────────────────
            if server_config.discover_all_prompts or server_config.prompts:
                try:
                    prompts = await client.list_prompts(auth)
                    if not server_config.discover_all_prompts and server_config.prompts:
                        allowed = set(server_config.prompts)
                        prompts = [p for p in prompts if p["name"] in allowed]

                    for prompt_def in prompts:
                        has_required = any(a.get("required") for a in prompt_def.get("arguments", []))
                        if has_required:
                            caps.skipped_prompts.append(
                                {
                                    "name": prompt_def["name"],
                                    "description": prompt_def.get("description", ""),
                                    "required_args": [a["name"] for a in prompt_def.get("arguments", [])],
                                }
                            )
                            continue
                        try:
                            rendered = await client.get_prompt(prompt_def["name"], {}, auth)
                            for msg in rendered:
                                if msg.get("role") == "system":
                                    caps.rendered_prompts.append(
                                        {
                                            "name": prompt_def["name"],
                                            "text": msg["content"],
                                        }
                                    )
                        except Exception as exc:
                            logger.warning(
                                "[%s] Could not render prompt '%s' from '%s': %s",
                                agent_name,
                                prompt_def["name"],
                                server_name,
                                exc,
                            )
                except (ConnectionError, TimeoutError, ValueError, RuntimeError, OSError) as exc:
                    logger.warning("[%s] Could not list prompts from '%s': %s", agent_name, server_name, exc)

            # ── Resources ──────────────────────────────────────
            if server_config.discover_all_resources or server_config.resources:
                try:
                    resources = await client.list_resources(auth)
                    if not server_config.discover_all_resources and server_config.resources:
                        allowed = set(server_config.resources)
                        resources = [r for r in resources if r["name"] in allowed or r.get("uri") in allowed]

                    for res in resources:
                        try:
                            content = await client.read_resource(res["uri"], auth)
                            caps.resource_contents[res["name"]] = content
                        except Exception as exc:
                            logger.warning(
                                "[%s] Could not read resource '%s' from '%s': %s",
                                agent_name,
                                res["uri"],
                                server_name,
                                exc,
                            )
                except (ConnectionError, TimeoutError, ValueError, RuntimeError, OSError) as exc:
                    logger.warning("[%s] Could not list resources from '%s': %s", agent_name, server_name, exc)

        # asyncio.gather runs coroutines on a single thread; they only
        # interleave at await points, so concurrent mutation of ``caps``
        # (list.append / dict.__setitem__) is safe without locks.
        await asyncio.gather(*(_render_server(i, cfg) for i, cfg in enumerate(self._configs)))
        return caps

    # ── MCP tools → litellm format ─────────────────────────────

    @staticmethod
    def mcp_tools_to_litellm(mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert MCP tool definitions to the litellm/OpenAI function-calling format."""
        result = []
        for tool in mcp_tools:
            schema = tool.get("schema") or tool.get("inputSchema") or {}
            if "type" not in schema:
                schema = {"type": "object", "properties": schema.get("properties", {})}
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description") or tool["name"],
                        "parameters": schema,
                    },
                }
            )
        return result
