"""
Concrete MCP client — supports Streamable HTTP and SSE transports.

Capabilities (tools, prompts, resources) are cached with a configurable TTL
so that discovery happens once and subsequent requests reuse the cache.
Only ``call_tool``, ``get_prompt``, and ``read_resource`` open fresh
connections on every invocation (they carry request-specific data).

Transport is selected via ``transport`` parameter:
  - ``"streamable_http"`` (default) — modern MCP protocol, uses POST
  - ``"sse"`` — legacy MCP protocol, uses GET for SSE stream + POST for messages
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

from ..core.mcp import MCPClient, MCPToolResult
from ..core.state import AuthContext

logger = logging.getLogger(__name__)

DEFAULT_CACHE_TTL = 300  # 5 minutes


@dataclass
class _CapabilitiesCache:
    """In-memory cache for MCP server capabilities."""

    tools: list[dict[str, Any]] = field(default_factory=list)
    prompts: list[dict[str, Any]] = field(default_factory=list)
    resources: list[dict[str, Any]] = field(default_factory=list)
    resource_contents: dict[str, str] = field(default_factory=dict)
    rendered_prompts: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    populated: bool = False
    timestamp: float = 0.0


class StreamableHttpMCPClient(MCPClient):
    """MCP client that connects via Streamable HTTP or SSE transport.

    Capabilities are discovered once and cached for ``cache_ttl`` seconds.
    """

    def __init__(
        self,
        url: str,
        server_type: str = "local",
        transport: str = "streamable_http",
        cache_ttl: float = DEFAULT_CACHE_TTL,
    ) -> None:
        self._url = url
        self._server_type = server_type
        self._transport = transport
        self._cache_ttl = cache_ttl
        self._cache = _CapabilitiesCache()

    @property
    def server_url(self) -> str:
        return self._url

    # ── Cache management ─────────────────────────────────────

    def _cache_valid(self) -> bool:
        """Return True if the cache is populated and not expired."""
        return self._cache.populated and (time.monotonic() - self._cache.timestamp) < self._cache_ttl

    def invalidate_cache(self) -> None:
        """Force re-discovery on next call."""
        self._cache = _CapabilitiesCache()

    async def warm_cache(self, auth: AuthContext) -> None:
        """
        Pre-populate the capabilities cache.

        Call this during startup or first interaction to avoid latency later.
        """
        if self._cache_valid():
            return
        await self._discover_and_cache(auth)

    async def _discover_and_cache(self, auth: AuthContext) -> None:
        """Discover all capabilities in a single connection and populate the cache."""
        logger.info("[MCP] Discovering capabilities from %s …", self._url)

        async with self._connect(auth) as session:
            # Tools
            try:
                tools_result = await session.list_tools()
                self._cache.tools = [
                    {
                        "name": t.name,
                        "description": t.description,
                        "schema": t.inputSchema,
                    }
                    for t in tools_result.tools
                ]
                logger.info("[MCP] Cached %d tools from %s", len(self._cache.tools), self._url)
            except Exception as exc:
                logger.warning("[MCP] list_tools failed for %s: %s", self._url, exc)

            # Prompts
            try:
                prompts_result = await session.list_prompts()
                self._cache.prompts = [
                    {
                        "name": p.name,
                        "description": p.description or "",
                        "arguments": [
                            {"name": a.name, "description": a.description or "", "required": a.required or False}
                            for a in (p.arguments or [])
                        ],
                    }
                    for p in prompts_result.prompts
                ]
                logger.info("[MCP] Cached %d prompts from %s", len(self._cache.prompts), self._url)

                # Pre-render prompts that have NO required arguments
                for prompt_def in self._cache.prompts:
                    has_required = any(a.get("required") for a in prompt_def.get("arguments", []))
                    if has_required:
                        continue
                    try:
                        result = await session.get_prompt(prompt_def["name"], {})
                        self._cache.rendered_prompts[prompt_def["name"]] = [
                            {
                                "role": m.role,
                                "content": getattr(m.content, "text", str(m.content)),
                            }
                            for m in result.messages
                        ]
                    except Exception as exc:
                        logger.warning("[MCP] Could not pre-render prompt '%s': %s", prompt_def["name"], exc)
            except Exception as exc:
                logger.warning("[MCP] list_prompts failed for %s: %s", self._url, exc)

            # Resources
            try:
                resources_result = await session.list_resources()
                self._cache.resources = [
                    {
                        "uri": str(r.uri),
                        "name": r.name,
                        "description": r.description or "",
                        "mime_type": r.mimeType or "",
                    }
                    for r in resources_result.resources
                ]
                logger.info("[MCP] Cached %d resources from %s", len(self._cache.resources), self._url)

                # Pre-read text resources
                for res in self._cache.resources:
                    try:
                        read_result = await session.read_resource(res["uri"])
                        parts: list[str] = []
                        for content in read_result.contents:
                            if hasattr(content, "text"):
                                parts.append(content.text)
                            elif hasattr(content, "blob"):
                                parts.append(f"[binary data, {len(content.blob)} bytes]")
                        self._cache.resource_contents[res["name"]] = "\n".join(parts)
                    except Exception as exc:
                        logger.warning("[MCP] Could not pre-read resource '%s': %s", res["uri"], exc)
            except Exception as exc:
                logger.warning("[MCP] list_resources failed for %s: %s", self._url, exc)

        self._cache.populated = True
        self._cache.timestamp = time.monotonic()
        logger.info(
            "[MCP] Cache populated for %s: %d tools, %d prompts (%d rendered), %d resources (%d read)",
            self._url,
            len(self._cache.tools),
            len(self._cache.prompts),
            len(self._cache.rendered_prompts),
            len(self._cache.resources),
            len(self._cache.resource_contents),
        )

    # ── Auth ─────────────────────────────────────────────────

    def _auth_headers(self, auth: AuthContext) -> dict[str, str]:
        """
        Build Authorization header.

        Always uses the original OAuth token (Bearer Passthrough) — both local
        and remote MCP servers forward it to the platform APIs.  The SSO JWT
        (if any) is only used for identity resolution, not for MCP calls.
        """
        return auth.bearer_header

    # ── Transport ────────────────────────────────────────────

    @asynccontextmanager
    async def _connect(self, auth: AuthContext, *, timeout: float = 30.0) -> AsyncGenerator[ClientSession, None]:
        """Open a transport connection and yield an initialized ClientSession."""
        import asyncio

        headers = self._auth_headers(auth)

        async with asyncio.timeout(timeout):
            if self._transport == "sse":
                async with sse_client(self._url, headers=headers) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        yield session
            else:
                async with streamablehttp_client(self._url, headers=headers) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        yield session

    # ── Tool execution (always live — not cached) ────────────

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        auth: AuthContext,
        *,
        timeout: float = 60.0,
    ) -> MCPToolResult:
        """Invoke a named tool on the MCP server."""
        logger.info(
            "[MCP] call_tool → %s | params: %s | url: %s [%s]",
            tool_name,
            arguments,
            self._url,
            self._transport,
        )

        async with self._connect(auth, timeout=timeout) as session:
            result = await session.call_tool(tool_name, arguments)
            content = [{"type": c.type, "text": getattr(c, "text", "")} for c in result.content]

            result_obj = MCPToolResult(
                content=content,
                is_error=result.isError or False,
            )

            logger.info(
                "[MCP] call_tool ← %s | is_error: %s | response: %s",
                tool_name,
                result_obj.is_error,
                result_obj.text[:500] if len(result_obj.text) > 500 else result_obj.text,
            )

            return result_obj

    # ── Cached discovery methods ─────────────────────────────

    async def list_tools(self, auth: AuthContext) -> list[dict[str, Any]]:
        """List all tools (cached)."""
        if not self._cache_valid():
            await self._discover_and_cache(auth)
        return self._cache.tools

    async def list_prompts(self, auth: AuthContext) -> list[dict[str, Any]]:
        """List all prompts (cached)."""
        if not self._cache_valid():
            await self._discover_and_cache(auth)
        return self._cache.prompts

    async def list_resources(self, auth: AuthContext) -> list[dict[str, Any]]:
        """List all resources (cached)."""
        if not self._cache_valid():
            await self._discover_and_cache(auth)
        return self._cache.resources

    # ── Prompt / Resource access ─────────────────────────────

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, str],
        auth: AuthContext,
    ) -> list[dict[str, Any]]:
        """Render a prompt template (cached for zero-arg prompts)."""
        # Return from cache if pre-rendered and no arguments provided
        if not arguments and name in self._cache.rendered_prompts:
            return self._cache.rendered_prompts[name]

        async with self._connect(auth) as session:
            result = await session.get_prompt(name, arguments)
            return [
                {
                    "role": m.role,
                    "content": getattr(m.content, "text", str(m.content)),
                }
                for m in result.messages
            ]

    async def read_resource(self, uri: str, auth: AuthContext) -> str:
        """Read resource content (cached if previously discovered)."""
        # Check cache by URI or name
        for res_name, content in self._cache.resource_contents.items():
            if res_name == uri:
                return content
        # Also check by matching URI in the resource list
        for res in self._cache.resources:
            if res["uri"] == uri and res["name"] in self._cache.resource_contents:
                return self._cache.resource_contents[res["name"]]

        # Not cached — fetch live
        async with self._connect(auth) as session:
            result = await session.read_resource(uri)
            parts: list[str] = []
            for content in result.contents:
                if hasattr(content, "text"):
                    parts.append(content.text)
                elif hasattr(content, "blob"):
                    parts.append(f"[binary data, {len(content.blob)} bytes]")
            return "\n".join(parts)
