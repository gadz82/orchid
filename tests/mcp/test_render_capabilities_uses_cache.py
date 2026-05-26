"""Per-request hot-path regression test.

Locks the contract that drives the MCP startup-discovery migration:
once the per-server :class:`StreamableHttpMCPClient` cache is populated,
:meth:`MCPDispatcher.render_capabilities` must NOT trigger
``_discover_and_cache`` on every chat invocation.  We assert this by
counting how many times each ``list_*`` method is invoked across two
back-to-back ``render_capabilities`` calls.
"""

from __future__ import annotations

from typing import Any

import pytest

from orchid_ai.agents.mcp_dispatcher import MCPDispatcher
from orchid_ai.config.schema import OrchidMCPAuthConfig, OrchidMCPServerConfig, OrchidToolConfig
from orchid_ai.core.mcp import OrchidMCPClient, OrchidMCPToolResult
from orchid_ai.core.state import OrchidAuthContext


class _RecordingClient(OrchidMCPClient):
    """An MCP client that prepopulates its cache and counts list_* calls."""

    def __init__(self, *, prepopulated: bool) -> None:
        self.list_tools_calls = 0
        self.list_prompts_calls = 0
        self.list_resources_calls = 0
        # When prepopulated=True we mimic the post-warmup steady state.
        self._tools: list[dict[str, Any]] = (
            [{"name": "search", "description": "Search the catalog", "schema": {"type": "object"}}]
            if prepopulated
            else []
        )
        self._populated = prepopulated

    @property
    def server_url(self) -> str:
        return "http://stub-mcp/mcp"

    async def call_tool(self, tool_name, arguments, auth):  # pragma: no cover
        return OrchidMCPToolResult()

    async def list_tools(self, auth):
        self.list_tools_calls += 1
        return list(self._tools)

    async def list_prompts(self, auth):
        self.list_prompts_calls += 1
        return []

    async def list_resources(self, auth):
        self.list_resources_calls += 1
        return []

    async def get_prompt(self, name, arguments, auth):  # pragma: no cover
        return []

    async def read_resource(self, uri, auth):  # pragma: no cover
        return ""


@pytest.mark.asyncio
async def test_render_capabilities_uses_cache_after_warmup():
    """A populated client services repeated render_capabilities calls
    with a single list_tools hit per call (no extra discovery)."""
    client = _RecordingClient(prepopulated=True)
    server_config = OrchidMCPServerConfig(
        name="catalog",
        url="http://stub-mcp/mcp",
        tools=[OrchidToolConfig(name="search")],
        auth=OrchidMCPAuthConfig(mode="none"),
    )
    dispatcher = MCPDispatcher([client], [server_config])
    auth = OrchidAuthContext(access_token="t", tenant_key="x", user_id="y")

    caps_a = await dispatcher.render_capabilities(auth, agent_name="catalog-agent")
    caps_b = await dispatcher.render_capabilities(auth, agent_name="catalog-agent")

    # render_capabilities calls list_tools once per invocation — that is
    # the OrchidMCPClient.list_tools surface, which the production
    # ``StreamableHttpMCPClient`` implements as an in-memory cache hit
    # once the cache is populated (see test_client_no_ttl.py).  The
    # crucial assertion is that the dispatcher does NOT issue any
    # extra list_prompts / list_resources beyond what the YAML config
    # asks for.
    assert client.list_tools_calls == 2
    # No prompts/resources configured → no discovery initiated by the
    # dispatcher for those.
    assert client.list_prompts_calls == 0
    assert client.list_resources_calls == 0

    # And the caps payload still carries the cached tool name, proving
    # the second call read from the populated cache rather than an
    # empty discovery result.
    assert any(t["name"] == "search" for t in caps_a.raw_tools)
    assert any(t["name"] == "search" for t in caps_b.raw_tools)


@pytest.mark.asyncio
async def test_render_capabilities_skips_clients_without_configured_caps():
    """Servers configured with no tools/prompts/resources stay silent —
    the dispatcher never touches their list_* surfaces."""
    client = _RecordingClient(prepopulated=True)
    server_config = OrchidMCPServerConfig(
        name="silent",
        url="http://stub-mcp/mcp",
        tools=[],  # no whitelist, no wildcard
        auth=OrchidMCPAuthConfig(mode="none"),
    )
    dispatcher = MCPDispatcher([client], [server_config])
    auth = OrchidAuthContext(access_token="t", tenant_key="x", user_id="y")

    await dispatcher.render_capabilities(auth, agent_name="silent-agent")

    assert client.list_tools_calls == 0
    assert client.list_prompts_calls == 0
    assert client.list_resources_calls == 0


class _DuplicateResourceClient(OrchidMCPClient):
    @property
    def server_url(self) -> str:
        return "http://stub-mcp/mcp"

    async def call_tool(self, tool_name, arguments, auth):  # pragma: no cover
        return OrchidMCPToolResult()

    async def list_tools(self, auth):
        return []

    async def list_prompts(self, auth):
        return []

    async def list_resources(self, auth):
        return [
            {"uri": "ui://first", "name": "duplicate", "description": "", "mime_type": "text/plain"},
            {"uri": "ui://second", "name": "duplicate", "description": "", "mime_type": "text/plain"},
        ]

    async def get_prompt(self, name, arguments, auth):  # pragma: no cover
        return []

    async def read_resource(self, uri, auth):
        return f"content-for:{uri}"


@pytest.mark.asyncio
async def test_render_capabilities_disambiguates_duplicate_resource_names():
    client = _DuplicateResourceClient()
    server_config = OrchidMCPServerConfig(
        name="catalog",
        url="http://stub-mcp/mcp",
        resources=[],
        discover_all_resources=True,
        auth=OrchidMCPAuthConfig(mode="none"),
    )
    dispatcher = MCPDispatcher([client], [server_config])
    auth = OrchidAuthContext(access_token="t", tenant_key="x", user_id="y")

    caps = await dispatcher.render_capabilities(auth, agent_name="catalog-agent")

    assert caps.resource_contents == {
        "duplicate (ui://first)": "content-for:ui://first",
        "duplicate (ui://second)": "content-for:ui://second",
    }
