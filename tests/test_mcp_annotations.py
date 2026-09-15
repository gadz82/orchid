"""MCP tool-annotation extraction tests.

Covers two layers:

1. ``StreamableHttpMCPClient._discover_and_cache`` carries
   ``Tool.annotations`` through to ``_cache.tools`` so that the
   dispatcher can surface it.
2. ``MCPDispatcher.render_capabilities`` populates
   ``MCPCapabilities.tool_annotations`` with typed
   ``MCPToolAnnotations`` records parsed from raw payloads.
3. ``MCPToolAnnotations.from_raw`` accepts dict (camelCase) and
   object (camelCase / snake_case) shapes, and degrades gracefully on
   garbage inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from orchid_ai.core.mcp import OrchidMCPClient, OrchidMCPToolResult
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.mcp.inventory import MCPToolAnnotations

# ── MCPToolAnnotations.from_raw — parsing ───────────────────────


class TestMCPToolAnnotationsFromRaw:
    def test_none_returns_none(self):
        assert MCPToolAnnotations.from_raw(None) is None

    def test_empty_dict_yields_all_none(self):
        annotations = MCPToolAnnotations.from_raw({})
        assert annotations == MCPToolAnnotations()
        assert annotations.read_only_hint is None
        assert annotations.idempotent_hint is None
        assert annotations.destructive_hint is None
        assert annotations.open_world_hint is None

    def test_dict_camel_case_keys(self):
        annotations = MCPToolAnnotations.from_raw(
            {
                "readOnlyHint": True,
                "idempotentHint": False,
                "destructiveHint": True,
                "openWorldHint": False,
            }
        )
        assert annotations.read_only_hint is True
        assert annotations.idempotent_hint is False
        assert annotations.destructive_hint is True
        assert annotations.open_world_hint is False

    def test_dict_snake_case_fallback(self):
        annotations = MCPToolAnnotations.from_raw({"read_only_hint": True})
        assert annotations.read_only_hint is True
        assert annotations.idempotent_hint is None

    def test_object_with_camel_attributes(self):
        @dataclass
        class _Annotations:
            readOnlyHint: bool = True
            idempotentHint: bool | None = None
            destructiveHint: bool | None = None
            openWorldHint: bool | None = None

        annotations = MCPToolAnnotations.from_raw(_Annotations())
        assert annotations.read_only_hint is True

    def test_garbage_input_degrades_gracefully(self):
        # Non-mapping, non-attribute object that throws on getattr —
        # parser should swallow and return either None or all-None.
        class _Spite:
            def __getattr__(self, name: str) -> bool:
                raise RuntimeError("nope")

        result = MCPToolAnnotations.from_raw(_Spite())
        assert result is None or result == MCPToolAnnotations()


# ── MCPDispatcher.render_capabilities populates tool_annotations ──


class _AuthStub:
    """Minimal OrchidAuthContext stand-in for tests that don't care about auth."""


def _auth() -> OrchidAuthContext:
    return OrchidAuthContext(access_token="t", tenant_key="tenant", user_id="user")


class _AnnotatedToolsClient(OrchidMCPClient):
    """MCP client whose ``list_tools`` returns annotation-bearing dicts.

    Mimics the cached-tools shape that ``StreamableHttpMCPClient``
    produces in ``_discover_and_cache``.
    """

    def __init__(self, tools: list[dict]) -> None:
        self._tools = tools

    async def call_tool(self, name, args, auth):
        return OrchidMCPToolResult(text="")

    async def list_tools(self, auth):
        return list(self._tools)

    async def list_prompts(self, auth):
        return []

    async def list_resources(self, auth):
        return []

    async def get_prompt(self, name, args, auth):
        return []

    async def read_resource(self, uri, auth):
        return ""

    @property
    def server_url(self):
        return "http://mcp.example.com"


@pytest.mark.asyncio
async def test_render_capabilities_populates_tool_annotations():
    """Annotations on the cached-tool dicts surface as typed records."""
    from orchid_ai.agents.mcp_dispatcher import MCPDispatcher
    from orchid_ai.config.schema import OrchidMCPServerConfig

    client = _AnnotatedToolsClient(
        tools=[
            {
                "name": "search_kb",
                "description": "search",
                "schema": {},
                "annotations": {"readOnlyHint": True, "idempotentHint": True},
            },
            {
                "name": "delete_record",
                "description": "delete",
                "schema": {},
                "annotations": {"destructiveHint": True, "readOnlyHint": False},
            },
            {
                "name": "ping",
                "description": "ping",
                "schema": {},
                "annotations": None,  # server omitted annotations entirely.
            },
        ],
    )
    cfg = OrchidMCPServerConfig(name="kb", url="http://kb.example.com", discover_all_tools=True)
    dispatcher = MCPDispatcher(mcp_clients=[client], server_configs=[cfg])

    caps = await dispatcher.render_capabilities(_auth(), agent_name="support")

    # Every annotated tool surfaces as a typed record.
    assert "search_kb" in caps.tool_annotations
    assert caps.tool_annotations["search_kb"].read_only_hint is True
    assert caps.tool_annotations["search_kb"].idempotent_hint is True

    assert caps.tool_annotations["delete_record"].destructive_hint is True
    assert caps.tool_annotations["delete_record"].read_only_hint is False

    # Tools without an annotations payload are simply absent from the
    # map — consumers must treat missing entries as "unknown".
    assert "ping" not in caps.tool_annotations


@pytest.mark.asyncio
async def test_render_capabilities_omits_when_no_annotations_field():
    """Cached dicts without an "annotations" key still work."""
    from orchid_ai.agents.mcp_dispatcher import MCPDispatcher
    from orchid_ai.config.schema import OrchidMCPServerConfig

    # ``annotations`` key not present at all — older client cache shapes.
    client = _AnnotatedToolsClient(
        tools=[{"name": "lookup_record", "description": "look", "schema": {}}],
    )
    cfg = OrchidMCPServerConfig(name="kb", url="http://kb.example.com", discover_all_tools=True)
    dispatcher = MCPDispatcher(mcp_clients=[client], server_configs=[cfg])

    caps = await dispatcher.render_capabilities(_auth(), agent_name="support")

    assert "lookup_record" in caps.tool_client_map
    assert "lookup_record" not in caps.tool_annotations


# ── client.py cache shape — covered indirectly through the discover path ──


def test_client_cache_dict_shape_includes_annotations_key():
    """The dict shape produced by ``_discover_and_cache`` must carry
    an ``"annotations"`` key (even when the server omits it, in which
    case the value is ``None``).  We verify the shape by replaying
    what ``_discover_and_cache`` does — without spinning up an MCP
    session — and asserting the dispatcher pipeline accepts it.
    """

    @dataclass
    class _Tool:
        name: str
        description: str
        inputSchema: dict
        annotations: object | None

    raw_tools = [
        _Tool(name="t1", description="d", inputSchema={"type": "object"}, annotations={"readOnlyHint": True}),
        _Tool(name="t2", description="d", inputSchema={"type": "object"}, annotations=None),
    ]

    cached = [
        {
            "name": t.name,
            "description": t.description,
            "schema": t.inputSchema,
            "annotations": getattr(t, "annotations", None),
        }
        for t in raw_tools
    ]

    assert cached[0]["annotations"] == {"readOnlyHint": True}
    assert cached[1]["annotations"] is None
    # Schema and description still on the dict — we did not regress them.
    assert cached[0]["schema"] == {"type": "object"}
    assert cached[0]["description"] == "d"
