"""Tests for StreamableHttpMCPClient dual-mode auth resolution."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

from orchid_ai.core.mcp import OrchidMCPAuthRequiredError, OrchidMCPTokenRecord, OrchidMCPToolResult
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.mcp.client import StreamableHttpMCPClient


def _auth() -> OrchidAuthContext:
    return OrchidAuthContext(access_token="graph-token", tenant_key="t1", user_id="u1")


def _token(
    *,
    expired: bool = False,
    refresh: bool = False,
) -> OrchidMCPTokenRecord:
    return OrchidMCPTokenRecord(
        server_name="ext-crm",
        tenant_id="t1",
        user_id="u1",
        access_token="oauth-token-123",
        refresh_token="refresh-abc" if refresh else "",
        expires_at=(time.time() - 60) if expired else (time.time() + 3600),
    )


class TestResolveAuthHeaders:
    @pytest.mark.asyncio
    async def test_none_mode_returns_empty_headers(self):
        client = StreamableHttpMCPClient(
            "http://localhost/mcp",
            auth_mode="none",
        )
        headers = await client._resolve_auth_headers(_auth())
        assert headers == {}

    @pytest.mark.asyncio
    async def test_passthrough_mode_returns_graph_bearer(self):
        client = StreamableHttpMCPClient(
            "http://localhost/mcp",
            auth_mode="passthrough",
        )
        headers = await client._resolve_auth_headers(_auth())
        assert headers == {"Authorization": "Bearer graph-token"}

    @pytest.mark.asyncio
    async def test_oauth_with_valid_token(self):
        store = AsyncMock()
        store.get_token = AsyncMock(return_value=_token())

        client = StreamableHttpMCPClient(
            "http://localhost/mcp",
            server_name="ext-crm",
            auth_mode="oauth",
            token_store=store,
        )
        headers = await client._resolve_auth_headers(_auth())
        assert headers == {"Authorization": "Bearer oauth-token-123"}
        store.get_token.assert_called_once_with("t1", "u1", "ext-crm")

    @pytest.mark.asyncio
    async def test_oauth_with_no_token_raises(self):
        store = AsyncMock()
        store.get_token = AsyncMock(return_value=None)

        client = StreamableHttpMCPClient(
            "http://localhost/mcp",
            server_name="ext-crm",
            auth_mode="oauth",
            token_store=store,
        )
        with pytest.raises(OrchidMCPAuthRequiredError) as exc_info:
            await client._resolve_auth_headers(_auth())
        assert exc_info.value.server_name == "ext-crm"

    @pytest.mark.asyncio
    async def test_oauth_with_no_token_store_raises(self):
        client = StreamableHttpMCPClient(
            "http://localhost/mcp",
            server_name="ext-crm",
            auth_mode="oauth",
            token_store=None,
        )
        with pytest.raises(OrchidMCPAuthRequiredError):
            await client._resolve_auth_headers(_auth())

    @pytest.mark.asyncio
    async def test_oauth_with_expired_no_refresh_raises(self):
        store = AsyncMock()
        store.get_token = AsyncMock(return_value=_token(expired=True, refresh=False))

        client = StreamableHttpMCPClient(
            "http://localhost/mcp",
            server_name="ext-crm",
            auth_mode="oauth",
            token_store=store,
        )
        with pytest.raises(OrchidMCPAuthRequiredError):
            await client._resolve_auth_headers(_auth())


class TestMCPAuthRequiredError:
    def test_has_server_name(self):
        err = OrchidMCPAuthRequiredError("my-server")
        assert err.server_name == "my-server"
        assert "my-server" in str(err)

    def test_is_exception(self):
        assert issubclass(OrchidMCPAuthRequiredError, Exception)


class TestMCPTokenRecord:
    def test_is_expired_when_past(self):
        record = OrchidMCPTokenRecord(
            server_name="s",
            tenant_id="t",
            user_id="u",
            access_token="tok",
            expires_at=time.time() - 60,
        )
        assert record.is_expired is True

    def test_is_not_expired_when_future(self):
        record = OrchidMCPTokenRecord(
            server_name="s",
            tenant_id="t",
            user_id="u",
            access_token="tok",
            expires_at=time.time() + 3600,
        )
        assert record.is_expired is False

    def test_is_not_expired_when_zero(self):
        record = OrchidMCPTokenRecord(
            server_name="s",
            tenant_id="t",
            user_id="u",
            access_token="tok",
            expires_at=0.0,
        )
        assert record.is_expired is False

    def test_is_refresh_available(self):
        record = OrchidMCPTokenRecord(
            server_name="s",
            tenant_id="t",
            user_id="u",
            access_token="tok",
            refresh_token="ref",
        )
        assert record.is_refresh_available is True

    def test_is_refresh_not_available(self):
        record = OrchidMCPTokenRecord(
            server_name="s",
            tenant_id="t",
            user_id="u",
            access_token="tok",
        )
        assert record.is_refresh_available is False

    def test_bearer_header(self):
        record = OrchidMCPTokenRecord(
            server_name="s",
            tenant_id="t",
            user_id="u",
            access_token="my-token",
        )
        assert record.bearer_header == {"Authorization": "Bearer my-token"}


class TestNormalizedStreamableHttpClient:
    """``_normalized_streamablehttp_client`` adapts mcp's varying yield arity."""

    def _fake_context_manager(self, streams):
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=streams)
        cm.__aexit__ = AsyncMock(return_value=None)
        return cm

    @pytest.mark.asyncio
    async def test_adapts_two_tuple_yield(self):
        client = StreamableHttpMCPClient("http://localhost/mcp", auth_mode="none")
        fake_cm = self._fake_context_manager(("read", "write"))

        with patch("orchid_ai.mcp.client.streamablehttp_client", return_value=fake_cm):
            async with client._normalized_streamablehttp_client("http://localhost/mcp", http_client="hc") as (rs, ws):
                assert rs == "read"
                assert ws == "write"

        fake_cm.__aexit__.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_adapts_three_tuple_yield(self):
        client = StreamableHttpMCPClient("http://localhost/mcp", auth_mode="none")
        fake_cm = self._fake_context_manager(("read", "write", "context"))

        with patch("orchid_ai.mcp.client.streamablehttp_client", return_value=fake_cm):
            async with client._normalized_streamablehttp_client("http://localhost/mcp", http_client="hc") as (rs, ws):
                assert rs == "read"
                assert ws == "write"


class TestDiscoverAndCache:
    """Capability discovery caches tools even when prompts/resources are unsupported."""

    @pytest.mark.asyncio
    async def test_tool_discovery_uses_input_schema(self):
        """Tools exposing ``input_schema`` (snake_case) are cached correctly."""
        from mcp.types import ListToolsResult, Tool

        client = StreamableHttpMCPClient("http://localhost/mcp", auth_mode="none")

        fake_session = AsyncMock()
        fake_session.list_tools = AsyncMock(
            return_value=ListToolsResult(
                tools=[
                    Tool(
                        name="get_confluence_page",
                        description="Fetch a page",
                        input_schema={"type": "object", "properties": {}},
                    ),
                ],
            ),
        )
        fake_session.list_prompts = AsyncMock(side_effect=Exception("Method not found"))
        fake_session.list_resources = AsyncMock(side_effect=Exception("Method not found"))

        @asynccontextmanager
        async def fake_connect(auth):
            yield fake_session

        with patch.object(client, "_connect", fake_connect):
            await client._discover_and_cache(_auth())

        assert client._cache.tools_discovered is True
        assert len(client._cache.tools) == 1
        assert client._cache.tools[0]["name"] == "get_confluence_page"
        assert client._cache.tools[0]["schema"] == {"type": "object", "properties": {}}
        assert client._cache.populated is True


class TestCallToolResult:
    """``call_tool`` adapts ``is_error`` / ``isError`` casing across ``mcp`` versions."""

    @pytest.mark.asyncio
    async def test_uses_snake_case_is_error(self):
        """Current ``mcp`` returns ``is_error`` (snake_case)."""
        from mcp.types import CallToolResult, TextContent

        client = StreamableHttpMCPClient("http://localhost/mcp", auth_mode="none")

        fake_session = AsyncMock()
        fake_session.call_tool = AsyncMock(
            return_value=CallToolResult(
                content=[TextContent(type="text", text="hello")],
                is_error=False,
            ),
        )

        @asynccontextmanager
        async def fake_connect(auth, timeout=30.0):
            yield fake_session

        with patch.object(client, "_connect", fake_connect):
            result = await client.call_tool("greet", {}, _auth())

        assert isinstance(result, OrchidMCPToolResult)
        assert result.is_error is False
        assert result.text == "hello"

    @pytest.mark.asyncio
    async def test_falls_back_to_camel_case_is_error(self):
        """Older ``mcp`` versions may return ``isError`` instead."""

        class LegacyResult:
            isError = True

            class Content:
                type = "text"
                text = "legacy failure"

            def __init__(self):
                self.content = [self.Content()]

        client = StreamableHttpMCPClient("http://localhost/mcp", auth_mode="none")

        fake_session = AsyncMock()
        fake_session.call_tool = AsyncMock(return_value=LegacyResult())

        @asynccontextmanager
        async def fake_connect(auth, timeout=30.0):
            yield fake_session

        with patch.object(client, "_connect", fake_connect):
            result = await client.call_tool("legacy", {}, _auth())

        assert result.is_error is True
        assert "legacy failure" in result.text
