"""Tests for StreamableHttpMCPClient dual-mode auth resolution."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from orchid_ai.core.mcp import OrchidMCPAuthRequiredError, OrchidMCPTokenRecord
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
