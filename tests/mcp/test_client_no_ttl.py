"""Tests for the session-bounded MCP capability cache.

The legacy 5-minute TTL was removed in the MCP startup-discovery
migration: capabilities live for the process lifetime and are flushed
only via :meth:`StreamableHttpMCPClient.invalidate_cache`.  These tests
nail down the new contract.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.mcp.client import StreamableHttpMCPClient


class TestCacheLifecycle:
    def test_cache_invalid_until_populated(self):
        client = StreamableHttpMCPClient("http://localhost/mcp", auth_mode="none")
        assert client._cache_valid() is False

    def test_cache_valid_once_populated(self):
        client = StreamableHttpMCPClient("http://localhost/mcp", auth_mode="none")
        client._cache.populated = True
        assert client._cache_valid() is True

    def test_cache_does_not_expire_over_time(self):
        # The legacy timestamp-based check is gone — once populated, the
        # cache stays valid for the lifetime of the process.
        client = StreamableHttpMCPClient("http://localhost/mcp", auth_mode="none")
        client._cache.populated = True
        # The cache dataclass no longer carries a ``timestamp`` field.
        assert not hasattr(client._cache, "timestamp")

    def test_invalidate_resets_populated(self):
        client = StreamableHttpMCPClient("http://localhost/mcp", auth_mode="none")
        client._cache.populated = True
        client._cache.tools = [{"name": "stale"}]
        client.invalidate_cache()
        assert client._cache.populated is False
        assert client._cache.tools == []

    def test_no_default_cache_ttl_constant(self):
        # ``DEFAULT_CACHE_TTL`` was the only hardcoded knob driving the
        # old expiry.  Asserting its absence prevents accidental
        # re-introduction.
        from orchid_ai.mcp import client as client_module

        assert not hasattr(client_module, "DEFAULT_CACHE_TTL")


class TestFailedResourceNegativeCache:
    """Failed ``read_resource`` calls are negative-cached for the
    process lifetime so we don't pay the round-trip on every request.

    Regression: a real-world notifications MCP server exposed a
    resource that returned ``ENOENT: no such file or directory`` on
    read.  Before this fix, every chat invocation re-attempted the
    read, costing ~1.9 s of wall clock per request.
    """

    def _auth(self) -> OrchidAuthContext:
        return OrchidAuthContext(access_token="t", tenant_key="x", user_id="y")

    def test_failed_uri_set_starts_empty(self):
        client = StreamableHttpMCPClient("http://localhost/mcp", auth_mode="none")
        assert client._cache.failed_resource_uris == set()

    def test_invalidate_clears_failed_uris(self):
        client = StreamableHttpMCPClient("http://localhost/mcp", auth_mode="none")
        client._cache.failed_resource_uris.add("ui://broken")
        client.invalidate_cache()
        assert client._cache.failed_resource_uris == set()

    @pytest.mark.asyncio
    async def test_read_resource_returns_empty_for_negative_cached_uri(self):
        # Pre-seed the negative cache and assert ``read_resource``
        # short-circuits without opening any transport connection.
        client = StreamableHttpMCPClient("http://localhost/mcp", auth_mode="none")
        client._cache.failed_resource_uris.add("ui://broken")

        connect_mock = AsyncMock(side_effect=AssertionError("must not open a connection"))
        client._connect = connect_mock  # type: ignore[assignment]

        result = await client.read_resource("ui://broken", self._auth())
        assert result == ""
        connect_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_live_read_failure_is_negative_cached(self):
        # First call: live fetch fails → URI added to failed set + exception re-raised.
        # Second call: short-circuits, no connection opened.
        client = StreamableHttpMCPClient("http://localhost/mcp", auth_mode="none")

        @asynccontextmanager
        async def _failing_connect(auth, *, timeout=30.0):
            class _Session:
                async def read_resource(self, uri):
                    raise RuntimeError("ENOENT: no such file or directory")

            yield _Session()

        client._connect = _failing_connect  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="ENOENT"):
            await client.read_resource("ui://broken", self._auth())
        assert "ui://broken" in client._cache.failed_resource_uris

        # Now swap the connect mock to ensure no second network call happens.
        client._connect = AsyncMock(side_effect=AssertionError("must not retry"))  # type: ignore[assignment]
        result = await client.read_resource("ui://broken", self._auth())
        assert result == ""

    @pytest.mark.asyncio
    async def test_discover_and_cache_negative_caches_pre_read_failures(self):
        # When ``_discover_and_cache`` encounters a resource it can list
        # but cannot read, the URI is added to ``failed_resource_uris``.
        client = StreamableHttpMCPClient("http://localhost/mcp", auth_mode="none")

        class _StubResource:
            def __init__(self, uri, name):
                self.uri = uri
                self.name = name
                self.description = ""
                self.mimeType = "text/plain"

        class _StubResources:
            def __init__(self, resources):
                self.resources = resources

        class _StubResult:
            def __init__(self, items):
                self.tools = items
                self.prompts = items

        @asynccontextmanager
        async def _stub_connect(auth, *, timeout=30.0):
            class _Session:
                async def list_tools(self):
                    return _StubResult([])

                async def list_prompts(self):
                    return _StubResult([])

                async def list_resources(self):
                    return _StubResources(
                        [
                            _StubResource("ui://good", "good"),
                            _StubResource("ui://broken", "broken"),
                        ]
                    )

                async def read_resource(self, uri):
                    if "broken" in uri:
                        raise RuntimeError("ENOENT")

                    # Return a result-like object with a contents list
                    class _Content:
                        text = "ok-content"

                    class _Result:
                        contents = [_Content()]

                    return _Result()

            yield _Session()

        client._connect = _stub_connect  # type: ignore[assignment]

        await client._discover_and_cache(self._auth())

        assert "ui://broken" in client._cache.failed_resource_uris
        assert "ui://good" not in client._cache.failed_resource_uris
        assert client._cache.resource_contents.get("good") == "ok-content"
        assert "broken" not in client._cache.resource_contents
