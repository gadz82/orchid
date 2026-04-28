"""Tests for the session-bounded MCP capability cache.

The legacy 5-minute TTL was removed in the MCP startup-discovery
migration: capabilities live for the process lifetime and are flushed
only via :meth:`StreamableHttpMCPClient.invalidate_cache`.  These tests
nail down the new contract.
"""

from __future__ import annotations

import warnings

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


class TestCacheTTLDeprecation:
    def test_passing_cache_ttl_emits_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            StreamableHttpMCPClient(
                "http://localhost/mcp",
                auth_mode="none",
                cache_ttl=600,
            )
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deprecations, "passing cache_ttl should emit a DeprecationWarning"
        assert "cache_ttl" in str(deprecations[0].message)

    def test_omitting_cache_ttl_emits_no_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            StreamableHttpMCPClient("http://localhost/mcp", auth_mode="none")
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deprecations == []
