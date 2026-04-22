"""Tests for OrchidMCPAuthConfig and OrchidMCPServerConfig auth field.

After the MCP 2025-03-26 authorization spec migration, the schema only
carries ``mode`` — everything else (endpoints, client credentials, scopes)
is discovered at runtime via the three-RFC chain in
``orchid_ai.mcp.discovery``.  YAML containing legacy fields is rejected
by Pydantic's default "ignore extra" behaviour silently, but they have
no effect.  These tests lock in the minimal surface.
"""

from __future__ import annotations

import pytest

from orchid_ai.config.schema import OrchidMCPAuthConfig, OrchidMCPServerConfig


class TestMCPAuthConfig:
    """OrchidMCPAuthConfig Pydantic model tests."""

    def test_default_mode_is_none(self):
        cfg = OrchidMCPAuthConfig()
        assert cfg.mode == "none"

    def test_passthrough_mode(self):
        cfg = OrchidMCPAuthConfig(mode="passthrough")
        assert cfg.mode == "passthrough"

    def test_oauth_mode_requires_nothing_else(self):
        """``mode: oauth`` is self-sufficient — discovery fills the rest."""
        cfg = OrchidMCPAuthConfig(mode="oauth")
        assert cfg.mode == "oauth"

    def test_invalid_mode_raises(self):
        with pytest.raises(Exception):
            OrchidMCPAuthConfig(mode="invalid")

    def test_legacy_static_fields_no_longer_exist(self):
        """The pre-2025-03-26 static fields are gone from the schema."""
        cfg = OrchidMCPAuthConfig(mode="oauth")
        for legacy in (
            "client_id",
            "client_secret_env",
            "authorization_endpoint",
            "token_endpoint",
            "issuer",
            "scopes",
        ):
            assert not hasattr(cfg, legacy), f"legacy field '{legacy}' still present — discovery is authoritative"


class TestMCPServerConfigAuth:
    """OrchidMCPServerConfig backward compatibility and auth integration."""

    def test_no_auth_field_defaults_to_none_mode(self):
        """Existing YAML without auth: still parses (backward compat)."""
        cfg = OrchidMCPServerConfig(name="my-server", url="http://localhost:3000/mcp")
        assert cfg.auth.mode == "none"

    def test_auth_passthrough_untouched(self):
        """Passthrough mode is unchanged by the spec migration (ADR-010)."""
        cfg = OrchidMCPServerConfig(
            name="internal",
            url="http://localhost:3000/mcp",
            auth=OrchidMCPAuthConfig(mode="passthrough"),
        )
        assert cfg.auth.mode == "passthrough"

    def test_auth_oauth_minimal(self):
        """``auth.mode: oauth`` alone is a valid config — discovery handles the rest."""
        cfg = OrchidMCPServerConfig(
            name="external-crm",
            url="https://crm.example.com/mcp",
            auth=OrchidMCPAuthConfig(mode="oauth"),
        )
        assert cfg.auth.mode == "oauth"

    def test_from_dict_no_auth(self):
        """Parse from dict (as YAML loader produces)."""
        data = {"name": "server1", "url": "http://localhost/mcp"}
        cfg = OrchidMCPServerConfig(**data)
        assert cfg.auth.mode == "none"

    def test_from_dict_with_oauth(self):
        data = {
            "name": "ext",
            "url": "https://ext.example.com/mcp",
            "auth": {"mode": "oauth"},
        }
        cfg = OrchidMCPServerConfig(**data)
        assert cfg.auth.mode == "oauth"

    def test_wildcards_still_work_with_auth(self):
        """Ensure the wildcard validator doesn't interfere with auth."""
        data = {
            "name": "ext",
            "url": "https://ext.example.com/mcp",
            "tools": "*",
            "auth": {"mode": "passthrough"},
        }
        cfg = OrchidMCPServerConfig(**data)
        assert cfg.discover_all_tools is True
        assert cfg.auth.mode == "passthrough"
