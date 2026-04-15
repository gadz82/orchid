"""Tests for MCPAuthConfig and MCPServerConfig auth field."""

from __future__ import annotations

import pytest

from orchid_ai.config.schema import MCPAuthConfig, MCPServerConfig


class TestMCPAuthConfig:
    """MCPAuthConfig Pydantic model tests."""

    def test_default_mode_is_none(self):
        cfg = MCPAuthConfig()
        assert cfg.mode == "none"

    def test_passthrough_mode(self):
        cfg = MCPAuthConfig(mode="passthrough")
        assert cfg.mode == "passthrough"

    def test_oauth_mode_with_issuer(self):
        cfg = MCPAuthConfig(
            mode="oauth",
            client_id="my-app",
            issuer="https://auth.example.com",
            scopes="openid api.read",
        )
        assert cfg.mode == "oauth"
        assert cfg.client_id == "my-app"
        assert cfg.issuer == "https://auth.example.com"
        assert cfg.scopes == "openid api.read"

    def test_oauth_mode_with_explicit_endpoints(self):
        cfg = MCPAuthConfig(
            mode="oauth",
            client_id="my-app",
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
        )
        assert cfg.authorization_endpoint == "https://auth.example.com/authorize"
        assert cfg.token_endpoint == "https://auth.example.com/token"

    def test_invalid_mode_raises(self):
        with pytest.raises(Exception):
            MCPAuthConfig(mode="invalid")

    def test_default_scopes(self):
        cfg = MCPAuthConfig()
        assert cfg.scopes == "openid"

    def test_default_empty_strings(self):
        cfg = MCPAuthConfig()
        assert cfg.client_id == ""
        assert cfg.authorization_endpoint == ""
        assert cfg.token_endpoint == ""
        assert cfg.issuer == ""


class TestMCPServerConfigAuth:
    """MCPServerConfig backward compatibility and auth integration."""

    def test_no_auth_field_defaults_to_none_mode(self):
        """Existing YAML without auth: still parses (backward compat)."""
        cfg = MCPServerConfig(name="my-server", url="http://localhost:3000/mcp")
        assert cfg.auth.mode == "none"
        assert cfg.auth.client_id == ""

    def test_auth_passthrough(self):
        cfg = MCPServerConfig(
            name="internal",
            url="http://localhost:3000/mcp",
            auth=MCPAuthConfig(mode="passthrough"),
        )
        assert cfg.auth.mode == "passthrough"

    def test_auth_oauth(self):
        cfg = MCPServerConfig(
            name="external-crm",
            url="https://crm.example.com/mcp",
            auth=MCPAuthConfig(
                mode="oauth",
                client_id="orchid-crm",
                issuer="https://auth.crm.example.com",
                scopes="openid crm.read",
            ),
        )
        assert cfg.auth.mode == "oauth"
        assert cfg.auth.client_id == "orchid-crm"

    def test_from_dict_no_auth(self):
        """Parse from dict (as YAML loader produces)."""
        data = {"name": "server1", "url": "http://localhost/mcp"}
        cfg = MCPServerConfig(**data)
        assert cfg.auth.mode == "none"

    def test_from_dict_with_auth(self):
        data = {
            "name": "ext",
            "url": "https://ext.example.com/mcp",
            "auth": {
                "mode": "oauth",
                "client_id": "my-app",
                "issuer": "https://auth.example.com",
            },
        }
        cfg = MCPServerConfig(**data)
        assert cfg.auth.mode == "oauth"
        assert cfg.auth.client_id == "my-app"

    def test_wildcards_still_work_with_auth(self):
        """Ensure the wildcard validator doesn't interfere with auth."""
        data = {
            "name": "ext",
            "url": "https://ext.example.com/mcp",
            "tools": "*",
            "auth": {"mode": "passthrough"},
        }
        cfg = MCPServerConfig(**data)
        assert cfg.discover_all_tools is True
        assert cfg.auth.mode == "passthrough"
