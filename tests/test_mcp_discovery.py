"""Tests for ``orchid_ai.mcp.discovery`` — RFC 9728 + RFC 8414 + RFC 7591 chain.

Covers:
  * ``extract_resource_metadata_url`` — header parsing edge cases.
  * ``_pick_auth_method`` + ``_intersect_preferred`` — auth-method selection.
  * ``OrchidMCPAuthDiscovery.ensure_registration`` — happy path with
    mocked httpx, cache hit on second call, DCR rejection surfaced as
    :class:`OrchidMCPDiscoveryError`, and the "auth server has no
    registration_endpoint" error path.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from orchid_ai.core.mcp import (
    OrchidMCPClientRegistration,
    OrchidMCPClientRegistrationStore,
    OrchidMCPDiscoveryError,
)
from orchid_ai.mcp.discovery import (
    OrchidMCPAuthDiscovery,
    _intersect_preferred,
    _pick_auth_method,
    extract_resource_metadata_url,
    probe_mcp_server_for_resource_metadata,
)

# ── Header parsing ───────────────────────────────────────────


class TestExtractResourceMetadataUrl:
    def test_missing_header(self):
        assert extract_resource_metadata_url("") is None
        assert extract_resource_metadata_url(None) is None  # type: ignore[arg-type]

    def test_no_resource_metadata_param(self):
        assert extract_resource_metadata_url('Bearer realm="foo"') is None

    def test_extracts_url(self):
        header = 'Bearer realm="mcp", resource_metadata="https://srv.example.com/.well-known/oauth-protected-resource"'
        assert extract_resource_metadata_url(header) == "https://srv.example.com/.well-known/oauth-protected-resource"

    def test_case_insensitive_param_name(self):
        header = 'Bearer Resource_Metadata="https://srv/foo"'
        assert extract_resource_metadata_url(header) == "https://srv/foo"


# ── Auth-method selection ────────────────────────────────────


class TestPickAuthMethod:
    def test_prefers_client_secret_post(self):
        assert _pick_auth_method(["client_secret_basic", "client_secret_post"]) == "client_secret_post"

    def test_falls_back_to_basic(self):
        assert _pick_auth_method(["client_secret_basic"]) == "client_secret_basic"

    def test_none_for_public_clients(self):
        assert _pick_auth_method(["none"]) == "none"

    def test_unknown_returns_first(self):
        assert _pick_auth_method(["private_key_jwt"]) == "private_key_jwt"

    def test_empty_defaults(self):
        assert _pick_auth_method([]) == "client_secret_post"
        assert _pick_auth_method(None) == "client_secret_post"


class TestIntersectPreferred:
    def test_keeps_advertised_subset(self):
        assert _intersect_preferred(
            ["authorization_code", "refresh_token", "client_credentials"],
            ("authorization_code", "refresh_token"),
        ) == ("authorization_code", "refresh_token")

    def test_preserves_preferred_order(self):
        assert _intersect_preferred(
            ["refresh_token", "authorization_code"],
            ("authorization_code", "refresh_token"),
        ) == ("authorization_code", "refresh_token")

    def test_empty_advertised_returns_preferred(self):
        assert _intersect_preferred([], ("authorization_code",)) == ("authorization_code",)


# ── In-memory registration store fixture ─────────────────────


class _InMemoryRegistrationStore(OrchidMCPClientRegistrationStore):
    def __init__(self) -> None:
        self._rows: dict[str, OrchidMCPClientRegistration] = {}

    async def init_db(self) -> None:  # pragma: no cover — unused in tests
        return

    async def close(self) -> None:  # pragma: no cover — unused
        return

    async def get(self, server_name: str) -> OrchidMCPClientRegistration | None:
        return self._rows.get(server_name)

    async def save(self, record: OrchidMCPClientRegistration) -> None:
        self._rows[record.server_name] = record

    async def delete(self, server_name: str) -> bool:
        return self._rows.pop(server_name, None) is not None


# ── End-to-end discovery (with mocked httpx) ─────────────────


def _json_response(data: Any, status_code: int = 200) -> MagicMock:
    """Build a MagicMock that quacks like ``httpx.Response``."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = str(data)
    resp.json = MagicMock(return_value=data)
    return resp


class TestProbeMcpServerForResourceMetadata:
    @pytest.mark.asyncio
    async def test_extracts_url_from_401_header(self):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 401
        resp.headers = {
            "www-authenticate": (
                'Bearer realm="mcp", resource_metadata="https://srv.example.com/.well-known/oauth-protected-resource"'
            ),
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=resp)
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch("httpx.AsyncClient", return_value=mock_client):
            url = await probe_mcp_server_for_resource_metadata(
                mcp_url="https://mcp.example.com/mcp",
                server_name="crm",
            )
        assert url == "https://srv.example.com/.well-known/oauth-protected-resource"

    @pytest.mark.asyncio
    async def test_non_401_response_raises(self):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.headers = {}
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=resp)
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            pytest.raises(OrchidMCPDiscoveryError, match="expected 401"),
        ):
            await probe_mcp_server_for_resource_metadata(
                mcp_url="https://mcp.example.com/mcp",
                server_name="crm",
            )

    @pytest.mark.asyncio
    async def test_401_without_www_authenticate_raises(self):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 401
        resp.headers = {}  # no WWW-Authenticate
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=resp)
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            pytest.raises(OrchidMCPDiscoveryError, match="resource_metadata"),
        ):
            await probe_mcp_server_for_resource_metadata(
                mcp_url="https://mcp.example.com/mcp",
                server_name="crm",
            )


class TestEnsureRegistration:
    @pytest.mark.asyncio
    async def test_full_chain_happy_path(self):
        store = _InMemoryRegistrationStore()
        discovery = OrchidMCPAuthDiscovery(
            store=store,
            redirect_uri="https://orchid.example.com/mcp/auth/callback",
            client_name="Test Orchid",
        )

        resource_meta = {
            "resource": "https://mcp.example.com/mcp",
            "authorization_servers": ["https://idp.example.com"],
        }
        as_meta = {
            "issuer": "https://idp.example.com",
            "authorization_endpoint": "https://idp.example.com/oauth2/authorize",
            "token_endpoint": "https://idp.example.com/oauth2/token",
            "registration_endpoint": "https://idp.example.com/oauth2/register",
            "scopes_supported": ["openid", "mcp.read"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"],
        }
        dcr_response = {
            "client_id": "dyn-client-abc",
            "client_secret": "s3kr3t",
            "client_id_issued_at": 1700000000,
            "client_secret_expires_at": 0,
        }

        # Each leg: resource metadata GET, AS metadata GET, DCR POST.
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=[_json_response(resource_meta), _json_response(as_meta)],
        )
        mock_client.post = AsyncMock(return_value=_json_response(dcr_response))
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch("httpx.AsyncClient", return_value=mock_client):
            record = await discovery.ensure_registration(
                server_name="crm-backend",
                resource_metadata_url="https://mcp.example.com/.well-known/oauth-protected-resource",
            )

        assert record.server_name == "crm-backend"
        assert record.client_id == "dyn-client-abc"
        assert record.client_secret == "s3kr3t"
        assert record.authorization_endpoint == "https://idp.example.com/oauth2/authorize"
        assert record.token_endpoint == "https://idp.example.com/oauth2/token"
        assert record.registration_endpoint == "https://idp.example.com/oauth2/register"
        # client_secret_post wins over basic even though both are supported.
        assert record.token_endpoint_auth_methods_supported == "client_secret_post"
        assert record.scopes_supported == "openid mcp.read"
        # Stored for the next call.
        assert (await store.get("crm-backend")) is record

        # Second call short-circuits — no new HTTP.
        mock_client.get.reset_mock()
        mock_client.post.reset_mock()
        cached = await discovery.ensure_registration(
            server_name="crm-backend",
            resource_metadata_url="https://mcp.example.com/.well-known/oauth-protected-resource",
        )
        assert cached.client_id == "dyn-client-abc"
        mock_client.get.assert_not_awaited()
        mock_client.post.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_registration_endpoint_raises(self):
        store = _InMemoryRegistrationStore()
        discovery = OrchidMCPAuthDiscovery(
            store=store,
            redirect_uri="https://orchid.example.com/mcp/auth/callback",
        )

        resource_meta = {"authorization_servers": ["https://idp.example.com"]}
        as_meta_no_dcr = {
            "issuer": "https://idp.example.com",
            "authorization_endpoint": "https://idp.example.com/oauth2/authorize",
            "token_endpoint": "https://idp.example.com/oauth2/token",
            # registration_endpoint missing
        }
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=[_json_response(resource_meta), _json_response(as_meta_no_dcr)],
        )
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            pytest.raises(OrchidMCPDiscoveryError, match="registration_endpoint"),
        ):
            await discovery.ensure_registration(
                server_name="crm-backend",
                resource_metadata_url="https://mcp.example.com/.well-known/oauth-protected-resource",
            )

        # Must NOT have cached anything on failure.
        assert (await store.get("crm-backend")) is None

    @pytest.mark.asyncio
    async def test_dcr_rejection_surfaced(self):
        store = _InMemoryRegistrationStore()
        discovery = OrchidMCPAuthDiscovery(
            store=store,
            redirect_uri="https://orchid.example.com/mcp/auth/callback",
        )

        resource_meta = {"authorization_servers": ["https://idp.example.com"]}
        as_meta = {
            "authorization_endpoint": "https://idp.example.com/oauth2/authorize",
            "token_endpoint": "https://idp.example.com/oauth2/token",
            "registration_endpoint": "https://idp.example.com/oauth2/register",
        }
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=[_json_response(resource_meta), _json_response(as_meta)],
        )
        mock_client.post = AsyncMock(
            return_value=_json_response({"error": "invalid_client_metadata"}, status_code=400),
        )
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            pytest.raises(OrchidMCPDiscoveryError, match="registration rejected"),
        ):
            await discovery.ensure_registration(
                server_name="crm-backend",
                resource_metadata_url="https://mcp.example.com/.well-known/oauth-protected-resource",
            )
        assert (await store.get("crm-backend")) is None

    @pytest.mark.asyncio
    async def test_empty_authorization_servers_raises(self):
        store = _InMemoryRegistrationStore()
        discovery = OrchidMCPAuthDiscovery(
            store=store,
            redirect_uri="https://orchid.example.com/mcp/auth/callback",
        )
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            return_value=_json_response({"authorization_servers": []}),
        )
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            pytest.raises(OrchidMCPDiscoveryError, match="authorization_servers"),
        ):
            await discovery.ensure_registration(
                server_name="srv",
                resource_metadata_url="https://mcp.example.com/.well-known/oauth-protected-resource",
            )
