"""Tests for OrchidSessionWarmer.

Pins the per-mode lifecycle contract: ``warm_unauthenticated`` runs
exclusively against ``auth.mode: none`` servers, ``warm_for_user`` is
idempotent per ``(tenant_key, user_id)`` and short-circuits without
re-issuing discovery RPCs, ``OrchidMCPAuthRequiredError`` is reported
as ``skipped`` rather than ``failed``.
"""

from __future__ import annotations

import asyncio

import pytest

from orchid_ai.config.schema import (
    OrchidAgentConfig,
    OrchidAgentsConfig,
    OrchidMCPAuthConfig,
    OrchidMCPServerConfig,
)
from orchid_ai.core.mcp import OrchidMCPAuthRequiredError, OrchidMCPClient, OrchidMCPToolResult
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.mcp.inventory import OrchidMCPServerInventory
from orchid_ai.mcp.session_warmer import OrchidSessionWarmer, OrchidWarmReport


# ── Test doubles ────────────────────────────────────────────────


class _FakeMCPClient(OrchidMCPClient):
    def __init__(
        self,
        url: str,
        *,
        warm_error: Exception | None = None,
        warm_delay: float = 0.0,
    ) -> None:
        self._url = url
        self._warm_error = warm_error
        self._warm_delay = warm_delay
        self.warm_calls: list[OrchidAuthContext] = []
        self.invalidated = 0

    @property
    def server_url(self) -> str:
        return self._url

    async def warm_cache(self, auth: OrchidAuthContext) -> None:
        if self._warm_delay:
            await asyncio.sleep(self._warm_delay)
        if self._warm_error is not None:
            raise self._warm_error
        self.warm_calls.append(auth)

    def invalidate_cache(self) -> None:
        self.invalidated += 1

    # Unused on the warming path — implement to satisfy the ABC.
    async def call_tool(self, tool_name, arguments, auth):  # pragma: no cover
        return OrchidMCPToolResult()

    async def list_tools(self, auth):  # pragma: no cover
        return []

    async def list_prompts(self, auth):  # pragma: no cover
        return []

    async def list_resources(self, auth):  # pragma: no cover
        return []

    async def get_prompt(self, name, arguments, auth):  # pragma: no cover
        return []

    async def read_resource(self, uri, auth):  # pragma: no cover
        return ""


class _StubAgent:
    def __init__(self, agent_config: OrchidAgentConfig, mcp_clients: list[_FakeMCPClient]) -> None:
        self._config = agent_config
        self.mcp_clients = mcp_clients


def _server(name: str, mode: str = "none", url: str | None = None) -> OrchidMCPServerConfig:
    return OrchidMCPServerConfig(
        name=name,
        url=url or f"https://{name}.example.com/mcp",
        auth=OrchidMCPAuthConfig(mode=mode),
    )


def _build_setup(
    *,
    server_specs: dict[str, str],
    warm_errors: dict[str, Exception] | None = None,
    warm_delays: dict[str, float] | None = None,
) -> tuple[OrchidSessionWarmer, dict[str, _FakeMCPClient]]:
    """Build a warmer + the underlying fake clients keyed by server name.

    Each declared server is wired to a single agent (one per server).
    """
    warm_errors = warm_errors or {}
    warm_delays = warm_delays or {}

    agents_dict: dict[str, _StubAgent] = {}
    clients_by_server: dict[str, _FakeMCPClient] = {}
    config_agents: dict[str, OrchidAgentConfig] = {}

    for server_name, mode in server_specs.items():
        srv = _server(server_name, mode=mode)
        client = _FakeMCPClient(
            srv.url,
            warm_error=warm_errors.get(server_name),
            warm_delay=warm_delays.get(server_name, 0.0),
        )
        agent_config = OrchidAgentConfig(
            description="t",
            prompt="t",
            mcp_servers=[srv],
        )
        agent_name = f"agent-for-{server_name}"
        agents_dict[agent_name] = _StubAgent(agent_config, [client])
        clients_by_server[server_name] = client
        config_agents[agent_name] = agent_config

    cfg = OrchidAgentsConfig(agents=config_agents)
    inv = OrchidMCPServerInventory.from_config(cfg)
    warmer = OrchidSessionWarmer(inv, agents_dict, per_server_timeout=0.5)
    return warmer, clients_by_server


def _auth(tenant: str = "t1", user: str = "u1") -> OrchidAuthContext:
    return OrchidAuthContext(access_token="token", tenant_key=tenant, user_id=user)


# ── Tests ──────────────────────────────────────────────────────


class TestWarmUnauthenticated:
    @pytest.mark.asyncio
    async def test_warms_only_none_mode_servers(self):
        warmer, clients = _build_setup(
            server_specs={
                "local": "none",
                "internal": "passthrough",
                "ext": "oauth",
            },
        )
        report = await warmer.warm_unauthenticated()

        assert report.warmed == ["local"]
        assert report.skipped == []
        assert report.failed == {}
        assert len(clients["local"].warm_calls) == 1
        assert len(clients["internal"].warm_calls) == 0
        assert len(clients["ext"].warm_calls) == 0

    @pytest.mark.asyncio
    async def test_failed_server_is_recorded_not_raised(self):
        warmer, _ = _build_setup(
            server_specs={"good": "none", "broken": "none"},
            warm_errors={"broken": RuntimeError("boom")},
        )
        report = await warmer.warm_unauthenticated()

        assert "good" in report.warmed
        assert "broken" in report.failed
        assert "boom" in report.failed["broken"]

    @pytest.mark.asyncio
    async def test_timeout_is_recorded_as_failed(self):
        warmer, _ = _build_setup(
            server_specs={"slow": "none"},
            warm_delays={"slow": 0.6},  # > per_server_timeout=0.5
        )
        report = await warmer.warm_unauthenticated()

        assert "slow" in report.failed
        assert "timeout" in report.failed["slow"].lower()

    @pytest.mark.asyncio
    async def test_uses_synthetic_auth_for_warming(self):
        warmer, clients = _build_setup(server_specs={"local": "none"})
        await warmer.warm_unauthenticated()

        recorded = clients["local"].warm_calls[0]
        # Synthetic auth values per the warmer contract.
        assert recorded.tenant_key == "0"
        assert recorded.user_id == "system"


class TestWarmForUser:
    @pytest.mark.asyncio
    async def test_warms_passthrough_and_oauth(self):
        warmer, clients = _build_setup(
            server_specs={
                "local": "none",
                "internal": "passthrough",
                "ext": "oauth",
            },
        )
        auth = _auth()
        report = await warmer.warm_for_user(auth)

        assert set(report.warmed) == {"internal", "ext"}
        assert "local" not in report.warmed
        # Sanity: per-user warm forwards the actual auth.
        assert clients["internal"].warm_calls[0].user_id == "u1"

    @pytest.mark.asyncio
    async def test_oauth_required_is_skipped_not_failed(self):
        warmer, _ = _build_setup(
            server_specs={"ext": "oauth"},
            warm_errors={"ext": OrchidMCPAuthRequiredError("ext")},
        )
        report = await warmer.warm_for_user(_auth())

        assert report.skipped == ["ext"]
        assert "ext" not in report.failed

    @pytest.mark.asyncio
    async def test_idempotent_per_user(self):
        warmer, clients = _build_setup(server_specs={"internal": "passthrough"})
        auth = _auth()

        first = await warmer.warm_for_user(auth)
        second = await warmer.warm_for_user(auth)

        assert "internal" in first.warmed
        # Second call short-circuits — empty report and no extra warm hit.
        assert second.warmed == []
        assert second.skipped == []
        assert second.failed == {}
        assert len(clients["internal"].warm_calls) == 1
        assert warmer.is_warmed(auth)

    @pytest.mark.asyncio
    async def test_distinct_users_warm_independently(self):
        warmer, clients = _build_setup(server_specs={"internal": "passthrough"})
        await warmer.warm_for_user(_auth(tenant="t1", user="u1"))
        await warmer.warm_for_user(_auth(tenant="t1", user="u2"))

        assert len(clients["internal"].warm_calls) == 2

    @pytest.mark.asyncio
    async def test_records_completion_after_partial_failure(self):
        warmer, _ = _build_setup(
            server_specs={"good": "passthrough", "bad": "passthrough"},
            warm_errors={"bad": RuntimeError("boom")},
        )
        auth = _auth()
        first = await warmer.warm_for_user(auth)

        assert first.warmed == ["good"]
        assert "bad" in first.failed
        # Second call still short-circuits — partial failures don't
        # invalidate the warmed-user record.
        second = await warmer.warm_for_user(auth)
        assert second.warmed == []
        assert second.failed == {}

    @pytest.mark.asyncio
    async def test_invalidate_user_drops_record_and_flushes_clients(self):
        warmer, clients = _build_setup(
            server_specs={"internal": "passthrough", "ext": "oauth"},
        )
        auth = _auth()
        await warmer.warm_for_user(auth)
        assert warmer.is_warmed(auth)

        warmer.invalidate_user(auth)
        assert not warmer.is_warmed(auth)
        assert clients["internal"].invalidated == 1
        assert clients["ext"].invalidated == 1

        # After invalidation we re-warm.
        await warmer.warm_for_user(auth)
        assert len(clients["internal"].warm_calls) == 2


class TestWarmOneForUser:
    @pytest.mark.asyncio
    async def test_warms_only_named_server(self):
        warmer, clients = _build_setup(
            server_specs={"a": "passthrough", "b": "passthrough"},
        )
        report = await warmer.warm_one_for_user(_auth(), "a")

        assert report.warmed == ["a"]
        assert len(clients["a"].warm_calls) == 1
        assert len(clients["b"].warm_calls) == 0

    @pytest.mark.asyncio
    async def test_unknown_server_yields_empty_report(self):
        warmer, _ = _build_setup(server_specs={"a": "passthrough"})
        report = await warmer.warm_one_for_user(_auth(), "missing")

        assert report.warmed == []
        assert report.skipped == []
        assert report.failed == {}


class TestInvalidateServer:
    def test_flushes_only_targeted_server(self):
        warmer, clients = _build_setup(
            server_specs={"a": "passthrough", "b": "passthrough"},
        )
        warmer.invalidate_server("a")
        assert clients["a"].invalidated == 1
        assert clients["b"].invalidated == 0


class TestReportShape:
    def test_warm_report_defaults_are_empty(self):
        report = OrchidWarmReport()
        assert report.warmed == []
        assert report.skipped == []
        assert report.failed == {}


# ── Multi-agent dedupe ─────────────────────────────────────────


class TestMultiAgentSharedServer:
    @pytest.mark.asyncio
    async def test_warms_each_agent_client_once(self):
        # Two agents share the same server NAME + URL but each holds its
        # own per-agent ``OrchidMCPClient`` instance — both must warm.
        srv = _server("shared", mode="passthrough")
        agent_a_cfg = OrchidAgentConfig(description="t", prompt="t", mcp_servers=[srv])
        agent_b_cfg = OrchidAgentConfig(description="t", prompt="t", mcp_servers=[srv])
        client_a = _FakeMCPClient(srv.url)
        client_b = _FakeMCPClient(srv.url)
        agents = {
            "a": _StubAgent(agent_a_cfg, [client_a]),
            "b": _StubAgent(agent_b_cfg, [client_b]),
        }
        cfg = OrchidAgentsConfig(agents={"a": agent_a_cfg, "b": agent_b_cfg})
        inv = OrchidMCPServerInventory.from_config(cfg)
        warmer = OrchidSessionWarmer(inv, agents)

        report = await warmer.warm_for_user(_auth())
        assert report.warmed == ["shared"]
        assert len(client_a.warm_calls) == 1
        assert len(client_b.warm_calls) == 1
