"""
:class:`OrchidSessionWarmer` — proactive MCP capability warm-up.

Each :class:`StreamableHttpMCPClient` already caches ``list_tools`` /
``list_prompts`` / ``list_resources`` responses (see ``client.py``).
What was missing was anyone driving ``warm_cache`` at the right time.
This module owns that single responsibility, with two crisp boundaries:

  * :meth:`warm_unauthenticated` — runs at process startup against
    ``auth.mode: none`` servers.  No user identity is required.
  * :meth:`warm_for_user` — runs once per ``(tenant_key, user_id)``
    process-lifetime against ``auth.mode: passthrough`` and
    ``auth.mode: oauth`` servers.  Idempotent: a second call with the
    same auth pair is a near-instant no-op.

The warmer never raises on per-server failure — every entry is
classified into ``warmed`` / ``skipped`` / ``failed`` and reported via
:class:`OrchidWarmReport` so the caller can log and move on.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from ..core.mcp import OrchidMCPAuthRequiredError, OrchidMCPClient
from ..core.state import OrchidAuthContext
from .inventory import OrchidMCPAuthMode, OrchidMCPServerEntry, OrchidMCPServerInventory

logger = logging.getLogger(__name__)


@dataclass
class OrchidWarmReport:
    """Per-call summary of which servers warmed, skipped, or failed.

    ``skipped`` carries the names of OAuth-mode servers where the user
    has not (yet) completed the authorization dance — that's a normal
    state, not an error.  ``failed`` carries the names of servers that
    raised any other exception, with the stringified reason.
    """

    warmed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


class OrchidSessionWarmer:
    """Drives ``OrchidMCPClient.warm_cache`` against the right clients.

    Idempotency keyed by ``(auth.tenant_key, auth.user_id)`` is the
    backstop that makes calling :meth:`warm_for_user` from multiple
    code paths (explicit ``POST /session/warm`` endpoint AND the lazy
    fire-and-forget hook on first authenticated request) cheap and safe.

    Parameters
    ----------
    inventory : OrchidMCPServerInventory
        Pre-built inventory of every declared MCP server.
    agents : dict[str, OrchidAgent]
        Top-level agents keyed by name — used to locate the actual
        per-agent client instances behind each inventory entry.
    per_server_timeout : float
        Maximum seconds to wait on any single server's warm-up before
        we give up and log it as failed.  Defaults to ``10.0`` —
        generous enough for slow remote MCP servers, tight enough that
        startup never hangs indefinitely.
    """

    def __init__(
        self,
        inventory: OrchidMCPServerInventory,
        agents: dict[str, object] | None = None,
        *,
        per_server_timeout: float = 10.0,
    ) -> None:
        self._inventory = inventory
        # ``object`` typing avoids a circular import with core.agent;
        # the inventory's ``clients_for`` only duck-types each value.
        self._agents: dict[str, object] = dict(agents or {})
        self._per_server_timeout = per_server_timeout
        self._warmed_users: set[tuple[str, str]] = set()

    # ── Public API ────────────────────────────────────────────

    async def warm_unauthenticated(self) -> OrchidWarmReport:
        """Warm every ``auth.mode: none`` server.

        Uses a synthetic :class:`OrchidAuthContext` because ``none``
        servers ignore auth headers entirely (see
        ``StreamableHttpMCPClient._resolve_auth_headers``); the synthetic
        context only exists to satisfy the
        ``warm_cache(auth)`` signature.
        """
        synthetic_auth = OrchidAuthContext(
            access_token="warmup",
            tenant_key="0",
            user_id="system",
        )
        entries = self._inventory.entries_with_mode("none")
        return await self._warm_entries(entries, synthetic_auth)

    async def warm_for_user(self, auth: OrchidAuthContext) -> OrchidWarmReport:
        """Warm ``passthrough`` and ``oauth`` servers for this user.

        Idempotent per ``(tenant_key, user_id)`` for the process
        lifetime; the second call returns an empty report immediately.
        """
        key = self._user_key(auth)
        if key in self._warmed_users:
            return OrchidWarmReport()

        entries = [entry for entry in self._inventory.entries() if entry.mode in ("passthrough", "oauth")]
        report = await self._warm_entries(entries, auth)

        # Record completion regardless of partial failures — the
        # backstop in ``orchid-api/auth.py`` only schedules one warm
        # per user pair, and the explicit endpoint should be a no-op on
        # the second call.  Re-warming after partial failure is the
        # responsibility of explicit invalidation.
        self._warmed_users.add(key)
        return report

    async def warm_one_for_user(
        self,
        auth: OrchidAuthContext,
        server_name: str,
    ) -> OrchidWarmReport:
        """Warm a single MCP server for the user.

        Used by the OAuth-callback handlers (orchid-api and
        orchid-cli ``mcp authorize``) right after a token is persisted,
        so the freshly-authorized server's tools are immediately
        available without waiting for the user's next chat.
        """
        entries = [entry for entry in self._inventory.entries() if entry.server_name == server_name]
        return await self._warm_entries(entries, auth)

    def is_warmed(self, auth: OrchidAuthContext) -> bool:
        """True when :meth:`warm_for_user` already ran for this user."""
        return self._user_key(auth) in self._warmed_users

    def invalidate_user(self, auth: OrchidAuthContext) -> None:
        """Drop the warmed-user record AND flush all per-user caches.

        Walks every passthrough/oauth client and calls
        :meth:`OrchidMCPClient.invalidate_cache` so the next request
        re-discovers capabilities (e.g. after a token rotation).
        """
        self._warmed_users.discard(self._user_key(auth))
        for entry in self._inventory.entries():
            if entry.mode == "none":
                continue
            for client in self._inventory.clients_for(entry, self._agents):  # type: ignore[arg-type]
                self._invalidate(client)

    def invalidate_server(self, server_name: str) -> None:
        """Flush every client backed by an inventory entry for ``server_name``."""
        for entry in self._inventory.entries():
            if entry.server_name != server_name:
                continue
            for client in self._inventory.clients_for(entry, self._agents):  # type: ignore[arg-type]
                self._invalidate(client)

    # ── Internals ─────────────────────────────────────────────

    @staticmethod
    def _user_key(auth: OrchidAuthContext) -> tuple[str, str]:
        return (auth.tenant_key, auth.user_id)

    @staticmethod
    def _invalidate(client: OrchidMCPClient) -> None:
        from ..core.mcp import OrchidCacheableMCPClient

        try:
            if isinstance(client, OrchidCacheableMCPClient):
                client.invalidate_cache()
        except Exception as exc:
            logger.warning("[OrchidSessionWarmer] invalidate_cache raised: %s", exc)

    async def _warm_one_entry(
        self,
        entry: OrchidMCPServerEntry,
        auth: OrchidAuthContext,
    ) -> tuple[str, str | None]:
        """Warm every client behind a single entry.

        Returns ``("warmed"|"skipped"|"failed", error_or_none)``.
        """
        clients = self._inventory.clients_for(entry, self._agents)  # type: ignore[arg-type]
        if not clients:
            # No materialised client for this entry — nothing to warm,
            # but we still report success so the entry shows up in
            # ``warmed`` rather than disappearing silently.
            return ("warmed", None)

        try:
            async with asyncio.timeout(self._per_server_timeout):
                # Sequential per-server: ``warm_cache`` for clients
                # sharing the same (server_name, url) hits the same
                # upstream and they all settle quickly.  Parallel-across-
                # entries is what gives us the wall-clock savings.
                for client in clients:
                    await client.warm_cache(auth)
        except OrchidMCPAuthRequiredError:
            return ("skipped", None)
        except TimeoutError:
            reason = f"timeout after {self._per_server_timeout:.1f}s"
            logger.warning(
                "[OrchidSessionWarmer] warm '%s' (%s) %s",
                entry.server_name,
                entry.mode,
                reason,
            )
            return ("failed", reason)
        except Exception as exc:
            # Some implementations may raise the auth error from inside
            # warm_cache — the outer except above catches the canonical
            # case; keep this branch resilient to defensive subclassing.
            if isinstance(exc, OrchidMCPAuthRequiredError):
                return ("skipped", None)
            logger.warning(
                "[OrchidSessionWarmer] warm '%s' (%s) failed: %s",
                entry.server_name,
                entry.mode,
                exc,
            )
            return ("failed", str(exc))

        return ("warmed", None)

    async def _warm_entries(
        self,
        entries: list[OrchidMCPServerEntry],
        auth: OrchidAuthContext,
    ) -> OrchidWarmReport:
        report = OrchidWarmReport()
        if not entries:
            return report

        results = await asyncio.gather(
            *(self._warm_one_entry(entry, auth) for entry in entries),
            return_exceptions=False,
        )

        for entry, (status, error) in zip(entries, results):
            if status == "warmed":
                report.warmed.append(entry.server_name)
            elif status == "skipped":
                report.skipped.append(entry.server_name)
            elif status == "failed":
                report.failed[entry.server_name] = error or "unknown error"

        if report.warmed or report.skipped or report.failed:
            logger.info(
                "[OrchidSessionWarmer] warm complete: warmed=%s, skipped=%s, failed=%s",
                report.warmed,
                report.skipped,
                report.failed,
            )
        return report


__all__ = [
    "OrchidMCPAuthMode",
    "OrchidSessionWarmer",
    "OrchidWarmReport",
]
