"""OrchidLifecycle — owns init / close, signal-emitter wiring, hot-reload.

M1 refactoring: extracted from the 1281-LOC ``Orchid`` god module.
Owns resource lifecycle management and hot-reload graph rebuild.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from orchid_ai.bootstrap import _BootstrapResult, _teardown_runtime
from orchid_ai.config.schema import OrchidAgentsConfig
from orchid_ai.config.watcher import OrchidConfigWatcherBase
from orchid_ai.core.agent import OrchidAgent
from orchid_ai.graph.graph import build_graph
from orchid_ai.mcp.inventory import OrchidMCPServerInventory
from orchid_ai.mcp.session_warmer import OrchidSessionWarmer
from orchid_ai.runtime import OrchidRuntime

logger = logging.getLogger(__name__)


class OrchidLifecycle:
    """Manages the Orchid resource lifecycle.

    Owns:
    - Graph construction and hot-reload rebuild.
    - Signal-emitter injection into agents.
    - Resource teardown (chat repo, MCP token store, checkpointer).
    """

    def __init__(
        self,
        config: OrchidAgentsConfig,
        runtime: OrchidRuntime,
        chat_repo: Any | None = None,
        mcp_token_store: Any | None = None,
        owns_resources: bool = False,
        config_storage: Any | None = None,
        config_watcher: OrchidConfigWatcherBase | None = None,
        config_file_hashes: dict[str, str] | None = None,
    ) -> None:
        self._config = config
        self._runtime = runtime
        self._chat_repo = chat_repo
        self._mcp_token_store = mcp_token_store
        self._owns_resources = owns_resources
        self._config_storage = config_storage
        self._config_watcher = config_watcher
        self._config_file_hashes = config_file_hashes
        self._closed = False
        self._agents: dict[str, OrchidAgent] = {}
        self._rebuild_lock = asyncio.Lock()

        # Build initial graph
        self._graph = build_graph(config=config, runtime=runtime, agents_out=self._agents)
        self._inventory = OrchidMCPServerInventory.from_config(config)
        self._session_warmer = OrchidSessionWarmer(self._inventory, self._agents)

    # ── Public accessors ────────────────────────────────────

    @property
    def graph(self) -> Any:
        """The compiled LangGraph graph."""
        return self._graph

    @property
    def runtime(self) -> OrchidRuntime:
        return self._runtime

    @property
    def config(self) -> OrchidAgentsConfig:
        return self._config

    @property
    def agents(self) -> dict[str, OrchidAgent]:
        """Instantiated agent instances (populated by graph builder)."""
        return self._agents

    @property
    def chat_repo(self) -> Any | None:
        return self._chat_repo

    @property
    def mcp_token_store(self) -> Any | None:
        return self._mcp_token_store

    @property
    def session_warmer(self) -> OrchidSessionWarmer:
        return self._session_warmer

    @property
    def server_inventory(self) -> OrchidMCPServerInventory:
        return self._inventory

    @property
    def config_storage(self) -> Any | None:
        return self._config_storage

    @property
    def is_closed(self) -> bool:
        return self._closed

    # ── Signal emitter ──────────────────────────────────────

    def inject_signal_emitter(self, emitter: Any) -> None:
        """Inject a signal emitter into the runtime and every agent instance."""
        self._runtime.signal_emitter = emitter
        for agent in self._agents.values():
            agent._signal_emitter = emitter

    # ── Hot-reload ─────────────────────────────────────────

    async def reload_config(self) -> bool:
        """Check for config changes and hot-reload if any are detected.

        Returns ``True`` when a change was detected and the graph was rebuilt.
        """
        watcher = self._config_watcher
        if watcher is None:
            return False

        new_snapshot = watcher.reload_if_changed()
        if new_snapshot is None:
            return False

        async with self._rebuild_lock:
            self._config = new_snapshot.config
            self._agents.clear()
            self._graph = build_graph(
                config=new_snapshot.config,
                runtime=self._runtime,
                agents_out=self._agents,
            )
            self._inventory = OrchidMCPServerInventory.from_config(new_snapshot.config)
            self._session_warmer = OrchidSessionWarmer(self._inventory, self._agents)
            # Re-inject signal emitter into the new agent instances
            if self._runtime.signal_emitter is not None:
                for agent in self._agents.values():
                    agent._signal_emitter = self._runtime.signal_emitter

        logger.info("[Orchid] Hot-reloaded graph with %d agent(s)", len(new_snapshot.config.agents))
        return True

    # ── Shutdown ────────────────────────────────────────────

    async def close(self) -> None:
        """Release resources owned by this lifecycle."""
        if self._closed:
            return
        self._closed = True

        if not self._owns_resources:
            return

        await _teardown_runtime(
            _BootstrapResult(
                runtime=self._runtime,
                config=self._config,
                chat_repo=self._chat_repo,
                mcp_token_store=self._mcp_token_store,
                mcp_client_registration_store=self._runtime.mcp_client_registration_store,
                mcp_gateway_state_store=self._runtime.mcp_gateway_client_store,
            )
        )
        self._mcp_token_store = None
        self._chat_repo = None
        if self._config_storage is not None:
            await self._config_storage.close()
        if self._runtime is not None:
            self._runtime.mcp_client_registration_store = None
            self._runtime.mcp_gateway_client_store = None
            self._runtime.mcp_gateway_auth_code_store = None
            self._runtime.mcp_gateway_token_store = None
