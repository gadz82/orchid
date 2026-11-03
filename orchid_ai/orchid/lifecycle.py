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
            agent.inject_signal_emitter(emitter)

    # ── Hot-reload ─────────────────────────────────────────

    async def reload_config(self) -> bool:
        """Check for config changes and hot-reload if any are detected.

        The swap is transactional: providers and the graph are built into
        temporaries, and the previous config/agents/graph/providers are kept if
        either build fails. Returns ``True`` only when the new set is installed.
        """
        watcher = self._config_watcher
        if watcher is None:
            return False

        async with self._rebuild_lock:
            if self._closed:
                return False
            new_snapshot = watcher.reload_if_changed()
            if new_snapshot is None:
                return False

            previous_config = self._config
            previous_agents = self._agents
            previous_graph = self._graph
            previous_inventory = self._inventory
            previous_warmer = self._session_warmer
            previous_models = self._runtime.decision_models
            decision_changed = _decision_config_changed(previous_config, new_snapshot.config)
            new_models = previous_models

            try:
                if decision_changed:
                    from orchid_ai.decision.factory import build_decision_models
                    from orchid_ai.llm_factory import build_chat_model

                    new_models = build_decision_models(
                        new_snapshot.config,
                        chat_model_factory=build_chat_model,
                    )
                    self._runtime.decision_models = new_models
                new_agents: dict[str, OrchidAgent] = {}
                new_graph = build_graph(
                    config=new_snapshot.config,
                    runtime=self._runtime,
                    agents_out=new_agents,
                )
            except Exception:
                logger.exception("[Lifecycle] hot reload failed; keeping the previous config and providers")
                self._runtime.decision_models = previous_models
                self._config = previous_config
                self._agents = previous_agents
                self._graph = previous_graph
                self._inventory = previous_inventory
                self._session_warmer = previous_warmer
                if decision_changed:
                    for model in new_models.values():
                        await _close_quietly(model)
                return False

            self._config = new_snapshot.config
            self._agents = new_agents
            self._graph = new_graph
            self._inventory = OrchidMCPServerInventory.from_config(new_snapshot.config)
            self._session_warmer = OrchidSessionWarmer(self._inventory, self._agents)
            if self._runtime.signal_emitter is not None:
                for agent in self._agents.values():
                    agent._signal_emitter = self._runtime.signal_emitter
            if decision_changed:
                for model in previous_models.values():
                    await _close_quietly(model)

        logger.info("[Orchid] Hot-reloaded graph with %d agent(s)", len(new_snapshot.config.agents))
        return True

    # ── Shutdown ────────────────────────────────────────────

    async def close(self) -> None:
        """Release resources owned by this lifecycle.

        The closed flag is set only after teardown succeeds, so a transient
        failure can be retried. The rebuild lock prevents a concurrent reload
        from installing providers after shutdown.
        """
        async with self._rebuild_lock:
            if self._closed:
                return
            if not self._owns_resources:
                self._closed = True
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
            self._closed = True


def _decision_config_changed(previous: OrchidAgentsConfig, current: OrchidAgentsConfig) -> bool:
    return (
        previous.decision_models != current.decision_models
        or previous.decisions != current.decisions
        or previous.allowed_decision_hosts != current.allowed_decision_hosts
    )


async def _close_quietly(model: Any) -> None:
    try:
        await model.aclose()
    except Exception:
        logger.warning("[Lifecycle] decision provider close failed", exc_info=True)
