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
from orchid_ai.graph.agent_executor import OrchidAgentExecutor
from orchid_ai.graph.decision_bindings import bind_decision_points
from orchid_ai.graph.graph import build_graph
from orchid_ai.mcp.inventory import OrchidMCPServerInventory
from orchid_ai.mcp.session_warmer import OrchidSessionWarmer
from orchid_ai.runtime import OrchidRuntime

logger = logging.getLogger(__name__)


def build_pipeline_registry(
    config: OrchidAgentsConfig,
    runtime: OrchidRuntime,
    agents: dict[str, OrchidAgent],
    *,
    agent_executors: dict[str, OrchidAgentExecutor] | None = None,
) -> tuple[Any | None, Any | None]:
    """Compile the ``pipelines:`` block and return ``(registry, deps)``.

    Built after ``build_graph`` so agent instances, their graph-wired executors
    and the built-in tool map exist and can be validated at build time. Both
    values are ``None`` when the block is absent or disabled. The dependencies
    are retained so the facade can pass them to every runner. A compile failure
    propagates: it is a configuration error, and the caller decides whether to
    keep the previous registry (reload) or fail construction.
    """
    pipelines = config.pipelines
    if pipelines is None or not pipelines.enabled:
        return None, None
    from orchid_ai.config.tool_registry import list_tools
    from orchid_ai.pipeline.definition import PipelineDependencies, compile_pipelines
    from orchid_ai.pipeline.registry import PipelineRegistry

    compiled: dict[str, Any] = {}
    deps = PipelineDependencies(
        agents=agents,
        decision_models=runtime.decision_models,
        tools={tool.name: tool for tool in list_tools()},
        signal_emitter=runtime.signal_emitter,
        agent_configs=dict(config.agents),
        agent_executors=dict(agent_executors or {}),
        pipeline_resolver=compiled.get,
    )
    definitions = compile_pipelines(pipelines, deps=deps)
    compiled.update(definitions)
    return PipelineRegistry(definitions.values()), deps


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
        self._pipeline_tools: list[Any] = []
        self._pipeline_tool_names: set[str] = set()
        self._rebuild_lock = asyncio.Lock()

        self._annotate_pipeline_tool_policies(config)

        # Build initial graph
        pipeline_executors: dict[str, OrchidAgentExecutor] = {}
        self._graph = build_graph(
            config=config,
            runtime=runtime,
            agents_out=self._agents,
            executors_out=pipeline_executors,
        )
        self._pipeline_registry, self._pipeline_dependencies = build_pipeline_registry(
            config,
            runtime,
            self._agents,
            agent_executors=pipeline_executors,
        )
        runtime.pipeline_registry = self._pipeline_registry
        self._inventory = OrchidMCPServerInventory.from_config(config)
        self._session_warmer = OrchidSessionWarmer(self._inventory, self._agents)
        self._pipeline_tools = self._build_pipeline_tools(
            self._pipeline_registry,
            self._pipeline_dependencies,
            config,
        )
        self._install_pipeline_tools(self._pipeline_tools)

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
    def pipeline_registry(self) -> Any | None:
        """Compiled pipeline registry, or ``None`` when no pipeline is configured."""
        return self._pipeline_registry

    @property
    def pipeline_dependencies(self) -> Any | None:
        """Build-time dependency map the registry was compiled with."""
        return self._pipeline_dependencies

    @property
    def pipeline_tools(self) -> list[Any]:
        """Pipeline tools currently registered by this lifecycle."""
        return list(self._pipeline_tools)

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
            previous_bindings = dict(self._runtime.decision_bindings)
            previous_registry = self._pipeline_registry
            previous_dependencies = self._pipeline_dependencies
            previous_tools = self._pipeline_tools
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
                # Refresh bindings against the new config/providers before the
                # graph is built, so graph wiring sees the new point set.
                self._annotate_pipeline_tool_policies(new_snapshot.config)
                bind_decision_points(new_snapshot.config, self._runtime)
                new_agents: dict[str, OrchidAgent] = {}
                new_executors: dict[str, OrchidAgentExecutor] = {}
                new_graph = build_graph(
                    config=new_snapshot.config,
                    runtime=self._runtime,
                    agents_out=new_agents,
                    executors_out=new_executors,
                )
                new_registry, new_dependencies = build_pipeline_registry(
                    new_snapshot.config,
                    self._runtime,
                    new_agents,
                    agent_executors=new_executors,
                )
                new_tools = self._build_pipeline_tools(new_registry, new_dependencies, new_snapshot.config)
                new_inventory = OrchidMCPServerInventory.from_config(new_snapshot.config)
                new_warmer = OrchidSessionWarmer(new_inventory, new_agents)
                removed = set(self._pipeline_tool_names) - {tool.name for tool in new_tools}
                self._check_removed_tools_in_agents(new_snapshot.config, removed)
                self._install_pipeline_tools(
                    new_tools,
                    previous_tools=self._pipeline_tools,
                    previous_names=self._pipeline_tool_names,
                )
            except Exception:
                logger.exception("[Lifecycle] hot reload failed; keeping the previous config and providers")
                self._runtime.decision_models = previous_models
                self._runtime.decision_bindings.clear()
                self._runtime.decision_bindings.update(previous_bindings)
                self._config = previous_config
                self._agents = previous_agents
                self._graph = previous_graph
                self._pipeline_registry = previous_registry
                self._pipeline_dependencies = previous_dependencies
                self._runtime.pipeline_registry = previous_registry
                self._restore_pipeline_tools(previous_tools)
                self._inventory = previous_inventory
                self._session_warmer = previous_warmer
                if decision_changed:
                    for model in new_models.values():
                        await _close_quietly(model)
                return False

            self._config = new_snapshot.config
            self._agents = new_agents
            self._graph = new_graph
            # The swap is atomic for new runs: an in-flight run holds its own
            # compiled definition, so replacing the registry cannot change it.
            self._pipeline_registry = new_registry
            self._pipeline_dependencies = new_dependencies
            self._runtime.pipeline_registry = new_registry
            self._inventory = new_inventory
            self._session_warmer = new_warmer
            if self._runtime.signal_emitter is not None:
                for agent in self._agents.values():
                    agent._signal_emitter = self._runtime.signal_emitter
            if decision_changed:
                for model in previous_models.values():
                    await _close_quietly(model)

        logger.info("[Orchid] Hot-reloaded graph with %d agent(s)", len(new_snapshot.config.agents))
        return True

    # ── Pipeline tools (ADR-034) ────────────────────────────

    @staticmethod
    def _annotate_pipeline_tool_policies(config: OrchidAgentsConfig) -> None:
        """Add generated pipeline tools to each agent's HITL/parallel policy.

        ``OrchidAgentConfig.approval_tools`` is computed from configured tools,
        MCP tools and external agents before the graph is built. Pipeline tools
        are generated later from ``pipelines`` definitions, so this pre-graph
        pass makes ``requires_approval`` and ``parallel_safe`` visible to the
        existing agentic loop and parallel scheduler.
        """
        pipelines = config.pipelines
        if pipelines is None or not pipelines.enabled:
            return
        from orchid_ai.config.schema_agent import iter_agents
        from orchid_ai.pipeline.tools import sanitize_tool_name

        policies: dict[str, tuple[bool, bool]] = {}
        for name, definition in pipelines.definitions.items():
            if not definition.enabled or not definition.expose.as_tool:
                continue
            expose = definition.expose
            tool_name = expose.tool_name or sanitize_tool_name(f"pipeline_{name}")
            policies[tool_name] = (expose.requires_approval, expose.parallel_safe)
        for _path, agent_config in iter_agents(config.agents):
            for tool_name, (requires_approval, parallel_safe) in policies.items():
                if tool_name not in agent_config.tools:
                    continue
                if requires_approval:
                    agent_config.approval_tools.add(tool_name)
                if parallel_safe:
                    agent_config.parallel_safe_builtin_tools.add(tool_name)

    def _build_pipeline_tools(
        self,
        registry: Any | None,
        dependencies: Any | None,
        config: OrchidAgentsConfig,
    ) -> list[Any]:
        """Build the tools for every exposed pipeline without registering them."""
        if registry is None:
            return []
        from orchid_ai.pipeline.launcher import PipelineLauncher
        from orchid_ai.pipeline.tools import PipelineTool

        launcher = PipelineLauncher(
            registry=registry,
            events_config=config.events,
            signal_emitter_getter=lambda: self._runtime.signal_emitter,
        )
        invoker = self._pipeline_invoker(registry, dependencies)
        tools: list[Any] = []
        for spec in registry.tool_specs():
            definition = registry.get(spec.pipeline)
            timeout = definition.limits.timeout_seconds if definition is not None else None
            tools.append(
                PipelineTool(
                    spec=spec,
                    invoker=invoker,
                    launcher=launcher,
                    default_timeout=timeout,
                )
            )
        return tools

    def _pipeline_invoker(self, registry: Any, dependencies: Any):
        """Return the sync invoker used by pipeline tools.

        The invoker resolves the definition at call time from the registry it
        was built with, so a tool installed by a reload always runs the new
        definition. ``interpretation`` defaults to ``caller`` for tool runs, as
        specified for conversational use.
        """

        async def invoke(
            name: str,
            input: dict[str, Any],
            *,
            auth: Any,
            chat_id: str | None,
            correlation_id: str | None,
            timeout: float | None,
        ):
            from orchid_ai.core.pipeline import OrchidPipelineNotFoundError
            from orchid_ai.pipeline.builder import PipelineExecution
            from orchid_ai.pipeline.runner import OrchidPipelineRunner

            definition = registry.get(name)
            if definition is None:
                raise OrchidPipelineNotFoundError(f"pipeline {name!r} is not configured")
            runner = OrchidPipelineRunner(
                definition,
                deps=dependencies,
                execution=PipelineExecution(
                    chat_model=self._runtime.chat_model,
                    reader=self._runtime.reader,
                    signal_emitter=self._runtime.signal_emitter,
                ),
            )
            interpretation = definition.definition.expose.interpretation or "caller"
            return await runner.run(
                input,
                auth=auth,
                chat_id=chat_id,
                correlation_id=correlation_id,
                mode="chat",
                timeout=timeout,
                include_graph=False,
                interpretation=interpretation,
            )

        return invoke

    # ORCHID-MEM: 01a1f37c-20cd-72b4-bc2c-6a062403dd5c — Pipeline tool lifecycle; build, collision-check and swap tools with the registry, owning only the names it created.
    def _install_pipeline_tools(
        self,
        tools: list[Any],
        *,
        previous_tools: list[Any] | None = None,
        previous_names: set[str] | None = None,
    ) -> None:
        """Atomically replace the lifecycle-owned pipeline tools.

        Collisions with tools this lifecycle does not own are configuration
        errors. Registration never clears the global registry; only names in
        ``previous_names`` are removed before the new set is installed.
        """
        from orchid_ai.config.tool_registry import TOOL_REGISTRY, register_tool, unregister
        from orchid_ai.core.pipeline import OrchidPipelineConfigError

        previous_tools = list(previous_tools or [])
        previous_names = set(previous_names or ())
        previous_by_name = {tool.name: tool for tool in previous_tools}
        new_names = {tool.name for tool in tools}
        if len(new_names) != len(tools):
            raise OrchidPipelineConfigError("pipeline tools must have unique names")

        current = TOOL_REGISTRY.get_all()
        for name, tool in previous_by_name.items():
            existing = current.get(name)
            if existing is not None and existing is not tool:
                raise OrchidPipelineConfigError(
                    f"pipeline tool {name!r} was replaced by another registration; refusing to overwrite it"
                )

        collisions = new_names & (set(current) - previous_names)
        if collisions:
            owners = {tool.name: tool.pipeline for tool in tools}
            detail = ", ".join(f"{name!r} (pipeline {owners[name]!r})" for name in sorted(collisions))
            raise OrchidPipelineConfigError(f"pipeline tools collide with registered tools: {detail}")

        try:
            for name, tool in previous_by_name.items():
                if current.get(name) is tool:
                    unregister(name)
            for tool in tools:
                register_tool(tool)
        except Exception:
            for name in new_names - previous_names:
                unregister(name)
            self._restore_pipeline_tools(previous_tools)
            raise
        self._pipeline_tools = list(tools)
        self._pipeline_tool_names = new_names

    def _restore_pipeline_tools(self, tools: list[Any]) -> None:
        """Restore exactly the previous tool instances after a failed swap."""
        from orchid_ai.config.tool_registry import TOOL_REGISTRY, register_tool, unregister

        target_names = {tool.name for tool in tools}
        current = TOOL_REGISTRY.get_all()
        for tool in self._pipeline_tools:
            if tool.name not in target_names and current.get(tool.name) is tool:
                unregister(tool.name)
        for tool in tools:
            register_tool(tool)
        self._pipeline_tools = list(tools)
        self._pipeline_tool_names = target_names

    def _unregister_pipeline_tools(self) -> None:
        from orchid_ai.config.tool_registry import TOOL_REGISTRY, unregister

        current = TOOL_REGISTRY.get_all()
        for tool in self._pipeline_tools:
            if current.get(tool.name) is tool:
                unregister(tool.name)
        self._pipeline_tools = []
        self._pipeline_tool_names = set()

    def _check_removed_tools_in_agents(
        self,
        config: OrchidAgentsConfig,
        removed: set[str],
    ) -> None:
        if not removed:
            return
        from orchid_ai.config.schema_agent import iter_agents
        from orchid_ai.core.pipeline import OrchidPipelineConfigError

        for agent_name, agent_config in iter_agents(config.agents):
            referenced = set(agent_config.tools) & removed
            if referenced:
                raise OrchidPipelineConfigError(
                    f"agent {agent_name!r} references removed pipeline tool(s) {sorted(referenced)}"
                )

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
            self._unregister_pipeline_tools()
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
