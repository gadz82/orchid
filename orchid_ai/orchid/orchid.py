"""
``Orchid`` — the public integrator entry point.

M1 refactoring: the 1281-LOC god module is now a thin facade that composes
three focused collaborators:

- ``OrchidConfigLoader`` — YAML / MD / hybrid auto-detection + loading.
- ``OrchidLifecycle``  — init / close, signal-emitter wiring, hot-reload.
- ``OrchidInvoker``     — invoke / stream / resume, including persistence.

The old 16-kwarg factory signatures are preserved for backward
compatibility — they populate ``OrchidFactoryOverrides`` internally.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from langchain_core.messages import BaseMessage
from langgraph.errors import GraphInterrupt

from orchid_ai.config.schema import OrchidAgentsConfig
from orchid_ai.config.watcher import OrchidConfigWatcherBase
from orchid_ai.core.content import OrchidContentSource
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.mcp.inventory import OrchidMCPServerInventory
from orchid_ai.mcp.session_warmer import OrchidSessionWarmer, OrchidWarmReport

# Collaborators (M1 refactoring)
from orchid_ai.orchid.config_loader import OrchidConfigLoader
from orchid_ai.orchid.invoker import OrchidInvoker, OrchidInvokeResult, OrchidPendingApproval
from orchid_ai.orchid.lifecycle import OrchidLifecycle
from orchid_ai.orchid.overrides import (
    CheckpointerOverrides,
    MCPStorageOverrides,
    OrchidFactoryOverrides,
    StartupOverrides,
    StorageOverrides,
)
from orchid_ai.persistence.base import OrchidChatStorage
from orchid_ai.runtime import OrchidRuntime

if TYPE_CHECKING:
    from .config.storage import OrchidConfigStorage
    from .core.agent import OrchidAgent
    from .core.mcp import OrchidMCPTokenStore

logger = logging.getLogger(__name__)


# ── Orchid ─────────────────────────────────────────────────


class Orchid:
    """Programmatic entry point for the Orchid framework.

    Lifecycle
    ---------
    Construct via the async factory :meth:`from_config_path` (recommended)
    or directly with a pre-built ``OrchidRuntime`` + ``OrchidAgentsConfig`` via
    :meth:`__init__`.  Always call :meth:`close` (or use ``async with``)
    to release database and checkpointer connections.

    Example — minimal::

        async with Orchid.from_config_path("orchid.yml") as client:
            result = await client.invoke(
                "What's the team schedule?",
                user_id="alice",
                tenant_id="acme",
            )
            print(result.response)

    Example — custom runtime override::

        runtime = OrchidRuntime(
            default_model="openai/gpt-4o",
            chat_model=ChatOpenAI(model="gpt-4o"),
            reader=my_qdrant_reader,
        )
        client = Orchid(
            config=load_config("agents.yaml"),
            runtime=runtime,
        )
        try:
            result = await client.invoke("Hello", user_id="bob")
        finally:
            await client.close()
    """

    def __init__(
        self,
        *,
        config: OrchidAgentsConfig,
        runtime: OrchidRuntime,
        chat_repo: OrchidChatStorage | None = None,
        mcp_token_store: OrchidMCPTokenStore | None = None,
        _owns_resources: bool = False,
        _config_file_hashes: dict[str, str] | None = None,
        _config_storage: OrchidConfigStorage | None = None,
        _config_watcher: OrchidConfigWatcherBase | None = None,
    ) -> None:
        """Low-level constructor — prefer :meth:`from_config_path` for most uses.

        Parameters
        ----------
        config : OrchidAgentsConfig
            Parsed and validated agent configuration.
        runtime : OrchidRuntime
            Pre-configured runtime (reader, chat_model, checkpointer, etc.).
        chat_repo : OrchidChatStorage | None
            Optional chat persistence backend.
        mcp_token_store : OrchidMCPTokenStore | None
            Optional per-server OAuth token store.
        _owns_resources : bool
            Internal flag — ``True`` means :meth:`close` should shut down
            chat_repo / mcp_token_store / checkpointer.
        _config_file_hashes : dict[str, str] | None
            File path → SHA-256 mapping for hot-reload change detection.
        _config_storage : OrchidConfigStorage | None
            Optional config storage backend.
        _config_watcher : OrchidConfigWatcherBase | None
            Optional config watcher for hot-reload.
        """
        self._config = config
        self._runtime = runtime
        self._chat_repo = chat_repo
        self._mcp_token_store = mcp_token_store
        self._owns_resources = _owns_resources

        # Load framework-managed plugins
        from orchid_ai.plugins import lazy_init_plugins

        lazy_init_plugins()

        # Compose the three M1 collaborators
        self._lifecycle = OrchidLifecycle(
            config=config,
            runtime=runtime,
            chat_repo=chat_repo,
            mcp_token_store=mcp_token_store,
            owns_resources=_owns_resources,
            config_storage=_config_storage,
            config_watcher=_config_watcher,
            config_file_hashes=_config_file_hashes,
        )
        self._invoker = OrchidInvoker(
            graph=self._lifecycle.graph,
            chat_repo=chat_repo,
            checkpointer=runtime.checkpointer,
        )

    # ── Construction helpers ─────────────────────────────────

    @classmethod
    async def from_config_path(
        cls,
        config_path: str,
        *,
        apply_yaml: bool = True,
        agents_config_path: str = "",
        model: str = "",
        vector_backend: str = "",
        qdrant_url: str = "",
        embedding_model: str = "",
        chat_storage_class: str = "",
        chat_db_dsn: str = "",
        chat_extra_migrations_package: str | None = None,
        mcp_token_store_class: str = "",
        mcp_token_store_dsn: str = "",
        mcp_client_registration_store_class: str = "",
        mcp_client_registration_store_dsn: str = "",
        mcp_gateway_state_store_class: str = "",
        mcp_gateway_state_store_dsn: str = "",
        checkpointer_type: str = "",
        checkpointer_dsn: str = "",
        startup_hook: str = "",
        startup_hook_kwargs: dict[str, Any] | None = None,
        content_sources: list[OrchidContentSource] | None = None,
        runtime_overrides: dict[str, Any] | None = None,
        skip_yaml_sections: set[str] | None = None,
        # New grouped overrides (M1)
        storage: StorageOverrides | None = None,
        mcp_storage: MCPStorageOverrides | None = None,
        checkpointer_overrides: CheckpointerOverrides | None = None,
        startup_overrides: StartupOverrides | None = None,
        factory_overrides: OrchidFactoryOverrides | None = None,
    ) -> Orchid:
        """Build a fully-initialised :class:`Orchid` from an ``orchid.yml`` path.

        The **mandatory** integrator bootstrap path.  All string parameters
        fall back first to environment variables then to sensible defaults.

        Backward-compat: the old 16-kwarg signature is preserved.  New
        callers can use the grouped ``storage=``, ``mcp_storage=``, etc.
        dataclasses instead.

        Parameters
        ----------
        config_path : str
            Path to ``orchid.yml`` (YAML) or ``orchid.md`` (Markdown).
            Auto-detected by file extension.
        apply_yaml : bool
            Skip the YAML → env step when ``False``.
        agents_config_path : str
            Path to ``agents.yaml``.
        model, vector_backend, qdrant_url, embedding_model : str
            Overrides for LLM + RAG settings.
        chat_storage_class, chat_db_dsn : str
            Override chat persistence backend.
        chat_extra_migrations_package : str | None
            Integrator-supplied migrations package.
        mcp_token_store_class, mcp_token_store_dsn : str
            Override MCP per-server OAuth token store.
        checkpointer_type, checkpointer_dsn : str
            Enable LangGraph state checkpointing.
        startup_hook : str
            Optional dotted path to a startup hook.
        startup_hook_kwargs : dict | None
            Extra kwargs forwarded to the startup hook.
        content_sources : list[OrchidContentSource] | None
            Content sources for the runtime.
        runtime_overrides : dict | None
            Extra keyword arguments forwarded to :class:`OrchidRuntime`.
        skip_yaml_sections : set[str] | None
            YAML sections to skip during env flattening.
        storage : StorageOverrides | None
            Grouped storage overrides (M1).
        mcp_storage : MCPStorageOverrides | None
            Grouped MCP storage overrides (M1).
        checkpointer_overrides : CheckpointerOverrides | None
            Grouped checkpointer overrides (M1).
        startup_overrides : StartupOverrides | None
            Grouped startup overrides (M1).
        factory_overrides : OrchidFactoryOverrides | None
            Full grouped overrides bundle (M1).  Individual kwargs are
            merged into this when also provided.

        Returns
        -------
        Orchid
            Ready-to-use facade.
        """
        # Merge old kwargs into grouped overrides (backward compat)
        overrides = factory_overrides or OrchidFactoryOverrides(
            model=model,
            vector_backend=vector_backend,
            qdrant_url=qdrant_url,
            embedding_model=embedding_model,
            storage=storage
            or StorageOverrides(
                chat_storage_class=chat_storage_class,
                chat_db_dsn=chat_db_dsn,
                chat_extra_migrations_package=chat_extra_migrations_package,
            ),
            mcp_storage=mcp_storage
            or MCPStorageOverrides(
                mcp_token_store_class=mcp_token_store_class,
                mcp_token_store_dsn=mcp_token_store_dsn,
                mcp_client_registration_store_class=mcp_client_registration_store_class,
                mcp_client_registration_store_dsn=mcp_client_registration_store_dsn,
                mcp_gateway_state_store_class=mcp_gateway_state_store_class,
                mcp_gateway_state_store_dsn=mcp_gateway_state_store_dsn,
            ),
            checkpointer=checkpointer_overrides
            or CheckpointerOverrides(
                checkpointer_type=checkpointer_type,
                checkpointer_dsn=checkpointer_dsn,
            ),
            startup=startup_overrides
            or StartupOverrides(
                startup_hook=startup_hook,
                startup_hook_kwargs=startup_hook_kwargs or {},
            ),
            content_sources=content_sources,
            runtime_overrides=runtime_overrides or {},
            skip_yaml_sections=skip_yaml_sections or set(),
        )

        # Handle individual kwargs that override grouped defaults
        if model:
            overrides.model = model
        if vector_backend:
            overrides.vector_backend = vector_backend
        if qdrant_url:
            overrides.qdrant_url = qdrant_url
        if embedding_model:
            overrides.embedding_model = embedding_model
        if chat_storage_class:
            overrides.storage.chat_storage_class = chat_storage_class
        if chat_db_dsn:
            overrides.storage.chat_db_dsn = chat_db_dsn
        if chat_extra_migrations_package:
            overrides.storage.chat_extra_migrations_package = chat_extra_migrations_package
        if mcp_token_store_class:
            overrides.mcp_storage.mcp_token_store_class = mcp_token_store_class
        if mcp_token_store_dsn:
            overrides.mcp_storage.mcp_token_store_dsn = mcp_token_store_dsn
        if mcp_client_registration_store_class:
            overrides.mcp_storage.mcp_client_registration_store_class = mcp_client_registration_store_class
        if mcp_client_registration_store_dsn:
            overrides.mcp_storage.mcp_client_registration_store_dsn = mcp_client_registration_store_dsn
        if mcp_gateway_state_store_class:
            overrides.mcp_storage.mcp_gateway_state_store_class = mcp_gateway_state_store_class
        if mcp_gateway_state_store_dsn:
            overrides.mcp_storage.mcp_gateway_state_store_dsn = mcp_gateway_state_store_dsn
        if checkpointer_type:
            overrides.checkpointer.checkpointer_type = checkpointer_type
        if checkpointer_dsn:
            overrides.checkpointer.checkpointer_dsn = checkpointer_dsn
        if startup_hook:
            overrides.startup.startup_hook = startup_hook
        if startup_hook_kwargs:
            overrides.startup.startup_hook_kwargs = startup_hook_kwargs
        if content_sources:
            overrides.content_sources = content_sources
        if runtime_overrides:
            overrides.runtime_overrides = runtime_overrides
        if skip_yaml_sections:
            overrides.skip_yaml_sections = skip_yaml_sections

        loaded = await OrchidConfigLoader.load(
            config_path=config_path,
            apply_yaml=apply_yaml,
            agents_config_path=agents_config_path,
            overrides=overrides,
        )

        return cls(
            config=loaded.config,
            runtime=loaded.bootstrap.runtime,
            chat_repo=loaded.bootstrap.chat_repo,
            mcp_token_store=loaded.bootstrap.mcp_token_store,
            _owns_resources=True,
            _config_storage=loaded.config_storage,
            _config_watcher=loaded.config_watcher,
            _config_file_hashes=loaded.config_file_hashes,
        )

    @classmethod
    async def from_md_config(
        cls,
        root_path: str,
        *,
        watch: bool = True,
        agents_dir: str = "",
        model: str = "",
        vector_backend: str = "",
        qdrant_url: str = "",
        embedding_model: str = "",
        chat_storage_class: str = "",
        chat_db_dsn: str = "",
        chat_extra_migrations_package: str | None = None,
        mcp_token_store_class: str = "",
        mcp_token_store_dsn: str = "",
        mcp_client_registration_store_class: str = "",
        mcp_client_registration_store_dsn: str = "",
        mcp_gateway_state_store_class: str = "",
        mcp_gateway_state_store_dsn: str = "",
        checkpointer_type: str = "",
        checkpointer_dsn: str = "",
        startup_hook: str = "",
        startup_hook_kwargs: dict[str, Any] | None = None,
        runtime_overrides: dict[str, Any] | None = None,
        skip_yaml_sections: set[str] | None = None,
    ) -> Orchid:
        """Build a fully-initialised :class:`Orchid` from ``orchid.md`` + ``agents/*.md``.

        The Markdown-equivalent of :meth:`from_config_path`.
        """
        overrides = OrchidFactoryOverrides(
            model=model,
            vector_backend=vector_backend,
            qdrant_url=qdrant_url,
            embedding_model=embedding_model,
            storage=StorageOverrides(
                chat_storage_class=chat_storage_class,
                chat_db_dsn=chat_db_dsn,
                chat_extra_migrations_package=chat_extra_migrations_package,
            ),
            mcp_storage=MCPStorageOverrides(
                mcp_token_store_class=mcp_token_store_class,
                mcp_token_store_dsn=mcp_token_store_dsn,
                mcp_client_registration_store_class=mcp_client_registration_store_class,
                mcp_client_registration_store_dsn=mcp_client_registration_store_dsn,
                mcp_gateway_state_store_class=mcp_gateway_state_store_class,
                mcp_gateway_state_store_dsn=mcp_gateway_state_store_dsn,
            ),
            checkpointer=CheckpointerOverrides(
                checkpointer_type=checkpointer_type,
                checkpointer_dsn=checkpointer_dsn,
            ),
            startup=StartupOverrides(
                startup_hook=startup_hook,
                startup_hook_kwargs=startup_hook_kwargs or {},
            ),
            runtime_overrides=runtime_overrides or {},
            skip_yaml_sections=skip_yaml_sections or set(),
        )

        loaded = await OrchidConfigLoader.load(
            config_path=root_path,
            apply_yaml=False,
            agents_config_path=agents_dir,
            overrides=overrides,
        )

        # from_md_config always sets up a watcher for hot-reload
        if loaded.config_watcher is None and watch:
            from .config.frontmatter import load_markdown_file
            from .config.md_loader import _resolve_agents_dir

            root_path_as_path = Path(root_path)
            root_md = load_markdown_file(root_path_as_path)
            agents_dir_resolved = _resolve_agents_dir(
                root_path_as_path, root_md.frontmatter, Path(agents_dir) if agents_dir else None
            )
            from .config.watcher import OrchidConfigWatcher

            loaded.config_watcher = OrchidConfigWatcher(
                root_path=root_path_as_path,
                agents_dir=agents_dir_resolved,
                initial_config=loaded.config,
                initial_hashes=loaded.config_file_hashes,
            )

        return cls(
            config=loaded.config,
            runtime=loaded.bootstrap.runtime,
            chat_repo=loaded.bootstrap.chat_repo,
            mcp_token_store=loaded.bootstrap.mcp_token_store,
            _owns_resources=True,
            _config_file_hashes=loaded.config_file_hashes,
            _config_watcher=loaded.config_watcher,
        )

    # ── Async context manager ───────────────────────────────

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # ── Public accessors ────────────────────────────────────

    @property
    def graph(self):
        """The compiled LangGraph graph."""
        return self._lifecycle.graph

    @property
    def runtime(self) -> OrchidRuntime:
        """The underlying ``OrchidRuntime``."""
        return self._lifecycle.runtime

    @property
    def config(self) -> OrchidAgentsConfig:
        """Parsed agents configuration."""
        return self._lifecycle.config

    @property
    def chat_repo(self) -> OrchidChatStorage | None:
        """Chat storage backend, or ``None``."""
        return self._lifecycle.chat_repo

    @property
    def session_warmer(self) -> OrchidSessionWarmer:
        """The :class:`OrchidSessionWarmer` bound to this client."""
        return self._lifecycle.session_warmer

    @property
    def server_inventory(self) -> OrchidMCPServerInventory:
        """Read-only inventory of every MCP server declared in the config."""
        return self._lifecycle.server_inventory

    async def warm_unauthenticated_capabilities(self) -> OrchidWarmReport:
        """Convenience for ``self.session_warmer.warm_unauthenticated()``."""
        return await self._lifecycle.session_warmer.warm_unauthenticated()

    @property
    def mcp_token_store(self) -> OrchidMCPTokenStore | None:
        """MCP per-server OAuth token store, or ``None``."""
        return self._lifecycle.mcp_token_store

    def inject_signal_emitter(self, emitter: Any) -> None:
        """Inject a signal emitter into the runtime and every agent instance."""
        self._lifecycle.inject_signal_emitter(emitter)

    @property
    def config_storage(self) -> OrchidConfigStorage | None:
        """Database-backed config storage, or ``None``."""
        return self._lifecycle.config_storage

    @property
    def _agents(self) -> dict[str, OrchidAgent]:
        """Backward-compat alias — delegates to ``OrchidLifecycle._agents``."""
        return self._lifecycle.agents

    # ── Hot-reload ─────────────────────────────────────────

    async def reload_config(self) -> bool:
        """Check for config changes and hot-reload if any are detected."""
        return await self._lifecycle.reload_config()

    # ── Internal helpers (delegated to OrchidInvoker for backward compat) ──

    async def _prepare_invocation(self, **kwargs) -> Any:
        return await self._invoker._prepare_invocation(**kwargs)

    async def _resolve_history(self, **kwargs) -> list[BaseMessage]:
        return await self._invoker._resolve_history(**kwargs)

    def _result_from_graph_output(self, result: dict[str, Any], chat_id: str) -> OrchidInvokeResult:
        return self._invoker._result_from_graph_output(result, chat_id)

    @staticmethod
    def _interrupt_to_result(exc: GraphInterrupt, chat_id: str) -> OrchidInvokeResult:
        return OrchidInvoker._interrupt_to_result(exc, chat_id)

    # ── Core operations (delegated to OrchidInvoker) ────────

    async def invoke(
        self,
        message: str,
        *,
        chat_id: str | None = None,
        user_id: str = "",
        tenant_id: str = "default",
        access_token: str = "",
        auth: OrchidAuthContext | None = None,
        history: list[BaseMessage] | None = None,
        persist: bool = True,
    ) -> OrchidInvokeResult:
        """Run a single request through the agent graph."""
        self._ensure_open()
        return await self._invoker.invoke(
            message=message,
            chat_id=chat_id,
            user_id=user_id,
            tenant_id=tenant_id,
            access_token=access_token,
            auth=auth,
            history=history,
            persist=persist,
        )

    async def resume(
        self,
        chat_id: str,
        *,
        auth: OrchidAuthContext | None = None,
        approved: bool = True,
        persist: bool = True,
    ) -> OrchidInvokeResult:
        """Continue a graph previously paused by ``interrupt()``.

        ``auth`` is the freshly-resolved identity from the authenticated
        resume request, injected into the run via config (never read from
        the checkpoint).
        """
        self._ensure_open()
        return await self._invoker.resume(
            chat_id=chat_id,
            auth=auth,
            approved=approved,
            persist=persist,
        )

    async def stream(
        self,
        message: str,
        *,
        chat_id: str | None = None,
        user_id: str = "",
        tenant_id: str = "default",
        access_token: str = "",
        auth: OrchidAuthContext | None = None,
        history: list[BaseMessage] | None = None,
        stream_mode: str | list[str] = "updates",
    ) -> AsyncIterator[tuple[str, Any]]:
        """Stream graph events as an async iterator."""
        self._ensure_open()
        async for mode, chunk in self._invoker.stream(
            message=message,
            chat_id=chat_id,
            user_id=user_id,
            tenant_id=tenant_id,
            access_token=access_token,
            auth=auth,
            history=history,
            stream_mode=stream_mode,
        ):
            yield mode, chunk

    # ── Shutdown ────────────────────────────────────────────

    async def close(self) -> None:
        """Release resources owned by this client."""
        await self._lifecycle.close()

    # ── Internal helpers ────────────────────────────────────

    def _ensure_open(self) -> None:
        if self._lifecycle.is_closed:
            raise RuntimeError("Orchid is closed; create a new instance before invoking.")


# ── Re-exports for backward compat ──────────────────────────
# These were previously module-level in orchid.py and are imported
# by tests and downstream code.

__all__ = [
    "CheckpointerOverrides",
    "MCPStorageOverrides",
    "Orchid",
    "OrchidFactoryOverrides",
    "OrchidInvokeResult",
    "OrchidPendingApproval",
    "StartupOverrides",
    "StorageOverrides",
]
