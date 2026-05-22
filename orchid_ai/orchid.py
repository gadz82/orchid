"""
``Orchid`` — the mandatory integrator entry point.

Every integrator declares an :class:`Orchid` instance explicitly. It is
the single canonical way to start the framework from Python: the library
does **not** expose a free ``build_runtime`` function anymore (that name
is now private), and ``orchid-api`` / ``orchid-cli`` route their own
bootstraps through this same class so all three surfaces stay in
lock-step.

Owns the full lifecycle: loads YAML config, builds the reader, chat
storage, MCP token store, optional checkpointer, and compiled graph.
Exposes a small imperative surface::

    async with Orchid.from_config_path("orchid.yml") as orchid:
        result = await orchid.invoke("Hello", user_id="alice")
        print(result.response)

Three call shapes are provided:
  * :meth:`invoke`  — blocking call, returns :class:`OrchidInvokeResult`
  * :meth:`stream`  — async iterator of events (token / agent / final)
  * :meth:`resume`  — continue after a human-in-the-loop interrupt

Always call :meth:`close` (or use ``async with``) to release database
and checkpointer connections.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from .bootstrap import _build_runtime
from .config.schema import OrchidAgentsConfig
from .config.watcher import OrchidConfigWatcherBase
from .core.content import OrchidContentSource
from .core.state import OrchidAuthContext
from .graph.graph import build_graph
from .mcp.inventory import OrchidMCPServerInventory
from .mcp.session_warmer import OrchidSessionWarmer, OrchidWarmReport
from .persistence.base import OrchidChatStorage
from .runtime import OrchidRuntime

if TYPE_CHECKING:
    from .config.storage import OrchidConfigStorage
    from .core.agent import OrchidAgent  # noqa: F401
    from .core.mcp import OrchidMCPTokenStore

logger = logging.getLogger(__name__)


# ── Result types ────────────────────────────────────────────────


@dataclass
class OrchidPendingApproval:
    """A single tool-approval request surfaced by ``interrupt()``."""

    tool: str
    args: dict[str, Any]
    agent: str
    interrupt_id: str


@dataclass
class OrchidInvokeResult:
    """Result of a single :meth:`Orchid.invoke` call.

    Attributes
    ----------
    response : str
        The synthesised final response.  Empty when ``interrupted=True``.
    chat_id : str
        The chat (and LangGraph thread) identifier used for this call.
    agents_used : list[str]
        Names of the agents that were activated during this run.
    messages : list[BaseMessage]
        Full message list as returned by the graph (after the run).
    interrupted : bool
        ``True`` when the graph paused for human-in-the-loop approval.
    approvals_needed : list[OrchidPendingApproval]
        Populated only when ``interrupted=True``.
    mcp_context : dict[str, Any]
        Raw MCP tool results produced during this run.
    rag_context : dict[str, Any]
        Retrieved RAG chunks per agent.
    """

    response: str
    chat_id: str
    agents_used: list[str] = field(default_factory=list)
    messages: list[BaseMessage] = field(default_factory=list)
    interrupted: bool = False
    approvals_needed: list[OrchidPendingApproval] = field(default_factory=list)
    mcp_context: dict[str, Any] = field(default_factory=dict)
    rag_context: dict[str, Any] = field(default_factory=dict)


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
        mcp_token_store: "OrchidMCPTokenStore | None" = None,
        _owns_resources: bool = False,
        _config_file_hashes: dict[str, str] | None = None,
        _config_storage: "OrchidConfigStorage | None" = None,
    ) -> None:
        """Low-level constructor — prefer :meth:`from_config_path` for most uses.

        Parameters
        ----------
        config : OrchidAgentsConfig
            Parsed and validated agent configuration.
        runtime : OrchidRuntime
            Pre-configured runtime (reader, chat_model, checkpointer, etc.).
        chat_repo : OrchidChatStorage | None
            Optional chat persistence backend.  When ``None``, the client does
            not persist messages — history must be passed explicitly.
        mcp_token_store : OrchidMCPTokenStore | None
            Optional per-server OAuth token store.  Populated on ``runtime``.
        _owns_resources : bool
            Internal flag — ``True`` means :meth:`close` should shut down
            chat_repo / mcp_token_store / checkpointer.  Set by the
            :meth:`from_config_path` factory.
        _config_file_hashes : dict[str, str] | None
            File path → SHA-256 mapping for hot-reload change detection.
            Set by the MD config loader; ``None`` when using YAML config.
        _config_storage : OrchidConfigStorage | None
            Optional config storage backend (e.g. database-backed agent
            configs).  When set, the facade closes it on :meth:`close`.
        """
        self._config = config
        self._runtime = runtime
        self._chat_repo = chat_repo
        self._mcp_token_store = mcp_token_store
        self._owns_resources = _owns_resources
        self._closed = False
        self._agents: dict[str, OrchidAgent] = {}
        self._graph = build_graph(config=config, runtime=runtime, agents_out=self._agents)
        self._inventory = OrchidMCPServerInventory.from_config(config)
        self._session_warmer = OrchidSessionWarmer(self._inventory, self._agents)
        self._config_file_hashes = _config_file_hashes
        self._config_watcher: OrchidConfigWatcherBase | None = None
        self._config_storage = _config_storage
        self._rebuild_lock = asyncio.Lock()

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
    ) -> "Orchid":
        """Build a fully-initialised :class:`Orchid` from an ``orchid.yml`` path.

        The **mandatory** integrator bootstrap path.  Mirrors the full
        parameter surface of the (now-private) library bootstrapper so
        ``orchid-api`` and ``orchid-cli`` can route through this same
        method and stay in lock-step with in-process integrators.

        Auto-detection: if ``config_path`` ends in ``.md`` or is a
        directory containing ``orchid.md``, the call delegates to
        :meth:`from_md_config`.

        All string parameters fall back first to environment variables
        (``LITELLM_MODEL``, ``VECTOR_BACKEND``, ``QDRANT_URL``, …) and
        then to sensible hardcoded defaults (arg > env > hardcoded).

        Parameters
        ----------
        config_path : str
            Path to ``orchid.yml`` (YAML) or ``orchid.md`` (Markdown).
            Auto-detected by file extension.  When non-empty and
            ``apply_yaml=True``, values are flattened into env vars
            (existing env wins) and ``ORCHID_CONFIG`` is exported so
            downstream components can find the file.  Pass ``""`` when
            the caller has already applied YAML to the environment
            (e.g. ``orchid-api``).
        apply_yaml : bool
            Skip the YAML → env step.  Useful when pydantic-settings (or
            the caller) already populated env vars at import time.
        agents_config_path : str
            Path to ``agents.yaml``.  Falls back to ``AGENTS_CONFIG_PATH``
            env / ``"agents.yaml"`` default.  Ignored when auto-detection
            selects Markdown config.
        model, vector_backend, qdrant_url, embedding_model : str
            Overrides for LLM + RAG settings.
        chat_storage_class, chat_db_dsn : str
            Override chat persistence backend.  Defaults to SQLite at
            ``~/.orchid/chats.db``.
        chat_extra_migrations_package : str | None
            Integrator-supplied migrations package (dotted import path)
            applied after the framework's migrations by both the chat
            storage and the MCP token store (shared DB).
        mcp_token_store_class, mcp_token_store_dsn : str
            Override MCP per-server OAuth token store.
        checkpointer_type, checkpointer_dsn : str
            Enable LangGraph state checkpointing.  Required for the HITL
            :meth:`resume` flow.  Valid types: ``"memory"``, ``"sqlite"``,
            ``"postgres"``, or a dotted class path.
        startup_hook : str
            Optional dotted path to an ``async(reader, settings) -> None``
            hook executed after the reader is built (e.g. for seeding RAG).
        startup_hook_kwargs : dict | None
            Extra kwargs forwarded to the startup hook.  The hook always
            receives ``reader=<OrchidVectorReader>`` in addition.
        runtime_overrides : dict | None
            Extra keyword arguments forwarded to :class:`OrchidRuntime`.
            Use this to inject a custom ``chat_model``, ``reader``, or
            ``mcp_client_factory`` that bypass the built-in factories.
        skip_yaml_sections : set[str] | None
            YAML sections to skip during env flattening (e.g.
            ``{"storage"}`` for the CLI convention).

        Returns
        -------
        Orchid
            Ready-to-use facade.  The caller owns the returned object
            and must call :meth:`close` (or use ``async with``) to
            release resources.
        """
        # ── Auto-detect Markdown config ──────────────────
        config_path_obj = Path(config_path) if config_path else None
        if config_path_obj is not None:
            if config_path_obj.suffix == ".md":
                return await cls.from_md_config(
                    config_path,
                    model=model,
                    vector_backend=vector_backend,
                    qdrant_url=qdrant_url,
                    embedding_model=embedding_model,
                    chat_storage_class=chat_storage_class,
                    chat_db_dsn=chat_db_dsn,
                    chat_extra_migrations_package=chat_extra_migrations_package,
                    mcp_token_store_class=mcp_token_store_class,
                    mcp_token_store_dsn=mcp_token_store_dsn,
                    mcp_client_registration_store_class=mcp_client_registration_store_class,
                    mcp_client_registration_store_dsn=mcp_client_registration_store_dsn,
                    mcp_gateway_state_store_class=mcp_gateway_state_store_class,
                    mcp_gateway_state_store_dsn=mcp_gateway_state_store_dsn,
                    checkpointer_type=checkpointer_type,
                    checkpointer_dsn=checkpointer_dsn,
                    startup_hook=startup_hook,
                    startup_hook_kwargs=startup_hook_kwargs,
                    content_sources=content_sources,
                    runtime_overrides=runtime_overrides,
                    skip_yaml_sections=skip_yaml_sections,
                )
            if config_path_obj.is_dir():
                md_candidate = config_path_obj / "orchid.md"
                if md_candidate.exists():
                    return await cls.from_md_config(
                        str(md_candidate),
                        model=model,
                        vector_backend=vector_backend,
                        qdrant_url=qdrant_url,
                        embedding_model=embedding_model,
                        chat_storage_class=chat_storage_class,
                        chat_db_dsn=chat_db_dsn,
                        chat_extra_migrations_package=chat_extra_migrations_package,
                        mcp_token_store_class=mcp_token_store_class,
                        mcp_token_store_dsn=mcp_token_store_dsn,
                        mcp_client_registration_store_class=mcp_client_registration_store_class,
                        mcp_client_registration_store_dsn=mcp_client_registration_store_dsn,
                        mcp_gateway_state_store_class=mcp_gateway_state_store_class,
                        mcp_gateway_state_store_dsn=mcp_gateway_state_store_dsn,
                        checkpointer_type=checkpointer_type,
                        checkpointer_dsn=checkpointer_dsn,
                        startup_hook=startup_hook,
                        startup_hook_kwargs=startup_hook_kwargs,
                        runtime_overrides=runtime_overrides,
                        skip_yaml_sections=skip_yaml_sections,
                    )

            # ── Hybrid: YAML root + MD agents ──────────────────
            if config_path_obj.suffix in (".yml", ".yaml") and config_path_obj.exists():
                agents_dir = Path(agents_config_path or os.environ.get("AGENTS_CONFIG_PATH", "agents"))
                if not agents_dir.is_absolute():
                    agents_dir = config_path_obj.parent / agents_dir
                if agents_dir.is_dir() and any(agents_dir.glob("*.md")):
                    return await _build_hybrid_config(
                        orchid_yml_path=config_path_obj,
                        agents_dir=agents_dir,
                        apply_yaml=apply_yaml,
                        skip_yaml_sections=skip_yaml_sections,
                        model=model,
                        vector_backend=vector_backend,
                        qdrant_url=qdrant_url,
                        embedding_model=embedding_model,
                        chat_storage_class=chat_storage_class,
                        chat_db_dsn=chat_db_dsn,
                        chat_extra_migrations_package=chat_extra_migrations_package,
                        mcp_token_store_class=mcp_token_store_class,
                        mcp_token_store_dsn=mcp_token_store_dsn,
                        mcp_client_registration_store_class=mcp_client_registration_store_class,
                        mcp_client_registration_store_dsn=mcp_client_registration_store_dsn,
                        mcp_gateway_state_store_class=mcp_gateway_state_store_class,
                        mcp_gateway_state_store_dsn=mcp_gateway_state_store_dsn,
                        checkpointer_type=checkpointer_type,
                        checkpointer_dsn=checkpointer_dsn,
                        startup_hook=startup_hook,
                        startup_hook_kwargs=startup_hook_kwargs,
                        runtime_overrides=runtime_overrides,
                    )

        result = await _build_runtime(
            **_build_runtime_kwargs(
                config_path=config_path,
                apply_yaml=apply_yaml,
                config=None,
                agents_config_path=agents_config_path,
                model=model,
                vector_backend=vector_backend,
                qdrant_url=qdrant_url,
                embedding_model=embedding_model,
                chat_storage_class=chat_storage_class,
                chat_db_dsn=chat_db_dsn,
                chat_extra_migrations_package=chat_extra_migrations_package,
                mcp_token_store_class=mcp_token_store_class,
                mcp_token_store_dsn=mcp_token_store_dsn,
                mcp_client_registration_store_class=mcp_client_registration_store_class,
                mcp_client_registration_store_dsn=mcp_client_registration_store_dsn,
                mcp_gateway_state_store_class=mcp_gateway_state_store_class,
                mcp_gateway_state_store_dsn=mcp_gateway_state_store_dsn,
                checkpointer_type=checkpointer_type,
                checkpointer_dsn=checkpointer_dsn,
                startup_hook=startup_hook,
                startup_hook_kwargs=startup_hook_kwargs,
                runtime_overrides=runtime_overrides,
                skip_yaml_sections=skip_yaml_sections,
            )
        )

        # Merge DB-sourced configs if config storage is enabled in the config
        config_store_instance = None
        if result.config.config_storage.enabled:
            from orchid_ai.config.storage_factory import build_config_storage

            cfg = result.config.config_storage
            config_store_instance = build_config_storage(cfg.class_path, cfg.dsn)
            await config_store_instance.init_db()
            db_configs = await config_store_instance.list_configs()
            if db_configs:
                result.config.merge_from_db(db_configs, strict=True)

        instance = cls(
            config=result.config,
            runtime=result.runtime,
            chat_repo=result.chat_repo,
            mcp_token_store=result.mcp_token_store,
            _owns_resources=True,
            _config_storage=config_store_instance,
        )
        return instance

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
    ) -> "Orchid":
        """Build a fully-initialised :class:`Orchid` from ``orchid.md`` + ``agents/*.md``.

        The Markdown-equivalent of :meth:`from_config_path`.  Loads the
        unified MD config, maps infrastructure keys to environment
        variables, and wires the full runtime stack.

        Parameters
        ----------
        root_path : str
            Path to the root ``orchid.md`` file.
        watch : bool
            When ``True`` (default), stores file hashes for future
            hot-reload change detection (Phase 2 watcher).
        agents_dir : str
            Directory containing per-agent ``.md`` files.  When empty,
            resolved from the root frontmatter's ``agents.agents_dir``
            field, then defaults to ``<root_dir>/agents``.
        model, vector_backend, qdrant_url, embedding_model : str
            Overrides for LLM + RAG settings.  Falls back to env vars
            (which may have been populated by ``md_infrastructure_to_env``)
            then to hardcoded defaults.
        chat_storage_class, chat_db_dsn : str
            Override chat persistence backend.
        chat_extra_migrations_package : str | None
            Integrator-supplied migrations package.
        mcp_token_store_class, mcp_token_store_dsn : str
            Override MCP per-server OAuth token store.
        checkpointer_type, checkpointer_dsn : str
            Enable LangGraph state checkpointing.
        startup_hook : str
            Optional dotted path to an ``async(**kwargs) -> None`` hook.
        startup_hook_kwargs : dict | None
            Extra kwargs forwarded to the startup hook.
        runtime_overrides : dict | None
            Extra keyword arguments forwarded to :class:`OrchidRuntime`.
        skip_yaml_sections : set[str] | None
            Sections to skip when mapping MD frontmatter keys to env
            vars (e.g. ``{"storage"}`` for the CLI convention).

        Returns
        -------
        Orchid
            Ready-to-use facade.
        """
        import os

        from .config.md_loader import load_md_config, md_infrastructure_to_env
        from .config.frontmatter import load_markdown_file

        root_path_as_path = Path(root_path)

        # ── 1. Load MD config ──────────────────────────
        agents_dir_override = Path(agents_dir) if agents_dir else None
        config, file_hashes = load_md_config(root_path_as_path, agents_dir=agents_dir_override)

        # ── 2. Apply infrastructure env vars ────────────
        root_md = load_markdown_file(root_path_as_path)
        env_vars = md_infrastructure_to_env(root_md.frontmatter, skip_sections=skip_yaml_sections)
        for k, v in env_vars.items():
            if k not in os.environ:
                os.environ[k] = v
        os.environ.setdefault("ORCHID_CONFIG", str(root_path_as_path))

        # ── 3. Build runtime with the pre-loaded config ─
        result = await _build_runtime(
            **_build_runtime_kwargs(
                config_path=str(root_path_as_path),
                apply_yaml=False,
                config=config,
                model=model,
                vector_backend=vector_backend,
                qdrant_url=qdrant_url,
                embedding_model=embedding_model,
                chat_storage_class=chat_storage_class,
                chat_db_dsn=chat_db_dsn,
                chat_extra_migrations_package=chat_extra_migrations_package,
                mcp_token_store_class=mcp_token_store_class,
                mcp_token_store_dsn=mcp_token_store_dsn,
                mcp_client_registration_store_class=mcp_client_registration_store_class,
                mcp_client_registration_store_dsn=mcp_client_registration_store_dsn,
                mcp_gateway_state_store_class=mcp_gateway_state_store_class,
                mcp_gateway_state_store_dsn=mcp_gateway_state_store_dsn,
                checkpointer_type=checkpointer_type,
                checkpointer_dsn=checkpointer_dsn,
                startup_hook=startup_hook,
                startup_hook_kwargs=startup_hook_kwargs,
                runtime_overrides=runtime_overrides,
                skip_yaml_sections=skip_yaml_sections,
            )
        )
        instance = cls(
            config=result.config,
            runtime=result.runtime,
            chat_repo=result.chat_repo,
            mcp_token_store=result.mcp_token_store,
            _owns_resources=True,
            _config_file_hashes=file_hashes if watch else None,
        )
        if watch:
            from .config.watcher import OrchidConfigWatcher

            agents_dir_resolved = _resolve_agents_dir_from_md(
                root_path_as_path, root_md.frontmatter, agents_dir_override
            )
            instance._config_watcher = OrchidConfigWatcher(
                root_path=root_path_as_path,
                agents_dir=agents_dir_resolved,
                initial_config=config,
                initial_hashes=file_hashes,
            )
        return instance

    # ── Async context manager ───────────────────────────────

    async def __aenter__(self) -> "Orchid":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # ── Public accessors ────────────────────────────────────

    @property
    def graph(self):
        """The compiled LangGraph graph."""
        return self._graph

    @property
    def runtime(self) -> OrchidRuntime:
        """The underlying ``OrchidRuntime`` (reader, chat_model, checkpointer, ...)."""
        return self._runtime

    @property
    def config(self) -> OrchidAgentsConfig:
        """Parsed agents configuration."""
        return self._config

    @property
    def chat_repo(self) -> OrchidChatStorage | None:
        """Chat storage backend, or ``None`` when running without persistence."""
        return self._chat_repo

    @property
    def session_warmer(self) -> OrchidSessionWarmer:
        """The :class:`OrchidSessionWarmer` bound to this client.

        Drives proactive warming of MCP capability caches at the right
        lifecycle boundaries.  Use
        :meth:`warm_unauthenticated_capabilities` for the convenience
        startup hook; integrators with a per-user session start (or
        OAuth-callback handler) call ``session_warmer.warm_for_user`` /
        ``session_warmer.warm_one_for_user`` directly.
        """
        return self._session_warmer

    @property
    def server_inventory(self) -> OrchidMCPServerInventory:
        """Read-only inventory of every MCP server declared in the config."""
        return self._inventory

    async def warm_unauthenticated_capabilities(self) -> OrchidWarmReport:
        """Convenience for ``self.session_warmer.warm_unauthenticated()``.

        Call this once at startup (orchid-api lifespan / orchid-cli
        bootstrap) so ``auth.mode: none`` MCP servers populate their
        capability caches before the first chat invocation.  Failures
        are reported in the returned :class:`OrchidWarmReport` and
        never raise — the caller logs and moves on.
        """
        return await self._session_warmer.warm_unauthenticated()

    @property
    def mcp_token_store(self) -> "OrchidMCPTokenStore | None":
        """MCP per-server OAuth token store, or ``None``.

        Downstream packages dereference ``orchid.mcp_token_store`` — in
        particular ``orchid-cli.commands.chat._send_message`` for the
        pre-flight MCP auth check, and
        ``orchid-api.context.AppContext.mcp_token_store`` as a
        read-through property for the ``get_mcp_token_store_optional``
        FastAPI dependency.  ``_mcp_token_store`` is already stored in
        :meth:`__init__`; this accessor exposes it through the public
        facade surface.
        """
        return self._mcp_token_store

    @property
    def config_storage(self) -> "OrchidConfigStorage | None":
        """Database-backed config storage, or ``None`` when not configured.

        Set automatically by :meth:`from_config_path` when
        ``config_storage_class`` and ``config_storage_dsn`` are provided.
        Exposes the store so integrators can call
        ``config_storage.upsert_config()``, ``config_storage.patch_config()``,
        etc. from outside the Orchid facade — for example from an API
        router that manages configs at runtime.
        """
        return self._config_storage

    # ── Hot-reload ─────────────────────────────────────────

    async def reload_config(self) -> bool:
        """Check for config changes and hot-reload if any are detected.

        Requires a config watcher (set up by :meth:`from_md_config` with
        ``watch=True``, or ``from_config_path`` with ``watch=True``).
        Graph rebuild is serialised via an ``asyncio.Lock`` so concurrent
        requests complete with the old config until the rebuild finishes.

        Returns
        -------
        bool
            ``True`` when a change was detected and the graph was rebuilt.
            ``False`` when nothing changed or no watcher is active.
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

        logger.info("[Orchid] Hot-reloaded graph with %d agent(s)", len(new_snapshot.config.agents))
        return True

    # ── Core operations ─────────────────────────────────────

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
        """Run a single request through the agent graph.

        Parameters
        ----------
        message : str
            The user's input.  Passed to the graph as a ``HumanMessage``.
        chat_id : str | None
            Chat / thread identifier.  A UUID is generated when ``None``.
            Also used as LangGraph ``thread_id`` for checkpointing.
        user_id, tenant_id : str
            Used to build a default :class:`OrchidAuthContext` when ``auth`` is
            not supplied.  Drive RAG scoping and chat ownership.
        access_token : str
            Bearer token forwarded to MCP servers (passthrough mode).
        auth : OrchidAuthContext | None
            Fully-formed auth context.  When provided, the ``user_id`` /
            ``tenant_id`` / ``access_token`` parameters are ignored.
        history : list[BaseMessage] | None
            Explicit conversation history.  When ``None`` and a chat_repo
            is configured, history is auto-loaded from persistence.  Not
            used when a checkpointer is active (the graph owns state).
        persist : bool
            When ``True`` (the default) and chat_repo is configured, the
            user message and assistant response are saved.  A chat row
            is auto-created on first use.

        Returns
        -------
        OrchidInvokeResult
            Either a normal response (``interrupted=False``) or, when the
            graph pauses for tool approval, an interrupt descriptor with
            ``interrupted=True`` and populated ``approvals_needed``.  Call
            :meth:`resume` to continue.
        """
        self._ensure_open()

        prepared = await self._prepare_invocation(
            message=message,
            chat_id=chat_id,
            user_id=user_id,
            tenant_id=tenant_id,
            access_token=access_token,
            auth=auth,
            history=history,
            persist=persist,
        )

        try:
            result = await self._graph.ainvoke(prepared.state, config=prepared.graph_config)
        except GraphInterrupt as exc:
            # HITL pause — don't persist; caller resumes via :meth:`resume`.
            return self._interrupt_to_result(exc, prepared.chat_id)

        response_text = result.get("final_response", "")
        agents_used = list(result.get("active_agents") or [])

        if persist and self._chat_repo is not None and response_text:
            await self._chat_repo.add_message(prepared.chat_id, "user", message)
            await self._chat_repo.add_message(
                prepared.chat_id,
                "assistant",
                response_text,
                agents_used=agents_used,
            )

        return self._result_from_graph_output(result, prepared.chat_id)

    async def resume(
        self,
        chat_id: str,
        *,
        approved: bool = True,
        persist: bool = True,
    ) -> OrchidInvokeResult:
        """Continue a graph previously paused by ``interrupt()``.

        Requires a checkpointer on the runtime.  The ``chat_id`` must
        match the one used when the graph was interrupted.

        Parameters
        ----------
        chat_id : str
            The chat/thread identifier of the paused graph.
        approved : bool
            ``True`` to execute the pending tool, ``False`` to reject it.
        persist : bool
            When ``True`` and chat_repo is configured, the synthesised
            assistant response is persisted.

        Raises
        ------
        RuntimeError
            If no checkpointer is configured (resume is impossible).
        """
        self._ensure_open()

        if self._runtime.checkpointer is None:
            raise RuntimeError(
                "Cannot resume without a checkpointer — "
                "construct the client with checkpointer_type='memory' (or 'sqlite'/'postgres')."
            )

        graph_config = {"configurable": {"thread_id": chat_id}}

        try:
            result = await self._graph.ainvoke(
                Command(resume={"approved": approved}),
                config=graph_config,
            )
        except GraphInterrupt as exc:
            return self._interrupt_to_result(exc, chat_id)

        response_text = result.get("final_response", "")
        agents_used = list(result.get("active_agents") or [])

        if persist and self._chat_repo is not None and response_text:
            await self._chat_repo.add_message(
                chat_id,
                "assistant",
                response_text,
                agents_used=agents_used,
            )

        return self._result_from_graph_output(result, chat_id)

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
        """Stream graph events as an async iterator.

        Each yielded item is ``(mode, chunk)`` using LangGraph's native
        stream shape.  Pass ``stream_mode="messages"`` for token-level
        streaming, ``"updates"`` (default) for per-node state updates,
        or a list to multiplex modes.

        Example::

            async for mode, chunk in client.stream("Hello", user_id="alice",
                                                   stream_mode=["messages"]):
                if mode == "messages":
                    token, meta = chunk
                    print(token.content, end="", flush=True)

        This method does not persist messages — the caller is responsible
        for saving both sides of the conversation once the stream drains
        (or uses its own persistence strategy).

        Checkpointer interaction
        ------------------------
        When the runtime has a ``BaseCheckpointSaver`` attached (HITL
        flows), the graph persists its own conversation state keyed by
        ``thread_id=chat_id``.  In that mode the client does **not**
        prepend history messages — passing the same ``chat_id`` across
        stream calls is enough for the graph to see prior turns.  When
        no checkpointer is configured and a ``chat_repo`` was wired,
        history is auto-loaded from the chat repo (same behaviour as
        :meth:`invoke`).
        """
        self._ensure_open()

        prepared = await self._prepare_invocation(
            message=message,
            chat_id=chat_id,
            user_id=user_id,
            tenant_id=tenant_id,
            access_token=access_token,
            auth=auth,
            history=history,
            persist=False,  # streaming never writes to the chat repo
        )

        async for mode, chunk in self._graph.astream(
            prepared.state,
            config=prepared.graph_config,
            stream_mode=stream_mode if isinstance(stream_mode, list) else [stream_mode],
        ):
            yield mode, chunk

    # ── Shutdown ────────────────────────────────────────────

    async def close(self) -> None:
        """Release resources owned by this client.

        Closes the chat repo, MCP token store, and checkpointer when this
        client was built via :meth:`from_config_path` (i.e. it owns them).
        Safe to call more than once.
        """
        if self._closed:
            return
        self._closed = True

        if not self._owns_resources:
            return

        from .bootstrap import _BootstrapResult, _teardown_runtime

        await _teardown_runtime(
            _BootstrapResult(
                runtime=self._runtime,
                config=self._config,
                chat_repo=self._chat_repo,  # type: ignore[arg-type]
                mcp_token_store=self._mcp_token_store,  # type: ignore[arg-type]
                mcp_client_registration_store=self._runtime.mcp_client_registration_store,  # type: ignore[arg-type]
                mcp_gateway_state_store=self._runtime.mcp_gateway_client_store,  # type: ignore[arg-type]
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

    # ── Internal helpers ────────────────────────────────────

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Orchid is closed; create a new instance before invoking.")

    async def _resolve_history(
        self,
        chat_id: str,
        explicit_history: list[BaseMessage] | None,
    ) -> list[BaseMessage]:
        """Resolve conversation history for the next invocation.

        Priority:
          1. Explicit ``history`` argument when provided.
          2. Loaded from chat_repo when available.
          3. Empty list.

        Skipped entirely when a checkpointer is active — the graph owns
        conversation state and we'd duplicate messages via ``add_messages``.
        """
        if self._runtime.checkpointer is not None:
            return []

        if explicit_history is not None:
            return list(explicit_history)

        if self._chat_repo is None:
            return []

        rows = await self._chat_repo.get_messages(chat_id, limit=50)
        out: list[BaseMessage] = []
        for row in rows:
            if row.role == "user":
                out.append(HumanMessage(content=row.content, id=row.id))
            elif row.role == "assistant":
                out.append(AIMessage(content=row.content, id=row.id))
        return out

    async def _prepare_invocation(
        self,
        *,
        message: str,
        chat_id: str | None,
        user_id: str,
        tenant_id: str,
        access_token: str,
        auth: OrchidAuthContext | None,
        history: list[BaseMessage] | None,
        persist: bool,
    ) -> _PreparedInvocation:
        """Assemble everything ``graph.ainvoke`` / ``astream`` need.

        Shared by :meth:`invoke` and :meth:`stream`.  Handles the four
        prelude concerns in one place:

          1. Build (or accept) the :class:`OrchidAuthContext`.
          2. Resolve the ``chat_id`` — when ``persist=True`` and no id
             is supplied, a new row is created in ``chat_repo`` so its
             backend-assigned id stays in sync with storage.  Otherwise
             a fresh UUID is generated locally (no DB write).
          3. Resolve conversation history (explicit arg, chat_repo, or
             none — skipped when a checkpointer owns state).
          4. Build the initial ``GraphState`` dict and ``thread_id`` config.
        """
        auth_ctx = auth or OrchidAuthContext(
            access_token=access_token,
            tenant_key=tenant_id,
            user_id=user_id,
        )

        if chat_id is None and persist and self._chat_repo is not None:
            # Let the backend assign the id so it stays in sync with storage.
            new_chat = await self._chat_repo.create_chat(
                tenant_id=auth_ctx.tenant_key,
                user_id=auth_ctx.user_id,
                title=message[:50],
            )
            effective_chat_id = new_chat.id
        else:
            effective_chat_id = chat_id or str(uuid.uuid4())

        resolved_history = await self._resolve_history(effective_chat_id, history)

        new_user_msg = HumanMessage(content=message)
        state: dict[str, Any] = {
            "messages": [*resolved_history, new_user_msg],
            "auth_context": auth_ctx,
            "chat_id": effective_chat_id,
        }
        graph_config = {"configurable": {"thread_id": effective_chat_id}}

        return _PreparedInvocation(
            auth_ctx=auth_ctx,
            chat_id=effective_chat_id,
            state=state,
            graph_config=graph_config,
        )

    def _result_from_graph_output(self, result: dict[str, Any], chat_id: str) -> OrchidInvokeResult:
        """Build an :class:`OrchidInvokeResult` from the graph's return payload.

        Shared by :meth:`invoke` and :meth:`resume`.
        """
        return OrchidInvokeResult(
            response=result.get("final_response", ""),
            chat_id=chat_id,
            agents_used=list(result.get("active_agents") or []),
            messages=list(result.get("messages") or []),
            mcp_context=dict(result.get("mcp_context") or {}),
            rag_context=dict(result.get("rag_context") or {}),
        )

    @staticmethod
    def _interrupt_to_result(exc: GraphInterrupt, chat_id: str) -> OrchidInvokeResult:
        """Convert a ``GraphInterrupt`` into an :class:`OrchidInvokeResult`.

        Kept as a :func:`staticmethod` so tests and subclasses can
        override the interrupt → result mapping without touching module
        state.
        """
        interrupts = exc.args[0] if exc.args else []
        approvals: list[OrchidPendingApproval] = []
        for i in interrupts:
            val = getattr(i, "value", None)
            if isinstance(val, dict):
                approvals.append(
                    OrchidPendingApproval(
                        tool=val.get("tool", ""),
                        args=val.get("args", {}),
                        agent=val.get("agent", ""),
                        interrupt_id=str(getattr(i, "id", "")),
                    )
                )
            else:
                approvals.append(
                    OrchidPendingApproval(
                        tool=str(val) if val is not None else "",
                        args={},
                        agent="",
                        interrupt_id=str(getattr(i, "id", "")),
                    )
                )
        return OrchidInvokeResult(
            response="",
            chat_id=chat_id,
            interrupted=True,
            approvals_needed=approvals,
        )


def _build_runtime_kwargs(
    *,
    config_path: str,
    apply_yaml: bool,
    config: Any,
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
    content_sources: Any = None,
    runtime_overrides: dict[str, Any] | None = None,
    skip_yaml_sections: set[str] | None = None,
) -> dict[str, Any]:
    """Build a kwargs dict for ``_build_runtime`` to avoid parameter drift."""
    return {
        "config_path": config_path,
        "apply_yaml": apply_yaml,
        "config": config,
        "agents_config_path": agents_config_path,
        "model": model,
        "vector_backend": vector_backend,
        "qdrant_url": qdrant_url,
        "embedding_model": embedding_model,
        "chat_storage_class": chat_storage_class,
        "chat_db_dsn": chat_db_dsn,
        "chat_extra_migrations_package": chat_extra_migrations_package,
        "mcp_token_store_class": mcp_token_store_class,
        "mcp_token_store_dsn": mcp_token_store_dsn,
        "mcp_client_registration_store_class": mcp_client_registration_store_class,
        "mcp_client_registration_store_dsn": mcp_client_registration_store_dsn,
        "mcp_gateway_state_store_class": mcp_gateway_state_store_class,
        "mcp_gateway_state_store_dsn": mcp_gateway_state_store_dsn,
        "checkpointer_type": checkpointer_type,
        "checkpointer_dsn": checkpointer_dsn,
        "startup_hook": startup_hook,
        "startup_hook_kwargs": startup_hook_kwargs,
        "content_sources": content_sources,
        "runtime_overrides": runtime_overrides,
        "skip_yaml_sections": skip_yaml_sections,
    }


def _resolve_agents_dir_from_md(
    root_path: Path,
    frontmatter: dict[str, Any],
    agents_dir_override: Path | None,
) -> Path:
    from .config.md_loader import _resolve_agents_dir

    return _resolve_agents_dir(root_path, frontmatter, agents_dir_override)


# ── Hybrid config factory (YAML root + MD agents) ──────────


async def _build_hybrid_config(
    *,
    orchid_yml_path: Path,
    agents_dir: Path,
    apply_yaml: bool,
    skip_yaml_sections: set[str] | None = None,
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
) -> Orchid:
    """Hybrid bootstrap: YAML root (``orchid.yml``) for infrastructure +
    top-level config, ``agents_dir/*.md`` for per-agent configs."""
    import yaml

    from .config.md_loader import (
        _AGENT_BEHAVIOUR_FIELDS,
        _load_agents,
        build_config_data_from_yaml,
    )
    from .config.schema import OrchidAgentsConfig
    from .config.yaml_env import apply_yaml_to_env

    # ── 1. Apply YAML infrastructure → env vars ─────────
    if apply_yaml and str(orchid_yml_path):
        os.environ.setdefault("ORCHID_CONFIG", str(orchid_yml_path))
        apply_yaml_to_env(str(orchid_yml_path), skip_sections=skip_yaml_sections)

    # ── 2. Read orchid.yml for top-level agent config ───
    yaml_data: dict[str, Any] = {}
    try:
        with open(orchid_yml_path, encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        pass

    # ── 3. Build top-level config data ───────────────────
    agent_configs, _ = _load_agents(agents_dir)
    config_data = build_config_data_from_yaml(yaml_data, agent_configs, _AGENT_BEHAVIOUR_FIELDS)

    # ── 4. Validate ──────────────────────────────────────
    config = OrchidAgentsConfig.model_validate(config_data)

    # ── 5. Build runtime with the pre-loaded config ──────
    result = await _build_runtime(
        **_build_runtime_kwargs(
            config_path=str(orchid_yml_path),
            apply_yaml=False,
            config=config,
            model=model,
            vector_backend=vector_backend,
            qdrant_url=qdrant_url,
            embedding_model=embedding_model,
            chat_storage_class=chat_storage_class,
            chat_db_dsn=chat_db_dsn,
            chat_extra_migrations_package=chat_extra_migrations_package,
            mcp_token_store_class=mcp_token_store_class,
            mcp_token_store_dsn=mcp_token_store_dsn,
            mcp_client_registration_store_class=mcp_client_registration_store_class,
            mcp_client_registration_store_dsn=mcp_client_registration_store_dsn,
            mcp_gateway_state_store_class=mcp_gateway_state_store_class,
            mcp_gateway_state_store_dsn=mcp_gateway_state_store_dsn,
            checkpointer_type=checkpointer_type,
            checkpointer_dsn=checkpointer_dsn,
            startup_hook=startup_hook,
            startup_hook_kwargs=startup_hook_kwargs,
            runtime_overrides=runtime_overrides,
            skip_yaml_sections=skip_yaml_sections,
        )
    )
    return Orchid(
        config=result.config,
        runtime=result.runtime,
        chat_repo=result.chat_repo,
        mcp_token_store=result.mcp_token_store,
        _owns_resources=True,
    )


@dataclass(frozen=True)
class _PreparedInvocation:
    """Return type of :meth:`Orchid._prepare_invocation`.

    Keeps the four prelude outputs in a typed bundle so call sites can
    refer to them by name instead of unpacking a tuple.  Frozen because
    the prelude is decided once per call and never needs to be mutated.
    """

    auth_ctx: OrchidAuthContext
    chat_id: str
    state: dict[str, Any]
    graph_config: dict[str, Any]
