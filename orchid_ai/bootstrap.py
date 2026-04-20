"""
Shared runtime bootstrap — single source of truth for building the
``OrchidRuntime`` + persistence + checkpointer stack from configuration.

Called by:
  * :class:`orchid_ai.client.OrchidClient.from_config_path` — in-process clients
  * ``orchid_api.lifecycle.setup_orchid`` — FastAPI server
  * ``orchid_cli.bootstrap.bootstrap``   — CLI

Keeping this logic in one place prevents the three entry points from
drifting.  Each caller adds only its own adapter-specific concerns
(tracing + HTTP client + identity resolver for the API, Typer context
for the CLI, etc.).

The orchestrator :func:`build_runtime` delegates to focused, unit-
testable helpers (:func:`_resolve_overrides`, :func:`_prepare_reader`,
:func:`_build_persistence`, :func:`_run_startup_hook`,
:func:`_attach_checkpointer`) — splitting a 200-line linear sequence
into a handful of 10-30 line functions with obvious contracts.

Example::

    from orchid_ai.bootstrap import build_runtime, teardown_runtime

    result = await build_runtime(config_path="orchid.yml")
    try:
        graph = build_graph(config=result.config, runtime=result.runtime)
        # ... use the graph ...
    finally:
        await teardown_runtime(result)
"""

from __future__ import annotations

import inspect
import logging
import os
from dataclasses import dataclass
from typing import Any

from .config.loader import load_config
from .config.schema import AgentsConfig
from .config.yaml_env import apply_yaml_to_env
from .core.mcp import MCPTokenStore
from .core.repository import VectorReader, VectorStoreAdmin
from .persistence.base import ChatStorage
from .persistence.factory import build_chat_storage
from .persistence.mcp_token_factory import build_mcp_token_store
from .rag.factory import build_reader
from .runtime import OrchidRuntime
from .utils import import_class

logger = logging.getLogger(__name__)

# ── Hardcoded defaults — used when neither args nor env supply a value ──

_DEFAULT_MODEL = "ollama/llama3.2"
_DEFAULT_VECTOR_BACKEND = "qdrant"
_DEFAULT_QDRANT_URL = "http://qdrant:6333"
_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
_DEFAULT_STORAGE_CLASS = "orchid_ai.persistence.sqlite.SQLiteChatStorage"
_DEFAULT_STORAGE_DSN = "~/.orchid/chats.db"
_DEFAULT_TOKEN_STORE_CLASS = "orchid_ai.persistence.mcp_token_sqlite.SQLiteMCPTokenStore"


@dataclass
class BootstrapResult:
    """Resources produced by :func:`build_runtime`.

    The caller owns every field and must release them — either by calling
    :func:`teardown_runtime`, or by shutting each resource down manually.
    """

    runtime: OrchidRuntime
    config: AgentsConfig
    chat_repo: ChatStorage
    mcp_token_store: MCPTokenStore


@dataclass(frozen=True)
class _ResolvedOverrides:
    """Settings resolved from ``arg → env → default`` precedence.

    Frozen because every field is decided once during resolution and
    treated as read-only by downstream helpers.  (``runtime_overrides``
    is a dict we still mutate via ``.pop`` — that's not ``dataclass``
    assignment, so it's fine.)
    """

    agents_config_path: str
    model: str
    vector_backend: str
    qdrant_url: str
    embedding_model: str
    storage_class: str
    storage_dsn: str
    extra_migrations_package: str | None
    token_store_class: str
    token_store_dsn: str
    checkpointer_type: str
    checkpointer_dsn: str
    startup_hook: str
    runtime_overrides: dict[str, Any]


async def build_runtime(
    *,
    config_path: str = "",
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
    checkpointer_type: str = "",
    checkpointer_dsn: str = "",
    startup_hook: str = "",
    startup_hook_kwargs: dict[str, Any] | None = None,
    runtime_overrides: dict[str, Any] | None = None,
    skip_yaml_sections: set[str] | None = None,
) -> BootstrapResult:
    """Build an :class:`OrchidRuntime` plus its persistence stack.

    Resolution order for every optional ``*`` parameter:
        explicit argument → environment variable → hardcoded default.

    Parameters
    ----------
    config_path : str
        Optional path to ``orchid.yml``.  When set and ``apply_yaml=True``
        the YAML is flattened into env vars (existing vars win).
    apply_yaml : bool
        Whether to run ``apply_yaml_to_env``.  Set to ``False`` when the
        caller already applied YAML (e.g. ``orchid-api`` does this at
        module import time via pydantic-settings).
    agents_config_path : str
        Path to ``agents.yaml``.  When empty, resolved from
        ``AGENTS_CONFIG_PATH`` env var or ``"agents.yaml"`` default.
    model, vector_backend, qdrant_url, embedding_model : str
        Primary LLM + RAG settings.
    chat_storage_class, chat_db_dsn : str
        Chat persistence backend.  Defaults to SQLite at
        ``~/.orchid/chats.db``.
    chat_extra_migrations_package : str | None
        Optional dotted import path of an integrator-supplied migrations
        package.  Applied after the framework's migrations by both the
        chat storage and the MCP token store (they share the DB).  See
        :class:`orchid_ai.persistence.migrations.runner.MigrationRunner`.
    mcp_token_store_class, mcp_token_store_dsn : str
        MCP per-server OAuth token store.  DSN defaults to the chat DB
        path (same file).
    checkpointer_type, checkpointer_dsn : str
        Optional LangGraph checkpointer for state persistence.  Required
        for the HITL resume flow.
    startup_hook : str
        Optional dotted path to an ``async def(**kwargs) -> None`` hook
        invoked after the reader is built (e.g. for seeding RAG).
    startup_hook_kwargs : dict | None
        Keyword arguments forwarded to the startup hook.  The hook always
        receives at least ``reader=<VectorReader>``.
    runtime_overrides : dict | None
        Extra keyword arguments forwarded to :class:`OrchidRuntime`.  Use
        this to inject a pre-built ``chat_model`` or a custom
        ``mcp_client_factory`` that bypasses the built-in defaults.
    skip_yaml_sections : set[str] | None
        Forwarded to :func:`apply_yaml_to_env`.  Use for callers (like the
        CLI) that want their own defaults to win over YAML for specific
        sections (e.g. ``{"storage"}``).
    """
    # ── 1. YAML → env (optional) ──────────────────────────────
    if apply_yaml and config_path:
        os.environ.setdefault("ORCHID_CONFIG", config_path)
        apply_yaml_to_env(config_path, skip_sections=skip_yaml_sections)

    # ── 2. Resolve settings precedence ────────────────────────
    overrides = _resolve_overrides(
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
        checkpointer_type=checkpointer_type,
        checkpointer_dsn=checkpointer_dsn,
        startup_hook=startup_hook,
        runtime_overrides=runtime_overrides,
    )

    # ── 3. Load agents config ─────────────────────────────────
    agents_config = load_config(overrides.agents_config_path)

    # ── 4. Reader + pre-created collections ───────────────────
    reader = await _prepare_reader(overrides, agents_config)

    # ── 5. Persistence (chat storage + MCP token store) ──────
    chat_repo, mcp_token_store = await _build_persistence(overrides)

    # ── 6. Startup hook (optional) ───────────────────────────
    await _run_startup_hook(overrides.startup_hook, reader, startup_hook_kwargs)

    # ── 7. Assemble runtime + optional checkpointer ──────────
    runtime = OrchidRuntime(
        default_model=overrides.model,
        reader=reader,
        mcp_token_store=mcp_token_store,
        **overrides.runtime_overrides,
    )
    await _attach_checkpointer(runtime, overrides.checkpointer_type, overrides.checkpointer_dsn)

    logger.info(
        "[Bootstrap] Ready — model=%s, backend=%s, storage=%s, agents=%s",
        overrides.model,
        overrides.vector_backend,
        overrides.storage_class.rsplit(".", 1)[-1],
        list(agents_config.agents.keys()),
    )

    return BootstrapResult(
        runtime=runtime,
        config=agents_config,
        chat_repo=chat_repo,
        mcp_token_store=mcp_token_store,
    )


# ── Focused step helpers ─────────────────────────────────────


def _resolve_overrides(
    *,
    agents_config_path: str,
    model: str,
    vector_backend: str,
    qdrant_url: str,
    embedding_model: str,
    chat_storage_class: str,
    chat_db_dsn: str,
    chat_extra_migrations_package: str | None,
    mcp_token_store_class: str,
    mcp_token_store_dsn: str,
    checkpointer_type: str,
    checkpointer_dsn: str,
    startup_hook: str,
    runtime_overrides: dict[str, Any] | None,
) -> _ResolvedOverrides:
    """Apply ``arg → env → default`` precedence and return a typed bundle."""
    storage_dsn = chat_db_dsn or os.environ.get("CHAT_DB_DSN", "") or _DEFAULT_STORAGE_DSN
    extra_pkg = chat_extra_migrations_package or os.environ.get("CHAT_EXTRA_MIGRATIONS_PACKAGE", "") or None
    return _ResolvedOverrides(
        agents_config_path=agents_config_path or os.environ.get("AGENTS_CONFIG_PATH", "agents.yaml"),
        model=model or os.environ.get("LITELLM_MODEL", _DEFAULT_MODEL),
        vector_backend=vector_backend or os.environ.get("VECTOR_BACKEND", _DEFAULT_VECTOR_BACKEND),
        qdrant_url=qdrant_url or os.environ.get("QDRANT_URL", _DEFAULT_QDRANT_URL),
        embedding_model=embedding_model or os.environ.get("EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL),
        storage_class=(chat_storage_class or os.environ.get("CHAT_STORAGE_CLASS", "") or _DEFAULT_STORAGE_CLASS),
        storage_dsn=storage_dsn,
        extra_migrations_package=extra_pkg,
        token_store_class=(
            mcp_token_store_class or os.environ.get("MCP_TOKEN_STORE_CLASS", "") or _DEFAULT_TOKEN_STORE_CLASS
        ),
        token_store_dsn=mcp_token_store_dsn or os.environ.get("MCP_TOKEN_STORE_DSN", "") or storage_dsn,
        checkpointer_type=checkpointer_type or os.environ.get("CHECKPOINTER_TYPE", ""),
        checkpointer_dsn=checkpointer_dsn or os.environ.get("CHECKPOINTER_DSN", ""),
        startup_hook=startup_hook or os.environ.get("STARTUP_HOOK", ""),
        runtime_overrides=dict(runtime_overrides or {}),
    )


async def _prepare_reader(
    overrides: _ResolvedOverrides,
    agents_config: AgentsConfig,
) -> VectorReader:
    """Build (or accept) the vector reader and pre-create namespaces.

    A caller-provided ``reader`` in ``runtime_overrides`` wins over the
    built-in factory.  When any agent declares ``rag.enabled`` but the
    reader isn't :class:`VectorStoreAdmin`, log a single warning so the
    operator knows retrievals will silently return empty.
    """
    reader = overrides.runtime_overrides.pop("reader", None) or build_reader(
        vector_backend=overrides.vector_backend,
        qdrant_url=overrides.qdrant_url,
        embedding_model=overrides.embedding_model,
    )

    namespaces = [a.rag.namespace for a in agents_config.agents.values() if a.rag.enabled and a.rag.namespace]
    if not namespaces:
        return reader

    if isinstance(reader, VectorStoreAdmin):
        await reader.ensure_collections([*namespaces, "uploads"])
    else:
        logger.warning(
            "[Bootstrap] %d agent(s) declare rag.enabled but reader %s does not implement "
            "VectorStoreAdmin — collections will not be auto-created; retrievals may return empty.",
            len(namespaces),
            type(reader).__name__,
        )
    return reader


async def _build_persistence(
    overrides: _ResolvedOverrides,
) -> tuple[ChatStorage, MCPTokenStore]:
    """Initialise chat storage and MCP token store (idempotent ``init_db``).

    Both stores share the integrator ``extra_migrations_package`` (they
    share the underlying DB), so a single YAML entry covers both.
    """
    chat_repo = build_chat_storage(
        class_path=overrides.storage_class,
        dsn=overrides.storage_dsn,
        extra_migrations_package=overrides.extra_migrations_package,
    )
    await chat_repo.init_db()

    mcp_token_store = build_mcp_token_store(
        class_path=overrides.token_store_class,
        dsn=overrides.token_store_dsn,
        extra_migrations_package=overrides.extra_migrations_package,
    )
    await mcp_token_store.init_db()
    return chat_repo, mcp_token_store


async def _run_startup_hook(
    hook_path: str,
    reader: VectorReader,
    extra_kwargs: dict[str, Any] | None,
) -> None:
    """Resolve and invoke the startup hook, validating its signature first.

    Contract: ``async def(reader, settings, **_) -> None``.  ``settings``
    is ``None`` when the caller is the client / CLI; the API passes its
    ``Settings`` instance via *extra_kwargs*.  Missing / invalid hooks
    raise :class:`TypeError` with a clear message (see
    :func:`_validate_startup_hook`).
    """
    if not hook_path:
        return

    hook_fn = import_class(hook_path)
    hook_kwargs: dict[str, Any] = {"reader": reader, "settings": None}
    hook_kwargs.update(extra_kwargs or {})
    _validate_startup_hook(hook_path, hook_fn, hook_kwargs)
    await hook_fn(**hook_kwargs)
    logger.info("[Bootstrap] Startup hook executed: %s", hook_path)


async def _attach_checkpointer(
    runtime: OrchidRuntime,
    checkpointer_type: str,
    checkpointer_dsn: str,
) -> None:
    """Mount a LangGraph checkpointer onto *runtime* when configured."""
    if not checkpointer_type:
        return
    from .checkpointing import build_checkpointer

    runtime.checkpointer = await build_checkpointer(
        checkpointer_type=checkpointer_type,
        dsn=checkpointer_dsn,
    )
    logger.info("[Bootstrap] Checkpointer: %s", type(runtime.checkpointer).__name__)


def _validate_startup_hook(path: str, hook_fn: Any, kwargs: dict[str, Any]) -> None:
    """Fail fast when the configured hook is unusable.

    Checks are ordered cheap-first so the most informative error wins:

      1. Callable — rejects strings, classes, modules.
      2. Async — rejects ``def`` (would silently return a coroutine).
      3. Signature — rejects hooks that can't accept the kwargs we pass.

    Hooks are expected to match ``async def(reader, settings, **_) -> None``.
    We don't require the full signature — only that ``bind_partial`` accepts
    our kwargs — so integrators can accept additional keyword-only params
    without breaking.
    """
    if not callable(hook_fn):
        raise TypeError(f"Startup hook '{path}' must be callable (got {type(hook_fn).__name__}).")

    if not inspect.iscoroutinefunction(hook_fn):
        raise TypeError(
            f"Startup hook '{path}' must be an async function (got {type(hook_fn).__name__}). Use `async def`."
        )

    try:
        sig = inspect.signature(hook_fn)
        sig.bind_partial(**kwargs)
    except TypeError as exc:
        raise TypeError(
            f"Startup hook '{path}' does not accept the expected kwargs "
            f"(reader, settings). Hook must be `async def(reader, settings, **_)`. "
            f"Bind error: {exc}"
        ) from exc


async def teardown_runtime(result: BootstrapResult) -> None:
    """Idempotent cleanup for resources produced by :func:`build_runtime`.

    Safe to call multiple times; each shutdown step short-circuits when
    its resource is already ``None`` or closed.
    """
    if result.runtime.checkpointer is not None:
        from .checkpointing import shutdown_checkpointer

        await shutdown_checkpointer(result.runtime.checkpointer)
        result.runtime.checkpointer = None

    if result.mcp_token_store is not None:
        await result.mcp_token_store.close()

    if result.chat_repo is not None:
        await result.chat_repo.close()
