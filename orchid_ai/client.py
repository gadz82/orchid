"""
OrchidClient — high-level programmatic entry point.

Lets any Python application (Django view, Celery task, background worker,
notebook, script) invoke the Orchid graph without going through HTTP or
the CLI.  Owns the full lifecycle: loads YAML config, builds the reader,
chat storage, MCP token store, optional checkpointer, and compiled graph,
and exposes a small imperative surface::

    async with OrchidClient.from_config_path("agents.yaml") as client:
        result = await client.invoke("Hello", user_id="alice")
        print(result.response)

Three call shapes are provided:
  * :meth:`invoke`  — blocking call, returns :class:`InvokeResult`
  * :meth:`stream`  — async iterator of events (token / agent / final)
  * :meth:`resume`  — continue after a human-in-the-loop interrupt

All heavy wiring mirrors ``orchid-api/lifecycle.py`` and
``orchid-cli/bootstrap.py`` — integrators get the same semantics whether
they call through HTTP, the CLI, or this in-process client.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from .config.loader import load_config
from .config.schema import AgentsConfig
from .core.repository import VectorStoreAdmin
from .core.state import AuthContext
from .graph.graph import build_graph
from .persistence.base import ChatStorage
from .persistence.factory import build_chat_storage
from .persistence.mcp_token_factory import build_mcp_token_store
from .rag.factory import build_reader
from .runtime import OrchidRuntime
from .utils import import_class

if TYPE_CHECKING:
    from .core.mcp import MCPTokenStore

logger = logging.getLogger(__name__)


# ── Result types ────────────────────────────────────────────────


@dataclass
class PendingApproval:
    """A single tool-approval request surfaced by ``interrupt()``."""

    tool: str
    args: dict[str, Any]
    agent: str
    interrupt_id: str


@dataclass
class InvokeResult:
    """Result of a single :meth:`OrchidClient.invoke` call.

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
    approvals_needed : list[PendingApproval]
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
    approvals_needed: list[PendingApproval] = field(default_factory=list)
    mcp_context: dict[str, Any] = field(default_factory=dict)
    rag_context: dict[str, Any] = field(default_factory=dict)


# ── OrchidClient ─────────────────────────────────────────────────


class OrchidClient:
    """Programmatic entry point for the Orchid framework.

    Lifecycle
    ---------
    Construct via the async factory :meth:`from_config_path` (recommended)
    or directly with a pre-built ``OrchidRuntime`` + ``AgentsConfig`` via
    :meth:`__init__`.  Always call :meth:`close` (or use ``async with``)
    to release database and checkpointer connections.

    Example — minimal::

        async with OrchidClient.from_config_path("orchid.yml") as client:
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
        client = OrchidClient(
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
        config: AgentsConfig,
        runtime: OrchidRuntime,
        chat_repo: ChatStorage | None = None,
        mcp_token_store: "MCPTokenStore | None" = None,
        _owns_resources: bool = False,
    ) -> None:
        """Low-level constructor — prefer :meth:`from_config_path` for most uses.

        Parameters
        ----------
        config : AgentsConfig
            Parsed and validated agent configuration.
        runtime : OrchidRuntime
            Pre-configured runtime (reader, chat_model, checkpointer, etc.).
        chat_repo : ChatStorage | None
            Optional chat persistence backend.  When ``None``, the client does
            not persist messages — history must be passed explicitly.
        mcp_token_store : MCPTokenStore | None
            Optional per-server OAuth token store.  Populated on ``runtime``.
        _owns_resources : bool
            Internal flag — ``True`` means :meth:`close` should shut down
            chat_repo / mcp_token_store / checkpointer.  Set by the
            :meth:`from_config_path` factory.
        """
        self._config = config
        self._runtime = runtime
        self._chat_repo = chat_repo
        self._mcp_token_store = mcp_token_store
        self._owns_resources = _owns_resources
        self._closed = False

        if mcp_token_store is not None and runtime.mcp_token_store is None:
            runtime.mcp_token_store = mcp_token_store

        self._graph = build_graph(config=config, runtime=runtime)

    # ── Construction helpers ─────────────────────────────────

    @classmethod
    async def from_config_path(
        cls,
        config_path: str,
        *,
        model: str = "",
        vector_backend: str = "",
        qdrant_url: str = "",
        embedding_model: str = "",
        chat_storage_class: str = "",
        chat_db_dsn: str = "",
        mcp_token_store_class: str = "",
        mcp_token_store_dsn: str = "",
        checkpointer_type: str = "",
        checkpointer_dsn: str = "",
        startup_hook: str = "",
        runtime_overrides: dict[str, Any] | None = None,
    ) -> "OrchidClient":
        """Build a fully-initialised client from an ``orchid.yml`` path.

        This is the highest-level entry point.  It mirrors the bootstrap
        logic of ``orchid-api`` and ``orchid-cli``: loads the YAML config,
        builds the reader, chat storage, MCP token store, optional
        checkpointer, and compiled graph.

        All string parameters fall back first to environment variables
        (``LITELLM_MODEL``, ``VECTOR_BACKEND``, ``QDRANT_URL``, ...) and
        then to sensible hardcoded defaults, mirroring the CLI resolution
        order (arg > env > hardcoded).

        Parameters
        ----------
        config_path : str
            Path to ``orchid.yml``.  Exported as ``ORCHID_CONFIG`` env var
            so downstream components can find it.
        model, vector_backend, qdrant_url, embedding_model : str
            Overrides for LLM + RAG settings.  Fall back to env / defaults.
        chat_storage_class, chat_db_dsn : str
            Override chat persistence backend.  Defaults to SQLite at
            ``~/.orchid/chats.db`` — the same default used by the CLI.
        mcp_token_store_class, mcp_token_store_dsn : str
            Override MCP per-server OAuth token store.  Defaults to SQLite
            on the same DB file as chat persistence.
        checkpointer_type, checkpointer_dsn : str
            Enable LangGraph state checkpointing.  Required for the HITL
            :meth:`resume` flow.  Valid types: ``"memory"``, ``"sqlite"``,
            ``"postgres"``, or a dotted class path.
        startup_hook : str
            Optional dotted path to an ``async(reader, settings) -> None``
            hook executed after the reader is built (e.g. for seeding RAG).
        runtime_overrides : dict[str, Any] | None
            Extra keyword arguments forwarded to ``OrchidRuntime``.  Use
            this to inject a custom ``chat_model``, ``reader``, or
            ``mcp_client_factory`` that bypass the built-in factories.

        Returns
        -------
        OrchidClient
            Ready-to-use client.  The caller owns the returned object and
            must call :meth:`close` (or use ``async with``) to release
            resources.
        """
        # Apply YAML to env (same behaviour as bootstrap / settings)
        from .config.yaml_env import apply_yaml_to_env

        if config_path:
            os.environ.setdefault("ORCHID_CONFIG", config_path)
            apply_yaml_to_env(config_path)

        agents_config_path = os.environ.get("AGENTS_CONFIG_PATH", "agents.yaml")
        agents_config = load_config(agents_config_path)

        # Resolve per-setting: explicit arg > env var > hardcoded default
        resolved_model = model or os.environ.get("LITELLM_MODEL", "ollama/llama3.2")
        resolved_backend = vector_backend or os.environ.get("VECTOR_BACKEND", "qdrant")
        resolved_qdrant = qdrant_url or os.environ.get("QDRANT_URL", "http://qdrant:6333")
        resolved_embedding = embedding_model or os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

        resolved_storage_class = (
            chat_storage_class
            or os.environ.get("CHAT_STORAGE_CLASS", "")
            or "orchid_ai.persistence.sqlite.SQLiteChatStorage"
        )
        resolved_storage_dsn = chat_db_dsn or os.environ.get("CHAT_DB_DSN", "") or "~/.orchid/chats.db"

        resolved_token_class = (
            mcp_token_store_class
            or os.environ.get("MCP_TOKEN_STORE_CLASS", "")
            or "orchid_ai.persistence.mcp_token_sqlite.SQLiteMCPTokenStore"
        )
        resolved_token_dsn = mcp_token_store_dsn or os.environ.get("MCP_TOKEN_STORE_DSN", "") or resolved_storage_dsn

        resolved_checkpointer_type = checkpointer_type or os.environ.get("CHECKPOINTER_TYPE", "")
        resolved_checkpointer_dsn = checkpointer_dsn or os.environ.get("CHECKPOINTER_DSN", "")
        resolved_startup_hook = startup_hook or os.environ.get("STARTUP_HOOK", "")

        overrides = dict(runtime_overrides or {})

        # ── Build reader (unless caller injected one) ────────
        reader = overrides.pop("reader", None) or build_reader(
            vector_backend=resolved_backend,
            qdrant_url=resolved_qdrant,
            embedding_model=resolved_embedding,
        )

        # Pre-create vector collections referenced by agents
        if isinstance(reader, VectorStoreAdmin):
            namespaces = [a.rag.namespace for a in agents_config.agents.values() if a.rag.enabled and a.rag.namespace]
            if namespaces:
                await reader.ensure_collections([*namespaces, "uploads"])

        # ── Chat persistence ────────────────────────────────
        chat_repo = build_chat_storage(class_path=resolved_storage_class, dsn=resolved_storage_dsn)
        await chat_repo.init_db()

        # ── MCP OAuth token store ───────────────────────────
        mcp_token_store = build_mcp_token_store(class_path=resolved_token_class, dsn=resolved_token_dsn)
        await mcp_token_store.init_db()

        # ── Startup hook (optional seeding / warm-up) ──────
        if resolved_startup_hook:
            hook_fn = import_class(resolved_startup_hook)
            await hook_fn(reader=reader, settings=None)
            logger.info("[OrchidClient] Startup hook executed: %s", resolved_startup_hook)

        # ── Runtime ─────────────────────────────────────────
        runtime = OrchidRuntime(
            default_model=resolved_model,
            reader=reader,
            mcp_token_store=mcp_token_store,
            **overrides,
        )

        # ── Optional checkpointer ──────────────────────────
        if resolved_checkpointer_type:
            from .checkpointing import build_checkpointer

            runtime.checkpointer = await build_checkpointer(
                checkpointer_type=resolved_checkpointer_type,
                dsn=resolved_checkpointer_dsn,
            )
            logger.info("[OrchidClient] Checkpointer: %s", type(runtime.checkpointer).__name__)

        logger.info(
            "[OrchidClient] Ready — model=%s, backend=%s, agents=%s",
            resolved_model,
            resolved_backend,
            list(agents_config.agents.keys()),
        )

        return cls(
            config=agents_config,
            runtime=runtime,
            chat_repo=chat_repo,
            mcp_token_store=mcp_token_store,
            _owns_resources=True,
        )

    # ── Async context manager ───────────────────────────────

    async def __aenter__(self) -> "OrchidClient":
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
    def config(self) -> AgentsConfig:
        """Parsed agents configuration."""
        return self._config

    @property
    def chat_repo(self) -> ChatStorage | None:
        """Chat storage backend, or ``None`` when running without persistence."""
        return self._chat_repo

    # ── Core operations ─────────────────────────────────────

    async def invoke(
        self,
        message: str,
        *,
        chat_id: str | None = None,
        user_id: str = "",
        tenant_id: str = "default",
        access_token: str = "",
        auth: AuthContext | None = None,
        history: list[BaseMessage] | None = None,
        persist: bool = True,
    ) -> InvokeResult:
        """Run a single request through the agent graph.

        Parameters
        ----------
        message : str
            The user's input.  Passed to the graph as a ``HumanMessage``.
        chat_id : str | None
            Chat / thread identifier.  A UUID is generated when ``None``.
            Also used as LangGraph ``thread_id`` for checkpointing.
        user_id, tenant_id : str
            Used to build a default :class:`AuthContext` when ``auth`` is
            not supplied.  Drive RAG scoping and chat ownership.
        access_token : str
            Bearer token forwarded to MCP servers (passthrough mode).
        auth : AuthContext | None
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
        InvokeResult
            Either a normal response (``interrupted=False``) or, when the
            graph pauses for tool approval, an interrupt descriptor with
            ``interrupted=True`` and populated ``approvals_needed``.  Call
            :meth:`resume` to continue.
        """
        self._ensure_open()

        auth_ctx = auth or AuthContext(
            access_token=access_token,
            tenant_key=tenant_id,
            user_id=user_id,
        )

        # Resolve / create the chat id
        if chat_id is None and persist and self._chat_repo is not None:
            # Let the backend assign the id so it stays in sync with storage
            new_chat = await self._chat_repo.create_chat(
                tenant_id=auth_ctx.tenant_key,
                user_id=auth_ctx.user_id,
                title=message[:50],
            )
            effective_chat_id = new_chat.id
        else:
            effective_chat_id = chat_id or str(uuid.uuid4())

        # Resolve history
        resolved_history = await self._resolve_history(effective_chat_id, history)

        initial_state = self._build_initial_state(
            auth_ctx=auth_ctx,
            chat_id=effective_chat_id,
            message=message,
            history=resolved_history,
        )

        graph_config = {"configurable": {"thread_id": effective_chat_id}}

        try:
            result = await self._graph.ainvoke(initial_state, config=graph_config)
        except GraphInterrupt as exc:
            # HITL pause — don't persist; caller resumes via :meth:`resume`.
            return _interrupt_to_result(exc, effective_chat_id)

        response_text = result.get("final_response", "")
        agents_used = list(result.get("active_agents") or [])

        if persist and self._chat_repo is not None and response_text:
            await self._chat_repo.add_message(effective_chat_id, "user", message)
            await self._chat_repo.add_message(
                effective_chat_id,
                "assistant",
                response_text,
                agents_used=agents_used,
            )

        return InvokeResult(
            response=response_text,
            chat_id=effective_chat_id,
            agents_used=agents_used,
            messages=list(result.get("messages") or []),
            mcp_context=dict(result.get("mcp_context") or {}),
            rag_context=dict(result.get("rag_context") or {}),
        )

    async def resume(
        self,
        chat_id: str,
        *,
        approved: bool = True,
        persist: bool = True,
    ) -> InvokeResult:
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
            return _interrupt_to_result(exc, chat_id)

        response_text = result.get("final_response", "")
        agents_used = list(result.get("active_agents") or [])

        if persist and self._chat_repo is not None and response_text:
            await self._chat_repo.add_message(
                chat_id,
                "assistant",
                response_text,
                agents_used=agents_used,
            )

        return InvokeResult(
            response=response_text,
            chat_id=chat_id,
            agents_used=agents_used,
            messages=list(result.get("messages") or []),
            mcp_context=dict(result.get("mcp_context") or {}),
            rag_context=dict(result.get("rag_context") or {}),
        )

    async def stream(
        self,
        message: str,
        *,
        chat_id: str | None = None,
        user_id: str = "",
        tenant_id: str = "default",
        access_token: str = "",
        auth: AuthContext | None = None,
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
        """
        self._ensure_open()

        auth_ctx = auth or AuthContext(
            access_token=access_token,
            tenant_key=tenant_id,
            user_id=user_id,
        )
        effective_chat_id = chat_id or str(uuid.uuid4())

        resolved_history = await self._resolve_history(effective_chat_id, history)

        initial_state = self._build_initial_state(
            auth_ctx=auth_ctx,
            chat_id=effective_chat_id,
            message=message,
            history=resolved_history,
        )

        graph_config = {"configurable": {"thread_id": effective_chat_id}}

        async for mode, chunk in self._graph.astream(
            initial_state,
            config=graph_config,
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

        if self._runtime.checkpointer is not None:
            from .checkpointing import shutdown_checkpointer

            await shutdown_checkpointer(self._runtime.checkpointer)
            self._runtime.checkpointer = None

        if self._mcp_token_store is not None:
            await self._mcp_token_store.close()
            self._mcp_token_store = None

        if self._chat_repo is not None:
            await self._chat_repo.close()
            self._chat_repo = None

    # ── Internal helpers ────────────────────────────────────

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("OrchidClient is closed; create a new instance before invoking.")

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

    def _build_initial_state(
        self,
        *,
        auth_ctx: AuthContext,
        chat_id: str,
        message: str,
        history: list[BaseMessage],
    ) -> dict[str, Any]:
        """Compose the state dict handed to ``graph.ainvoke`` / ``astream``."""
        messages: list[BaseMessage] = [*history, HumanMessage(content=message)]
        state: dict[str, Any] = {
            "messages": messages,
            "auth_context": auth_ctx,
            "chat_id": chat_id,
        }
        return state


# ── Module-level helpers ───────────────────────────────────────


def _interrupt_to_result(exc: GraphInterrupt, chat_id: str) -> InvokeResult:
    """Convert a ``GraphInterrupt`` into an :class:`InvokeResult`."""
    interrupts = exc.args[0] if exc.args else []
    approvals: list[PendingApproval] = []
    for i in interrupts:
        val = getattr(i, "value", None)
        if isinstance(val, dict):
            approvals.append(
                PendingApproval(
                    tool=val.get("tool", ""),
                    args=val.get("args", {}),
                    agent=val.get("agent", ""),
                    interrupt_id=str(getattr(i, "id", "")),
                )
            )
        else:
            approvals.append(
                PendingApproval(
                    tool=str(val) if val is not None else "",
                    args={},
                    agent="",
                    interrupt_id=str(getattr(i, "id", "")),
                )
            )
    return InvokeResult(
        response="",
        chat_id=chat_id,
        interrupted=True,
        approvals_needed=approvals,
    )
