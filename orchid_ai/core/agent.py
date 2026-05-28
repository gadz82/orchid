"""
Base agent abstraction — Open/Closed Principle.

Adding a new agent = subclass OrchidAgent + register in Composition Root.
Nothing else needs to change.

This module uses ONLY stdlib types for its own definitions.  The
``chat_model`` parameter is typed as ``Any`` because the concrete
LangChain ``BaseChatModel`` is an external dependency that must not
leak into ``core/``.  At runtime the field holds a ``BaseChatModel``
that supports ``ainvoke(messages) -> AIMessage``.

Shared helpers (``extract_user_query``, ``fetch_rag_context``,
``summarise``) are provided as concrete methods so that both
``GenericAgent`` and custom agents can reuse them without duplication.

Platform-agnostic: no vendor-specific references.
"""

from __future__ import annotations

import asyncio
import contextvars
import datetime as _dt
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Literal

from . import helpers as _helpers
from .content import OrchidContentSource
from .mcp import OrchidMCPClient
from .repository import OrchidVectorReader
from .scopes import OrchidRAGScope
from .state import OrchidAgentState

__all__ = ["OrchidAgent", "OrchidAgentRunContext"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrchidAgentRunContext:
    """Per-request state for a single agent activation.

    Bound to the asyncio task via :data:`_run_ctx_var` so concurrent
    activations of the same :class:`OrchidAgent` instance (LangGraph
    parallel fan-out, mini-agent forks, events runner) do not race on
    shared instance attributes.
    """

    auth: Any | None = None
    correlation_id: str | None = None
    chat_id: str | None = None
    message_id: str | None = None


_EMPTY_RUN_CONTEXT = OrchidAgentRunContext()

_run_ctx_var: contextvars.ContextVar[OrchidAgentRunContext] = contextvars.ContextVar(
    "orchid.agent_run_context",
    default=_EMPTY_RUN_CONTEXT,
)


class OrchidAgent(ABC):
    """
    Abstract base for all domain agents.

    The Supervisor uses ``name`` and ``description`` for auto-discovery:
    it builds the routing prompt dynamically from every registered agent.
    """

    def __init__(
        self,
        *,
        model_id: str = "",
        reader: OrchidVectorReader,
        mcp_clients: list[OrchidMCPClient] | None = None,
        chat_model: Any | None = None,
        **_kwargs: Any,
    ):
        # ``**_kwargs`` absorbs framework-injected extras the graph
        # builder always passes (currently ``config`` and
        # ``summary_config``).  Subclasses that need those values —
        # ``GenericAgent`` and any consumer subclass — accept them
        # explicitly in their own ``__init__``; subclasses that don't
        # care simply ignore them.  This keeps the base ABC stable
        # while letting the composition root hand every agent the same
        # full kwargs set without inspect.signature sniffing.
        self.model_id: str = model_id
        self.reader = reader
        self.mcp_clients = mcp_clients or []
        self._chat_model = chat_model  # BaseChatModel (duck-typed to avoid core/ deps)
        self._upload_namespace: str = _kwargs.pop("upload_namespace", "uploads")
        self._content_sources: list[OrchidContentSource] = _kwargs.pop("content_sources", None) or []

        # Events-layer wiring (Pollen + Bloom — §15.1).  The
        # signal emitter is injected once at startup by
        # ``Orchid.inject_signal_emitter`` and read-only from then
        # on, so it is safe as an instance attribute.  Per-request
        # state (auth, chat_id, message_id, correlation_id) lives
        # on :data:`_run_ctx_var` — see :meth:`set_run_context`.
        self._signal_emitter: Any | None = None

    # ── Per-request run context (ContextVar-backed) ─────────
    #
    # The four ``_current_*`` properties below proxy to
    # :data:`_run_ctx_var`, an asyncio-task-local ContextVar.  Two
    # concurrent activations of the same agent (e.g. LangGraph
    # ``Send()`` fan-out, mini-agent forks, the events runner
    # firing the same agent for two signals at once) each see
    # their own run context — no cross-talk.
    #
    # Setters are preserved for test fixtures and back-compat;
    # production code should prefer :meth:`set_run_context` /
    # :meth:`reset_run_context` (Token API) for explicit
    # restoration on return.

    @property
    def _current_auth(self) -> Any | None:
        return _run_ctx_var.get().auth

    @_current_auth.setter
    def _current_auth(self, value: Any | None) -> None:
        _run_ctx_var.set(replace(_run_ctx_var.get(), auth=value))

    @property
    def _current_correlation_id(self) -> str | None:
        return _run_ctx_var.get().correlation_id

    @_current_correlation_id.setter
    def _current_correlation_id(self, value: str | None) -> None:
        _run_ctx_var.set(replace(_run_ctx_var.get(), correlation_id=value))

    @property
    def _current_chat_id(self) -> str | None:
        return _run_ctx_var.get().chat_id

    @_current_chat_id.setter
    def _current_chat_id(self, value: str | None) -> None:
        _run_ctx_var.set(replace(_run_ctx_var.get(), chat_id=value))

    @property
    def _current_message_id(self) -> str | None:
        return _run_ctx_var.get().message_id

    @_current_message_id.setter
    def _current_message_id(self, value: str | None) -> None:
        _run_ctx_var.set(replace(_run_ctx_var.get(), message_id=value))

    def set_run_context(self, ctx: OrchidAgentRunContext) -> contextvars.Token:
        """Bind *ctx* to the current asyncio task and return a token.

        Pair every call with :meth:`reset_run_context` in a
        ``try/finally`` to restore the previous binding on return.
        The graph wrapper (``_AgentNodeWrapper._run_agent``) drives
        this around each agent activation so concurrent fan-out of
        the same agent stays isolated.
        """
        return _run_ctx_var.set(ctx)

    def reset_run_context(self, token: contextvars.Token) -> None:
        """Restore the run context bound before :meth:`set_run_context`."""
        _run_ctx_var.reset(token)

    # ── Identity ────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique agent identifier, e.g. 'knowledge-base'."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Human-readable description used by the Supervisor LLM
        to decide whether to activate this agent.
        e.g. 'Handles knowledge-base lookups, document retrieval, and FAQs.'
        """
        ...

    @property
    def rag_namespace(self) -> str:
        """Vector store namespace, e.g. 'knowledge-base'.

        Override if the agent uses RAG.  Default: empty string (no RAG).
        """
        return ""

    @property
    def upload_namespace(self) -> str:
        """Vector store namespace for uploaded documents, e.g. 'education-uploads'.

        Queried by :meth:`fetch_all_rag_context` alongside ``rag_namespace``.
        Default: ``"uploads"`` (backward compatible).  Set via
        ``OrchidRuntime.upload_namespace`` at deployment level.
        """
        return self._upload_namespace

    @property
    def content_sources(self) -> list[OrchidContentSource]:
        return self._content_sources

    # ── Execution ───────────────────────────────────────────

    @abstractmethod
    async def run(self, state: OrchidAgentState) -> OrchidAgentState:
        """
        Agent-specific logic.
        Receives the full graph state, returns the updated state.

        The ``auth_context`` in state carries the Bearer token.
        Pass it to ``self.mcp_clients`` when calling tools.
        """
        ...

    # ── Peer wiring (cross-agent skill steps) ────────────────

    def needs_peer_wiring(self) -> bool:
        """Return ``True`` if this agent requires cross-agent peer wiring.

        Subclasses that support delegation to other agents via skill
        steps must override this method and return ``True`` when at
        least one skill step references another agent.
        """
        return False

    def wire_peers(self, peers: dict[str, "OrchidAgent"]) -> None:
        """Receive a mapping of peer agent names to instances.

        Called by the graph builder after all agents are instantiated
        but before the graph is compiled.  Subclasses that override
        :meth:`needs_peer_wiring` should also override this method to
        store the peer mapping.
        """
        pass

    # ── Shared helpers ──────────────────────────────────────

    @staticmethod
    def extract_user_query(state: OrchidAgentState) -> str:
        """Walk messages in reverse to find the last human message."""
        for msg in reversed(state.get("messages", [])):
            # Duck-type check: LangChain message objects expose .type
            if hasattr(msg, "type") and msg.type == "human":
                return str(msg.content)
            if type(msg).__name__ == "HumanMessage":
                return str(msg.content)
        return ""

    @staticmethod
    def extract_conversation_history(
        state: OrchidAgentState,
        *,
        max_turns: int = 10,
        max_chars: int | None = None,
        skip_prefixes: tuple[str, ...] = ("[Supervisor",),
        strip_prefixes: tuple[str, ...] = (),
        truncation_strategy: str = "hard",
    ) -> list[dict[str, str]]:
        """Extract recent conversation history from graph state.

        Delegates to :func:`core.helpers.extract_conversation_history`.
        """
        return _helpers.extract_conversation_history(
            state,
            max_turns=max_turns,
            max_chars=max_chars,
            skip_prefixes=skip_prefixes,
            strip_prefixes=strip_prefixes,
            truncation_strategy=truncation_strategy,
        )

    @staticmethod
    async def compress_conversation_history(
        history: list[dict[str, str]],
        *,
        chat_model: Any,
        recent_turns: int = 3,
        running_summary: str | None = None,
        structured_output: bool = False,
        compression_system_prompt: str | None = None,
        compression_user_prompt: str | None = None,
        extension_user_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """Compress older conversation turns into a summary, keeping recent ones verbatim.

        Delegates to :func:`orchid_ai.core.helpers.compress_conversation_history`.
        See that function for full documentation.
        """
        return await _helpers.compress_conversation_history(
            history,
            chat_model=chat_model,
            recent_turns=recent_turns,
            running_summary=running_summary,
            structured_output=structured_output,
            compression_system_prompt=compression_system_prompt,
            compression_user_prompt=compression_user_prompt,
            extension_user_prompt=extension_user_prompt,
        )

    async def fetch_rag_context(
        self,
        query: str,
        scope: OrchidRAGScope,
        *,
        namespace: str | None = None,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant documents from the vector store.

        Delegates to :func:`core.helpers.fetch_rag_context`.
        """
        return await _helpers.fetch_rag_context(
            query,
            scope,
            reader=self.reader,
            namespace=namespace or self.rag_namespace,
            k=k,
            agent_name=self.name,
        )

    async def summarise(
        self,
        query: str,
        mcp_data: dict[str, Any],
        rag_data: list[dict[str, Any]],
        *,
        system_prompt: str,
        temperature: float = 0.2,
        conversation_history: list[dict[str, str]] | None = None,
        prior_tool_context: dict[str, Any] | None = None,
        history_reminder: str | None = None,
        prior_results_header: str | None = None,
        rag_section_header: str | None = None,
        user_content_template: str | None = None,
        prior_results_max_chars: int = 4000,
        **_kwargs: Any,
    ) -> str:
        """Use LLM to produce a human-readable summary of RAG + MCP data.

        Delegates to :func:`core.helpers.summarise`.  See that function
        for full documentation.  The ``*_header`` / ``*_template`` /
        ``*_reminder`` overrides forward straight through so callers
        threading per-agent
        :class:`~orchid_ai.config.schema.OrchidAgentPromptConfig` values
        reach the helper without re-implementing the assembly.

        Raises ``RuntimeError`` if no ``chat_model`` was injected.
        """
        if not self._chat_model:
            raise RuntimeError(
                f"[{self.name}] Cannot summarise: no chat model injected. Pass chat_model= when constructing the agent."
            )
        return await _helpers.summarise(
            query,
            mcp_data,
            rag_data,
            system_prompt=system_prompt,
            chat_model=self._chat_model,
            temperature=temperature,
            conversation_history=conversation_history,
            prior_tool_context=prior_tool_context,
            history_reminder=history_reminder,
            prior_results_header=prior_results_header,
            rag_section_header=rag_section_header,
            user_content_template=user_content_template,
            prior_results_max_chars=prior_results_max_chars,
        )

    async def fetch_all_rag_context(
        self,
        query: str,
        scope: OrchidRAGScope,
        *,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve from both the domain namespace AND the uploads namespace.

        Merges results by score and returns the top-k.
        """
        domain_docs, upload_docs = await asyncio.gather(
            self.fetch_rag_context(query, scope, namespace=self.rag_namespace, k=k),
            self.fetch_rag_context(query, scope, namespace=self.upload_namespace, k=k),
        )

        combined = domain_docs + upload_docs
        combined.sort(key=lambda d: d.get("score", 0), reverse=True)
        return combined[:k]

    # ── Signal emission (events layer — §15.1) ──────────────

    async def emit_signal(
        self,
        signal_type: str,
        payload: dict[str, Any],
        *,
        dedupe_key: str | None = None,
        identity: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        chat_id: str | Literal["self"] | None = None,
        chat_binding_mode: Literal["append_final_message", "append_with_metadata"] = "append_final_message",
        chat_binding_on_failure: Literal["post_error", "silent"] = "post_error",
        chat_binding_source_message_id: str | None = None,
    ) -> Any:
        """Emit a signal from inside this agent's run (§15.1).

        Defaults derived from the current run context:

        - ``source`` = ``f"internal:agent:{self.name}"``
        - ``tenant_key`` / ``user_id`` = inherited from
          ``self._current_auth`` (set by the graph wrapper around
          :meth:`run`)
        - ``correlation_id`` = current chat_id or job run_id
        - ``identity_claim`` = inherited
          (``act_as_user`` with the current user_id) unless
          overridden via ``identity``

        Pass ``chat_id="self"`` to bind the resulting Bloom to the
        chat this agent is currently running in.  Pass an explicit
        ``chat_id`` to bind to a different chat — the resolved auth
        of the matching trigger MUST have write permission on it,
        which the runner's ``_resolve_chat_binding`` re-verifies at
        runtime.

        Returns whatever the configured emitter returns — typically
        a :class:`SignalIngestResult` from
        :meth:`OrchidSignalDispatcher.ingest`.

        Raises ``RuntimeError`` when:

        - events are disabled (``self._signal_emitter`` is ``None``);
        - ``chat_id="self"`` is used outside a chat run.
        """
        if self._signal_emitter is None:
            raise RuntimeError("Signal emitter not injected — events disabled (set events.enabled=true in agents.yaml)")

        # Resolve the binding dict.  ``self`` is the friendly literal
        # for "bind to the chat I'm running in".
        binding: dict[str, Any] | None = None
        if chat_id is not None:
            resolved = self._current_chat_id if chat_id == "self" else chat_id
            if resolved is None:
                raise RuntimeError("chat_id='self' is only valid inside a chat run")
            # ``source_message_id`` anchors the in-chat live
            # progress card under the user message that produced
            # the binding (§LS5).  Auto-populate ONLY when
            # ``chat_id=='self'`` — across chats, anchoring to a
            # message id that lives in a different thread is
            # nonsense, so the field stays ``None`` unless the
            # caller passed it explicitly.
            if chat_binding_source_message_id is not None:
                source_msg_id: str | None = chat_binding_source_message_id
            elif chat_id == "self":
                source_msg_id = self._current_message_id
            else:
                source_msg_id = None
            binding = {
                "chat_id": resolved,
                "mode": chat_binding_mode,
                "on_failure": chat_binding_on_failure,
                "source_message_id": source_msg_id,
            }

        # Defer the SignalEnvelope import to runtime so ``core/agent.py``
        # does not pull ``core/events/`` into module-load time for
        # consumers that haven't enabled events.
        from .events.signal import SignalEnvelope

        auth = self._current_auth
        tenant_key = getattr(auth, "tenant_key", "default") if auth else "default"
        user_id = getattr(auth, "user_id", None) if auth else None

        # Default identity claim — consumer override wins.  When no
        # auth is present (graph hasn't wired the current run yet)
        # we fall back to a minimal claim with an empty user_id;
        # the trigger registry's mint probe rejects misconfigured
        # triggers at boot, so this only matters for wiring tests.
        if identity is None:
            identity_claim = {
                "mode": "act_as_user",
                "user_id": user_id or "",
            }
        else:
            identity_claim = dict(identity)

        envelope = SignalEnvelope(
            type=signal_type,
            payload=dict(payload),
            source=f"internal:agent:{self.name}",
            occurred_at=_dt.datetime.now(tz=_dt.UTC),
            tenant_key=tenant_key,
            user_id=user_id,
            correlation_id=correlation_id or self._current_correlation_id,
            dedupe_key=dedupe_key,
            identity_claim=identity_claim,
            chat_binding=binding,
        )
        return await self._signal_emitter.emit(envelope)

    # ── Built-in tool access ────────────────────────────────

    async def call_builtin_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """
        Call a registered built-in tool by name.

        Available to all agents (GenericAgent and custom subclasses).
        The tool must be registered in the tool registry (via ``agents.yaml``
        top-level ``tools:`` section or programmatic ``register_tool()``).
        """
        from ..config.tool_registry import build_tool_input, get_tool

        tool = get_tool(tool_name)
        output = await tool.invoke(build_tool_input(tool, **kwargs))
        return output.result
