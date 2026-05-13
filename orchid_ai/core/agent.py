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
import datetime as _dt
import logging
from abc import ABC, abstractmethod
from typing import Any, Literal

from . import helpers as _helpers
from .mcp import OrchidMCPClient
from .repository import OrchidVectorReader
from .scopes import OrchidRAGScope
from .state import OrchidAgentState

__all__ = ["OrchidAgent"]

logger = logging.getLogger(__name__)


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

        # ── Events-layer wiring (Pollen + Bloom — §15.1) ─────
        # The graph builder injects these when ``events.enabled:
        # true``; otherwise they stay ``None`` and any
        # ``await self.emit_signal(...)`` raises a clear
        # ``RuntimeError`` rather than silently dropping.
        # Concrete graph wiring lands in Phase 4's lifespan; the
        # API ships now so consumers can plan against it.
        self._signal_emitter: Any | None = None
        self._current_auth: Any | None = None
        self._current_correlation_id: str | None = None
        self._current_chat_id: str | None = None
        # The id of the user message that triggered the current
        # chat turn.  Set by the graph wrapper alongside
        # ``_current_chat_id`` so that ``emit_signal(chat_id="self")``
        # can auto-populate ``ChatBinding.source_message_id``
        # (§LS5).  ``None`` outside chat turns (job runs, scheduled
        # firings) — chained Bloom emissions therefore default to
        # ``source_message_id=None`` and the frontend renders them
        # in the bottom dock fallback.
        self._current_message_id: str | None = None

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
    ) -> list[dict[str, str]]:
        """Extract recent conversation history from graph state.

        Returns a list of ``{"role": "user"|"assistant", "content": ...}``
        dicts suitable for injection into an LLM message list.

        The last user message is **excluded** — it should be appended
        separately as the current query to avoid duplication.

        Parameters
        ----------
        state : OrchidAgentState
            Full graph state containing ``messages``.
        max_turns : int
            Maximum number of user/assistant exchanges to keep.
            Older messages are trimmed to avoid blowing up the context window.
        max_chars : int | None
            When set, individual message content is truncated to this
            many characters (with an ``…`` suffix).  ``None`` means
            no truncation.
        skip_prefixes : tuple[str, ...]
            Messages whose content starts with any of these prefixes are
            dropped entirely (e.g. internal supervisor routing messages).
        strip_prefixes : tuple[str, ...]
            Prefixes to strip from assistant messages (e.g. ``"[MyAgent]\\n"``).
            Only the first matching prefix is stripped.
        """
        all_messages = state.get("messages", [])
        if not all_messages:
            return []

        history: list[dict[str, str]] = []
        for msg in all_messages:
            # Duck-type: LangChain messages expose .type and .content
            msg_type = getattr(msg, "type", None) or type(msg).__name__.lower()
            content = str(msg.content) if hasattr(msg, "content") else str(msg)

            if not content.strip():
                continue

            # Skip internal messages (e.g. supervisor routing)
            if any(content.startswith(prefix) for prefix in skip_prefixes):
                continue

            if msg_type in ("human", "humanmessage"):
                if max_chars is not None and len(content) > max_chars:
                    content = content[:max_chars] + "…"
                history.append({"role": "user", "content": content})
            elif msg_type in ("ai", "aimessage"):
                # Strip known agent prefixes for a clean dialogue
                for prefix in strip_prefixes:
                    if content.startswith(prefix):
                        content = content[len(prefix) :]
                        break
                if max_chars is not None and len(content) > max_chars:
                    content = content[:max_chars] + "…"
                history.append({"role": "assistant", "content": content})

        # Drop the last user message — it will be added separately as the current query
        if history and history[-1]["role"] == "user":
            history = history[:-1]

        # Keep only the most recent turns to avoid exceeding context limits
        max_messages = max_turns * 2
        if len(history) > max_messages:
            history = history[-max_messages:]

        return history

    @staticmethod
    async def compress_conversation_history(
        history: list[dict[str, str]],
        *,
        chat_model: Any,
        recent_turns: int = 3,
    ) -> list[dict[str, str]]:
        """Compress older conversation turns into a summary, keeping recent ones verbatim.

        Implements the **sliding-window with summarization** pattern:
        messages older than ``recent_turns`` are condensed into a single
        system-style summary paragraph by calling the LLM, while the
        most recent ``recent_turns`` exchanges are preserved as-is.

        If the history fits within ``recent_turns * 2`` messages, it is
        returned unchanged (no LLM call).

        Parameters
        ----------
        history : list[dict[str, str]]
            Full conversation history from ``extract_conversation_history()``.
        chat_model : BaseChatModel
            LangChain chat model for the summarization call (duck-typed).
        recent_turns : int
            Number of recent user/assistant exchange pairs to keep verbatim.
            Default ``10`` (= 20 messages).

        Returns
        -------
        list[dict[str, str]]
            Compressed history: one ``{"role": "assistant", "content":
            "Conversation summary: ..."}`` entry followed by the recent
            verbatim messages.  If no compression was needed, returns the
            original history unchanged.
        """
        recent_count = recent_turns * 2
        if len(history) <= recent_count:
            return history

        older = history[:-recent_count]
        recent = history[-recent_count:]

        # Build a simple transcript for the LLM to summarize
        transcript_lines = [f"{m['role'].upper()}: {m['content']}" for m in older]
        transcript = "\n".join(transcript_lines)

        summary_prompt = (
            "Summarize the following conversation excerpt in 2-4 sentences. "
            "Focus on: (1) the key topics discussed, (2) any entities or "
            "names mentioned, (3) actions taken or decisions made, (4) any "
            "outstanding questions. Be factual and concise.\n\n"
            f"{transcript}"
        )

        try:
            result = await chat_model.ainvoke(
                [
                    {"role": "system", "content": "You are a conversation summarizer. Output only the summary."},
                    {"role": "user", "content": summary_prompt},
                ],
                temperature=0.0,
            )
            summary_text = result.content or ""
        except Exception as exc:
            logger.warning("History compression failed (%s), falling back to truncation", exc)
            # Fallback: just keep the recent turns (no summary)
            return recent

        compressed: list[dict[str, str]] = [
            {"role": "assistant", "content": f"[Conversation summary]\n{summary_text.strip()}"},
        ]
        compressed.extend(recent)
        return compressed

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
            self.fetch_rag_context(query, scope, namespace="uploads", k=k),
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
        from ..config.tool_registry import call_tool

        return await call_tool(tool_name, **kwargs)
