"""
Base agent abstraction — Open/Closed Principle (ADR-008).

Adding a new agent = subclass BaseAgent + register in Composition Root.
Nothing else needs to change.

This module uses ONLY stdlib types.  The `llm` parameter is typed as
`Any` because the concrete LLM wrapper (LiteLLM) is a third-party
dependency that must not leak into core/.

Shared helpers (``extract_user_query``, ``fetch_rag_context``,
``summarise``) are provided as concrete methods so that both
``GenericAgent`` and custom agents can reuse them without duplication.

Platform-agnostic: no vendor-specific references.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from .llm_provider import LLMProvider
from .mcp import MCPClient
from .repository import VectorReader
from .state import AgentState
from ..rag.scopes import RAGScope

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base for all domain agents.

    The Supervisor uses ``name`` and ``description`` for auto-discovery:
    it builds the routing prompt dynamically from every registered agent.
    """

    def __init__(
        self,
        *,
        llm: Any,
        reader: VectorReader,
        mcp_clients: list[MCPClient] | None = None,
        llm_service: LLMProvider | None = None,
    ):
        self.llm = llm
        self.reader = reader
        self.mcp_clients = mcp_clients or []
        self._llm_service = llm_service

    # ── Identity ────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique agent identifier, e.g. 'notifications'."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Human-readable description used by the Supervisor LLM
        to decide whether to activate this agent.
        e.g. 'Manages notifications: create, list, templates.'
        """
        ...

    @property
    def rag_namespace(self) -> str:
        """Vector store namespace, e.g. 'notifications'.

        Override if the agent uses RAG.  Default: empty string (no RAG).
        """
        return ""

    # ── Execution ───────────────────────────────────────────

    @abstractmethod
    async def run(self, state: AgentState) -> AgentState:
        """
        Agent-specific logic.
        Receives the full graph state, returns the updated state.

        The ``auth_context`` in state carries the Bearer token
        (ADR-010).  Pass it to ``self.mcp_clients`` when calling tools.
        """
        ...

    # ── Shared helpers ──────────────────────────────────────

    @staticmethod
    def extract_user_query(state: AgentState) -> str:
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
        state: AgentState,
        *,
        max_turns: int = 10,
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
        state : AgentState
            Full graph state containing ``messages``.
        max_turns : int
            Maximum number of user/assistant exchanges to keep.
            Older messages are trimmed to avoid blowing up the context window.
        skip_prefixes : tuple[str, ...]
            Messages whose content starts with any of these prefixes are
            dropped entirely (e.g. internal supervisor routing messages).
        strip_prefixes : tuple[str, ...]
            Prefixes to strip from assistant messages (e.g. ``"[Notifications Agent]\\n"``).
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
                history.append({"role": "user", "content": content})
            elif msg_type in ("ai", "aimessage"):
                # Strip known agent prefixes for a clean dialogue
                for prefix in strip_prefixes:
                    if content.startswith(prefix):
                        content = content[len(prefix) :]
                        break
                history.append({"role": "assistant", "content": content})

        # Drop the last user message — it will be added separately as the current query
        if history and history[-1]["role"] == "user":
            history = history[:-1]

        # Keep only the most recent turns to avoid exceeding context limits
        max_messages = max_turns * 2
        if len(history) > max_messages:
            history = history[-max_messages:]

        return history

    async def fetch_rag_context(
        self,
        query: str,
        scope: RAGScope,
        *,
        namespace: str | None = None,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant documents from the vector store.

        Parameters
        ----------
        query : str
            User query to embed and search.
        scope : RAGScope
            Hierarchical scope for filtering (tenant → user → chat → agent).
        namespace : str | None
            Qdrant collection name.  Falls back to ``self.rag_namespace``.
        k : int
            Number of documents to retrieve.
        """
        ns = namespace or self.rag_namespace
        try:
            results = await self.reader.retrieve(
                query=query,
                namespace=ns,
                k=k,
                scope=scope,
            )
            rag_docs = [
                {
                    "content": r.document.content,
                    "score": round(r.score, 3),
                    "metadata": {
                        mk: mv for mk, mv in r.document.metadata.items() if mk not in ("content", "embedding")
                    },
                }
                for r in results
            ]
            logger.info("[%s] RAG retrieved %d docs (scope=%s)", self.name, len(rag_docs), scope.tenant_id)
            return rag_docs
        except (ConnectionError, TimeoutError, OSError, ValueError) as exc:
            logger.warning("[%s] RAG retrieval failed: %s", self.name, exc)
            return []

    async def summarise(
        self,
        query: str,
        mcp_data: dict[str, Any],
        rag_data: list[dict[str, Any]],
        *,
        system_prompt: str,
        model: str | None = None,
        temperature: float = 0.2,
        conversation_history: list[dict[str, str]] | None = None,
        prior_tool_context: dict[str, Any] | None = None,
    ) -> str:
        """
        Use LLM to produce a human-readable summary of RAG + MCP data.

        When ``conversation_history`` is provided (from
        ``extract_conversation_history``), it is injected between the
        system prompt and the current user query so the LLM has
        multi-turn context (e.g. knows which entity the user is
        referring to with "tell me more about the second one").

        When ``prior_tool_context`` is provided, it is appended to the
        system prompt so the LLM knows what tools returned in previous
        invocations (e.g. a ``create_notification`` response with IDs
        and details from a prior turn).

        Requires an injected ``LLMProvider`` (DIP-compliant).

        Parameters
        ----------
        query : str
            The current user query.
        mcp_data : dict
            Tool call results to summarise.
        rag_data : list[dict]
            RAG retrieval results for background context.
        system_prompt : str
            System prompt for the LLM.
        model : str | None
            LLM model override; falls back to ``self.llm``.
        temperature : float
            Sampling temperature.
        conversation_history : list[dict] | None
            Prior conversation turns from ``extract_conversation_history()``.
            Each entry is ``{"role": "user"|"assistant", "content": ...}``.
        prior_tool_context : dict | None
            Tool results from previous agent invocations (from
            ``state["mcp_context"]``).  Injected into the system prompt
            so the LLM has grounding on what was previously fetched or
            created.

        Raises
        ------
        RuntimeError
            If no ``LLMProvider`` was injected during construction.
        """
        if not self._llm_service:
            raise RuntimeError(
                f"[{self.name}] Cannot summarise: no LLMProvider injected. "
                "Pass llm_service= when constructing the agent."
            )

        _model = model or (self.llm if isinstance(self.llm, str) else str(self.llm))

        # ── Build enriched system prompt ──
        enriched_system = system_prompt

        # Add multi-turn focus instruction when history is available
        if conversation_history:
            enriched_system += (
                "\n\nIMPORTANT: The conversation history below shows prior exchanges. "
                "Always focus on the user's LATEST message and its relationship to "
                "the most recent topic. Do NOT change topic or introduce unrelated "
                "content unless the user explicitly asks for something new."
            )

        # Append prior tool results so the LLM knows what was returned in earlier turns
        if prior_tool_context:
            prior_json = json.dumps(prior_tool_context, indent=2, default=str)[:4000]
            enriched_system += f"\n\n--- Previous Tool Results (from prior turns) ---\n{prior_json}"

        rag_section = ""
        if rag_data:
            rag_section = "Background knowledge (from RAG):\n" + json.dumps(rag_data, indent=2, default=str) + "\n\n"

        user_content = (
            f"User query: {query}\n\n{rag_section}Live data (from API):\n{json.dumps(mcp_data, indent=2, default=str)}"
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": enriched_system},
        ]

        # Inject conversation history so the LLM has multi-turn context
        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_content})

        return await self._llm_service.complete(
            _model,
            messages,
            temperature=temperature,
        )

    async def fetch_all_rag_context(
        self,
        query: str,
        scope: RAGScope,
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

    # ── Built-in tool access (ADR-017) ──────────────────────

    async def call_builtin_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """
        Call a registered built-in tool by name.

        Available to all agents (GenericAgent and custom subclasses).
        The tool must be registered in the tool registry (via ``agents.yaml``
        top-level ``tools:`` section or programmatic ``register_tool()``).
        """
        from ..config.tool_registry import call_tool

        return await call_tool(tool_name, **kwargs)
