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
    ) -> str:
        """
        Use LLM to produce a human-readable summary of RAG + MCP data.

        Requires an injected ``LLMProvider`` (DIP-compliant).

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

        rag_section = ""
        if rag_data:
            rag_section = "Background knowledge (from RAG):\n" + json.dumps(rag_data, indent=2, default=str) + "\n\n"

        user_content = (
            f"User query: {query}\n\n{rag_section}Live data (from API):\n{json.dumps(mcp_data, indent=2, default=str)}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

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
