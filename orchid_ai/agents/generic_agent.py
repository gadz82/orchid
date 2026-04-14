"""
GenericAgent — config-driven agent that requires no custom Python code.

Implements the standard 6-step flow entirely from YAML configuration (ADR-016, ADR-017):
  1. RAG retrieval (tenant-aware, ADR-014)
  2. Skill check — if query matches an agent skill, run it and skip to step 6
  3. MCP tool calls (per server, per strategy)
  4. Built-in tool calls (ADR-017)
  5. Dynamic RAG injection (ADR-005 feedback loop)
  6. LLM summarisation using the prompt from config

Tool call strategies (MCP):
  - ``all`` — call every whitelisted tool, collect results
  - ``sequential`` — call tools in order, chaining context forward
  - ``llm_decides`` — ask the LLM which tools to call and with what args

Collaborators (SRP — each extracted into its own module):
  - ``SkillDetector``  — matches user queries to agent-level skills via LLM
  - ``MCPDispatcher``  — orchestrates tool calls across MCP servers
  - ``SkillExecutor``  — runs multi-step agent-level skills
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import AIMessage

from ..config.schema import AgentConfig
from ..core.agent import BaseAgent
from ..core.llm_provider import LLMProvider
from ..core.mcp import MCPClient
from ..core.repository import VectorReader
from ..core.state import AgentState, AuthContext
from ..rag.dynamic import inject_to_rag
from ..rag.scopes import RAGScope

from .mcp_dispatcher import MCPDispatcher
from .skill_detector import SkillDetector
from .skill_executor import SkillExecutor

logger = logging.getLogger(__name__)

_monotonic = time.monotonic  # cache TTL uses monotonic clock (not wall clock)


class GenericAgent(BaseAgent):
    """
    A concrete agent whose behavior is fully defined by an ``AgentConfig``.

    No subclassing needed — add agents by editing ``agents.yaml``.
    """

    def __init__(
        self,
        *,
        config: AgentConfig,
        llm: Any,
        reader: VectorReader,
        mcp_clients: list[MCPClient] | None = None,
        agent_peers: dict[str, Any] | None = None,
        llm_service: LLMProvider | None = None,
        summary_config: dict[str, Any] | None = None,
    ):
        super().__init__(llm=llm, reader=reader, mcp_clients=mcp_clients, llm_service=llm_service)
        self._config = config
        self._agent_peers: dict[str, Any] = agent_peers or {}
        self._summary_config: dict[str, Any] | None = summary_config

        # ── Create collaborators ──
        if llm_service:
            self._skill_detector: SkillDetector | None = SkillDetector(
                llm_service,
                config.llm.model if config.llm else llm,
            )
        else:
            self._skill_detector = None

        self._mcp_dispatcher = MCPDispatcher(self.mcp_clients, config.mcp_servers)
        self._skill_executor = SkillExecutor(
            agent_name=config.name,
            mcp_dispatcher=self._mcp_dispatcher,
            builtin_tool_caller=self.call_builtin_tool,
            agent_peers=self._agent_peers,
        )

    # ── Identity (from YAML config) ──────────────────────────

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def description(self) -> str:
        return self._config.description

    @property
    def rag_namespace(self) -> str:
        return self._config.rag.namespace

    # ── Execution ────────────────────────────────────────────

    async def run(self, state: AgentState) -> AgentState:
        """Execute the 6-step pipeline: RAG → skill → MCP → builtins → inject → summarise."""
        auth: AuthContext | None = state.get("auth_context")
        if not auth:
            return {
                "messages": [AIMessage(content=f"[{self.name}] Error: no auth context")],
            }

        query = self.extract_user_query(state)
        scope = self._build_scope(auth, state)

        rag_data = await self._step_rag_retrieval(query, scope)
        cached_tools = await self._step_cache_check(scope)
        mcp_data = await self._step_tool_calls(query, auth, cached_tools)
        await self._step_dynamic_injection(mcp_data, scope)
        summary = await self._step_summarise(query, mcp_data, rag_data, state)

        return {
            "messages": [AIMessage(content=f"[{self.name.title()} Agent]\n{summary}")],
            "mcp_context": {self.name: mcp_data},
            "rag_context": {self.name: rag_data},
        }

    # ── Pipeline steps ──────────────────────────────────────────

    def _build_scope(self, auth: AuthContext, state: AgentState) -> RAGScope:
        """Build hierarchical RAG scope from auth + state."""
        return RAGScope(
            tenant_id=auth.tenant_key,
            user_id=auth.user_id,
            chat_id=state.get("chat_id", ""),
            agent_id=self.name,
        )

    async def _step_rag_retrieval(
        self,
        query: str,
        scope: RAGScope,
    ) -> list[dict[str, Any]]:
        """Step 1: RAG retrieval (domain namespace + uploads)."""
        if not self._config.rag.enabled:
            return []
        return await self.fetch_all_rag_context(query, scope, k=self._config.rag.k)

    async def _step_cache_check(self, scope: RAGScope) -> dict[str, Any]:
        """Step 1.5: Check RAG for cached tool results within TTL."""
        if not (self._config.rag.enabled and self._config.injectable_tool_ttls):
            return {}
        return await self._check_tool_cache(scope)

    async def _step_tool_calls(
        self,
        query: str,
        auth: AuthContext,
        cached_tools: dict[str, Any],
    ) -> dict[str, Any]:
        """Steps 2–4: Skill check → MCP tools → built-in tools, with cache merge."""
        # Step 2: skill check
        skill_name = await self._detect_skill(query)
        if skill_name:
            logger.info("[%s] Running agent skill '%s'", self.name, skill_name)
            skill = self._config.skills[skill_name]
            mcp_data = await self._skill_executor.run_skill(skill_name, skill.steps, query, auth)
        else:
            # Step 3: MCP tool calls
            llm_config = self._config.llm
            mcp_data = await self._mcp_dispatcher.fetch(
                query,
                auth,
                agent_name=self.name,
                llm_model=llm_config.model if llm_config else None,
                llm_service=self._llm_service,
                skip_tools=set(cached_tools.keys()),
            )
            # Step 4: built-in tool calls
            builtin_data = await self._run_builtin_tools(
                query,
                mcp_data,
                skip_tools=set(cached_tools.keys()),
            )
            mcp_data.update(builtin_data)

        # Merge cached results
        if cached_tools:
            mcp_data = {**cached_tools, **mcp_data}
        return mcp_data

    async def _step_dynamic_injection(
        self,
        mcp_data: dict[str, Any],
        scope: RAGScope,
    ) -> None:
        """Step 5: Dynamic RAG injection for tools with inject_to_rag=True."""
        if not (self._config.rag.enabled and self._config.injectable_tools):
            return
        injectable = {k: v for k, v in mcp_data.items() if k in self._config.injectable_tools}
        if injectable:
            await inject_to_rag(
                self.reader,
                mcp_data=injectable,
                namespace=self._config.rag.namespace,
                scope=scope,
                source_tool=self.name,
            )

    async def _step_summarise(
        self,
        query: str,
        mcp_data: dict[str, Any],
        rag_data: list[dict[str, Any]],
        state: AgentState | None = None,
    ) -> str:
        """Step 6: LLM summarisation with conversation history and prior tool context."""
        llm_config = self._config.llm

        # Extract conversation history so the LLM knows what was
        # previously discussed (e.g. "tell me more" or "the second one").
        history = self.extract_conversation_history(state) if state else []

        # Compress older history when sliding-window summarization is
        # enabled in the supervisor config.  The config is not directly
        # available here, so we accept any ``_summary_config`` injected
        # at construction.  When absent, no compression happens.
        if history and self._summary_config and self._llm_service:
            history = await self.compress_conversation_history(
                history,
                llm_service=self._llm_service,
                model=self._summary_config.get("model") or (llm_config.model if llm_config else ""),
                recent_turns=self._summary_config.get("recent_turns", 3),
            )

        # Extract prior tool results from state so the LLM has grounding
        # on what was previously fetched or created by this agent.
        prior_ctx = (state.get("mcp_context") or {}).get(self.name) if state else None

        return await self.summarise(
            query,
            mcp_data,
            rag_data,
            system_prompt=self._config.prompt,
            model=llm_config.model if llm_config else None,
            temperature=llm_config.temperature if llm_config else 0.2,
            conversation_history=history or None,
            prior_tool_context=prior_ctx,
        )

    # ── Built-in tools (ADR-017) ─────────────────────────────

    async def _run_builtin_tools(
        self,
        query: str,
        mcp_data: dict[str, Any],
        *,
        skip_tools: set[str] | None = None,
    ) -> dict[str, Any]:
        """Run all built-in tools declared for this agent."""
        results: dict[str, Any] = {}
        if not self._config.tools:
            return results

        for tool_name in self._config.tools:
            key = f"builtin_{tool_name}"
            if skip_tools and key in skip_tools:
                logger.info("[%s] Built-in tool '%s' skipped (cache hit)", self.name, tool_name)
                continue
            try:
                result = await self.call_builtin_tool(tool_name, query=query, context=mcp_data)
                results[key] = result
                logger.info("[%s] Built-in tool '%s' succeeded", self.name, tool_name)
            except (ValueError, TypeError, KeyError, RuntimeError, OSError) as exc:
                logger.error("[%s] Built-in tool '%s' failed: %s", self.name, tool_name, exc)
                results[f"{key}_error"] = str(exc)

        return results

    # ── RAG tool cache ────────────────────────────────────────

    async def _check_tool_cache(self, scope: RAGScope) -> dict[str, Any]:
        """Check RAG for cached tool results within TTL. Returns {tool_name: content}."""
        import asyncio

        if not self._config.injectable_tool_ttls:
            return {}

        async def _lookup(tool_name: str, ttl: int) -> tuple[str, Any]:
            min_time = _monotonic() - ttl
            result = await self.reader.lookup_cached_tool_results(
                namespace=self._config.rag.namespace,
                scope=scope,
                tool_name=tool_name,
                min_injected_at=min_time,
            )
            if result is not None:
                logger.info("[%s] Cache hit for tool '%s' (TTL=%ds)", self.name, tool_name, ttl)
            return tool_name, result

        pairs = await asyncio.gather(*(_lookup(name, ttl) for name, ttl in self._config.injectable_tool_ttls.items()))
        return {name: val for name, val in pairs if val is not None}

    # ── Skill detection ──────────────────────────────────────

    async def _detect_skill(self, query: str) -> str | None:
        """
        Determine if the query matches an agent-level skill.

        Requires ``SkillDetector`` (backed by ``LLMProvider``).
        """
        if not self._config.skills:
            return None

        if not self._skill_detector:
            logger.warning("[%s] No SkillDetector available — skill detection skipped", self.name)
            return None

        return await self._skill_detector.detect(query, self._config.skills)

    # Tool call strategies (_call_all_tools, _call_tools_sequential,
    # _call_tools_llm_decides) have been extracted to agents/strategies.py
    # following the Strategy pattern (OCP).  See ``get_strategy()``.

    # MCP interaction, skill execution, and capability discovery have been
    # extracted to MCPDispatcher, SkillExecutor, and SkillDetector
    # following the Single Responsibility Principle (SRP).
