"""
GenericAgent — config-driven agent that requires no custom Python code.

Implements the standard pipeline entirely from YAML configuration:
  1. RAG retrieval (tenant-aware)
  2. Skill check — if query matches an agent skill, run it and skip to step 5
  3. Agentic tool-calling loop — unified MCP + built-in tools with native
     ``tool_calls`` protocol, duplicate detection, and per-call error handling
  4. Dynamic RAG injection (feedback loop)
  5. LLM summarisation using the prompt from config (skipped when the
     agentic loop already produced a final text response)

Collaborators (SRP — each extracted into its own module):
  - ``SkillDetector``  — matches user queries to agent-level skills via LLM
  - ``MCPDispatcher``  — discovers MCP capabilities and routes tool calls
  - ``SkillExecutor``  — runs multi-step agent-level skills
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.messages import AIMessage

from ..config.schema import AgentConfig
from ..core.agent import BaseAgent
from ..core.mcp import MCPClient
from ..core.repository import VectorReader
from ..core.state import AgentState, AuthContext
from ..rag.dynamic import inject_to_rag
from ..rag.scopes import RAGScope

from .mcp_dispatcher import MCPCapabilities, MCPDispatcher
from .skill_detector import SkillDetector
from .skill_executor import SkillExecutor

logger = logging.getLogger(__name__)

_wall_clock = time.time  # cache TTL must use wall clock — dynamic.py stores time.time()


class GenericAgent(BaseAgent):
    """
    A concrete agent whose behavior is fully defined by an ``AgentConfig``.

    No subclassing needed — add agents by editing ``agents.yaml``.
    """

    def __init__(
        self,
        *,
        config: AgentConfig,
        reader: VectorReader,
        mcp_clients: list[MCPClient] | None = None,
        agent_peers: dict[str, Any] | None = None,
        chat_model: Any | None = None,
        summary_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(reader=reader, mcp_clients=mcp_clients, chat_model=chat_model, **kwargs)
        self._config = config
        self._agent_peers: dict[str, Any] = agent_peers or {}
        self._summary_config: dict[str, Any] | None = summary_config

        # ── Create collaborators ──
        if chat_model:
            self._skill_detector: SkillDetector | None = SkillDetector(chat_model)
        else:
            self._skill_detector = None

        self._mcp_dispatcher = MCPDispatcher(self.mcp_clients, config.mcp_servers)
        self._skill_executor = SkillExecutor(
            agent_name=config.name,
            mcp_dispatcher=self._mcp_dispatcher,
            builtin_tool_caller=self.call_builtin_tool,
            agent_peers=self._agent_peers,
        )

    def set_agent_peers(self, peers: dict[str, Any]) -> None:
        """Set peer agents for cross-agent skill steps.

        Called by the graph builder after all agents are instantiated.
        Propagates to the internal ``SkillExecutor`` automatically.
        """
        self._agent_peers = peers
        self._skill_executor._agent_peers = peers

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
        """Execute the pipeline: RAG → skill check → agentic loop → inject → summarise."""
        auth: AuthContext | None = state.get("auth_context")
        if not auth:
            return {
                "messages": [AIMessage(content=f"[{self.name}] Error: no auth context")],
            }

        raw_query = self.extract_user_query(state)

        # Reformulate query using conversation history for better search/tool precision
        if self._config.rag.reformulate_queries and self._chat_model:
            query = await self.reformulate_query(raw_query, state)
        else:
            query = raw_query

        scope = self._build_scope(auth, state)

        rag_data = await self._step_rag_retrieval(query, scope)
        cached_tools = await self._step_cache_check(scope)

        # Skill check — if matched, run the skill and skip the agentic loop
        skill_name = await self._detect_skill(query)
        if skill_name:
            logger.info("[%s] Running agent skill '%s'", self.name, skill_name)
            skill = self._config.skills[skill_name]
            mcp_data = await self._skill_executor.run_skill(skill_name, skill.steps, query, auth)
            final_text = None
        else:
            # Unified agentic loop: MCP + built-in tools in a single
            # native tool_calls conversation with duplicate detection.
            final_text, mcp_data = await self._agentic_tool_loop(
                query,
                auth,
                state,
                rag_data,
                skip_tools=set(cached_tools.keys()),
            )
            if cached_tools:
                mcp_data = {**cached_tools, **mcp_data}

        await self._step_dynamic_injection(mcp_data, scope)

        # When the agentic loop produced a final text response the
        # summarisation step is redundant — the LLM already synthesised.
        if final_text:
            summary = final_text
        else:
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
        """Step 1: RAG retrieval (domain namespace + uploads).

        When ``retriever_type`` is ``multi_query``, the LLM generates
        query variations for broader recall before merging results.
        """
        if not self._config.rag.enabled:
            return []

        if self._config.rag.retriever_type == "multi_query" and self._chat_model:
            return await self._multi_query_rag(query, scope, k=self._config.rag.k)

        return await self.fetch_all_rag_context(query, scope, k=self._config.rag.k)

    async def _multi_query_rag(
        self,
        query: str,
        scope: RAGScope,
        *,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Multi-query RAG: generate query variations, retrieve in parallel, merge."""
        import asyncio as _asyncio

        from ..rag.retriever import multi_query_retrieve

        domain_results, upload_results = await _asyncio.gather(
            multi_query_retrieve(query, self.reader, self.rag_namespace, scope, self._chat_model, k=k),
            multi_query_retrieve(query, self.reader, "uploads", scope, self._chat_model, k=k),
        )

        combined = []
        for r in domain_results + upload_results:
            combined.append(
                {
                    "content": r.document.page_content,
                    "score": round(r.score, 3),
                    "metadata": {
                        mk: mv for mk, mv in r.document.metadata.items() if mk not in ("content", "embedding")
                    },
                }
            )

        combined.sort(key=lambda d: d.get("score", 0), reverse=True)
        return combined[:k]

    async def _step_cache_check(self, scope: RAGScope) -> dict[str, Any]:
        """Step 1.5: Check RAG for cached tool results within TTL."""
        if not (self._config.rag.enabled and self._config.injectable_tool_ttls):
            return {}
        return await self._check_tool_cache(scope)

    async def _step_dynamic_injection(
        self,
        mcp_data: dict[str, Any],
        scope: RAGScope,
    ) -> None:
        """Dynamic RAG injection for tools with inject_to_rag=True."""
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
        """LLM summarisation with conversation history and prior tool context."""
        llm_config = self._config.llm

        history = (
            self.extract_conversation_history(
                state,
                strip_prefixes=self._compute_agent_prefixes(),
            )
            if state
            else []
        )

        # Compress older history via sliding-window summarization when enabled.
        if history and self._summary_config and self._chat_model:
            history = await self.compress_conversation_history(
                history,
                chat_model=self._chat_model,
                recent_turns=self._summary_config.get("recent_turns", 3),
            )

        prior_ctx = (state.get("mcp_context") or {}).get(self.name) if state else None

        return await self.summarise(
            query,
            mcp_data,
            rag_data,
            system_prompt=self._config.prompt,
            temperature=llm_config.temperature if llm_config else 0.2,
            conversation_history=history or None,
            prior_tool_context=prior_ctx,
        )

    # ── Agent prefix computation ─────────────────────────────────

    def _compute_agent_prefixes(self) -> tuple[str, ...]:
        """Generate ``[{Name} Agent]\\n`` prefixes for all known agents.

        Used by :meth:`extract_conversation_history` to strip internal
        agent name tags from historical messages.
        """
        prefixes = [f"[{self.name.title()} Agent]\n"]
        for peer_name in self._agent_peers:
            prefixes.append(f"[{peer_name.title()} Agent]\n")
        return tuple(prefixes)

    # ── Agentic tool-calling loop ────────────────────────────────

    async def _agentic_tool_loop(
        self,
        query: str,
        auth: AuthContext,
        state: AgentState | None,
        rag_data: list[dict[str, Any]],
        *,
        skip_tools: set[str] | None = None,
    ) -> tuple[str | None, dict[str, Any]]:
        """Unified MCP + built-in tool loop using native ``tool_calls``.

        Delegates to :class:`AgenticLoop` which owns the multi-round
        lifecycle (duplicate tracking, max-round safety, HITL interrupts).

        Returns ``(final_text, tool_results)``.  ``final_text`` is ``None``
        when the loop exhausted rounds without producing text (caller
        should fall back to summarisation).
        """
        from .agentic_loop import AgenticLoop
        from .tools import build_langchain_tools

        llm_config = self._config.llm
        if not self._chat_model:
            return None, {}

        # ── Discover MCP capabilities ───────────────────────
        caps = await self._mcp_dispatcher.render_capabilities(auth, agent_name=self.name)

        # ── Build unified tool list ─────────────────────────
        builtin_tool_names, builtin_tool_defs = self._builtin_tools_to_litellm(skip_tools)
        mcp_tool_defs = MCPDispatcher.mcp_tools_to_litellm(
            [
                t
                for t in caps.raw_tools
                if t["name"] not in builtin_tool_names and not (skip_tools and t["name"] in skip_tools)
            ],
        )
        all_tool_defs = mcp_tool_defs + builtin_tool_defs
        if not all_tool_defs:
            return None, {}

        # ── Build LangChain tool wrappers ───────────────────
        lc_tools = build_langchain_tools(
            builtin_names=builtin_tool_names,
            builtin_tool_defs=builtin_tool_defs,
            mcp_tool_defs=mcp_tool_defs,
            mcp_tool_client_map=caps.tool_client_map,
            auth=auth,
            agent_name=self.name,
            approval_tools=self._config.approval_tools or None,
        )
        tool_map: dict[str, Any] = {t.name: t for t in lc_tools}

        # ── Build messages ──────────────────────────────────
        system_prompt = self._build_agentic_system_prompt(caps, rag_data, state)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if state:
            messages.extend(
                self.extract_conversation_history(
                    state,
                    strip_prefixes=self._compute_agent_prefixes(),
                )
            )
        messages.append({"role": "user", "content": query})

        # ── Run the loop ────────────────────────────────────
        loop = AgenticLoop(
            agent_name=self.name,
            chat_model=self._chat_model,
            tool_map=tool_map,
            all_tool_defs=all_tool_defs,
            temperature=llm_config.temperature if llm_config else 0.2,
        )
        return await loop.run(messages)

    # ── System prompt builder for the agentic loop ──────────────

    def _build_agentic_system_prompt(
        self,
        caps: MCPCapabilities,
        rag_data: list[dict[str, Any]],
        state: AgentState | None,
    ) -> str:
        """Build a rich system prompt from config + MCP metadata + RAG context."""
        parts = [self._config.prompt]

        # Prior tool results from previous turns
        prior_ctx = (state.get("mcp_context") or {}).get(self.name) if state else None
        if prior_ctx:
            parts.append("\n--- Previous Tool Results (from prior turns) ---")
            parts.append(json.dumps(prior_ctx, indent=2, default=str)[:4000])

        # Rendered MCP prompts (zero-arg prompts evaluated at discovery time)
        if caps.rendered_prompts:
            for prompt in caps.rendered_prompts:
                parts.append(f"\n--- MCP Prompt: {prompt['name']} ---\n{prompt['text']}")

        # Prompts that require arguments — listed so the LLM knows they exist
        if caps.skipped_prompts:
            for sp in caps.skipped_prompts:
                parts.append(
                    f"\n[Available prompt: {sp['name']}] {sp['description']} "
                    f"(requires: {', '.join(sp['required_args'])})"
                )

        # MCP resource contents
        if caps.resource_contents:
            parts.append("\n--- Available Resources ---")
            for name, content in caps.resource_contents.items():
                parts.append(f"\n[{name}]\n{content[:2000]}")

        # RAG context
        if rag_data:
            parts.append("\n--- Background Knowledge (RAG) ---")
            parts.append(json.dumps(rag_data, indent=2, default=str)[:3000])

        return "\n".join(parts)

    # ── Built-in tools → litellm format ──────────────────────────

    def _builtin_tools_to_litellm(
        self,
        skip_tools: set[str] | None = None,
    ) -> tuple[set[str], list[dict[str, Any]]]:
        """Convert registered built-in tools to litellm function-calling format.

        Returns ``(builtin_tool_names, litellm_tool_defs)``.
        """
        from ..config.tool_registry import get_tool

        names: set[str] = set()
        defs: list[dict[str, Any]] = []

        for tool_name in self._config.tools:
            if skip_tools and (tool_name in skip_tools or f"builtin_{tool_name}" in skip_tools):
                continue
            try:
                entry = get_tool(tool_name)
            except KeyError:
                continue

            names.add(tool_name)

            # Build JSON Schema properties from ToolParameter metadata
            properties: dict[str, Any] = {}
            required: list[str] = []
            for p in entry.parameters.values():
                prop: dict[str, str] = {
                    "type": _tool_param_type_to_json_schema(p.type),
                    "description": p.description,
                }
                properties[p.name] = prop
                if p.required:
                    required.append(p.name)

            schema: dict[str, Any] = {"type": "object", "properties": properties}
            if required:
                schema["required"] = required

            defs.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": entry.description,
                        "parameters": schema,
                    },
                }
            )

        return names, defs

    # ── Built-in tool call helper ────────────────────────────────

    async def _call_builtin_tool(
        self,
        fn_name: str,
        fn_args: dict[str, Any],
        auth: AuthContext,
    ) -> str:
        """Call a built-in tool, injecting auth_context, and return JSON result."""
        try:
            result = await self.call_builtin_tool(fn_name, auth_context=auth, **fn_args)
            result_text = json.dumps(result, indent=2, default=str)
            logger.info("[%s] Built-in tool '%s' succeeded", self.name, fn_name)
            return result_text
        except Exception as exc:
            logger.error("[%s] Built-in tool '%s' error: %s", self.name, fn_name, exc, exc_info=True)
            return f"[Tool error] {exc}"

    # ── RAG tool cache ────────────────────────────────────────

    async def _check_tool_cache(self, scope: RAGScope) -> dict[str, Any]:
        """Check RAG for cached tool results within TTL. Returns {tool_name: content}."""
        import asyncio as _asyncio

        if not self._config.injectable_tool_ttls:
            return {}

        async def _lookup(tool_name: str, ttl: int) -> tuple[str, Any]:
            min_time = _wall_clock() - ttl
            result = await self.reader.lookup_cached_tool_results(
                namespace=self._config.rag.namespace,
                scope=scope,
                tool_name=tool_name,
                min_injected_at=min_time,
            )
            if result is not None:
                logger.info("[%s] Cache hit for tool '%s' (TTL=%ds)", self.name, tool_name, ttl)
            return tool_name, result

        pairs = await _asyncio.gather(*(_lookup(name, ttl) for name, ttl in self._config.injectable_tool_ttls.items()))
        return {name: val for name, val in pairs if val is not None}

    # ── Skill detection ──────────────────────────────────────

    async def _detect_skill(self, query: str) -> str | None:
        """Determine if the query matches an agent-level skill."""
        if not self._config.skills:
            return None
        if not self._skill_detector:
            logger.warning("[%s] No SkillDetector available — skill detection skipped", self.name)
            return None
        return await self._skill_detector.detect(query, self._config.skills)


# ── Module-level helpers ─────────────────────────────────────────

_PARAM_TYPE_MAP: dict[str, str] = {
    "string": "string",
    "str": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
}


def _tool_param_type_to_json_schema(param_type: str) -> str:
    """Map a ``ToolParameter.type`` string to a JSON Schema type."""
    return _PARAM_TYPE_MAP.get(param_type.lower(), "string")
