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
  - ``SkillDetector``      — matches user queries to agent-level skills via LLM
  - ``MCPDispatcher``      — discovers MCP capabilities and routes tool calls
  - ``SkillExecutor``      — runs multi-step agent-level skills
  - ``RagPipeline``        — owns RAG retrieval, cache check, dynamic injection
  - ``SystemPromptBuilder`` — assembles the agentic-loop system prompt
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.messages import AIMessage

from ..config.schema import OrchidAgentConfig
from ..core.agent import OrchidAgent
from ..core.graph_store import OrchidGraphStore
from ..core.helpers import filter_summary_messages
from ..core.mcp import OrchidMCPClient
from ..observability.mini_agent_events import make_event_message
from ..core.repository import OrchidVectorReader
from ..core.retrieval import apply_pre_strategy
from ..core.state import OrchidAgentState, OrchidAuthContext
from ..rag.scopes import OrchidRAGScope
from ..rag.transformers import (
    TRANSFORMER_REGISTRY,
    get_query_transformer,
    resolve_transformer_kwargs,
)

from .mcp_dispatcher import MCPCapabilities, MCPDispatcher
from .prompt_builder import SystemPromptBuilder
from .rag_pipeline import RagPipeline
from .skill_detector import SkillDetector
from .skill_executor import SkillExecutor

logger = logging.getLogger(__name__)
perf_logger = logging.getLogger("orchid.perf")


class GenericAgent(OrchidAgent):
    """
    A concrete agent whose behavior is fully defined by an ``OrchidAgentConfig``.

    No subclassing needed — add agents by editing ``agents.yaml``.
    """

    def __init__(
        self,
        *,
        config: OrchidAgentConfig,
        reader: OrchidVectorReader,
        mcp_clients: list[OrchidMCPClient] | None = None,
        agent_peers: dict[str, Any] | None = None,
        chat_model: Any | None = None,
        summary_config: dict[str, Any] | None = None,
        graph_store: OrchidGraphStore | None = None,
        **kwargs: Any,
    ):
        super().__init__(reader=reader, mcp_clients=mcp_clients, chat_model=chat_model, **kwargs)
        self._config = config
        self._agent_peers: dict[str, Any] = agent_peers or {}
        self._summary_config: dict[str, Any] | None = summary_config
        # graph store injected by the graph builder so the
        # ``graph_rag`` retrieval strategy can traverse entities and
        # relations.  ``None`` (or a NullGraphStore) makes
        # GraphRAGRetrieval fall back to SimpleRetrieval.
        self._graph_store: OrchidGraphStore | None = graph_store

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
            content_sources=self._content_sources,
            max_skill_depth=config.max_skill_depth,
        )
        self._rag_pipeline = RagPipeline(
            reader=reader,
            chat_model=chat_model,
            graph_store=graph_store,
        )
        self._prompt_builder = SystemPromptBuilder(config.prompt_sections)

    def needs_peer_wiring(self) -> bool:
        """Return ``True`` if any skill step references another agent."""
        return any(step.agent is not None for skill in self._config.skills.values() for step in skill.steps)

    def wire_peers(self, peers: dict[str, Any]) -> None:
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

    async def run(self, state: OrchidAgentState) -> OrchidAgentState:
        """Execute the pipeline: RAG → skill check → agentic loop → inject → summarise."""
        auth: OrchidAuthContext | None = state.get("auth_context")
        if not auth:
            return {
                "messages": [AIMessage(content=f"[{self.name}] Error: no auth context")],
            }

        raw_query = self.extract_user_query(state)

        # Apply pre_strategy=True query transformers in order so the
        # rewritten query feeds RAG retrieval, the agentic loop, and
        # the summarisation step alike (single source of truth — see
        # the pre_strategy / strategy-internal split).  Prompt
        # overrides (when configured) are resolved from
        # ``rag.retrieval.transformer_prompts`` and threaded through
        # the registry's kwargs forwarding.
        transformer_names = self._config.rag.retrieval.query_transformers or []
        prompts_cfg = self._config.rag.retrieval.transformer_prompts
        pre_transformers = [
            get_query_transformer(name, **resolve_transformer_kwargs(name, prompts_cfg))
            for name in transformer_names
            if TRANSFORMER_REGISTRY[name].pre_strategy
        ]
        if pre_transformers and self._chat_model:
            transform_start = time.perf_counter()
            query = await apply_pre_strategy(
                pre_transformers,
                raw_query,
                chat_model=self._chat_model,
                history=self.extract_conversation_history(
                    state,
                    max_turns=5,
                    max_chars=500,
                    truncation_strategy=self._get_truncation_strategy(),
                ),
            )
            transform_elapsed = (time.perf_counter() - transform_start) * 1000
            perf_logger.info(
                "[PERF][agent=%s] step=pre_strategy_transformers took %.1f ms (n=%d)",
                self.name,
                transform_elapsed,
                len(pre_transformers),
            )
        else:
            query = raw_query

        scope = self._build_scope(auth, state)

        rag_start = time.perf_counter()
        rag_data = await self._step_rag_retrieval(query, scope)
        rag_elapsed = (time.perf_counter() - rag_start) * 1000
        perf_logger.info(
            "[PERF][agent=%s] step=rag_retrieval took %.1f ms (docs=%d, k=%d, namespace=%s)",
            self.name,
            rag_elapsed,
            len(rag_data),
            self._config.rag.k,
            self._config.rag.namespace,
        )

        cache_start = time.perf_counter()
        cached_tools = await self._step_cache_check(scope)
        cache_elapsed = (time.perf_counter() - cache_start) * 1000
        if self._config.injectable_tool_ttls:
            perf_logger.info(
                "[PERF][agent=%s] step=cache_check took %.1f ms (cache_hits=%d)",
                self.name,
                cache_elapsed,
                len(cached_tools),
            )

        # Skill check — if matched, run the skill and skip the agentic loop
        skill_start = time.perf_counter()
        skill_name = await self._detect_skill(query)
        skill_detect_elapsed = (time.perf_counter() - skill_start) * 1000
        if self._config.skills:
            perf_logger.info(
                "[PERF][agent=%s] step=skill_detect took %.1f ms (matched=%s)",
                self.name,
                skill_detect_elapsed,
                skill_name or "none",
            )
        if skill_name:
            logger.info("[%s] Running agent skill '%s'", self.name, skill_name)
            skill = self._config.skills[skill_name]
            loop_events = [
                make_event_message(
                    "skill.adopted",
                    {
                        "agent": self.name,
                        "skill": skill_name,
                    },
                )
            ]
            skill_run_start = time.perf_counter()
            mcp_data = await self._skill_executor.run_skill(skill_name, skill.steps, query, auth)
            skill_run_elapsed = (time.perf_counter() - skill_run_start) * 1000
            perf_logger.info(
                "[PERF][agent=%s] step=skill_run name=%s took %.1f ms (tool_results=%d)",
                self.name,
                skill_name,
                skill_run_elapsed,
                len(mcp_data),
            )
            final_text = None
        else:
            # Unified agentic loop: MCP + built-in tools in a single
            # native tool_calls conversation with duplicate detection.
            loop_start = time.perf_counter()
            final_text, mcp_data, loop_events = await self._agentic_tool_loop(
                query,
                auth,
                state,
                rag_data,
                skip_tools=set(cached_tools.keys()),
                content_sources=self._content_sources,
            )
            loop_elapsed = (time.perf_counter() - loop_start) * 1000
            perf_logger.info(
                "[PERF][agent=%s] step=agentic_loop took %.1f ms (tool_results=%d, produced_text=%s)",
                self.name,
                loop_elapsed,
                len(mcp_data),
                bool(final_text),
            )
            if cached_tools:
                mcp_data = {**cached_tools, **mcp_data}

        inject_start = time.perf_counter()
        await self._step_dynamic_injection(mcp_data, scope)
        inject_elapsed = (time.perf_counter() - inject_start) * 1000
        if self._config.injectable_tools:
            perf_logger.info(
                "[PERF][agent=%s] step=dynamic_injection took %.1f ms",
                self.name,
                inject_elapsed,
            )

        # When the agentic loop produced a final text response the
        # summarisation step is redundant — the LLM already synthesised.
        if final_text:
            summary = final_text
        else:
            summarise_start = time.perf_counter()
            summary = await self._step_summarise(query, mcp_data, rag_data, state)
            summarise_elapsed = (time.perf_counter() - summarise_start) * 1000
            perf_logger.info(
                "[PERF][agent=%s] step=summarise took %.1f ms (out_chars=%d)",
                self.name,
                summarise_elapsed,
                len(summary),
            )

        await self._store_agent_turn(state, summary)

        output_messages: list = [AIMessage(content=f"[{self.name.title()} Agent]\n{summary}")]
        for evt in loop_events:
            output_messages.insert(0, evt)
        return {
            "messages": output_messages,
            "mcp_context": {self.name: mcp_data},
            "rag_context": {self.name: rag_data},
        }

    # ── Pipeline steps ──────────────────────────────────────────

    def _get_truncation_strategy(self) -> str:
        if self._summary_config:
            return self._summary_config.get("truncation_strategy", "hard")
        return "hard"

    async def _store_agent_turn(
        self,
        state: OrchidAgentState | None,
        response: str,
    ) -> None:
        if state is None or self._summary_config is None:
            return
        memory = self._summary_config.get("memory")
        if memory is None:
            return
        from .memory_rag import OrchidRAGConversationMemory

        if not isinstance(memory, OrchidRAGConversationMemory):
            return
        try:
            chat_id = state.get("chat_id", "")
            if not chat_id:
                return
            auth = state.get("auth_context")
            tenant_id = auth.tenant_key if auth else "default"
            user_id = auth.user_id if auth else ""
            query = self.extract_user_query(state)
            if query:
                await memory.store_conversation_turn(
                    chat_id=chat_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    turn={"role": "user", "content": query},
                    metadata={"turn_type": "agent", "agent": self.name},
                )
            if response:
                await memory.store_conversation_turn(
                    chat_id=chat_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    turn={"role": "assistant", "content": response},
                    metadata={"turn_type": "agent", "agent": self.name},
                )
        except Exception:
            pass

    def _build_scope(self, auth: OrchidAuthContext, state: OrchidAgentState) -> OrchidRAGScope:
        """Build hierarchical RAG scope from auth + state."""
        return OrchidRAGScope(
            tenant_id=auth.tenant_key,
            user_id=auth.user_id,
            chat_id=state.get("chat_id", ""),
            agent_id=self.name,
        )

    async def _step_rag_retrieval(
        self,
        query: str,
        scope: OrchidRAGScope,
    ) -> list[dict[str, Any]]:
        """Step 1: RAG retrieval — delegated to RagPipeline."""
        return await self._rag_pipeline.retrieve(
            query=query,
            scope=scope,
            rag_namespace=self.rag_namespace,
            k=self._config.rag.k,
            enabled=self._config.rag.enabled,
            retrieval_strategy=self._config.rag.retrieval.strategy,
            retrieval_config=self._config.rag.retrieval,
            exclude_dynamic=self._config.rag.retrieval.exclude_dynamic,
        )

    async def _step_cache_check(self, scope: OrchidRAGScope) -> dict[str, Any]:
        """Step 1.5: Check RAG for cached tool results within TTL."""
        return await self._rag_pipeline.check_cache(
            scope=scope,
            rag_namespace=self._config.rag.namespace,
            enabled=self._config.rag.enabled,
            tool_ttls=self._config.injectable_tool_ttls,
        )

    async def _step_dynamic_injection(
        self,
        mcp_data: dict[str, Any],
        scope: OrchidRAGScope,
    ) -> None:
        """Dynamic RAG injection — per-tool ``effective_rag``."""
        await self._rag_pipeline.inject(
            mcp_data=mcp_data,
            scope=scope,
            rag_namespace=self._config.rag.namespace,
            enabled=self._config.rag.enabled,
            injectable_tools=self._config.injectable_tools,
            effective_rag_resolver=self._config.effective_rag,
        )

    async def _step_summarise(
        self,
        query: str,
        mcp_data: dict[str, Any],
        rag_data: list[dict[str, Any]],
        state: OrchidAgentState | None = None,
    ) -> str:
        """LLM summarisation with conversation history and prior tool context."""
        llm_config = self._config.llm

        history = (
            self.extract_conversation_history(
                state,
                strip_prefixes=self._compute_agent_prefixes(),
                truncation_strategy=self._get_truncation_strategy(),
            )
            if state
            else []
        )

        # Compress older history via sliding-window summarization when enabled.
        if history and self._summary_config and self._chat_model:
            running_summary: str | None = None
            memory = self._summary_config.get("memory")
            if memory is not None and state:
                chat_id = state.get("chat_id", "")
                if chat_id:
                    try:
                        running_summary = await memory.get_running_summary(chat_id)
                    except Exception:
                        pass
            history = await self.compress_conversation_history(
                history,
                chat_model=self._chat_model,
                recent_turns=self._summary_config.get("recent_turns", 3),
                running_summary=running_summary,
                structured_output=self._summary_config.get("structured_output", False),
            )
            # Persist the updated summary
            if memory is not None and running_summary is not None and state:
                chat_id = state.get("chat_id", "")
                if chat_id:
                    try:
                        delta = filter_summary_messages(history)
                        await memory.update_running_summary(
                            chat_id,
                            delta,
                            running_summary,
                        )
                    except Exception:
                        pass

        prior_ctx = (state.get("mcp_context") or {}).get(self.name) if state else None

        # Thread per-agent summarise prompt overrides through to the
        # helper so the LLM-facing surface respects the same
        # ``prompt_sections`` block that drives the agentic-loop
        # builder.
        sections = self._config.prompt_sections
        return await self.summarise(
            query,
            mcp_data,
            rag_data,
            system_prompt=self._config.prompt,
            temperature=llm_config.temperature if llm_config else 0.2,
            conversation_history=history or None,
            prior_tool_context=prior_ctx,
            history_reminder=sections.summarise_history_reminder,
            prior_results_header=sections.summarise_prior_results_header,
            rag_section_header=sections.summarise_rag_section_header,
            user_content_template=sections.summarise_user_template,
            prior_results_max_chars=sections.summarise_prior_results_max_chars,
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
        auth: OrchidAuthContext,
        state: OrchidAgentState | None,
        rag_data: list[dict[str, Any]],
        *,
        skip_tools: set[str] | None = None,
        content_sources: Any = None,
    ) -> tuple[str | None, dict[str, Any], list]:
        """Unified MCP + built-in tool loop using native ``tool_calls``.

        Delegates to :class:`AgenticLoop` which owns the multi-round
        lifecycle (duplicate tracking, max-round safety, HITL interrupts).

        Returns ``(final_text, tool_results, events)``.  ``final_text`` is
        ``None`` when the loop exhausted rounds without producing text
        (caller should fall back to summarisation).  ``events`` is a list
        of lifecycle ``SystemMessage`` objects (``tool.started``,
        ``tool.finished``) collected during dispatch.
        """
        from .agentic_loop import AgenticLoop
        from .tools import build_langchain_tools

        llm_config = self._config.llm
        if not self._chat_model:
            return None, {}, []

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
            return None, {}, []

        # ── Build LangChain tool wrappers ───────────────────
        lc_tools = build_langchain_tools(
            builtin_names=builtin_tool_names,
            builtin_tool_defs=builtin_tool_defs,
            mcp_tool_defs=mcp_tool_defs,
            mcp_tool_client_map=caps.tool_client_map,
            auth=auth,
            agent_name=self.name,
            approval_tools=self._config.approval_tools or None,
            content_sources=content_sources,
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
                    truncation_strategy=self._get_truncation_strategy(),
                )
            )
        messages.append({"role": "user", "content": query})

        # ── Resolve parallel-dispatch safety map ──
        parallel_safety = self._resolve_parallel_safety(
            tool_map=tool_map,
            builtin_tool_names=builtin_tool_names,
            caps=caps,
        )

        # ── Run the loop ────────────────────────────────────
        loop = AgenticLoop(
            agent_name=self.name,
            chat_model=self._chat_model,
            tool_map=tool_map,
            all_tool_defs=all_tool_defs,
            temperature=llm_config.temperature if llm_config else 0.2,
            parallel_safety=parallel_safety,
            max_tool_rounds=self._config.max_tool_rounds,
            max_consecutive_dupes=self._config.max_consecutive_dupes,
        )
        final_text, tool_results = await loop.run(messages)
        return final_text, tool_results, loop._events

    # ── Parallel-dispatch safety resolver ───────────────────────

    def _resolve_parallel_safety(
        self,
        *,
        tool_map: dict[str, Any],
        builtin_tool_names: set[str],
        caps: MCPCapabilities,
    ) -> dict[str, bool] | None:
        """Resolve which tools may run in parallel within one round.

        Returns ``None`` when the agent has not opted in to
        ``parallel_tools`` — the loop then runs strictly sequentially
        (today's behaviour).  When opted in, returns a per-tool-name
        bool computed via this precedence:

        1. ``requires_approval=True`` → ``False`` (HITL must serialise).
        2. Built-in tool → ``True`` iff its name is in the agent's
           precomputed ``parallel_safe_builtin_tools`` set; else ``False``.
        3. MCP tool with explicit YAML ``parallel_safe`` (``True``/
           ``False``) → use it.
        4. MCP tool without YAML override → ``True`` iff the server
           advertised ``readOnlyHint=True`` for that tool; else ``False``.
        """
        from .tool_utils import resolve_parallel_safety

        return resolve_parallel_safety(
            tool_map=tool_map,
            builtin_tool_names=builtin_tool_names,
            caps=caps,
            parallel_tools_enabled=bool(self._config.parallel_tools),
            approval_tools=self._config.approval_tools,
            parallel_safe_builtin_tools=self._config.parallel_safe_builtin_tools,
            mcp_parallel_overrides=self._mcp_parallel_overrides(),
        )

    def _mcp_parallel_overrides(self) -> dict[str, bool | None]:
        """Collect explicit per-tool ``parallel_safe`` overrides from YAML.

        Walks the agent's ``mcp_servers[*].tools`` and returns a flat
        ``{tool_name: True|False|None}`` map.  Wildcarded servers
        (``tools: "*"``) leave their tools out of this map — those
        tools fall through to the MCP-annotation branch in
        :meth:`_resolve_parallel_safety`.
        """
        overrides: dict[str, bool | None] = {}
        for server in self._config.mcp_servers:
            for tool in server.tools:
                # ``parallel_safe`` is ``None`` by default, meaning
                # "no explicit override — defer to the MCP annotation".
                if tool.parallel_safe is None:
                    continue
                overrides[tool.name] = tool.parallel_safe
        return overrides

    # ── System prompt builder for the agentic loop ──────────────

    def _build_agentic_system_prompt(
        self,
        caps: MCPCapabilities,
        rag_data: list[dict[str, Any]],
        state: OrchidAgentState | None,
    ) -> str:
        """Build a rich system prompt from config + MCP metadata + RAG context.

        Delegated to :class:`SystemPromptBuilder`.
        """
        builder = getattr(self, "_prompt_builder", None)
        if builder is None:
            # Fallback for tests that bypass __init__ (GenericAgent.__new__)
            from orchid_ai.agents.prompt_builder import SystemPromptBuilder

            builder = SystemPromptBuilder(self._config.prompt_sections)
        return builder.build(
            base_prompt=self._config.prompt,
            caps=caps,
            rag_data=rag_data,
            state=state,
            agent_name=self.name,
            rag_max_context_chars=self._config.rag.max_context_chars or 3000,
        )

    # ── Built-in tools → litellm format ──────────────────────────

    def _builtin_tools_to_litellm(
        self,
        skip_tools: set[str] | None = None,
    ) -> tuple[set[str], list[dict[str, Any]]]:
        """Convert registered built-in tools to litellm function-calling format.

        Returns ``(builtin_tool_names, litellm_tool_defs)``.
        """
        from .tool_utils import tools_to_litellm_format

        return tools_to_litellm_format(self._config.tools, skip_tools=skip_tools)

    # ── Built-in tool call helper ────────────────────────────────

    async def _call_builtin_tool(
        self,
        fn_name: str,
        fn_args: dict[str, Any],
        auth: OrchidAuthContext,
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

    # ── Skill detection ──────────────────────────────────────

    async def _detect_skill(self, query: str) -> str | None:
        """Determine if the query matches an agent-level skill."""
        if not self._config.skills:
            return None
        if not self._skill_detector:
            logger.warning("[%s] No SkillDetector available — skill detection skipped", self.name)
            return None
        return await self._skill_detector.detect(query, self._config.skills)
