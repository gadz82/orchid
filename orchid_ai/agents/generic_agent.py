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
from ..core.llm_provider import LLMProvider
from ..core.mcp import MCPClient
from ..core.repository import VectorReader
from ..core.state import AgentState, AuthContext
from ..rag.dynamic import inject_to_rag
from ..rag.scopes import RAGScope

from .mcp_dispatcher import MCPCapabilities, MCPDispatcher
from .skill_detector import SkillDetector
from .skill_executor import SkillExecutor

logger = logging.getLogger(__name__)

_monotonic = time.monotonic  # cache TTL uses monotonic clock (not wall clock)


class GenericAgent(BaseAgent):
    """
    A concrete agent whose behavior is fully defined by an ``AgentConfig``.

    No subclassing needed — add agents by editing ``agents.yaml``.
    """

    # ── Agentic loop safety limits ──────────────────────────────
    _MAX_TOOL_ROUNDS: int = 15
    _MAX_CONSECUTIVE_DUPES: int = 2

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
        """Execute the pipeline: RAG → skill check → agentic loop → inject → summarise."""
        auth: AuthContext | None = state.get("auth_context")
        if not auth:
            return {
                "messages": [AIMessage(content=f"[{self.name}] Error: no auth context")],
            }

        query = self.extract_user_query(state)
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
        """Step 1: RAG retrieval (domain namespace + uploads)."""
        if not self._config.rag.enabled:
            return []
        return await self.fetch_all_rag_context(query, scope, k=self._config.rag.k)

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
        if history and self._summary_config and self._llm_service:
            history = await self.compress_conversation_history(
                history,
                llm_service=self._llm_service,
                model=self._summary_config.get("model") or (llm_config.model if llm_config else ""),
                recent_turns=self._summary_config.get("recent_turns", 3),
            )

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

        Combines all available tools (MCP-discovered and built-in) into a
        single tool list, then runs a multi-turn conversation where the LLM
        decides which tools to call.  Includes duplicate call detection and
        per-call error handling.

        Returns ``(final_text, tool_results)``.  ``final_text`` is the LLM's
        final text response (or ``None`` if the loop exhausted its rounds
        without producing one — the caller should fall back to summarisation).

        .. note:: Relationship to MCP tool-call strategies

           This loop **always** uses "LLM decides" semantics — the LLM
           selects which tools to call via the native ``tool_calls``
           protocol.  The YAML ``tool_call_strategy`` setting (``all``,
           ``sequential``, ``llm_decides``) is **not consulted** here;
           those strategies are honored by :meth:`MCPDispatcher.fetch`,
           which is used only during **skill execution**
           (:class:`SkillExecutor`).  For regular (non-skill) queries,
           the agentic loop is the sole execution path and the strategy
           field is effectively ignored.
        """
        # Agentic loop requires full response objects (tool_calls),
        # so we use litellm directly — LLMProvider.complete() only returns str.
        import litellm
        from ..llm import get_llm_kwargs

        llm_config = self._config.llm
        model = llm_config.model if llm_config else (self.llm if isinstance(self.llm, str) else str(self.llm))
        tool_results: dict[str, Any] = {}

        # ── Discover MCP capabilities ───────────────────────
        caps = await self._mcp_dispatcher.render_capabilities(auth, agent_name=self.name)

        # ── Build unified tool list (litellm format) ────────
        # Built-in tools win over MCP tools with the same name.
        builtin_tool_names, builtin_tool_defs = self._builtin_tools_to_litellm(skip_tools)

        mcp_tool_defs = MCPDispatcher.mcp_tools_to_litellm(
            [
                t
                for t in caps.raw_tools
                if t["name"] not in builtin_tool_names and not (skip_tools and t["name"] in skip_tools)
            ],
        )

        litellm_tools = mcp_tool_defs + builtin_tool_defs

        if not litellm_tools:
            # No tools available — return immediately, summarisation will handle it.
            return None, tool_results

        # ── Build system prompt ─────────────────────────────
        system_prompt = self._build_agentic_system_prompt(caps, rag_data, state)

        # ── Build conversation messages ─────────────────────
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if state:
            messages.extend(
                self.extract_conversation_history(
                    state,
                    strip_prefixes=self._compute_agent_prefixes(),
                )
            )
        messages.append({"role": "user", "content": query})

        # ── Duplicate call tracking ─────────────────────────
        seen_calls: dict[str, str] = {}  # "name|args_json" → cached result text
        consecutive_dupes = 0

        # ── Loop ────────────────────────────────────────────
        for round_num in range(self._MAX_TOOL_ROUNDS):
            call_kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": llm_config.temperature if llm_config else 0.2,
                **get_llm_kwargs(model),
            }

            # Strip tools after too many consecutive duplicate calls
            # to force the LLM to produce a text response.
            if consecutive_dupes >= self._MAX_CONSECUTIVE_DUPES:
                logger.warning(
                    "[%s] %d consecutive duplicate calls — forcing text-only response",
                    self.name,
                    consecutive_dupes,
                )
            elif litellm_tools:
                call_kwargs["tools"] = litellm_tools
                call_kwargs["tool_choice"] = "auto"

            # ── LLM call with error handling ────────────────
            try:
                response = await litellm.acompletion(**call_kwargs)
            except Exception as exc:
                error_msg = str(exc)
                logger.error(
                    "[%s] LLM API error in round %d: %s",
                    self.name,
                    round_num,
                    error_msg,
                    exc_info=True,
                )
                if "503" in error_msg or "high demand" in error_msg.lower():
                    return (
                        "Currently experiencing high demand. Please try again shortly.",
                        tool_results,
                    )
                if "rate limit" in error_msg.lower():
                    return (
                        "Rate limit reached. Please try again in a few moments.",
                        tool_results,
                    )
                return (
                    f"Error processing request: {error_msg[:200]}. Please try again later.",
                    tool_results,
                )

            choice = response.choices[0]
            assistant_msg = choice.message
            messages.append(assistant_msg.model_dump(exclude_none=True))

            # ── No tool calls → final text response ─────────
            tool_calls = getattr(assistant_msg, "tool_calls", None)
            if not tool_calls:
                final_text = assistant_msg.content or ""
                logger.info("[%s] LLM responded after %d tool round(s)", self.name, round_num)
                return final_text, tool_results

            # ── Execute tool calls ──────────────────────────
            for tc in tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}
                    logger.warning("[%s] Failed to parse arguments for tool '%s'", self.name, fn_name)

                logger.info(
                    "[%s] Tool call #%d → %s | args: %s",
                    self.name,
                    round_num + 1,
                    fn_name,
                    fn_args,
                )

                # ── Duplicate detection ─────────────────────
                call_key = f"{fn_name}|{json.dumps(fn_args, sort_keys=True)}"
                if call_key in seen_calls:
                    consecutive_dupes += 1
                    result_text = (
                        "You already called this tool with the same parameters. "
                        "Here is the previous result (do NOT call it again — "
                        "summarise this data for the user instead):\n\n" + seen_calls[call_key]
                    )
                    logger.warning(
                        "[%s] Duplicate tool call #%d → %s (%d consecutive)",
                        self.name,
                        round_num + 1,
                        fn_name,
                        consecutive_dupes,
                    )
                # ── Route to built-in tool ──────────────────
                elif fn_name in builtin_tool_names:
                    consecutive_dupes = 0
                    result_text = await self._call_builtin_tool(fn_name, fn_args, auth)
                    seen_calls[call_key] = result_text
                # ── Route to MCP tool ───────────────────────
                elif fn_name in caps.tool_client_map:
                    consecutive_dupes = 0
                    client, _server_cfg = caps.tool_client_map[fn_name]
                    try:
                        result = await client.call_tool(fn_name, fn_args, auth)
                        result_text = result.text
                        if result.is_error:
                            result_text = f"[Tool error] {result_text}"
                            logger.warning(
                                "[%s] Tool #%d ← %s ERROR: %s", self.name, round_num + 1, fn_name, result_text[:300]
                            )
                        else:
                            seen_calls[call_key] = result_text
                    except Exception as exc:
                        result_text = f"[Tool error] {exc}"
                        logger.error(
                            "[%s] Tool #%d ← %s EXCEPTION: %s", self.name, round_num + 1, fn_name, exc, exc_info=True
                        )
                else:
                    result_text = f"[Error] Unknown tool '{fn_name}'"
                    logger.error("[%s] Unknown tool '%s' in agentic loop", self.name, fn_name)

                tool_results[fn_name] = result_text
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    }
                )

        # Safety: max rounds exceeded
        logger.warning("[%s] Hit max tool rounds (%d)", self.name, self._MAX_TOOL_ROUNDS)
        return None, tool_results

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
