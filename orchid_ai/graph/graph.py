"""
LangGraph graph definition — Composition Root.

All dependency wiring happens here:
  - Agents are instantiated from YAML config (GenericAgent or custom class)
  - MCP clients are created from YAML ``mcp_servers`` definitions
  - The NullVectorReader is used until the RAG pipeline is ready
  - The Supervisor receives agent descriptions for dynamic routing
  - Guardrails are wired as graph nodes (global) and agent wrappers (per-agent)

Graph topology (parallel vs sequential, guardrails):

  ┌──────────────────────┐
  │  input_guardrails     │  ← Global input rails (content safety, PII, injection)
  └──────────┬───────────┘
             │ blocked → blocked_response → END
             ▼
  ┌──────────────────────┐
  │  supervisor           │ ◀───────────┐
  └──────────┬───────────┘              │
             │ Send() fan-out           │
             ├───────────┐              │
             ▼           ▼              │
  ┌────────────┐  ┌────────────┐       │
  │ agent A    │  │ agent B    │       │  (per-agent rails wrap each node)
  └─────┬──────┘  └─────┬──────┘       │
        └───────────────┘───────────────┘
                         │
                         ▼
  ┌──────────────────────┐
  │  output_guardrails    │  ← Global output rails (PII redaction, groundedness)
  └──────────┬───────────┘
             ▼
            END

  The Supervisor decides the mode per request based on agent dependencies.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from ..agents.mini_agent_aggregator import aggregator_node_factory
from ..agents.mini_agent_node import mini_agent_node_factory
from ..config.schema import OrchidAgentConfig, OrchidAgentsConfig, OrchidGuardrailsConfig
from ..config.registry import get_class
from ..config.tool_registry import load_tools_from_config
from ..core.agent import OrchidAgent
from ..core.guardrails import (
    OrchidGuardrailAction,
    OrchidGuardrailChain,
    OrchidGuardrailContext,
    OrchidGuardrailDirection,
)
from ..core.repository import OrchidVectorReader

from langchain_core.language_models import BaseChatModel
from ..mcp.auth_registry import OrchidMCPAuthRegistry
from ..runtime import MCPClientFactory, OrchidRuntime
from .state import GraphState
from .supervisor import create_supervisor_node, route_to_agents

logger = logging.getLogger(__name__)
perf_logger = logging.getLogger("orchid.perf")


def _latest_human_message_id(state: GraphState) -> str | None:
    """Find the id of the most recent ``HumanMessage`` in graph state.

    Returns ``None`` when the messages list is empty, when no
    ``HumanMessage`` is present, or when the message has no
    ``.id`` attribute (LangChain typically assigns one but the
    field is optional in the base class).  Used by
    :func:`_create_agent_node` to anchor in-chat live progress
    cards under the message that triggered the turn (§LS5).
    """
    messages = state.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            mid = getattr(msg, "id", None)
            if isinstance(mid, str) and mid:
                return mid
            return None
    return None


class _AgentNodeWrapper:
    """LangGraph node that wraps an :class:`OrchidAgent` with per-agent guardrails
    and the optional mini-agent decomposer hook.

    Extracted from the :func:`_create_agent_node` closure so each concern
    lives in its own method.

    SRP — each method has one reason to change:
    - Input guardrails
    - Mini-agent decomposer
    - Agent execution
    - Output guardrails
    """

    def __init__(
        self,
        agent: OrchidAgent,
        input_guardrails: OrchidGuardrailChain | None = None,
        output_guardrails: OrchidGuardrailChain | None = None,
        agent_config: OrchidAgentConfig | None = None,
    ) -> None:
        self._agent = agent
        self._input_guardrails = input_guardrails
        self._output_guardrails = output_guardrails
        self._agent_config = agent_config
        self.__name__ = f"{agent.name}_agent"

    async def __call__(self, state: GraphState) -> GraphState:
        auth = state.get("auth_context")

        blocked = await self._run_input_guardrails(state, auth)
        if blocked is not None:
            return blocked

        decomposer_update = await self._run_decomposer(state, auth)
        if decomposer_update is not None:
            decomposer_update.setdefault("active_agents", [])
            return decomposer_update

        agent_result = await self._run_agent(state, auth)

        await self._run_output_guardrails(state, auth, agent_result)

        agent_result["active_agents"] = []
        return agent_result

    async def _run_input_guardrails(self, state: GraphState, auth: Any) -> GraphState | None:
        if not self._input_guardrails or self._input_guardrails.empty:
            return None

        query = self._agent.extract_user_query(state)
        ctx = OrchidGuardrailContext(
            direction=OrchidGuardrailDirection.INPUT,
            agent_name=self._agent.name,
            tenant_key=auth.tenant_key if auth else "default",
            user_id=auth.user_id if auth else "",
            chat_id=state.get("chat_id", ""),
        )
        result = await self._input_guardrails.evaluate(query, ctx)
        if result.blocked:
            logger.warning(
                "[Guardrails] Agent '%s' input blocked by '%s': %s",
                self._agent.name,
                result.guardrail_name,
                result.message,
            )
            return {
                "messages": [AIMessage(content=f"[{self._agent.name.title()} Agent] {result.message}")],
                "active_agents": [],
            }
        return None

    async def _run_decomposer(self, state: GraphState, auth: Any) -> GraphState | None:
        if self._agent_config is None or not self._agent_config.mini_agent.enabled or auth is None:
            return None

        from ..agents.mini_agent_decomposer import maybe_decompose

        decomp_start = time.perf_counter()
        update = await maybe_decompose(
            agent_config=self._agent_config,
            chat_model=getattr(self._agent, "_chat_model", None),
            mcp_clients=getattr(self._agent, "mcp_clients", None) or [],
            auth=auth,
            state=state,
        )
        decomp_elapsed = (time.perf_counter() - decomp_start) * 1000
        perf_logger.info(
            "[PERF][agent=%s] step=decomposer took %.1f ms (forked=%s)",
            self._agent.name,
            decomp_elapsed,
            bool(update and "mini_agent_decisions" in update),
        )
        return update

    async def _run_agent(self, state: GraphState, auth: Any) -> GraphState:
        prev_chat_id = self._agent._current_chat_id
        prev_message_id = self._agent._current_message_id
        self._agent._current_chat_id = state.get("chat_id") or None
        self._agent._current_message_id = _latest_human_message_id(state)

        agent_start = time.perf_counter()
        perf_logger.info("[PERF][agent=%s] >>> START", self._agent.name)
        try:
            result = await self._agent.run(state)
        except Exception as exc:
            logger.error(
                "[Graph] Agent '%s' raised an unhandled exception: %s",
                self._agent.name,
                exc,
                exc_info=True,
            )
            result = {
                "messages": [
                    AIMessage(
                        content=(
                            f"[{self._agent.name.title()} Agent] I'm temporarily unable to process "
                            "your request. Please try again in a few moments."
                        )
                    )
                ],
            }
        finally:
            self._agent._current_chat_id = prev_chat_id
            self._agent._current_message_id = prev_message_id
        agent_elapsed = (time.perf_counter() - agent_start) * 1000
        perf_logger.info("[PERF][agent=%s] <<< DONE total=%.1f ms", self._agent.name, agent_elapsed)
        return result

    async def _run_output_guardrails(self, state: GraphState, auth: Any, agent_result: GraphState) -> None:
        if not self._output_guardrails or self._output_guardrails.empty:
            return

        agent_messages = agent_result.get("messages", [])
        if not agent_messages:
            return

        response_text = str(agent_messages[-1].content) if hasattr(agent_messages[-1], "content") else ""
        ctx = OrchidGuardrailContext(
            direction=OrchidGuardrailDirection.OUTPUT,
            agent_name=self._agent.name,
            tenant_key=auth.tenant_key if auth else "default",
            user_id=auth.user_id if auth else "",
            chat_id=state.get("chat_id", ""),
            metadata={"rag_context": agent_result.get("rag_context", {}).get(self._agent.name, [])},
        )
        result = await self._output_guardrails.evaluate(response_text, ctx)
        if result.blocked:
            logger.warning(
                "[Guardrails] Agent '%s' output blocked by '%s': %s",
                self._agent.name,
                result.guardrail_name,
                result.message,
            )
            agent_result["messages"] = [AIMessage(content=f"[{self._agent.name.title()} Agent] {result.message}")]
        elif result.action == OrchidGuardrailAction.REDACT and result.redacted_content is not None:
            logger.info("[Guardrails] Agent '%s' output redacted by '%s'", self._agent.name, result.guardrail_name)
            agent_result["messages"] = [AIMessage(content=result.redacted_content)]


def _create_agent_node(
    agent: OrchidAgent,
    input_guardrails: OrchidGuardrailChain | None = None,
    output_guardrails: OrchidGuardrailChain | None = None,
    agent_config: OrchidAgentConfig | None = None,
):
    """
    Wrap a OrchidAgent into a LangGraph node function.

    When per-agent guardrails are configured, input is checked before
    ``agent.run()`` and output is checked after.

    When ``agent_config.mini_agent.enabled`` is true, the wrapper
    additionally runs the decomposer hook BEFORE
    ``agent.run()``.  If the decomposer chooses to fork, the wrapper
    returns the decision state update (with no AIMessage) and the
    graph's conditional edge fans out into mini-agents.

    Returns an :class:`_AgentNodeWrapper` — a callable LangGraph node
    that delegates to focused per-concern methods.
    """
    return _AgentNodeWrapper(agent, input_guardrails, output_guardrails, agent_config)


def _instantiate_agent(
    name: str,
    agent_config: OrchidAgentConfig,
    default_model: str,
    reader: OrchidVectorReader,
    default_chat_model: BaseChatModel | None = None,
    default_fallback: str | None = None,
    default_retry: int = 0,
    mcp_client_factory: MCPClientFactory | None = None,
    summary_config: dict[str, Any] | None = None,
    graph_store: Any | None = None,
    content_sources: Any | None = None,
    upload_namespace: str = "uploads",
) -> OrchidAgent:
    """
    Create an agent instance from its YAML config.

    If ``class_path`` is set, the referenced class is used.
    Otherwise, ``GenericAgent`` handles the standard flow.

    Each agent gets its own ``BaseChatModel`` when its ``llm`` config
    differs from the default (different model, fallback, or retry).
    Otherwise, the shared ``default_chat_model`` is reused.
    """
    from ..llm_factory import build_chat_model

    # Create MCP clients from config via the pluggable factory
    factory = mcp_client_factory or OrchidRuntime().get_mcp_client_factory()
    mcp_clients = [factory(server) for server in agent_config.mcp_servers]

    # Resolve the agent class
    cls = get_class(agent_config.class_path)

    # Determine the LLM model, fallback, and retry for this agent
    agent_llm = agent_config.llm
    agent_model = agent_llm.model if agent_llm else default_model
    agent_fallback = (agent_llm.fallback_model if agent_llm else None) or default_fallback
    agent_retry = agent_llm.retry_attempts if agent_llm else default_retry

    # Build per-agent chat model if it differs from default, otherwise reuse shared one
    needs_own_model = agent_llm and (
        agent_model != default_model or agent_fallback != default_fallback or agent_retry != default_retry
    )
    if needs_own_model:
        agent_chat_model = build_chat_model(
            agent_model,
            temperature=agent_llm.temperature if agent_llm else 0.2,
            fallback_model=agent_fallback,
            retry_attempts=agent_retry,
        )
    else:
        agent_chat_model = default_chat_model

    # All kwargs are passed — OrchidAgent accepts **_kwargs so subclasses
    # pick what they need and ignore the rest.  No inspect.signature sniffing.
    kwargs: dict[str, Any] = {
        "model_id": agent_model,
        "reader": reader,
        "mcp_clients": mcp_clients,
        "config": agent_config,
    }
    if agent_chat_model:
        kwargs["chat_model"] = agent_chat_model
    if summary_config:
        kwargs["summary_config"] = summary_config
    if graph_store is not None:
        kwargs["graph_store"] = graph_store
    if content_sources:
        kwargs["content_sources"] = content_sources
    kwargs["upload_namespace"] = upload_namespace

    return cls(**kwargs)


def _build_subgraph(
    parent_name: str,
    agent_config: OrchidAgentConfig,
    default_model: str,
    reader: OrchidVectorReader,
    default_chat_model: BaseChatModel | None = None,
    default_fallback: str | None = None,
    default_retry: int = 0,
    mcp_client_factory: MCPClientFactory | None = None,
    graph_store: Any | None = None,
    content_sources: Any | None = None,
    upload_namespace: str = "uploads",
) -> Any:
    """
    Build a sub-graph for an agent with children.

    Returns a compiled LangGraph sub-graph that the parent graph
    treats as a single node.
    """
    children_agents: list[OrchidAgent] = []
    for child_name, child_config in (agent_config.children or {}).items():
        child_agent = _instantiate_agent(
            child_name,
            child_config,
            default_model,
            reader,
            default_chat_model,
            default_fallback,
            default_retry,
            mcp_client_factory,
            graph_store=graph_store,
            content_sources=content_sources,
            upload_namespace=upload_namespace,
        )
        children_agents.append(child_agent)

    # Build sub-graph with its own supervisor
    child_descriptions = {a.name: a.description for a in children_agents}
    sub_supervisor = create_supervisor_node(default_model, child_descriptions, chat_model=default_chat_model)

    sg = StateGraph(GraphState)
    sg.add_node("supervisor", sub_supervisor)

    for child in children_agents:
        node_name = f"{child.name}_agent"
        sg.add_node(node_name, _create_agent_node(child))
        sg.add_edge(node_name, "supervisor")

    sg.set_entry_point("supervisor")
    sg.add_conditional_edges("supervisor", route_to_agents)

    compiled = sg.compile()
    logger.info(
        "[Graph] sub-graph '%s' compiled with children: %s",
        parent_name,
        list(child_descriptions.keys()),
    )
    return compiled


def _build_guardrail_chains(
    guardrails_config: OrchidGuardrailsConfig,
) -> tuple[OrchidGuardrailChain, OrchidGuardrailChain]:
    """Build input and output guardrail chains from YAML config."""
    from ..guardrails.registry import build_guardrail_chain

    input_configs = [{"type": r.type, "fail_action": r.fail_action, **r.config} for r in guardrails_config.input]
    output_configs = [{"type": r.type, "fail_action": r.fail_action, **r.config} for r in guardrails_config.output]

    return (
        build_guardrail_chain(input_configs),
        build_guardrail_chain(output_configs),
    )


def _create_global_input_guardrail_node(chain: OrchidGuardrailChain):
    """Create a LangGraph node that runs global input guardrails."""

    async def input_guardrails_node(state: GraphState) -> GraphState:
        """Global input guardrails — runs before the supervisor."""
        from ..core.agent import OrchidAgent

        query = OrchidAgent.extract_user_query(state)
        if not query:
            return state

        auth = state.get("auth_context")
        ctx = OrchidGuardrailContext(
            direction=OrchidGuardrailDirection.INPUT,
            tenant_key=auth.tenant_key if auth else "default",
            user_id=auth.user_id if auth else "",
            chat_id=state.get("chat_id", ""),
        )

        result = await chain.evaluate(query, ctx)
        if result.blocked:
            logger.warning("[Guardrails] Global input blocked by '%s': %s", result.guardrail_name, result.message)
            return {
                "messages": [AIMessage(content=result.message)],
                "final_response": result.message,
                "active_agents": [],
                "pending_agents": [],
            }

        if result.action == OrchidGuardrailAction.REDACT and result.redacted_content is not None:
            logger.info("[Guardrails] Global input redacted by '%s'", result.guardrail_name)
            # Replace the last human message with redacted version
            from langchain_core.messages import HumanMessage

            messages = list(state.get("messages", []))
            if messages and isinstance(messages[-1], HumanMessage):
                messages[-1] = HumanMessage(content=result.redacted_content)
            return {"messages": messages}

        return state

    return input_guardrails_node


def _create_global_output_guardrail_node(chain: OrchidGuardrailChain):
    """Create a LangGraph node that runs global output guardrails."""

    async def output_guardrails_node(state: GraphState) -> GraphState:
        """Global output guardrails — runs after synthesis, before END."""
        final = state.get("final_response")
        if not final:
            return state

        auth = state.get("auth_context")
        ctx = OrchidGuardrailContext(
            direction=OrchidGuardrailDirection.OUTPUT,
            tenant_key=auth.tenant_key if auth else "default",
            user_id=auth.user_id if auth else "",
            chat_id=state.get("chat_id", ""),
            metadata={"rag_context": state.get("rag_context", {})},
        )

        result = await chain.evaluate(final, ctx)
        if result.blocked:
            logger.warning("[Guardrails] Global output blocked by '%s': %s", result.guardrail_name, result.message)
            return {
                "messages": [AIMessage(content=result.message)],
                "final_response": result.message,
            }

        if result.action == OrchidGuardrailAction.REDACT and result.redacted_content is not None:
            logger.info("[Guardrails] Global output redacted by '%s'", result.guardrail_name)
            return {
                "messages": [AIMessage(content=result.redacted_content)],
                "final_response": result.redacted_content,
            }

        return state

    return output_guardrails_node


def _make_fork_router(parent_name: str):
    """Build the conditional-edge function for a mini-agent-enabled parent.

    Reads ``state["mini_agent_decisions"][parent_name]`` (written by
    the parent ``GenericAgent.run()`` when the decomposer chose to
    fork) and returns either:

    - ``"supervisor"`` — no fork (parent emitted its own AIMessage),
      or no decision was recorded (defensive fallthrough).
    - A list of ``Send(f"{parent_name}_mini", payload)`` — one per
      sub-task.  Each payload carries the per-Send sentinels the
      mini node reads to identify its sub-task.
    """

    def fork_router(state: GraphState):
        decisions = state.get("mini_agent_decisions") or {}
        decision = decisions.get(parent_name)
        if not decision or not decision.get("should_fork"):
            return "supervisor"

        sub_tasks = decision.get("sub_tasks") or []
        if not sub_tasks:
            return "supervisor"

        sends: list[Send] = []
        for sub_task in sub_tasks:
            mini_id = sub_task.get("id") or f"mini_{len(sends)}"
            tool_subset = sub_task.get("resolved_tool_subset") or sub_task.get("allowed_tools") or []
            sends.append(
                Send(
                    f"{parent_name}_mini",
                    {
                        **state,
                        "_active_mini_parent": parent_name,
                        "_active_mini_id": mini_id,
                        "_active_mini_subtask": sub_task,
                        "_active_mini_tool_subset": list(tool_subset),
                    },
                ),
            )
        logger.info(
            "[Route] %s parent forking into %d mini-agent(s)",
            parent_name,
            len(sends),
        )
        return sends

    fork_router.__name__ = f"{parent_name}_fork_router"
    return fork_router


def _route_after_input_guardrails(state: GraphState) -> str:
    """Conditional edge: skip supervisor if input guardrails blocked."""
    if state.get("final_response"):
        return END
    return "supervisor"


def _route_after_supervisor(state: GraphState) -> str:
    """Conditional edge: route to output guardrails or agents."""
    if state.get("final_response"):
        return "output_guardrails"
    return "route_agents"


def build_graph(
    *,
    config: OrchidAgentsConfig,
    runtime: OrchidRuntime,
    agents_out: dict[str, OrchidAgent] | None = None,
) -> Any:  # returns CompiledGraph
    """
    Build and compile the full agent graph from YAML configuration.

    Parameters
    ----------
    config : OrchidAgentsConfig
        Parsed and validated YAML configuration.
    runtime : OrchidRuntime
        Pre-configured runtime with all dependencies (reader, LLM provider,
        MCP client factory).
    agents_out : dict[str, OrchidAgent] | None
        Optional mutable dict that will be populated with the
        instantiated top-level :class:`OrchidAgent` instances keyed by
        name.  Lets callers (notably :class:`orchid_ai.Orchid`) reach
        the same client instances the graph wired in for proactive
        cache warming via
        :class:`orchid_ai.mcp.session_warmer.OrchidSessionWarmer`.
        Child agents inside compiled subgraphs are not exposed.
    """
    from ..llm_factory import build_chat_model as _build_chat_model

    reader = runtime.get_reader()
    # graph store reaches the agent so ``graph_rag`` retrieval
    # can traverse entities/edges.  Returns a NullGraphStore
    # when the runtime didn't wire one — GraphRAGRetrieval detects
    # that via ``is_null`` and falls back to SimpleRetrieval.
    graph_store = runtime.get_graph_store()
    default_model = runtime.default_model

    # ── Resolve default LLM config from YAML ──
    default_fallback = config.defaults.llm.fallback_model
    default_retry = config.defaults.llm.retry_attempts
    if runtime.chat_model is not None:
        # User provided a pre-built chat model — use it as-is
        default_chat_model: BaseChatModel = runtime.chat_model
    else:
        default_chat_model = _build_chat_model(
            default_model,
            fallback_model=default_fallback,
            retry_attempts=default_retry,
        )

    # ── Build MCP auth registry (scans all agents for OAuth servers) ──
    auth_registry = OrchidMCPAuthRegistry.from_config(config)
    runtime.mcp_auth_registry = auth_registry

    # ── Build MCP factory (enhanced with token_store for OAuth servers) ──
    mcp_factory: MCPClientFactory = runtime.get_mcp_client_factory()

    # ── Register built-in tools from config ──
    if config.tools:
        load_tools_from_config(config.tools)
        logger.info("[Graph] registered %d built-in tools", len(config.tools))

    # ── Build global guardrail chains ──
    global_input_chain, global_output_chain = _build_guardrail_chains(config.guardrails)
    has_global_input_rails = not global_input_chain.empty
    has_global_output_rails = not global_output_chain.empty

    if has_global_input_rails:
        logger.info("[Graph] global input guardrails: %s", global_input_chain)
    if has_global_output_rails:
        logger.info("[Graph] global output guardrails: %s", global_output_chain)

    # ── Instantiate agents from config ──
    agents: list[OrchidAgent] = []
    agent_guardrails: dict[str, tuple[OrchidGuardrailChain, OrchidGuardrailChain]] = {}
    subgraph_nodes: dict[str, Any] = {}

    # Build agent descriptions directly from config (no proxy needed)
    agent_descriptions: dict[str, str] = {name: cfg.description for name, cfg in config.agents.items()}

    # Build summary config dict from OrchidSupervisorConfig (passed to GenericAgent)
    sup = config.supervisor
    summary_cfg: dict[str, Any] | None = None
    if sup.history_summary_enabled:
        summary_cfg = {
            "model": sup.history_summary_model or default_model,
            "recent_turns": sup.history_summary_recent_turns,
        }

    # ── Build conversation memory instance ──────────────────
    memory: Any = None
    memory_strategy = sup.memory.strategy
    if memory_strategy in ("running_summary", "rag_augmented") and runtime.chat_storage is not None:
        from ..agents.memory import OrchidInMemoryConversationMemory
        from ..core.memory import NullConversationMemory

        # Build a cheap chat model for summary extension calls
        memory_model = sup.memory.summary_model or sup.history_summary_model or default_model
        memory_chat_model: BaseChatModel = _build_chat_model(
            memory_model,
            fallback_model=default_fallback,
            retry_attempts=0,
        )
        try:
            if memory_strategy == "rag_augmented":
                from ..agents.memory_rag import OrchidRAGConversationMemory

                memory = OrchidRAGConversationMemory(
                    chat_storage=runtime.chat_storage,
                    chat_model=memory_chat_model,
                    reader=runtime.get_reader(),
                    writer=runtime.get_writer(),
                    structured_output=sup.memory.structured_output,
                )
                logger.info(
                    "[Graph] memory strategy=rag_augmented model=%s namespace=%s k=%d",
                    memory_model,
                    sup.memory.rag_namespace,
                    sup.memory.rag_k,
                )
            else:
                memory = OrchidInMemoryConversationMemory(
                    chat_storage=runtime.chat_storage,
                    chat_model=memory_chat_model,
                    structured_output=sup.memory.structured_output,
                )
                logger.info(
                    "[Graph] memory strategy=running_summary model=%s persist=%s structured=%s",
                    memory_model,
                    sup.memory.persist_summary,
                    sup.memory.structured_output,
                )
            if summary_cfg is not None:
                summary_cfg["memory"] = memory
                summary_cfg["structured_output"] = sup.memory.structured_output
            else:
                summary_cfg = {
                    "model": sup.history_summary_model or default_model,
                    "recent_turns": sup.history_summary_recent_turns,
                    "memory": memory,
                    "structured_output": sup.memory.structured_output,
                }
        except Exception as exc:
            logger.warning("[Graph] Failed to initialise memory: %s", exc)
            memory = NullConversationMemory()
    elif memory_strategy in ("running_summary", "rag_augmented") and runtime.chat_storage is None:
        logger.warning(
            "[Graph] memory.strategy=%s configured but no chat_storage in runtime — memory disabled",
            memory_strategy,
        )
    if memory is None:
        from ..core.memory import NullConversationMemory

        memory = NullConversationMemory()

    if summary_cfg is not None:
        summary_cfg["truncation_strategy"] = sup.memory.truncation_strategy
        summary_cfg["truncation_max_chars"] = sup.memory.truncation_max_chars

    for agent_name, agent_config in config.agents.items():
        if agent_config.children:
            # Agent with children → build a sub-graph
            subgraph = _build_subgraph(
                agent_name,
                agent_config,
                default_model,
                reader,
                default_chat_model,
                default_fallback,
                default_retry,
                mcp_factory,
                graph_store=graph_store,
                content_sources=runtime.content_sources,
                upload_namespace=runtime.upload_namespace,
            )
            subgraph_nodes[agent_name] = subgraph
        else:
            agent = _instantiate_agent(
                agent_name,
                agent_config,
                default_model,
                reader,
                default_chat_model,
                default_fallback,
                default_retry,
                mcp_factory,
                summary_config=summary_cfg,
                graph_store=graph_store,
                content_sources=runtime.content_sources,
                upload_namespace=runtime.upload_namespace,
            )
            agents.append(agent)

        # Build per-agent guardrail chains
        if agent_config.guardrails.input or agent_config.guardrails.output:
            input_chain, output_chain = _build_guardrail_chains(agent_config.guardrails)
            agent_guardrails[agent_name] = (input_chain, output_chain)
            logger.info(
                "[Graph] agent '%s' guardrails: input=%s, output=%s",
                agent_name,
                input_chain,
                output_chain,
            )

    # ── Wire agent peers (for cross-agent skill steps) ──
    agent_map: dict[str, OrchidAgent] = {a.name: a for a in agents}

    # Expose the materialised agents to the caller so the Orchid
    # facade can build an OrchidSessionWarmer over the same client
    # instances we just wired into the graph.
    if agents_out is not None:
        agents_out.update(agent_map)
    for agent in agents:
        if not agent.needs_peer_wiring():
            continue
        peers = {name: peer for name, peer in agent_map.items() if name != agent.name}
        agent.wire_peers(peers)
        logger.info(
            "[Graph] agent '%s' wired with peers: %s",
            agent.name,
            list(peers.keys()),
        )

    # ── Supervisor chat model (may have its own fallback) ──
    sup_fallback = sup.fallback_model or default_fallback
    if sup.fallback_model and sup.fallback_model != default_fallback:
        supervisor_chat_model = _build_chat_model(
            default_model,
            fallback_model=sup_fallback,
            retry_attempts=default_retry,
        )
    else:
        supervisor_chat_model = default_chat_model

    # ── Optional routing/advance chat model (cheaper, for short calls) ──
    # When ``supervisor.routing_model`` is set, build a separate
    # ``BaseChatModel`` for the supervisor's routing + sequential-
    # advance phases.  Synthesis still uses ``supervisor_chat_model``.
    routing_chat_model: BaseChatModel | None = None
    if sup.routing_model and sup.routing_model != default_model:
        routing_chat_model = _build_chat_model(
            sup.routing_model,
            fallback_model=sup_fallback,
            retry_attempts=default_retry,
        )
        logger.info("[Graph] supervisor routing_model=%s (separate from synthesis model)", sup.routing_model)

    # ── Supervisor ──
    supervisor_node = create_supervisor_node(
        default_model,
        agent_descriptions,
        chat_model=supervisor_chat_model,
        orchestrator_skills=config.skills or None,
        supervisor_config=config.supervisor,
        routing_chat_model=routing_chat_model,
        memory=memory,
    )

    # ── Build graph ──
    g = StateGraph(GraphState)

    # Add global input guardrails node (before supervisor)
    if has_global_input_rails:
        g.add_node("input_guardrails", _create_global_input_guardrail_node(global_input_chain))

    g.add_node("supervisor", supervisor_node)

    # Add global output guardrails node (after synthesis)
    if has_global_output_rails:
        g.add_node("output_guardrails", _create_global_output_guardrail_node(global_output_chain))

    for agent in agents:
        node_name = f"{agent.name}_agent"
        ag = agent_guardrails.get(agent.name)
        input_chain = ag[0] if ag else None
        output_chain = ag[1] if ag else None
        agent_config = config.agents.get(agent.name)
        g.add_node(
            node_name,
            _create_agent_node(agent, input_chain, output_chain, agent_config=agent_config),
        )

        # Synthesise the mini + aggregator nodes only when the
        # agent has opted in.  Non-opt-in agents keep today's wiring
        # (a single ``{name}_agent → supervisor`` edge) — zero overhead.
        # Works for ANY ``OrchidAgent`` subclass — ``GenericAgent`` and
        # custom classes both expose ``_chat_model`` + ``mcp_clients``
        # via the base class.  The mini node ignores the parent's
        # ``run()`` and runs its own focused agentic loop, so a custom
        # class's specialised flow is bypassed inside minis by design.
        mini_enabled = bool(agent_config and agent_config.mini_agent.enabled)
        parent_chat_model = getattr(agent, "_chat_model", None)
        if mini_enabled and parent_chat_model is not None:
            mini_node = mini_agent_node_factory(
                parent_config=agent_config,
                chat_model=parent_chat_model,
                mcp_clients=getattr(agent, "mcp_clients", None) or [],
            )
            aggregator_node = aggregator_node_factory(
                parent_config=agent_config,
                chat_model=parent_chat_model,
            )
            g.add_node(f"{agent.name}_mini", mini_node)
            g.add_node(f"{agent.name}_aggregator", aggregator_node)
            g.add_conditional_edges(
                node_name,
                _make_fork_router(agent.name),
                [f"{agent.name}_mini", "supervisor"],
            )
            g.add_edge(f"{agent.name}_mini", f"{agent.name}_aggregator")
            g.add_edge(f"{agent.name}_aggregator", "supervisor")
            logger.info(
                "[Graph] agent '%s' wired with mini-agent topology (max_count=%d, class=%s)",
                agent.name,
                agent_config.mini_agent.max_count,
                type(agent).__name__,
            )
        else:
            if mini_enabled and parent_chat_model is None:
                logger.warning(
                    "[Graph] agent '%s' has mini_agent.enabled=true but no chat_model "
                    "is wired — mini-agent topology disabled for this agent",
                    agent.name,
                )
            g.add_edge(node_name, "supervisor")

    # Add subgraph nodes (agents with children)
    for agent_name, subgraph in subgraph_nodes.items():
        node_name = f"{agent_name}_agent"
        g.add_node(node_name, subgraph)
        g.add_edge(node_name, "supervisor")

    # ── Wire entry point and edges ──
    if has_global_input_rails:
        g.set_entry_point("input_guardrails")
        g.add_conditional_edges("input_guardrails", _route_after_input_guardrails)
    else:
        g.set_entry_point("supervisor")

    # Supervisor routes to agents (or sets final_response for direct answers)
    g.add_conditional_edges("supervisor", route_to_agents)

    # Output guardrails: intercept final_response before END
    if has_global_output_rails:
        # The output guardrails node is reached via route_to_agents when
        # final_response is set.  We wire it to END.
        g.add_edge("output_guardrails", END)

    compiled = g.compile(checkpointer=runtime.checkpointer)
    if runtime.checkpointer:
        logger.info(
            "[Graph] compiled with checkpointer=%s, agents=%s",
            type(runtime.checkpointer).__name__,
            list(agent_descriptions.keys()),
        )
    else:
        logger.info("[Graph] compiled with agents: %s", list(agent_descriptions.keys()))
    return compiled
