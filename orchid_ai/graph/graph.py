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

import hashlib
import logging
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from ..config.registry import get_class
from ..config.schema import OrchidAgentConfig, OrchidAgentsConfig
from ..config.tool_registry import load_tools_from_config
from ..core.agent import OrchidAgent, OrchidAgentRunContext
from ..core.guardrails import (
    OrchidGuardrailAction,
    OrchidGuardrailChain,
    OrchidGuardrailContext,
    OrchidGuardrailDirection,
)
from ..core.repository import OrchidVectorReader
from ..core.run_config import auth_from_config
from ..mcp.auth_registry import OrchidMCPAuthRegistry
from ..runtime import MCPClientFactory, OrchidRuntime
from ..tools.external_cli_config import load_external_agents_from_config
from .decision_bindings import resolve_decision_binding
from .guardrail_wiring import _GuardrailWiring
from .mini_agent_wiring import _MiniAgentWiring
from .model_routing import ModelRoutingSelector, TurnBudgetRegistry, build_routed_chat_model
from .routing import build_routing_strategy
from .state import GraphState
from .supervisor import create_supervisor_node

# Backward-compat re-exports (moved to dedicated modules in M2 refactoring)
_make_fork_router = _MiniAgentWiring.make_fork_router

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


def _turn_key(state: GraphState) -> str:
    """Stable key for one user turn across every agent node in a graph run.

    The latest human message and its count do not change while the supervisor
    and agents loop within a request, so the first agent creates the shared
    turn budget and later agents spend from it. The query text is hashed rather
    than stored, and the message id distinguishes repeated identical prompts.
    """
    chat_id = state.get("chat_id") or ""
    human_count = 0
    last_human: Any = None
    for message in state.get("messages") or []:
        if isinstance(message, HumanMessage):
            human_count += 1
            last_human = message

    text = str(getattr(last_human, "content", "")) if last_human is not None else ""
    message_id = (getattr(last_human, "id", "") or "") if last_human is not None else ""
    digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]
    return f"{chat_id}:{human_count}:{message_id}:{digest}"


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
        model_router: ModelRoutingSelector | None = None,
        turn_budgets: TurnBudgetRegistry | None = None,
    ) -> None:
        self._agent = agent
        self._input_guardrails = input_guardrails
        self._output_guardrails = output_guardrails
        self._agent_config = agent_config
        self._model_router = model_router
        self._turn_budgets = turn_budgets
        self.__name__ = f"{agent.name}_agent"

    async def __call__(self, state: GraphState, config: RunnableConfig = None) -> GraphState:
        auth = auth_from_config(config)

        # Per-agent input rails run before the routing point sees the query, so
        # blocked content is never shipped to a decision provider (F2).
        blocked = await self._run_input_guardrails(state, auth)
        if blocked is not None:
            return blocked

        if self._model_router is None:
            return await self._run_after_input_guardrails(state, auth)

        # Ask the ``model_routing`` point once per agent run and bind the
        # selection for the duration of this coroutine. The routed chat model
        # reads it on every LLM call, so concurrent runs (and every other
        # agent) keep their own selection. The budget is shared across every
        # agent node in the same turn (F4).
        query = self._agent.extract_user_query(state)
        budget = self._turn_budgets.for_turn(_turn_key(state)) if self._turn_budgets is not None else None
        async with self._model_router.bound(query, budget=budget):
            return await self._run_after_input_guardrails(state, auth)

    async def _run_after_input_guardrails(self, state: GraphState, auth: Any) -> GraphState:
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
            chat_model=self._agent.chat_model,
            mcp_clients=self._agent.mcp_clients or [],
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
        # Bind per-request state on a ContextVar so two concurrent
        # activations of the same agent (LangGraph ``Send()``
        # fan-out, mini-agent forks, events runner) stay isolated.
        # ``correlation_id`` inherits the existing binding — the
        # wrapper does not own it.
        token = self._agent.set_run_context(
            OrchidAgentRunContext(
                auth=auth,
                correlation_id=self._agent._current_correlation_id,
                chat_id=state.get("chat_id") or None,
                message_id=_latest_human_message_id(state),
            )
        )

        agent_start = time.perf_counter()
        perf_logger.info("[PERF][agent=%s] >>> START", self._agent.name)
        try:
            result = await self._agent.run(state)
        except GraphBubbleUp:
            # HITL ``interrupt()`` and LangGraph control-flow signals must
            # bubble up to the graph runtime, not be swallowed as agent errors.
            raise
        except Exception as exc:
            logger.exception(
                "[Graph] Agent '%s' raised an unhandled exception",
                self._agent.name,
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
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            self._agent.reset_run_context(token)
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
    model_router: ModelRoutingSelector | None = None,
    turn_budgets: TurnBudgetRegistry | None = None,
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
    return _AgentNodeWrapper(agent, input_guardrails, output_guardrails, agent_config, model_router, turn_budgets)


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
    skill_decision_model: Any | None = None,
    skill_confidence_threshold: float = 0.75,
    skill_on_low_confidence: str = "llm",
    tool_decision_model: Any | None = None,
    tool_confidence_threshold: float = 0.75,
    tool_max_candidates: int = 32,
    rag_decision_model: Any | None = None,
    rag_confidence_threshold: float = 0.75,
    rag_strategies: list[str] | None = None,
    retrieval_strategy_binding: Any | None = None,
    decision_model: Any | None = None,
    model_routing_binding: Any | None = None,
    model_routing_max_calls: int = 4,
    model_routing_max_total_ms: int | None = None,
    model_routing_out: dict[str, ModelRoutingSelector] | None = None,
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

    # Wrap the configured model when the ``model_routing`` point is enabled.
    # The wrapper delegates every call to the model selected for the current
    # agent run and falls back to ``agent_chat_model`` (the configured default)
    # outside a graph run or when no selection is made.
    model_router: ModelRoutingSelector | None = None
    if model_routing_binding is not None:
        model_router = ModelRoutingSelector(
            model_routing_binding,
            default_model=agent_model,
            fallback_model=agent_fallback,
            agent_name=name,
            max_calls_per_turn=model_routing_max_calls,
            max_total_ms=model_routing_max_total_ms,
        )
        agent_chat_model = build_routed_chat_model(
            model_router,
            default_chat_model=agent_chat_model,
            temperature=agent_llm.temperature if agent_llm else 0.2,
            fallback_model=agent_fallback,
            retry_attempts=agent_retry,
        )
        if model_routing_out is not None:
            model_routing_out[name] = model_router
        logger.info(
            "[Graph] agent '%s' model routing enabled (catalog=%s, default=%s)",
            name,
            model_router.catalog,
            agent_model,
        )

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
    if skill_decision_model is not None:
        kwargs["skill_decision_model"] = skill_decision_model
        kwargs["skill_confidence_threshold"] = skill_confidence_threshold
        kwargs["skill_on_low_confidence"] = skill_on_low_confidence
    if tool_decision_model is not None:
        kwargs["tool_decision_model"] = tool_decision_model
        kwargs["tool_confidence_threshold"] = tool_confidence_threshold
        kwargs["tool_max_candidates"] = tool_max_candidates
    if rag_decision_model is not None:
        kwargs["rag_decision_model"] = rag_decision_model
        kwargs["rag_confidence_threshold"] = rag_confidence_threshold
        kwargs["rag_strategies"] = rag_strategies
    if retrieval_strategy_binding is not None:
        kwargs["retrieval_strategy_binding"] = retrieval_strategy_binding
    if decision_model is not None:
        kwargs["decision_model"] = decision_model

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
    root_config: OrchidAgentsConfig | None = None,
    runtime: Any | None = None,
    turn_budgets: TurnBudgetRegistry | None = None,
) -> Any:
    """
    Build a sub-graph for an agent with children.

    Returns a compiled LangGraph sub-graph that the parent graph
    treats as a single node.
    """
    children_agents: list[OrchidAgent] = []
    child_model_routers: dict[str, ModelRoutingSelector] = {}
    model_routing_max_calls = root_config.decisions.budget.max_calls_per_turn if root_config is not None else 4
    model_routing_max_total_ms = root_config.decisions.budget.max_total_ms if root_config is not None else None
    for child_name, child_config in (agent_config.children or {}).items():
        child_skill_binding = (
            root_config.decisions.resolve("skills", child_config.decisions) if root_config is not None else None
        )
        child_skill_model = (
            runtime.get_decision_model(child_skill_binding.model)
            if runtime is not None and child_skill_binding is not None and child_skill_binding.model
            else None
        )
        child_retrieval_binding = (
            resolve_decision_binding(
                root_config,
                runtime,
                "retrieval_strategy",
                agent_name=f"{parent_name}.{child_name}",
            )
            if root_config is not None and runtime is not None
            else None
        )
        child_model_routing_binding = (
            resolve_decision_binding(
                root_config,
                runtime,
                "model_routing",
                agent_name=f"{parent_name}.{child_name}",
            )
            if root_config is not None and runtime is not None
            else None
        )
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
            skill_decision_model=child_skill_model,
            skill_confidence_threshold=(
                child_skill_binding.confidence_threshold if child_skill_binding is not None else 0.75
            ),
            skill_on_low_confidence=(
                child_skill_binding.on_low_confidence if child_skill_binding is not None else "llm"
            ),
            retrieval_strategy_binding=child_retrieval_binding,
            model_routing_binding=child_model_routing_binding,
            model_routing_max_calls=model_routing_max_calls,
            model_routing_max_total_ms=model_routing_max_total_ms,
            model_routing_out=child_model_routers,
        )
        children_agents.append(child_agent)

    # Build sub-graph with its own supervisor
    child_descriptions = {a.name: a.description for a in children_agents}
    sub_supervisor = create_supervisor_node(default_model, child_descriptions, chat_model=default_chat_model)

    sg = StateGraph(GraphState)
    sg.add_node("supervisor", sub_supervisor)

    for child in children_agents:
        node_name = f"{child.name}_agent"
        sg.add_node(
            node_name,
            _create_agent_node(
                child,
                model_router=child_model_routers.get(child.name),
                turn_budgets=turn_budgets,
            ),
        )
        sg.add_edge(node_name, "supervisor")

    sg.set_entry_point("supervisor")
    sg.add_conditional_edges("supervisor", _make_route_to_agents(has_output_guardrails=False))

    compiled = sg.compile()
    logger.info(
        "[Graph] sub-graph '%s' compiled with children: %s",
        parent_name,
        list(child_descriptions.keys()),
    )
    return compiled


def _route_after_input_guardrails(state: GraphState) -> str:
    """Conditional edge: skip supervisor if input guardrails blocked."""
    if state.get("final_response"):
        return END
    return "supervisor"


def _make_route_to_agents(has_output_guardrails: bool = False):
    """Build a conditional edge function for the supervisor node.

    Captures whether the graph has a global output-guardrails node so the
    sentinel key ``_has_output_guardrails`` does not need to be threaded
    through state manually.
    """

    def route_to_agents(state: GraphState) -> list[Send] | str:
        active = state.get("active_agents", [])
        if active:
            mode = state.get("execution_mode", "parallel")
            if mode == "parallel":
                logger.info("[Route] parallel dispatch -> %s", active)
                return [Send(f"{agent}_agent", state) for agent in active]
            logger.info("[Route] sequential dispatch -> %s", active[0])
            return f"{active[0]}_agent"

        if state.get("final_response"):
            if has_output_guardrails:
                return "output_guardrails"
            return END

        if state.get("pending_agents"):
            return "supervisor"

        if has_output_guardrails:
            return "output_guardrails"
        return END

    return route_to_agents


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

    # Wire security setting from config → runtime so the MCP factory
    # can enforce the passthrough host allowlist.
    if config.allowed_passthrough_hosts and not runtime.allowed_passthrough_hosts:
        runtime.allowed_passthrough_hosts = config.allowed_passthrough_hosts

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

    # ── Register external-agent CLI tools ──
    if getattr(config, "external_agents", None):
        load_external_agents_from_config(config.external_agents, chat_model=default_chat_model)
        logger.info("[Graph] registered %d external-agent tools", len(config.external_agents))

    # ── Build global guardrail chains ──
    global_input_chain, global_output_chain = _GuardrailWiring.build_chains(config.guardrails)
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
    # Agent name → selector for a per-agent ``model_routing`` binding. Node
    # wrappers read it to ask the point once per request and bind the selected
    # model for that run.
    model_routing_routers: dict[str, ModelRoutingSelector] = {}
    model_routing_max_calls = config.decisions.budget.max_calls_per_turn
    model_routing_max_total_ms = config.decisions.budget.max_total_ms
    # One budget per graph turn, shared by every agent node (F4).
    turn_budgets = TurnBudgetRegistry(
        max_calls_per_turn=model_routing_max_calls,
        max_total_ms=model_routing_max_total_ms,
    )

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
                root_config=config,
                runtime=runtime,
                turn_budgets=turn_budgets,
            )
            subgraph_nodes[agent_name] = subgraph
        else:
            skill_binding = config.decisions.resolve("skills", agent_config.decisions)
            skill_model = (
                runtime.get_decision_model(skill_binding.model)
                if skill_binding is not None and skill_binding.model
                else None
            )
            if skill_binding is not None and skill_binding.model and skill_model is None:
                logger.warning(
                    "[Graph] agent '%s' skills provider %r is configured but missing from the runtime",
                    agent_name,
                    skill_binding.model,
                )
            agent_decision_model = (
                runtime.get_decision_model(agent_config.decision_model) if agent_config.decision_model else None
            )
            if agent_config.decision_model and agent_decision_model is None:
                logger.warning(
                    "[Graph] agent '%s' decision_model %r is configured but missing from the runtime",
                    agent_name,
                    agent_config.decision_model,
                )
            rag_binding = config.decisions.resolve("rag", agent_config.decisions)
            rag_model = (
                runtime.get_decision_model(rag_binding.model) if rag_binding is not None and rag_binding.model else None
            )
            if rag_binding is not None and rag_binding.model and rag_model is None:
                logger.warning(
                    "[Graph] agent '%s' RAG provider %r is configured but missing from the runtime",
                    agent_name,
                    rag_binding.model,
                )
            tool_binding = config.decisions.resolve("tools", agent_config.decisions)
            tool_model = (
                runtime.get_decision_model(tool_binding.model)
                if tool_binding is not None and tool_binding.model
                else None
            )
            if tool_binding is not None and tool_binding.model and tool_model is None:
                logger.warning(
                    "[Graph] agent '%s' tools provider %r is configured but missing from the runtime",
                    agent_name,
                    tool_binding.model,
                )
            retrieval_binding = resolve_decision_binding(
                config,
                runtime,
                "retrieval_strategy",
                agent_name=agent_name,
            )
            model_routing_binding = resolve_decision_binding(
                config,
                runtime,
                "model_routing",
                agent_name=agent_name,
            )
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
                skill_decision_model=skill_model,
                skill_confidence_threshold=(skill_binding.confidence_threshold if skill_binding is not None else 0.8),
                retrieval_strategy_binding=retrieval_binding,
                model_routing_binding=model_routing_binding,
                model_routing_max_calls=model_routing_max_calls,
                model_routing_max_total_ms=model_routing_max_total_ms,
                model_routing_out=model_routing_routers,
            )
            agents.append(agent)

        # Build per-agent guardrail chains
        if agent_config.guardrails.input or agent_config.guardrails.output:
            input_chain, output_chain = _GuardrailWiring.build_chains(agent_config.guardrails)
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
        routing=build_routing_strategy(config, runtime),
    )

    # ── Build graph ──
    g = StateGraph(GraphState)

    # Add global input guardrails node (before supervisor)
    if has_global_input_rails:
        g.add_node("input_guardrails", _GuardrailWiring.create_global_input_node(global_input_chain))

    g.add_node("supervisor", supervisor_node)

    # Add global output guardrails node (after synthesis)
    if has_global_output_rails:
        g.add_node("output_guardrails", _GuardrailWiring.create_global_output_node(global_output_chain))

    for agent in agents:
        node_name = f"{agent.name}_agent"
        ag = agent_guardrails.get(agent.name)
        input_chain = ag[0] if ag else None
        output_chain = ag[1] if ag else None
        agent_config = config.agents.get(agent.name)
        g.add_node(
            node_name,
            _create_agent_node(
                agent,
                input_chain,
                output_chain,
                agent_config=agent_config,
                model_router=model_routing_routers.get(agent.name),
                turn_budgets=turn_budgets,
            ),
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
        parent_chat_model = agent.chat_model
        if mini_enabled and parent_chat_model is not None:
            _MiniAgentWiring.wire_mini_topology(
                graph=g,
                agent_name=agent.name,
                agent_config=agent_config,
                agent_chat_model=parent_chat_model,
                agent_mcp_clients=agent.mcp_clients,
                node_name=node_name,
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
    g.add_conditional_edges("supervisor", _make_route_to_agents(has_output_guardrails=has_global_output_rails))

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
