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
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from ..agents.delegation import DelegationCatalog
from ..config.registry import get_class
from ..config.schema import OrchidAgentConfig, OrchidAgentsConfig
from ..config.tool_registry import load_tools_from_config
from ..core.agent import OrchidAgent
from ..core.guardrails import OrchidGuardrailChain
from ..core.repository import OrchidVectorReader
from ..core.run_config import auth_from_config
from ..mcp.auth_registry import OrchidMCPAuthRegistry
from ..runtime import MCPClientFactory, OrchidRuntime
from ..tools.external_cli_config import load_external_agents_from_config
from .agent_executor import OrchidAgentExecutor, latest_human_message_id, turn_key
from .decision_bindings import resolve_decision_binding
from .guardrail_wiring import _GuardrailWiring
from .mini_agent_wiring import _MiniAgentWiring
from .model_routing import ModelRoutingSelector, TurnBudgetRegistry, build_routed_chat_model
from .routing import build_routing_strategy
from .state import GraphState
from .supervisor import create_supervisor_node

# Aliases kept because existing importers reach these helpers through this
# module; their implementation lives next to the shared execution policy.
_make_fork_router = _MiniAgentWiring.make_fork_router
_latest_human_message_id = latest_human_message_id
_turn_key = turn_key

logger = logging.getLogger(__name__)


class _AgentNodeWrapper:
    """LangGraph node that wraps an :class:`OrchidAgent` with the shared executor.

    The executor owns run-context binding, per-agent guardrails, model routing
    and error translation. This node keeps only the LangGraph-specific pieces:
    reading auth from the ``RunnableConfig``, deriving the run ids from state,
    and enabling the decomposer hook whose fork decision the graph's ``Send``
    topology consumes. Inline mini-agent orchestration is disabled here — the
    chat graph keeps its own fan-out.
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
        self._executor = OrchidAgentExecutor(
            agent,
            input_guardrails=input_guardrails,
            output_guardrails=output_guardrails,
            agent_config=agent_config,
            model_router=model_router,
            turn_budgets=turn_budgets,
            inline_mini_agents=False,
        )
        self.__name__ = f"{agent.name}_agent"

    async def __call__(self, state: GraphState, config: RunnableConfig = None) -> GraphState:
        return await self._executor.run(
            state,
            auth=auth_from_config(config),
            chat_id=state.get("chat_id") or None,
            message_id=latest_human_message_id(state),
            correlation_id=self._agent._current_correlation_id,
        )


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
    that delegates execution to the shared :class:`OrchidAgentExecutor`.
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
    ingestion_binding: Any | None = None,
    decision_model: Any | None = None,
    tool_choice_binding: Any | None = None,
    agentic_loop_binding: Any | None = None,
    model_routing_binding: Any | None = None,
    model_routing_max_calls: int = 4,
    model_routing_max_total_ms: int | None = None,
    model_routing_out: dict[str, ModelRoutingSelector] | None = None,
    memory_binding: Any | None = None,
    delegation_binding: Any | None = None,
    delegation_catalog: DelegationCatalog | None = None,
    delegation_fail_closed: bool = False,
    delegation_external_agents: frozenset[str] | None = None,
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
    from ..agents.generic_agent import GenericAgent

    # Delegation is a GenericAgent pipeline concern. Legacy custom agents may
    # have a strict constructor and override ``run()``; they stay outside the
    # gate by design, so never inject the delegation kwargs into them.
    supports_delegation = issubclass(cls, GenericAgent)

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
    if ingestion_binding is not None:
        kwargs["ingestion_binding"] = ingestion_binding
    if tool_choice_binding is not None:
        kwargs["tool_choice_binding"] = tool_choice_binding
    if agentic_loop_binding is not None:
        kwargs["agentic_loop_binding"] = agentic_loop_binding
    if memory_binding is not None:
        kwargs["memory_binding"] = memory_binding
    if supports_delegation and (delegation_binding is not None or delegation_fail_closed):
        # Inject the delegation contract only when the point is enabled. A
        # missing provider still injects the fail-closed sentinel so the agent
        # suppresses external tools instead of silently falling back to legacy.
        kwargs["delegation_binding"] = delegation_binding
        kwargs["delegation_fail_closed"] = delegation_fail_closed
        if delegation_catalog is not None:
            kwargs["delegation_catalog"] = delegation_catalog
        if delegation_external_agents is not None:
            kwargs["delegation_external_agents"] = delegation_external_agents
    if decision_model is not None:
        kwargs["decision_model"] = decision_model

    return cls(**kwargs)


def _resolve_delegation_assets(
    config: OrchidAgentsConfig | None,
    runtime: Any | None,
    agent_name: str,
    agent_config: OrchidAgentConfig,
) -> tuple[Any | None, DelegationCatalog, bool, frozenset[str]]:
    """Resolve the ``delegation`` binding, catalog, and fail-closed marker.

    ``resolve_decision_binding`` collapses two materially different states into
    ``None``: the point is disabled/unconfigured, or it is enabled but its
    provider is missing from the runtime. The first keeps legacy behaviour; the
    second must fail closed so external tools stay suppressed. The resolved
    agent config distinguishes them. The returned external-name set is the full
    fleet declaration, not the allowlisted catalog, so skills can never reach
    an external tool outside this agent's ``tools`` list.
    """
    if config is None:
        return None, DelegationCatalog(), False, frozenset()

    external_agents = frozenset(name for name in config.external_agents if isinstance(name, str) and name)
    catalog = DelegationCatalog.from_config(
        config.external_agents,
        allowed_tools=agent_config.tools,
    )
    if runtime is None:
        return None, catalog, False, external_agents

    binding_config = config.resolve_decision("delegation", agent_name)
    if binding_config is None:
        return None, catalog, False, external_agents
    binding = resolve_decision_binding(config, runtime, "delegation", agent_name=agent_name)
    return binding, catalog, binding is None, external_agents


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
    memory_binding: Any | None = None,
    summary_config: dict[str, Any] | None = None,
    memory: Any | None = None,
) -> Any:
    """
    Build a sub-graph for an agent with children.

    Returns a compiled LangGraph sub-graph that the parent graph
    treats as a single node.
    """
    children_agents: list[OrchidAgent] = []
    child_model_routers: dict[str, ModelRoutingSelector] = {}
    #: Child path → (input chain, output chain). Nested agents get the same
    #: per-agent rail treatment as top-level agents; the root config/runtime
    #: resolves a child-specific ``guardrails`` binding.
    child_guardrails: dict[str, tuple[OrchidGuardrailChain, OrchidGuardrailChain]] = {}
    # A parent with children is represented by this subgraph, not by an
    # ``_AgentNodeWrapper``. Its own rails run at the subgraph boundary so they
    # are not silently skipped.
    parent_input_chain: OrchidGuardrailChain | None = None
    parent_output_chain: OrchidGuardrailChain | None = None
    if agent_config.guardrails.input or agent_config.guardrails.output:
        parent_input_chain, parent_output_chain = _GuardrailWiring.build_chains(
            agent_config.guardrails,
            config=root_config,
            runtime=runtime,
            agent_name=parent_name,
        )
        logger.info(
            "[Graph] parent agent '%s' guardrails: input=%s, output=%s",
            parent_name,
            parent_input_chain,
            parent_output_chain,
        )
    model_routing_max_calls = root_config.decisions.budget.max_calls_per_turn if root_config is not None else 4
    model_routing_max_total_ms = root_config.decisions.budget.max_total_ms if root_config is not None else None
    for child_name, child_config in (agent_config.children or {}).items():
        child_path = f"{parent_name}.{child_name}"
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
                agent_name=child_path,
            )
            if root_config is not None and runtime is not None
            else None
        )
        child_ingestion_binding = (
            resolve_decision_binding(
                root_config,
                runtime,
                "ingestion",
                agent_name=child_path,
            )
            if root_config is not None and runtime is not None
            else None
        )
        child_model_routing_binding = (
            resolve_decision_binding(
                root_config,
                runtime,
                "model_routing",
                agent_name=child_path,
            )
            if root_config is not None and runtime is not None
            else None
        )
        child_tool_choice_binding = (
            resolve_decision_binding(
                root_config,
                runtime,
                "tool_choice",
                agent_name=child_path,
            )
            if root_config is not None and runtime is not None
            else None
        )
        child_agentic_loop_binding = (
            resolve_decision_binding(
                root_config,
                runtime,
                "agentic_loop",
                agent_name=child_path,
            )
            if root_config is not None and runtime is not None
            else None
        )
        (
            child_delegation_binding,
            child_delegation_catalog,
            child_delegation_fail_closed,
            child_delegation_external_agents,
        ) = _resolve_delegation_assets(
            root_config,
            runtime,
            child_path,
            child_config,
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
            ingestion_binding=child_ingestion_binding,
            model_routing_binding=child_model_routing_binding,
            model_routing_max_calls=model_routing_max_calls,
            model_routing_max_total_ms=model_routing_max_total_ms,
            model_routing_out=child_model_routers,
            tool_choice_binding=child_tool_choice_binding,
            agentic_loop_binding=child_agentic_loop_binding,
            memory_binding=memory_binding,
            delegation_binding=child_delegation_binding,
            delegation_catalog=child_delegation_catalog,
            delegation_fail_closed=child_delegation_fail_closed,
            delegation_external_agents=child_delegation_external_agents,
            summary_config=summary_config,
        )
        children_agents.append(child_agent)

        if child_config.guardrails.input or child_config.guardrails.output:
            child_input_chain, child_output_chain = _GuardrailWiring.build_chains(
                child_config.guardrails,
                config=root_config,
                runtime=runtime,
                agent_name=child_path,
            )
            child_guardrails[child_name] = (child_input_chain, child_output_chain)
            logger.info(
                "[Graph] sub-agent '%s' guardrails: input=%s, output=%s",
                child_path,
                child_input_chain,
                child_output_chain,
            )

    # Build sub-graph with its own supervisor. The fleet-scoped memory point and
    # the root supervisor's memory config govern this supervisor too; without
    # them the sub-supervisor would fall back to defaults and never persist.
    child_descriptions = {a.name: a.description for a in children_agents}
    sub_supervisor = create_supervisor_node(
        default_model,
        child_descriptions,
        chat_model=default_chat_model,
        supervisor_config=root_config.supervisor if root_config is not None else None,
        memory=memory,
        memory_binding=memory_binding,
    )

    has_parent_input = parent_input_chain is not None and not parent_input_chain.empty
    has_parent_output = parent_output_chain is not None and not parent_output_chain.empty

    sg = StateGraph(GraphState)
    if has_parent_input:
        sg.add_node(
            "input_guardrails",
            _GuardrailWiring.create_global_input_node(parent_input_chain, agent_name=parent_name),
        )
    sg.add_node("supervisor", sub_supervisor)
    if has_parent_output:
        sg.add_node(
            "output_guardrails",
            _GuardrailWiring.create_global_output_node(parent_output_chain, agent_name=parent_name),
        )

    for child in children_agents:
        node_name = f"{child.name}_agent"
        child_chain = child_guardrails.get(child.name)
        sg.add_node(
            node_name,
            _create_agent_node(
                child,
                child_chain[0] if child_chain else None,
                child_chain[1] if child_chain else None,
                model_router=child_model_routers.get(child.name),
                turn_budgets=turn_budgets,
            ),
        )
        sg.add_edge(node_name, "supervisor")

    if has_parent_input:
        sg.set_entry_point("input_guardrails")
        sg.add_conditional_edges("input_guardrails", _route_after_input_guardrails)
    else:
        sg.set_entry_point("supervisor")
    sg.add_conditional_edges("supervisor", _make_route_to_agents(has_output_guardrails=has_parent_output))
    if has_parent_output:
        sg.add_edge("output_guardrails", END)

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


def _merge_guardrail_chains(*chains: OrchidGuardrailChain | None) -> OrchidGuardrailChain | None:
    """Combine guardrail chains in order, dropping empty inputs.

    Pipeline jobs have no supervisor or global guardrail node, so their executor
    must run the global and per-agent rules in one chain. ``None`` is returned
    when every input is empty, which keeps the executor's fast path.
    """
    rails = [rail for chain in chains if chain is not None for rail in chain.guardrails]
    if not rails:
        return None
    return OrchidGuardrailChain(rails)


# ORCHID-MEM: 01a1ef10-9ed5-7bee-8cb7-9ecc818e02af — build_graph exposes leaf pipeline executors with guardrails, routing and inline minis.
def build_graph(
    *,
    config: OrchidAgentsConfig,
    runtime: OrchidRuntime,
    agents_out: dict[str, OrchidAgent] | None = None,
    executors_out: dict[str, OrchidAgentExecutor] | None = None,
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
    executors_out : dict[str, OrchidAgentExecutor] | None
        Optional mutable dict populated with one executor per **top-level leaf**
        agent. Each executor merges the global and per-agent guardrail chains,
        carries the agent's model router and enables inline mini-agents, so
        non-chat callers (pipeline agent jobs) reuse the chat execution policy.
        Parent agents are omitted because their subgraph owns the execution.
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
    # Global rails run before an agent is selected, so a decision-backed rail
    # here uses the fleet-level ``decisions.points.guardrails`` binding only.
    # Per-agent overrides are resolved when that agent's chain is built below.
    global_input_chain, global_output_chain = _GuardrailWiring.build_chains(
        config.guardrails,
        config=config,
        runtime=runtime,
    )
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

    # Fleet-scoped ``memory`` point: one binding for the supervisor and every
    # agent. Unconfigured/disabled resolves to ``None`` and leaves the
    # configured summary/persistence behaviour untouched. The point governs the
    # route and agent seams only; synthesis/advance keep their legacy writes.
    memory_binding = resolve_decision_binding(config, runtime, "memory")
    if memory_binding is not None:
        logger.info("[Graph] memory decision point enabled (provider=%s)", memory_binding.provider)

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
                memory_binding=memory_binding,
                summary_config=summary_cfg,
                memory=memory,
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
            ingestion_binding = resolve_decision_binding(
                config,
                runtime,
                "ingestion",
                agent_name=agent_name,
            )
            model_routing_binding = resolve_decision_binding(
                config,
                runtime,
                "model_routing",
                agent_name=agent_name,
            )
            tool_choice_binding = resolve_decision_binding(
                config,
                runtime,
                "tool_choice",
                agent_name=agent_name,
            )
            agentic_loop_binding = resolve_decision_binding(
                config,
                runtime,
                "agentic_loop",
                agent_name=agent_name,
            )
            (
                delegation_binding,
                delegation_catalog,
                delegation_fail_closed,
                delegation_external_agents,
            ) = _resolve_delegation_assets(
                config,
                runtime,
                agent_name,
                agent_config,
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
                ingestion_binding=ingestion_binding,
                model_routing_binding=model_routing_binding,
                model_routing_max_calls=model_routing_max_calls,
                model_routing_max_total_ms=model_routing_max_total_ms,
                model_routing_out=model_routing_routers,
                tool_choice_binding=tool_choice_binding,
                agentic_loop_binding=agentic_loop_binding,
                memory_binding=memory_binding,
                delegation_binding=delegation_binding,
                delegation_catalog=delegation_catalog,
                delegation_fail_closed=delegation_fail_closed,
                delegation_external_agents=delegation_external_agents,
            )
            agents.append(agent)

        # Build per-agent guardrail chains. A decision-backed rail resolves the
        # ``guardrails`` point for this agent (agent override, else the global
        # binding); its ``on_failure`` policy applies when no verdict arrives.
        # Parent agents are represented by their subgraph and build their chains
        # at that boundary, so skip the unused top-level copy here.
        if not agent_config.children and (agent_config.guardrails.input or agent_config.guardrails.output):
            input_chain, output_chain = _GuardrailWiring.build_chains(
                agent_config.guardrails,
                config=config,
                runtime=runtime,
                agent_name=agent_name,
            )
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
        memory_binding=memory_binding,
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
        if executors_out is not None and agent_config is not None and not agent_config.children:
            # Pipeline jobs are independent activations, so they get a fresh
            # routing budget from the selector instead of the chat graph's
            # per-turn registry (which would let identical jobs share a cap).
            executors_out[agent.name] = OrchidAgentExecutor(
                agent,
                input_guardrails=_merge_guardrail_chains(global_input_chain, input_chain),
                output_guardrails=_merge_guardrail_chains(output_chain, global_output_chain),
                agent_config=agent_config,
                model_router=model_routing_routers.get(agent.name),
                inline_mini_agents=True,
            )
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
