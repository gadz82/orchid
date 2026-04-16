"""
LangGraph graph definition — Composition Root (ADR-008, ADR-016, ADR-018).

All dependency wiring happens here:
  - Agents are instantiated from YAML config (GenericAgent or custom class)
  - MCP clients are created from YAML ``mcp_servers`` definitions
  - The NullVectorReader is used until the RAG pipeline is ready
  - The Supervisor receives agent descriptions for dynamic routing
  - Guardrails are wired as graph nodes (global) and agent wrappers (per-agent)

Graph topology (ADR-013 — parallel vs sequential, ADR-018 — guardrails):

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

import inspect
import logging
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph

from ..agents.generic_agent import GenericAgent
from ..config.schema import AgentConfig, AgentsConfig, GuardrailsConfig
from ..config.registry import get_class
from ..config.tool_registry import load_tools_from_config
from ..core.agent import BaseAgent
from ..core.guardrails import GuardrailAction, GuardrailChain, GuardrailContext, GuardrailDirection
from ..core.repository import VectorReader

from langchain_core.language_models import BaseChatModel
from ..mcp.auth_registry import MCPAuthRegistry
from ..runtime import MCPClientFactory, OrchidRuntime
from .state import GraphState
from .supervisor import create_supervisor_node, route_to_agents

logger = logging.getLogger(__name__)


def _create_agent_node(
    agent: BaseAgent,
    input_guardrails: GuardrailChain | None = None,
    output_guardrails: GuardrailChain | None = None,
):
    """
    Wrap a BaseAgent into a LangGraph node function (closure).

    When per-agent guardrails are configured, input is checked before
    ``agent.run()`` and output is checked after.
    """

    async def node(state: GraphState) -> GraphState:
        auth = state.get("auth_context")

        # ── Per-agent INPUT guardrails ──
        if input_guardrails and not input_guardrails.empty:
            query = agent.extract_user_query(state)
            ctx = GuardrailContext(
                direction=GuardrailDirection.INPUT,
                agent_name=agent.name,
                tenant_key=auth.tenant_key if auth else "default",
                user_id=auth.user_id if auth else "",
                chat_id=state.get("chat_id", ""),
            )
            result = await input_guardrails.evaluate(query, ctx)
            if result.blocked:
                logger.warning(
                    "[Guardrails] Agent '%s' input blocked by '%s': %s",
                    agent.name,
                    result.guardrail_name,
                    result.message,
                )
                return {
                    "messages": [AIMessage(content=f"[{agent.name.title()} Agent] {result.message}")],
                    "active_agents": [],
                }

        # ── Run agent ──
        try:
            agent_result = await agent.run(state)
        except Exception as exc:
            logger.error(
                "[Graph] Agent '%s' raised an unhandled exception: %s",
                agent.name,
                exc,
                exc_info=True,
            )
            agent_result = {
                "messages": [
                    AIMessage(
                        content=(
                            f"[{agent.name.title()} Agent] I'm temporarily unable to process "
                            "your request. Please try again in a few moments."
                        )
                    )
                ],
            }

        # ── Per-agent OUTPUT guardrails ──
        if output_guardrails and not output_guardrails.empty:
            # Extract agent's response text from messages
            agent_messages = agent_result.get("messages", [])
            if agent_messages:
                response_text = str(agent_messages[-1].content) if hasattr(agent_messages[-1], "content") else ""
                ctx = GuardrailContext(
                    direction=GuardrailDirection.OUTPUT,
                    agent_name=agent.name,
                    tenant_key=auth.tenant_key if auth else "default",
                    user_id=auth.user_id if auth else "",
                    chat_id=state.get("chat_id", ""),
                    metadata={"rag_context": agent_result.get("rag_context", {}).get(agent.name, [])},
                )
                result = await output_guardrails.evaluate(response_text, ctx)
                if result.blocked:
                    logger.warning(
                        "[Guardrails] Agent '%s' output blocked by '%s': %s",
                        agent.name,
                        result.guardrail_name,
                        result.message,
                    )
                    agent_result["messages"] = [AIMessage(content=f"[{agent.name.title()} Agent] {result.message}")]
                elif result.action == GuardrailAction.REDACT and result.redacted_content is not None:
                    logger.info("[Guardrails] Agent '%s' output redacted by '%s'", agent.name, result.guardrail_name)
                    agent_result["messages"] = [AIMessage(content=result.redacted_content)]

        # Clear active_agents so the supervisor knows this agent is done
        # and can proceed to synthesis or advance the sequential pipeline.
        agent_result["active_agents"] = []
        return agent_result

    node.__name__ = f"{agent.name}_agent"  # helps LangSmith tracing
    return node


def _instantiate_agent(
    name: str,
    agent_config: AgentConfig,
    default_model: str,
    reader: VectorReader,
    default_chat_model: BaseChatModel | None = None,
    default_fallback: str | None = None,
    default_retry: int = 0,
    mcp_client_factory: MCPClientFactory | None = None,
    summary_config: dict[str, Any] | None = None,
) -> BaseAgent:
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

    # Check which parameters the class accepts (GenericAgent accepts config + chat_model,
    # custom classes may not)
    sig = inspect.signature(cls.__init__)
    accepts_config = "config" in sig.parameters
    accepts_chat_model = "chat_model" in sig.parameters
    accepts_summary_config = "summary_config" in sig.parameters

    kwargs: dict[str, Any] = {"llm": agent_model, "reader": reader, "mcp_clients": mcp_clients}
    if accepts_config:
        kwargs["config"] = agent_config
    if accepts_chat_model and agent_chat_model:
        kwargs["chat_model"] = agent_chat_model
    if accepts_summary_config and summary_config:
        kwargs["summary_config"] = summary_config

    return cls(**kwargs)


def _build_subgraph(
    parent_name: str,
    agent_config: AgentConfig,
    default_model: str,
    reader: VectorReader,
    default_chat_model: BaseChatModel | None = None,
    default_fallback: str | None = None,
    default_retry: int = 0,
    mcp_client_factory: MCPClientFactory | None = None,
) -> Any:
    """
    Build a sub-graph for an agent with children.

    Returns a compiled LangGraph sub-graph that the parent graph
    treats as a single node.
    """
    children_agents: list[BaseAgent] = []
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
    guardrails_config: GuardrailsConfig,
) -> tuple[GuardrailChain, GuardrailChain]:
    """Build input and output guardrail chains from YAML config."""
    from ..guardrails.registry import build_guardrail_chain

    input_configs = [{"type": r.type, "fail_action": r.fail_action, **r.config} for r in guardrails_config.input]
    output_configs = [{"type": r.type, "fail_action": r.fail_action, **r.config} for r in guardrails_config.output]

    return (
        build_guardrail_chain(input_configs),
        build_guardrail_chain(output_configs),
    )


def _create_global_input_guardrail_node(chain: GuardrailChain):
    """Create a LangGraph node that runs global input guardrails."""

    async def input_guardrails_node(state: GraphState) -> GraphState:
        """Global input guardrails — runs before the supervisor."""
        from ..core.agent import BaseAgent

        query = BaseAgent.extract_user_query(state)
        if not query:
            return state

        auth = state.get("auth_context")
        ctx = GuardrailContext(
            direction=GuardrailDirection.INPUT,
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

        if result.action == GuardrailAction.REDACT and result.redacted_content is not None:
            logger.info("[Guardrails] Global input redacted by '%s'", result.guardrail_name)
            # Replace the last human message with redacted version
            from langchain_core.messages import HumanMessage

            messages = list(state.get("messages", []))
            if messages and isinstance(messages[-1], HumanMessage):
                messages[-1] = HumanMessage(content=result.redacted_content)
            return {"messages": messages}

        return state

    return input_guardrails_node


def _create_global_output_guardrail_node(chain: GuardrailChain):
    """Create a LangGraph node that runs global output guardrails."""

    async def output_guardrails_node(state: GraphState) -> GraphState:
        """Global output guardrails — runs after synthesis, before END."""
        final = state.get("final_response")
        if not final:
            return state

        auth = state.get("auth_context")
        ctx = GuardrailContext(
            direction=GuardrailDirection.OUTPUT,
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

        if result.action == GuardrailAction.REDACT and result.redacted_content is not None:
            logger.info("[Guardrails] Global output redacted by '%s'", result.guardrail_name)
            return {
                "messages": [AIMessage(content=result.redacted_content)],
                "final_response": result.redacted_content,
            }

        return state

    return output_guardrails_node


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
    config: AgentsConfig,
    runtime: OrchidRuntime,
) -> Any:  # returns CompiledGraph
    """
    Build and compile the full agent graph from YAML configuration (ADR-016, ADR-018).

    Parameters
    ----------
    config : AgentsConfig
        Parsed and validated YAML configuration.
    runtime : OrchidRuntime
        Pre-configured runtime with all dependencies (reader, LLM provider,
        MCP client factory).
    """
    from ..llm_factory import build_chat_model as _build_chat_model

    # ── Enable LLM response caching if configured ──
    if config.defaults.cache_enabled:
        from langchain_core.caches import InMemoryCache
        from langchain_core.globals import set_llm_cache

        set_llm_cache(InMemoryCache())
        logger.info("[Graph] LLM response caching enabled (InMemoryCache)")

    reader = runtime.get_reader()
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
    auth_registry = MCPAuthRegistry.from_config(config)
    runtime.mcp_auth_registry = auth_registry

    # ── Build MCP factory (enhanced with token_store for OAuth servers) ──
    if runtime.mcp_client_factory:
        mcp_factory: MCPClientFactory = runtime.mcp_client_factory
    else:
        from ..runtime import _default_mcp_client_factory

        token_store = runtime.mcp_token_store

        def mcp_factory(cfg):  # type: ignore[misc]
            return _default_mcp_client_factory(cfg, token_store=token_store)

    # ── Register built-in tools from config ──
    if config.tools:
        load_tools_from_config(config.tools)
        logger.info("[Graph] registered %d built-in tools", len(config.tools))

    # ── Build global guardrail chains (ADR-018) ──
    global_input_chain, global_output_chain = _build_guardrail_chains(config.guardrails)
    has_global_input_rails = not global_input_chain.empty
    has_global_output_rails = not global_output_chain.empty

    if has_global_input_rails:
        logger.info("[Graph] global input guardrails: %s", global_input_chain)
    if has_global_output_rails:
        logger.info("[Graph] global output guardrails: %s", global_output_chain)

    # ── Instantiate agents from config ──
    agents: list[BaseAgent] = []
    agent_guardrails: dict[str, tuple[GuardrailChain, GuardrailChain]] = {}
    subgraph_nodes: dict[str, Any] = {}

    # Build agent descriptions directly from config (no proxy needed)
    agent_descriptions: dict[str, str] = {name: cfg.description for name, cfg in config.agents.items()}

    # Build summary config dict from SupervisorConfig (passed to GenericAgent)
    sup = config.supervisor
    summary_cfg: dict[str, Any] | None = None
    if sup.history_summary_enabled:
        summary_cfg = {
            "model": sup.history_summary_model or default_model,
            "recent_turns": sup.history_summary_recent_turns,
        }

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
    agent_map: dict[str, BaseAgent] = {a.name: a for a in agents}
    for agent in agents:
        if not isinstance(agent, GenericAgent):
            continue
        # Check if any skill step references another agent
        needs_peers = any(step.agent is not None for skill in agent._config.skills.values() for step in skill.steps)
        if needs_peers:
            agent._agent_peers = {name: peer for name, peer in agent_map.items() if name != agent.name}
            # Also update the skill executor's peer reference
            agent._skill_executor._agent_peers = agent._agent_peers
            logger.info(
                "[Graph] agent '%s' wired with peers: %s",
                agent.name,
                list(agent._agent_peers.keys()),
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

    # ── Supervisor ──
    supervisor_node = create_supervisor_node(
        default_model,
        agent_descriptions,
        chat_model=supervisor_chat_model,
        orchestrator_skills=config.skills or None,
        supervisor_config=config.supervisor,
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
        g.add_node(node_name, _create_agent_node(agent, input_chain, output_chain))
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
