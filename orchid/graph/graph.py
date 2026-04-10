"""
LangGraph graph definition — Composition Root (ADR-008, ADR-016).

All dependency wiring happens here:
  - Agents are instantiated from YAML config (GenericAgent or custom class)
  - MCP clients are created from YAML ``mcp_servers`` definitions
  - The NullVectorReader is used until the RAG pipeline is ready
  - The Supervisor receives agent descriptions for dynamic routing

Graph topology (ADR-013 — parallel vs sequential):

  PARALLEL MODE                         SEQUENTIAL MODE
  ══════════════                        ════════════════

  ┌──────────────┐                      ┌──────────────┐
  │  supervisor   │ ◀───────────┐       │  supervisor   │ ◀─────────────┐
  └──────┬───────┘              │       └──────┬───────┘               │
         │ Send() fan-out       │              │ single dispatch        │
         ├───────────┐          │              ▼                        │
         ▼           ▼          │       ┌─────────────────┐            │
  ┌────────────┐ ┌────────┐    │       │ learning_agent   │            │
  │ learning   │ │ notif  │    │       └────────┬─────────┘            │
  └─────┬──────┘ └───┬────┘    │                │                      │
        │            │         │                ▼                      │
        └────────────┘─────────┘       ┌──────────────┐               │
                                       │  supervisor   │ (advance)    │
                                       └──────┬───────┘               │
                                              │                        │
                                              ▼                        │
                                       ┌──────────────────────┐       │
                                       │ notifications_agent   │       │
                                       └──────────┬───────────┘       │
                                                  │                    │
                                                  └────────────────────┘
                                       supervisor (synthesis) → END

  The Supervisor decides the mode per request based on agent dependencies.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

from langgraph.graph import StateGraph

from ..agents.generic_agent import GenericAgent
from ..config.registry import get_class
from ..config.schema import AgentConfig, AgentsConfig
from ..config.tool_registry import load_tools_from_config
from ..core.agent import BaseAgent
from ..core.llm_provider import LLMProvider
from ..core.repository import VectorReader
from ..runtime import MCPClientFactory, OrchidRuntime
from .state import GraphState
from .supervisor import create_supervisor_node, route_to_agents

logger = logging.getLogger(__name__)


def _create_agent_node(agent: BaseAgent):
    """Wrap a BaseAgent into a LangGraph node function (closure)."""

    async def node(state: GraphState) -> GraphState:
        result = await agent.run(state)
        # Clear active_agents so the supervisor knows this agent is done
        # and can proceed to synthesis or advance the sequential pipeline.
        result["active_agents"] = []
        return result

    node.__name__ = f"{agent.name}_agent"  # helps LangSmith tracing
    return node


def _instantiate_agent(
    name: str,
    agent_config: AgentConfig,
    default_model: str,
    reader: VectorReader,
    llm_service: LLMProvider | None = None,
    mcp_client_factory: MCPClientFactory | None = None,
) -> BaseAgent:
    """
    Create an agent instance from its YAML config.

    If ``class_path`` is set, the referenced class is used.
    Otherwise, ``GenericAgent`` handles the standard flow.
    """
    # Create MCP clients from config via the pluggable factory
    factory = mcp_client_factory or OrchidRuntime().get_mcp_client_factory()
    mcp_clients = [factory(server) for server in agent_config.mcp_servers]

    # Resolve the agent class
    cls = get_class(agent_config.class_path)

    # Determine the LLM model
    model = agent_config.llm.model if agent_config.llm else default_model

    # Check which parameters the class accepts (GenericAgent accepts config + llm_service,
    # custom classes may not)
    sig = inspect.signature(cls.__init__)
    accepts_config = "config" in sig.parameters
    accepts_llm_service = "llm_service" in sig.parameters

    kwargs: dict[str, Any] = {"llm": model, "reader": reader, "mcp_clients": mcp_clients}
    if accepts_config:
        kwargs["config"] = agent_config
    if accepts_llm_service and llm_service:
        kwargs["llm_service"] = llm_service

    return cls(**kwargs)


def _build_subgraph(
    parent_name: str,
    agent_config: AgentConfig,
    default_model: str,
    reader: VectorReader,
    llm_service: LLMProvider | None = None,
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
            child_name, child_config, default_model, reader, llm_service, mcp_client_factory,
        )
        children_agents.append(child_agent)

    # Build sub-graph with its own supervisor
    child_descriptions = {a.name: a.description for a in children_agents}
    sub_supervisor = create_supervisor_node(default_model, child_descriptions, llm_service=llm_service)

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


def build_graph(
    *,
    config: AgentsConfig,
    runtime: OrchidRuntime | None = None,
    default_model: str = "",
    reader: VectorReader | None = None,
) -> Any:  # returns CompiledGraph
    """
    Build and compile the full agent graph from YAML configuration (ADR-016).

    Parameters
    ----------
    config : AgentsConfig
        Parsed and validated YAML configuration.
    runtime : OrchidRuntime | None
        Pre-configured runtime with all dependencies.  When provided,
        ``default_model`` and ``reader`` kwargs are ignored.
    default_model : str
        LiteLLM model identifier.  Ignored when ``runtime`` is provided.
        Deprecated — prefer passing ``OrchidRuntime``.
    reader : VectorReader | None
        Vector store reader.  Ignored when ``runtime`` is provided.
        Deprecated — prefer passing ``OrchidRuntime``.
    """
    # Resolve runtime — build from kwargs for backward compatibility
    if runtime is None:
        runtime = OrchidRuntime(
            default_model=default_model or "ollama/llama3.2",
            reader=reader,
        )

    reader = runtime.get_reader()
    default_model = runtime.default_model
    llm_service: LLMProvider = runtime.get_llm_service()
    mcp_factory: MCPClientFactory = runtime.get_mcp_client_factory()

    # ── Register built-in tools from config ──
    if config.tools:
        load_tools_from_config(config.tools)
        logger.info("[Graph] registered %d built-in tools", len(config.tools))

    # ── Instantiate agents from config ──
    agents: list[BaseAgent] = []
    subgraph_nodes: dict[str, Any] = {}

    # Build agent descriptions directly from config (no proxy needed)
    agent_descriptions: dict[str, str] = {
        name: cfg.description for name, cfg in config.agents.items()
    }

    for agent_name, agent_config in config.agents.items():
        if agent_config.children:
            # Agent with children → build a sub-graph
            subgraph = _build_subgraph(
                agent_name, agent_config, default_model, reader,
                llm_service, mcp_factory,
            )
            subgraph_nodes[agent_name] = subgraph
            # No proxy needed — we already have the description from config
        else:
            agent = _instantiate_agent(
                agent_name, agent_config, default_model, reader,
                llm_service, mcp_factory,
            )
            agents.append(agent)

    # ── Wire agent peers (for cross-agent skill steps) ──
    agent_map: dict[str, BaseAgent] = {a.name: a for a in agents}
    for agent in agents:
        if not isinstance(agent, GenericAgent):
            continue
        # Check if any skill step references another agent
        needs_peers = any(
            step.agent is not None
            for skill in agent._config.skills.values()
            for step in skill.steps
        )
        if needs_peers:
            agent._agent_peers = {
                name: peer for name, peer in agent_map.items()
                if name != agent.name
            }
            # Also update the skill executor's peer reference
            agent._skill_executor._agent_peers = agent._agent_peers
            logger.info(
                "[Graph] agent '%s' wired with peers: %s",
                agent.name,
                list(agent._agent_peers.keys()),
            )

    # ── Supervisor ──
    supervisor_node = create_supervisor_node(
        default_model,
        agent_descriptions,
        llm_service=llm_service,
        orchestrator_skills=config.skills or None,
        supervisor_config=config.supervisor,
    )

    # ── Build graph ──
    g = StateGraph(GraphState)
    g.add_node("supervisor", supervisor_node)

    for agent in agents:
        node_name = f"{agent.name}_agent"
        g.add_node(node_name, _create_agent_node(agent))
        g.add_edge(node_name, "supervisor")

    # Add subgraph nodes (agents with children)
    for agent_name, subgraph in subgraph_nodes.items():
        node_name = f"{agent_name}_agent"
        g.add_node(node_name, subgraph)
        g.add_edge(node_name, "supervisor")

    g.set_entry_point("supervisor")
    g.add_conditional_edges("supervisor", route_to_agents)

    compiled = g.compile()
    logger.info("[Graph] compiled with agents: %s", list(agent_descriptions.keys()))
    return compiled
