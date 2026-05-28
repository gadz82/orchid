"""_MiniAgentWiring — builds mini-agent topology (fork router, mini + aggregator nodes).

M2 refactoring: extracted from the 931-LOC ``graph/graph.py`` build_graph
function.  Owns one concern — mini-agent fan-out wiring.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.types import Send

from ..config.schema import OrchidAgentConfig
from .state import GraphState

logger = logging.getLogger(__name__)


class _MiniAgentWiring:
    """Builds mini-agent topology: fork router, mini node, aggregator node."""

    @staticmethod
    def make_fork_router(parent_name: str):
        """Build the conditional-edge function for a mini-agent-enabled parent.

        Reads ``state["mini_agent_decisions"][parent_name]`` and returns
        either ``"supervisor"`` (no fork) or a list of ``Send`` objects.
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

    @staticmethod
    def wire_mini_topology(
        graph: Any,  # StateGraph
        agent_name: str,
        agent_config: OrchidAgentConfig,
        agent_chat_model: Any,
        agent_mcp_clients: list | None,
        node_name: str,
    ) -> None:
        """Wire mini + aggregator nodes onto the graph.

        Called by ``build_graph`` when an agent has ``mini_agent.enabled=true``.
        """
        from ..agents.mini_agent_aggregator import aggregator_node_factory
        from ..agents.mini_agent_node import mini_agent_node_factory

        mini_node = mini_agent_node_factory(
            parent_config=agent_config,
            chat_model=agent_chat_model,
            mcp_clients=agent_mcp_clients or [],
        )
        aggregator_node = aggregator_node_factory(
            parent_config=agent_config,
            chat_model=agent_chat_model,
        )
        graph.add_node(f"{agent_name}_mini", mini_node)
        graph.add_node(f"{agent_name}_aggregator", aggregator_node)
        graph.add_conditional_edges(
            node_name,
            _MiniAgentWiring.make_fork_router(agent_name),
            [f"{agent_name}_mini", "supervisor"],
        )
        graph.add_edge(f"{agent_name}_mini", f"{agent_name}_aggregator")
        graph.add_edge(f"{agent_name}_aggregator", "supervisor")
        logger.info(
            "[Graph] agent '%s' wired with mini-agent topology (max_count=%d, class=%s)",
            agent_name,
            agent_config.mini_agent.max_count,
            type(agent_chat_model).__name__,
        )
