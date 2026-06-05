"""_GuardrailWiring — builds global and per-agent guardrail chains + nodes.

M2 refactoring: extracted from the 931-LOC ``graph/graph.py`` build_graph
function.  Owns one concern — guardrail chain construction and node wiring.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from ..config.schema import OrchidGuardrailsConfig
from ..core.guardrails import (
    OrchidGuardrailAction,
    OrchidGuardrailChain,
    OrchidGuardrailContext,
    OrchidGuardrailDirection,
)
from ..core.run_config import auth_from_config
from .state import GraphState

logger = logging.getLogger(__name__)


class _GuardrailWiring:
    """Builds guardrail chains and LangGraph nodes from YAML config."""

    @staticmethod
    def build_chains(
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

    @staticmethod
    def create_global_input_node(chain: OrchidGuardrailChain):
        """Create a LangGraph node that runs global input guardrails."""

        async def input_guardrails_node(state: GraphState, config: RunnableConfig | None = None) -> GraphState:
            from ..core.agent import OrchidAgent

            query = OrchidAgent.extract_user_query(state)
            if not query:
                return state

            auth = auth_from_config(config)
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
                messages = list(state.get("messages", []))
                if messages and isinstance(messages[-1], HumanMessage):
                    messages[-1] = HumanMessage(content=result.redacted_content)
                return {"messages": messages}

            return state

        return input_guardrails_node

    @staticmethod
    def create_global_output_node(chain: OrchidGuardrailChain):
        """Create a LangGraph node that runs global output guardrails."""

        async def output_guardrails_node(state: GraphState, config: RunnableConfig | None = None) -> GraphState:
            final = state.get("final_response")
            if not final:
                return state

            auth = auth_from_config(config)
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
