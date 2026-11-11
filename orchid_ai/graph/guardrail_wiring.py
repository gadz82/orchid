"""_GuardrailWiring — builds global and per-agent guardrail chains + nodes.

M2 refactoring: extracted from the 931-LOC ``graph/graph.py`` build_graph
function.  Owns one concern — guardrail chain construction and node wiring.
"""

from __future__ import annotations

import logging
from typing import Any

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

    #: Registered guardrail type that consumes the ``guardrails`` decision point.
    _DECISION_RAIL_TYPE = "decision"
    _DECISION_POINT = "guardrails"

    @staticmethod
    def build_chains(
        guardrails_config: OrchidGuardrailsConfig,
        *,
        config: Any = None,
        runtime: Any = None,
        agent_name: str | None = None,
    ) -> tuple[OrchidGuardrailChain, OrchidGuardrailChain]:
        """Build input and output guardrail chains from YAML config.

        Decision-backed rails need a resolved ``guardrails`` binding. The point
        is ``agent``-scoped, so resolution depends on where the chain runs:

        * ``agent_name`` is set (per-agent rails): the agent override is used
          when declared, otherwise the fleet-level binding is inherited.
        * ``agent_name`` is ``None`` (global rails): the fleet-level binding is
          used. Global rails run before an agent is known, so they cannot use a
          per-agent override.

        When no binding is configured for the scope, the decision rail keeps
        ``binding=None`` and applies its own ``on_failure`` policy on every
        check. Callers that do not pass ``config``/``runtime`` (for example
        deterministic-only tests) can never resolve a binding; the rail then
        behaves the same way.
        """
        from ..guardrails.registry import build_guardrail_chain

        binding, configuration_reason = _GuardrailWiring._resolve_decision_binding(
            guardrails_config,
            config=config,
            runtime=runtime,
            agent_name=agent_name,
        )
        input_configs = _GuardrailWiring._rule_configs(
            guardrails_config.input,
            binding,
            configuration_reason,
        )
        output_configs = _GuardrailWiring._rule_configs(
            guardrails_config.output,
            binding,
            configuration_reason,
        )

        return (
            build_guardrail_chain(input_configs),
            build_guardrail_chain(output_configs),
        )

    @staticmethod
    def _resolve_decision_binding(
        guardrails_config: OrchidGuardrailsConfig,
        *,
        config: Any,
        runtime: Any,
        agent_name: str | None,
    ) -> tuple[Any, str]:
        """Resolve the ``guardrails`` point once per chain build, when needed.

        Returns ``(binding, reason)``. When no binding is available, ``reason``
        distinguishes an unconfigured scope (``not_configured``), an explicitly
        disabled binding (``disabled``), and a configured provider missing from
        the runtime (``provider_missing``) for the decision log.
        """
        rules = [*guardrails_config.input, *guardrails_config.output]
        if not any(rule.type == _GuardrailWiring._DECISION_RAIL_TYPE for rule in rules):
            return None, "not_configured"

        if config is None or runtime is None:
            logger.warning(
                "[Guardrails] decision rail at %s scope has no config/runtime; "
                "it will apply its on_failure policy on every check",
                f"agent '{agent_name}'" if agent_name else "global",
            )
            return None, "not_configured"

        from .decision_bindings import resolve_decision_binding

        binding_config = config.resolve_decision(_GuardrailWiring._DECISION_POINT, agent_name)
        if binding_config is None:
            reason = "disabled" if _GuardrailWiring._binding_is_disabled(config, agent_name) else "not_configured"
            logger.warning(
                "[Guardrails] decision rail at %s scope has no usable %r binding (%s); "
                "it will apply its on_failure policy on every check",
                f"agent '{agent_name}'" if agent_name else "global",
                _GuardrailWiring._DECISION_POINT,
                reason,
            )
            return None, reason

        # Resolve the provider here so a missing provider produces exactly one
        # build warning; ``resolve_decision_binding`` would log its own copy.
        if not binding_config.model or runtime.get_decision_model(binding_config.model) is None:
            logger.warning(
                "[Guardrails] decision rail at %s scope has no usable %r binding (provider_missing); "
                "it will apply its on_failure policy on every check",
                f"agent '{agent_name}'" if agent_name else "global",
                _GuardrailWiring._DECISION_POINT,
            )
            return None, "provider_missing"

        binding = resolve_decision_binding(
            config,
            runtime,
            _GuardrailWiring._DECISION_POINT,
            agent_name=agent_name,
        )
        if binding is not None:
            return binding, ""

        logger.warning(
            "[Guardrails] decision rail at %s scope has no usable %r binding (provider_missing); "
            "it will apply its on_failure policy on every check",
            f"agent '{agent_name}'" if agent_name else "global",
            _GuardrailWiring._DECISION_POINT,
        )
        return None, "provider_missing"

    @staticmethod
    def _binding_is_disabled(config: Any, agent_name: str | None) -> bool:
        """Whether the scope's ``guardrails`` binding was explicitly disabled."""
        from ..config.schema_decision import iter_agents

        agent_binding = None
        if agent_name is not None:
            for path, agent in iter_agents(config.agents):
                if path == agent_name or path.rsplit(".", 1)[-1] == agent_name:
                    agent_binding = agent.decisions.points.get(_GuardrailWiring._DECISION_POINT)
                    break
        if agent_binding is not None and agent_binding.enabled is False:
            return True
        if agent_binding is not None:
            return False
        global_binding = config.decisions.points.get(_GuardrailWiring._DECISION_POINT)
        return global_binding is not None and global_binding.enabled is False

    @staticmethod
    def _rule_configs(rules: list[Any], binding: Any, configuration_reason: str) -> list[dict[str, Any]]:
        """Serialise rules for the registry, leaving deterministic rails untouched.

        Existing rails keep the historical flattened shape so their behaviour
        does not change. A decision rail receives a nested ``config`` block with
        the resolved binding, the point name, its failure policy, and the
        resolution reason for the decision log.

        Known pre-existing limitation (tracked for P-010): the flattened shape
        means ``config:`` values declared on deterministic rails are not passed
        to their constructors by the graph wiring. This prompt deliberately
        preserves that behaviour; fixing it would change existing rails.
        """
        configs: list[dict[str, Any]] = []
        for rule in rules:
            if rule.type == _GuardrailWiring._DECISION_RAIL_TYPE:
                configs.append(
                    {
                        "type": rule.type,
                        "fail_action": rule.fail_action,
                        "on_failure": rule.on_failure,
                        "config": {
                            **rule.config,
                            "binding": binding,
                            "point": _GuardrailWiring._DECISION_POINT,
                            "configuration_reason": configuration_reason,
                        },
                    }
                )
                continue
            configs.append({"type": rule.type, "fail_action": rule.fail_action, **rule.config})
        return configs

    @staticmethod
    def create_global_input_node(chain: OrchidGuardrailChain, *, agent_name: str = ""):
        """Create a LangGraph node that runs global input guardrails.

        ``agent_name`` is set when the chain wraps a parent agent's subgraph,
        so the guardrail context and log label stay agent-scoped. The user turn
        is found by scanning backwards: the supervisor appends a dispatch
        ``AIMessage`` before agent nodes run.
        """

        async def input_guardrails_node(state: GraphState, config: RunnableConfig = None) -> GraphState:
            from ..core.agent import OrchidAgent

            query = OrchidAgent.extract_user_query(state)
            if not query:
                return state

            auth = auth_from_config(config)
            ctx = OrchidGuardrailContext(
                direction=OrchidGuardrailDirection.INPUT,
                agent_name=agent_name,
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

            if (
                result.action == OrchidGuardrailAction.REDACT
                and result.redacted_content is not None
                and result.redacted_content != query
            ):
                messages = list(state.get("messages", []))
                index = next(
                    (i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)),
                    None,
                )
                if index is not None:
                    replacement = messages[index].model_copy(update={"content": result.redacted_content})
                    logger.info("[Guardrails] Global input redacted by '%s'", result.guardrail_name)
                    return {"messages": [replacement]}

            return state

        return input_guardrails_node

    @staticmethod
    def create_global_output_node(chain: OrchidGuardrailChain, *, agent_name: str = ""):
        """Create a LangGraph node that runs global output guardrails."""

        async def output_guardrails_node(state: GraphState, config: RunnableConfig = None) -> GraphState:
            final = state.get("final_response")
            if not final:
                return state

            auth = auth_from_config(config)
            ctx = OrchidGuardrailContext(
                direction=OrchidGuardrailDirection.OUTPUT,
                agent_name=agent_name,
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
