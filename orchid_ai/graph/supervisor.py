"""
Supervisor node — intent analysis + routing.

The Supervisor is the ONLY entry point of the graph.
It does NOT have access to MCP servers or vector stores.
Its job:
  1. Analyse user intent via LLM
  2. Choose execution mode: **parallel** or **sequential**
  3. Route to one or more specialised sub-agents
  4. Delegate sequential advancement to :class:`SequentialAdvancer`
  5. Delegate final synthesis to :class:`ResponseSynthesizer`

Execution modes:
  - parallel   → all agents run simultaneously via Send() fan-out
  - sequential → agents run one at a time; each round's output is
                 visible to the next agent via mcp_context

Prompts are configurable via ``OrchidSupervisorConfig`` in agents.yaml.
"""

from __future__ import annotations

import logging
import time
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END
from langgraph.types import Send
from pydantic import BaseModel, Field

from ..config.schema import OrchidOrchestratorSkillConfig, OrchidSupervisorConfig
from ..core.agent import OrchidAgent
from .sequential_advancer import SequentialAdvancer
from .state import GraphState
from .synthesizer import ResponseSynthesizer

__all__ = [
    "OrchidRoutingDecision",
    "ROUTING_SYSTEM_PROMPT",
    "create_supervisor_node",
    "route_to_agents",
]


# ── Structured output model for routing ──────────────────────


class OrchidRoutingDecision(BaseModel):
    """LLM-generated routing decision — guaranteed valid via structured output."""

    reasoning: str = Field(description="Brief analysis of the user's intent")
    execution: Literal["parallel", "sequential", "skill"] = Field(
        default="parallel",
        description="Execution mode: parallel (independent agents), sequential (dependent), or skill (pre-defined workflow)",
    )
    agents: list[str] = Field(
        default_factory=list,
        description="Agent names to activate (empty if direct_response or skill)",
    )
    skill: str | None = Field(
        default=None,
        description="Skill name to invoke (only when execution='skill')",
    )
    direct_response: str | None = Field(
        default=None,
        description="Direct response to the user (only when no agent is needed)",
    )


logger = logging.getLogger(__name__)
perf_logger = logging.getLogger("orchid.perf")

# ── Prompt template for routing ──────────────────────────────

ROUTING_SYSTEM_PROMPT = """\
You are the Supervisor of the {assistant_name}.
Your role is to analyse the user's request and decide which specialised
sub-agents to activate.  You do NOT have access to any external tool or API.

Available agents:
{agent_descriptions}

EXECUTION MODES:
- "parallel"   — agents run simultaneously.  Use when they are INDEPENDENT
                  (e.g. looking up data AND listing available options).
- "sequential" — agents run one after another, in the order you specify.
                  Use when one agent's output is NEEDED by the next
                  (e.g. first find data, THEN act on it).

AVAILABLE SKILLS (pre-defined multi-agent workflows):
{skill_descriptions}

RULES:
- Route to one or more agents when the request requires domain-specific data or actions.
- Choose the right execution mode based on agent dependencies.
- If a pre-defined SKILL matches the user's request, prefer it over manual routing.
- If you can answer directly (greeting, general question), set direct_response.
- For follow-up messages like "yes", "tell me more", "go ahead", ALWAYS re-route
  to the SAME agent(s) that handled the previous turn.  Never return empty agents
  for a follow-up question.

FIELD INSTRUCTIONS (you MUST follow these):
- "reasoning": Explain WHY you chose these agents (1-2 sentences).
- "execution": One of "parallel", "sequential", or "skill".
- "agents": List of agent NAMES to activate. MUST NOT be empty unless you set
  direct_response. Example: ["menu"] or ["menu", "orders"].
- "skill": Skill name ONLY when execution="skill". Otherwise null.
- "direct_response": Your answer ONLY when no agent is needed (greetings,
  general knowledge). Otherwise null.
"""


# ── Factory ──────────────────────────────────────────────────


def create_supervisor_node(
    model: str,
    agent_descriptions: dict[str, str],
    chat_model: BaseChatModel | None = None,
    orchestrator_skills: dict[str, OrchidOrchestratorSkillConfig] | None = None,
    supervisor_config: OrchidSupervisorConfig | None = None,
    routing_chat_model: BaseChatModel | None = None,
):
    """
    Return a LangGraph node function with *model*, *agent_descriptions*,
    *chat_model*, and *orchestrator_skills* captured via closure — no
    module-level globals (Composition Root).

    When ``routing_chat_model`` is provided, the supervisor uses it for
    the cheap routing + sequential-advance phases (short structured
    classifications and one-line handoff messages) instead of the
    main ``chat_model`` reserved for synthesis.  Falls back to
    ``chat_model`` when ``None``.
    """
    skills = orchestrator_skills or {}
    sup_config = supervisor_config or OrchidSupervisorConfig()
    # Route + advance phases use the cheaper model when configured;
    # synthesis always uses the main chat_model.
    route_chat_model = routing_chat_model or chat_model

    advancer = SequentialAdvancer(
        model=model,
        agent_descriptions=agent_descriptions,
        supervisor_config=sup_config,
        chat_model=route_chat_model,
    )
    synthesizer = ResponseSynthesizer(
        model=model,
        supervisor_config=sup_config,
        chat_model=chat_model,
    )

    async def supervisor_node(state: GraphState) -> GraphState:
        pending = state.get("pending_agents", [])
        has_mcp_data = bool(state.get("mcp_context"))
        start = time.perf_counter()

        # ── Case 1: Sequential pipeline in progress — advance to next agent ──
        if pending and has_mcp_data:
            perf_logger.info(
                "[PERF][supervisor] phase=advance_sequential next=%s pending=%s",
                pending[0],
                pending[1:],
            )
            result = await advancer.advance(state, pending)
            elapsed = (time.perf_counter() - start) * 1000
            perf_logger.info("[PERF][supervisor] phase=advance_sequential took %.1f ms", elapsed)
            return result

        # ── Case 2: All agents done (no pending) + data collected → synthesise ──
        if has_mcp_data and not pending and not state.get("active_agents"):
            perf_logger.info(
                "[PERF][supervisor] phase=synthesise (mcp_keys=%s)",
                list(state.get("mcp_context", {}).keys()),
            )
            result = await synthesizer.synthesise(state)
            elapsed = (time.perf_counter() - start) * 1000
            perf_logger.info("[PERF][supervisor] phase=synthesise took %.1f ms", elapsed)
            return result

        # ── Case 3: First entry — analyse intent and route ──
        perf_logger.info("[PERF][supervisor] phase=route (initial intent analysis)")
        result = await _route(state, model, agent_descriptions, skills, sup_config, route_chat_model)
        elapsed = (time.perf_counter() - start) * 1000
        perf_logger.info(
            "[PERF][supervisor] phase=route took %.1f ms (active=%s pending=%s)",
            elapsed,
            result.get("active_agents", []),
            result.get("pending_agents", []),
        )
        return result

    return supervisor_node


# ── Routing function (conditional edges) ─────────────────────


def route_to_agents(state: GraphState) -> list[Send] | str:
    """
    LangGraph conditional-edge function.

    Returns:
      - ``"output_guardrails"`` when ``final_response`` is set AND output
        guardrails are configured (the graph wires this node).
      - ``END`` when ``final_response`` is set and no output guardrails exist.
      - ``"supervisor"`` when there are pending agents to advance.
      - A list of ``Send`` objects for parallel fan-out.
      - A single ``Send`` for sequential execution.
    """
    if state.get("final_response"):
        # If the graph has an output_guardrails node, route there.
        # We use a sentinel in state to detect this (set by build_graph).
        if state.get("_has_output_guardrails"):
            return "output_guardrails"
        return END

    active = state.get("active_agents", [])
    if not active:
        # No active agents but pending → re-enter supervisor to advance
        if state.get("pending_agents"):
            return "supervisor"
        if state.get("_has_output_guardrails"):
            return "output_guardrails"
        return END

    mode = state.get("execution_mode", "parallel")

    if mode == "parallel":
        # Fan-out: all active agents run simultaneously
        logger.info("[Route] parallel dispatch → %s", active)
        return [Send(f"{agent}_agent", state) for agent in active]
    else:
        # Sequential: run only the first active agent
        logger.info("[Route] sequential dispatch → %s", active[0])
        return f"{active[0]}_agent"


# ── Routing phase ────────────────────────────────────────────


# ── Routing helpers ────────────────────────────────────────────


def _inject_auth_hints(state: GraphState) -> str:
    """Build an auth-status hint string for the routing system prompt.

    When MCP servers require OAuth tokens that are not yet available,
    the supervisor should know which agents are degraded so it can
    route around them.
    """
    mcp_auth_status = state.get("mcp_auth_status", {})
    unauthorized = [name for name, ok in mcp_auth_status.items() if not ok]
    if not unauthorized:
        return ""
    return (
        f"\n\nNOTE: The following external services require user authorization "
        f"and are currently unavailable: {', '.join(unauthorized)}. "
        f"Agents that depend solely on these services may have limited capabilities."
    )


async def _extract_and_compress_history(
    state: GraphState,
    sup: OrchidSupervisorConfig,
    chat_model: BaseChatModel | None = None,
) -> list[dict[str, str]]:
    """Extract conversation history, optionally compressing older turns."""
    history = OrchidAgent.extract_conversation_history(
        state,
        max_turns=sup.history_max_turns,
        max_chars=sup.history_max_chars,
    )
    if history and sup.history_summary_enabled and chat_model:
        history = await OrchidAgent.compress_conversation_history(
            history,
            chat_model=chat_model,
            recent_turns=sup.history_summary_recent_turns,
        )
    # Filter conversation summaries — internal compression artifacts
    return [m for m in history if not m.get("content", "").startswith("[Conversation summary]")] if history else []


def _validate_skill_activation(
    skill_name: str,
    skills: dict[str, OrchidOrchestratorSkillConfig],
    agent_descriptions: dict[str, str],
) -> GraphState | None:
    """Validate and expand an orchestrator skill into a state update.

    Returns a state update (dict) on success, or ``None`` when the
    skill name is unknown (caller should fall back to agent routing).
    """
    if skill_name not in skills:
        logger.warning("[Supervisor] Unknown skill '%s', falling back to agent routing", skill_name)
        return None

    skill = skills[skill_name]
    skill_agents = [step.agent for step in skill.steps]
    skill_instructions_map = {step.agent: step.instruction for step in skill.steps if step.instruction}

    valid_skill_agents = [a for a in skill_agents if a in agent_descriptions]
    if not valid_skill_agents:
        fallback = f"Skill '{skill_name}' references unknown agents."
        return {
            "messages": [AIMessage(content=fallback)],
            "final_response": fallback,
            "active_agents": [],
            "pending_agents": [],
        }

    first, *rest = valid_skill_agents
    logger.info(
        "[Supervisor] orchestrator skill '%s': %s",
        skill_name,
        " → ".join(valid_skill_agents),
    )
    return {
        "active_agents": [first],
        "pending_agents": rest,
        "execution_mode": "sequential",
        "skill_instructions": skill_instructions_map,
        "messages": [AIMessage(content=(f"[Supervisor] Skill '{skill_name}': {' → '.join(valid_skill_agents)}"))],
    }


def _recover_agent_names(
    reasoning: str,
    agent_descriptions: dict[str, str],
) -> list[str]:
    """Recover agent names from the LLM's reasoning text.

    Small models sometimes return an empty agents list but mention
    agent names in their reasoning field.  This extracts them as a
    best-effort fallback.
    """
    reasoning_lower = reasoning.lower()
    recovered: list[str] = []
    for name in agent_descriptions:
        if name in reasoning_lower:
            recovered.append(name)
    if recovered:
        logger.warning(
            "[Supervisor] Recovered agent names from reasoning: %s (original agents list was empty)",
            recovered,
        )
    return recovered


async def _route(
    state: GraphState,
    model: str,
    agent_descriptions: dict[str, str],
    orchestrator_skills: dict[str, OrchidOrchestratorSkillConfig] | None = None,
    supervisor_config: OrchidSupervisorConfig | None = None,
    chat_model: BaseChatModel | None = None,
) -> GraphState:
    """Analyse user intent, choose execution mode, and activate agents."""
    desc_text = "\n".join(f"- **{name}**: {desc}" for name, desc in agent_descriptions.items())

    skills = orchestrator_skills or {}
    if skills:
        skill_text = "\n".join(f'- "{name}": {skill.description}' for name, skill in skills.items())
    else:
        skill_text = "(none defined)"

    sup = supervisor_config or OrchidSupervisorConfig()
    routing_template = sup.routing_system_prompt or ROUTING_SYSTEM_PROMPT

    auth_hint = _inject_auth_hints(state)

    system = routing_template.format(
        assistant_name=sup.assistant_name,
        agent_descriptions=desc_text + auth_hint,
        skill_descriptions=skill_text,
    )

    clean_history = await _extract_and_compress_history(state, sup, chat_model)

    llm_messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    if clean_history:
        llm_messages.extend(clean_history)

    # Add the current user query
    user_query = OrchidAgent.extract_user_query(state)
    if user_query:
        llm_messages.append({"role": "user", "content": user_query})

    try:
        if not chat_model:
            raise RuntimeError("Supervisor requires a BaseChatModel. Pass chat_model= when building the graph.")

        structured_model = chat_model.with_structured_output(OrchidRoutingDecision)
        llm_start = time.perf_counter()
        decision: OrchidRoutingDecision = await structured_model.ainvoke(llm_messages, temperature=0)
        llm_elapsed = (time.perf_counter() - llm_start) * 1000
        perf_logger.info("[PERF][supervisor.route] structured LLM call took %.1f ms", llm_elapsed)
        logger.info("[Supervisor] routing decision: %s", decision.model_dump_json())
    except Exception as exc:
        # Handle LLM API failures gracefully
        logger.error("[Supervisor] LLM API error during routing: %s", exc, exc_info=True)
        error_msg = str(exc)
        if "503" in error_msg or "high demand" in error_msg.lower():
            response_text = (
                "I'm currently experiencing high demand and cannot process your request. "
                "Please try again in a few moments."
            )
        elif "rate limit" in error_msg.lower():
            response_text = "I've hit my rate limit. Please try again in a few moments."
        else:
            response_text = f"I encountered an error: {error_msg[:200]}. Please try again later."

        # Return a direct response to user
        return {
            "messages": [AIMessage(content=response_text)],
            "next": "END",
            "routing_metadata": {"error": "llm_api_failure", "details": error_msg},
        }

    agents: list[str] = decision.agents
    direct: str | None = decision.direct_response
    execution: str = decision.execution

    # ── Orchestrator skill activation ──
    if execution == "skill":
        skill_result = _validate_skill_activation(
            decision.skill or "",
            orchestrator_skills or {},
            agent_descriptions,
        )
        if skill_result is not None:
            return skill_result

    # ── Direct response (no sub-agent needed) ──
    if direct and not agents:
        return {
            "messages": [AIMessage(content=direct)],
            "final_response": direct,
            "active_agents": [],
            "pending_agents": [],
        }

    # ── Validate agent names ──
    valid = [a for a in agents if a in agent_descriptions]

    # Recovery: if the LLM returned empty agents but mentioned an agent name
    # in the reasoning (common with small models), extract it.
    if not valid and not direct:
        valid = _recover_agent_names(decision.reasoning, agent_descriptions)

    if not valid:
        fallback = "I'm not sure how to help with that request. Could you rephrase or provide more details?"
        return {
            "messages": [AIMessage(content=fallback)],
            "final_response": fallback,
            "active_agents": [],
            "pending_agents": [],
        }

    # ── Dispatch based on execution mode ──
    if execution == "sequential" and len(valid) > 1:
        # Sequential: activate only the FIRST, queue the rest
        first, *rest = valid
        logger.info(
            "[Supervisor] sequential pipeline: %s → then %s",
            first,
            rest,
        )
        return {
            "active_agents": [first],
            "pending_agents": rest,
            "execution_mode": "sequential",
            "messages": [AIMessage(content=(f"[Supervisor] Sequential pipeline: {' → '.join(valid)}"))],
        }
    else:
        # Parallel (default): activate all at once
        logger.info("[Supervisor] parallel dispatch: %s", valid)
        return {
            "active_agents": valid,
            "pending_agents": [],
            "execution_mode": "parallel",
            "messages": [AIMessage(content=f"[Supervisor] Parallel dispatch: {', '.join(valid)}")],
        }
