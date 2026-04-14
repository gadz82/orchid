"""
Supervisor node — intent analysis + routing + synthesis (ADR-013).

The Supervisor is the ONLY entry point of the graph.
It does NOT have access to MCP servers or vector stores.
Its job:
  1. Analyse user intent via LLM
  2. Choose execution mode: **parallel** or **sequential**
  3. Route to one or more specialised sub-agents
  4. On re-entry after a sequential step, advance the pipeline
  5. After all agents complete, synthesise a final response

Execution modes (ADR-013):
  - parallel   → all agents run simultaneously via Send() fan-out
  - sequential → agents run one at a time; each round's output is
                 visible to the next agent via mcp_context

Prompts are configurable via ``SupervisorConfig`` in agents.yaml.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END
from langgraph.types import Send

from ..core.agent import BaseAgent
from ..core.llm_provider import LLMProvider
from ..config.schema import OrchestratorSkillConfig, SupervisorConfig
from .state import GraphState

logger = logging.getLogger(__name__)

# ── Prompt templates ─────────────────────────────────────────

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
- If you can answer directly (greeting, general question), do so.

Respond ONLY with a JSON object (no markdown fences):

To invoke a pre-defined skill:
{{
  "reasoning": "brief analysis of the user's intent",
  "execution": "skill",
  "skill": "skill_name",
  "agents": [],
  "direct_response": null
}}

To route to agents manually:
{{
  "reasoning": "brief analysis of the user's intent and why this execution mode",
  "execution": "parallel" or "sequential",
  "agents": ["agent_name_1", "agent_name_2"],
  "direct_response": null
}}

If no agent is needed:
{{
  "reasoning": "...",
  "execution": "parallel",
  "agents": [],
  "direct_response": "Your direct answer here"
}}
"""

SEQUENTIAL_ADVANCE_SYSTEM_PROMPT = """\
You are the Supervisor of the {assistant_name}.
A sub-agent has just completed a step in a sequential pipeline.

Previous agent results are in the conversation.
The NEXT agent in the pipeline is: **{next_agent}** ({next_description}).
Remaining pipeline: {remaining}
{skill_instruction_section}
Your job: write a brief handoff message that summarises what was found so far
and what the next agent should focus on.  This message will be visible to the
next agent in the conversation history.

Be concise — one or two sentences.
"""

SYNTHESIS_SYSTEM_PROMPT = """\
You are the Supervisor of the {assistant_name}.
The specialised sub-agents have completed their work.

IMPORTANT: Only answer the user's LATEST question or request.
The conversation history is provided for context only — do NOT
repeat or re-answer previous questions.

Combine agent results into a single coherent answer for the LATEST query.
Be concise but complete.  If data was retrieved, summarise it meaningfully.
Do NOT mention internal routing or agent names to the user.
"""


# ── Factory ──────────────────────────────────────────────────


def create_supervisor_node(
    model: str,
    agent_descriptions: dict[str, str],
    llm_service: LLMProvider | None = None,
    orchestrator_skills: dict[str, OrchestratorSkillConfig] | None = None,
    supervisor_config: SupervisorConfig | None = None,
):
    """
    Return a LangGraph node function with *model*, *agent_descriptions*,
    *llm_service*, and *orchestrator_skills* captured via closure — no
    module-level globals (ADR-008 Composition Root).
    """
    skills = orchestrator_skills or {}
    sup_config = supervisor_config or SupervisorConfig()

    async def supervisor_node(state: GraphState) -> GraphState:
        pending = state.get("pending_agents", [])
        has_mcp_data = bool(state.get("mcp_context"))

        # ── Case 1: Sequential pipeline in progress — advance to next agent ──
        if pending and has_mcp_data:
            return await _advance_sequential(state, model, agent_descriptions, pending, sup_config, llm_service)

        # ── Case 2: All agents done (no pending) + data collected → synthesise ──
        if has_mcp_data and not pending and not state.get("active_agents"):
            return await _synthesise(state, model, sup_config, llm_service)

        # ── Case 3: First entry — analyse intent and route ──
        return await _route(state, model, agent_descriptions, skills, sup_config, llm_service)

    return supervisor_node


# ── Routing function (conditional edges) ─────────────────────


def route_to_agents(state: GraphState) -> list[Send] | str:
    """
    LangGraph conditional-edge function (ADR-013, ADR-018).

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


# ── Internal helpers ─────────────────────────────────────────


def _filter_internal_messages(
    messages: list[BaseMessage],
    *,
    skip_prefixes: tuple[str, ...] = ("[Supervisor",),
) -> list[BaseMessage]:
    """Remove internal routing messages (e.g. supervisor dispatches) from a message list."""
    filtered: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, AIMessage) or (hasattr(msg, "type") and msg.type == "ai"):
            content = str(msg.content) if hasattr(msg, "content") else str(msg)
            if any(content.startswith(prefix) for prefix in skip_prefixes):
                continue
        filtered.append(msg)
    return filtered


def _to_llm_messages(
    system: str,
    state_messages: list[BaseMessage],
) -> list[dict[str, str]]:
    """Convert LangGraph messages to the [{role, content}] format used by LLMProvider."""
    llm_msgs: list[dict[str, str]] = [{"role": "system", "content": system}]
    for msg in state_messages:
        if isinstance(msg, HumanMessage):
            llm_msgs.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage):
            llm_msgs.append({"role": "assistant", "content": str(msg.content)})
    return llm_msgs


async def _llm_complete(
    llm_service: LLMProvider | None,
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    response_format: dict[str, str] | None = None,
) -> str:
    """Call LLM via the injected LLMProvider."""
    if not llm_service:
        raise RuntimeError("Supervisor requires an LLMProvider. Pass llm_service= when building the graph.")
    return await llm_service.complete(
        model,
        messages,
        temperature=temperature,
        response_format=response_format,
    )


async def _route(
    state: GraphState,
    model: str,
    agent_descriptions: dict[str, str],
    orchestrator_skills: dict[str, OrchestratorSkillConfig] | None = None,
    supervisor_config: SupervisorConfig | None = None,
    llm_service: LLMProvider | None = None,
) -> GraphState:
    """Analyse user intent, choose execution mode, and activate agents."""
    desc_text = "\n".join(f"- **{name}**: {desc}" for name, desc in agent_descriptions.items())

    skills = orchestrator_skills or {}
    if skills:
        skill_text = "\n".join(f'- "{name}": {skill.description}' for name, skill in skills.items())
    else:
        skill_text = "(none defined)"

    sup = supervisor_config or SupervisorConfig()
    routing_template = sup.routing_system_prompt or ROUTING_SYSTEM_PROMPT
    system = routing_template.format(
        assistant_name=sup.assistant_name,
        agent_descriptions=desc_text,
        skill_descriptions=skill_text,
    )
    # Filter internal messages so the routing LLM sees a clean dialogue
    clean_messages = _filter_internal_messages(state.get("messages", []))
    llm_messages = _to_llm_messages(system, clean_messages)

    try:
        raw = await _llm_complete(
            llm_service,
            model,
            llm_messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        logger.info("[Supervisor] routing decision: %s", raw)
    except (ConnectionError, TimeoutError, ValueError, RuntimeError, OSError) as exc:
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

    try:
        decision = json.loads(raw)
    except json.JSONDecodeError:
        decision = {"agents": [], "direct_response": raw, "reasoning": "JSON parse error"}

    agents: list[str] = decision.get("agents", [])
    direct: str | None = decision.get("direct_response")
    execution: str = decision.get("execution", "parallel")

    # ── Orchestrator skill activation ──
    if execution == "skill":
        skill_name = decision.get("skill", "")
        if skill_name in (orchestrator_skills or {}):
            skill = orchestrator_skills[skill_name]
            skill_agents = [step.agent for step in skill.steps]
            skill_instructions_map = {step.agent: step.instruction for step in skill.steps if step.instruction}

            # Validate all agents in the skill exist
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
                "messages": [
                    AIMessage(content=(f"[Supervisor] Skill '{skill_name}': {' → '.join(valid_skill_agents)}"))
                ],
            }
        else:
            logger.warning("[Supervisor] Unknown skill '%s', falling back to agent routing", skill_name)

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


async def _advance_sequential(
    state: GraphState,
    model: str,
    agent_descriptions: dict[str, str],
    pending: list[str],
    supervisor_config: SupervisorConfig | None = None,
    llm_service: LLMProvider | None = None,
) -> GraphState:
    """
    Advance the sequential pipeline: activate the next agent,
    with a handoff message that summarises what was found so far.
    """
    next_agent = pending[0]
    remaining = pending[1:]

    next_desc = agent_descriptions.get(next_agent, "")
    remaining_str = " → ".join(remaining) if remaining else "(last step)"

    # Include skill instruction for the next agent if available
    skill_instructions = state.get("skill_instructions", {})
    instruction = skill_instructions.get(next_agent, "")
    skill_instruction_section = f"\nSKILL INSTRUCTION for {next_agent}: {instruction}\n" if instruction else ""

    # Generate a handoff message so the next agent has context
    sup = supervisor_config or SupervisorConfig()
    advance_template = sup.sequential_advance_prompt or SEQUENTIAL_ADVANCE_SYSTEM_PROMPT
    system = advance_template.format(
        assistant_name=sup.assistant_name,
        next_agent=next_agent,
        next_description=next_desc,
        remaining=remaining_str,
        skill_instruction_section=skill_instruction_section,
    )

    # Build clean history using the shared framework helper
    history = BaseAgent.extract_conversation_history(
        state,
        max_turns=sup.history_max_turns,
        max_chars=sup.history_max_chars,
    )

    # Compress older turns when sliding-window summarization is enabled
    if history and sup.history_summary_enabled and llm_service:
        history = await BaseAgent.compress_conversation_history(
            history,
            llm_service=llm_service,
            model=sup.history_summary_model or model,
            recent_turns=sup.history_summary_recent_turns,
        )

    llm_messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    llm_messages.extend(history)

    # Inject current MCP data so the LLM knows what was collected
    mcp_ctx = state.get("mcp_context", {})
    if mcp_ctx:
        context_blob = json.dumps(mcp_ctx, indent=2, default=str)
        llm_messages.append(
            {
                "role": "user",
                "content": f"Data collected so far:\n```json\n{context_blob}\n```",
            }
        )

    try:
        handoff = await _llm_complete(llm_service, model, llm_messages, temperature=0.2)
        logger.info(
            "[Supervisor] sequential advance → %s (pending: %s): %s",
            next_agent,
            remaining,
            handoff[:100],
        )
    except (ConnectionError, TimeoutError, ValueError, RuntimeError, OSError) as exc:
        # Handle LLM API failures gracefully - use default handoff
        logger.error("[Supervisor] LLM API error during sequential handoff: %s", exc, exc_info=True)
        handoff = f"Continue with {next_agent} to address the user's request."

    return {
        "active_agents": [next_agent],
        "pending_agents": remaining,
        "execution_mode": "sequential",
        "messages": [AIMessage(content=f"[Supervisor → {next_agent}] {handoff}")],
    }


async def _synthesise(
    state: GraphState,
    model: str,
    supervisor_config: SupervisorConfig | None = None,
    llm_service: LLMProvider | None = None,
) -> GraphState:
    """Combine sub-agent results into a final user-facing response."""
    sup = supervisor_config or SupervisorConfig()
    all_messages = state.get("messages", [])

    # ── Split messages: prior history vs current turn ──
    # Current turn = last user message + all messages after it
    last_user_idx = -1
    for i, msg in enumerate(all_messages):
        if isinstance(msg, HumanMessage) or (hasattr(msg, "type") and msg.type == "human"):
            last_user_idx = i

    if last_user_idx > 0:
        current_turn = all_messages[last_user_idx:]
    else:
        current_turn = all_messages

    # ── Build LLM messages ──
    synthesis_template = sup.synthesis_system_prompt or SYNTHESIS_SYSTEM_PROMPT
    synthesis_prompt = synthesis_template.format(assistant_name=sup.assistant_name)
    llm_messages: list[dict[str, str]] = [
        {"role": "system", "content": synthesis_prompt},
    ]

    # Use the shared framework helper for clean, configurable history.
    # extract_conversation_history already filters [Supervisor messages,
    # excludes the last user message, and respects turn/char limits.
    history = BaseAgent.extract_conversation_history(
        state,
        max_turns=sup.history_max_turns,
        max_chars=sup.history_max_chars,
    )

    # Compress older turns when sliding-window summarization is enabled
    if history and sup.history_summary_enabled and llm_service:
        history = await BaseAgent.compress_conversation_history(
            history,
            llm_service=llm_service,
            model=sup.history_summary_model or model,
            recent_turns=sup.history_summary_recent_turns,
        )

    if history:
        llm_messages.append(
            {
                "role": "user",
                "content": "Previous conversation (for context only — do NOT re-answer):\n"
                + "\n".join(f"  {m['role']}: {m['content']}" for m in history),
            }
        )

    # Add the current turn messages, filtering out internal routing noise
    for msg in current_turn:
        content = str(msg.content) if hasattr(msg, "content") else str(msg)
        if isinstance(msg, HumanMessage) or (hasattr(msg, "type") and msg.type == "human"):
            llm_messages.append({"role": "user", "content": content})
        elif isinstance(msg, AIMessage) or (hasattr(msg, "type") and msg.type == "ai"):
            # Skip internal supervisor routing messages from current turn
            if content.startswith("[Supervisor"):
                continue
            llm_messages.append({"role": "assistant", "content": content})

    # Inject MCP context as additional grounding
    mcp_ctx = state.get("mcp_context", {})
    if mcp_ctx:
        context_blob = json.dumps(mcp_ctx, indent=2, default=str)
        llm_messages.append(
            {
                "role": "user",
                "content": f"Sub-agent data (for reference):\n```json\n{context_blob}\n```",
            }
        )

    try:
        final = await _llm_complete(llm_service, model, llm_messages, temperature=0.3)
        logger.info("[Supervisor] synthesis complete (%d chars)", len(final))
    except (ConnectionError, TimeoutError, ValueError, RuntimeError, OSError) as exc:
        # Handle LLM API failures gracefully
        logger.error("[Supervisor] LLM API error during synthesis: %s", exc, exc_info=True)
        error_msg = str(exc)
        if "503" in error_msg or "high demand" in error_msg.lower():
            final = (
                "I'm currently experiencing high demand and cannot synthesize the results. "
                "Please try again in a few moments."
            )
        elif "rate limit" in error_msg.lower():
            final = "I've hit my rate limit. Please try again in a few moments."
        else:
            final = (
                f"I encountered an error while synthesizing the response: {error_msg[:200]}. Please try again later."
            )

    return {
        "messages": [AIMessage(content=final)],
        "final_response": final,
        "active_agents": [],
        "pending_agents": [],
    }
