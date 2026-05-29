"""Sequential pipeline advancer.

When the supervisor finds it has been re-entered with pending agents and
collected mcp_context (i.e. the previous step has finished), it delegates
to :class:`SequentialAdvancer` to:

  1. Build a brief LLM-generated handoff message for the next agent,
  2. Activate that agent and shrink the pending list.

Extracted from ``supervisor.py`` so the supervisor is responsible only
for orchestration, not for the LLM-prompting details of every phase.
"""

from __future__ import annotations

import json
import logging
import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from ..config.schema import OrchidSupervisorConfig
from ..core.agent import OrchidAgent
from ..core.helpers import filter_summary_messages
from ..core.memory import OrchidConversationMemory
from ._supervisor_helpers import _llm_complete
from .state import GraphState

logger = logging.getLogger(__name__)
perf_logger = logging.getLogger("orchid.perf")


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


class SequentialAdvancer:
    """Advances a sequential agent pipeline by one step.

    Each invocation of :meth:`advance` activates the next agent in the
    pipeline and emits a handoff banner that summarises what previous
    agents collected. Exists as a class (not a free function) so the
    supervisor's collaborator graph is explicit and the LLM-prompting
    details stay out of the supervisor.
    """

    def __init__(
        self,
        *,
        model: str,
        agent_descriptions: dict[str, str],
        supervisor_config: OrchidSupervisorConfig,
        chat_model: BaseChatModel | None,
        memory: OrchidConversationMemory | None = None,
    ) -> None:
        self._model = model
        self._agent_descriptions = agent_descriptions
        self._supervisor_config = supervisor_config
        self._chat_model = chat_model
        self._memory = memory

    async def advance(self, state: GraphState, pending: list[str]) -> GraphState:
        """Activate the next agent in *pending* and emit a handoff message."""
        next_agent = pending[0]
        remaining = pending[1:]

        next_desc = self._agent_descriptions.get(next_agent, "")
        remaining_str = " → ".join(remaining) if remaining else "(last step)"

        skill_instructions = state.get("skill_instructions", {})
        instruction = skill_instructions.get(next_agent, "")
        skill_instruction_section = f"\nSKILL INSTRUCTION for {next_agent}: {instruction}\n" if instruction else ""

        sup = self._supervisor_config
        advance_template = sup.sequential_advance_prompt or SEQUENTIAL_ADVANCE_SYSTEM_PROMPT
        system = advance_template.format(
            assistant_name=sup.assistant_name,
            next_agent=next_agent,
            next_description=next_desc,
            remaining=remaining_str,
            skill_instruction_section=skill_instruction_section,
        )

        history = OrchidAgent.extract_conversation_history(
            state,
            max_turns=sup.history_max_turns,
            max_chars=sup.history_max_chars,
            truncation_strategy=sup.memory.truncation_strategy,
        )

        if history and sup.history_summary_enabled and self._chat_model:
            running_summary: str | None = None
            if self._memory is not None and sup.memory.strategy in ("running_summary", "rag_augmented"):
                chat_id = state.get("chat_id", "")
                if chat_id:
                    try:
                        running_summary = await self._memory.get_running_summary(chat_id)
                    except Exception:
                        pass
            history = await OrchidAgent.compress_conversation_history(
                history,
                chat_model=self._chat_model,
                recent_turns=sup.history_summary_recent_turns,
                running_summary=running_summary,
                structured_output=sup.memory.structured_output,
            )
            # Persist the updated summary
            if self._memory is not None and sup.memory.strategy in ("running_summary", "rag_augmented"):
                chat_id = state.get("chat_id", "")
                if chat_id:
                    try:
                        delta = filter_summary_messages(history)
                        await self._memory.update_running_summary(
                            chat_id,
                            delta,
                            running_summary,
                        )
                    except Exception:
                        pass

        clean_history = (
            [m for m in history if not m.get("content", "").startswith("[Conversation summary]")] if history else []
        )

        llm_messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        llm_messages.extend(clean_history)

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
            llm_start = time.perf_counter()
            handoff = await _llm_complete(self._chat_model, self._model, llm_messages, temperature=0.2)
            llm_elapsed = (time.perf_counter() - llm_start) * 1000
            perf_logger.info("[PERF][supervisor.advance] handoff LLM call took %.1f ms", llm_elapsed)
            logger.info(
                "[Supervisor] sequential advance → %s (pending: %s): %s",
                next_agent,
                remaining,
                handoff[:100],
            )
        except Exception as exc:
            logger.error("[Supervisor] LLM API error during sequential handoff: %s", exc, exc_info=True)
            handoff = f"Continue with {next_agent} to address the user's request."

        return {
            "active_agents": [next_agent],
            "pending_agents": remaining,
            "execution_mode": "sequential",
            "messages": [AIMessage(content=f"[Supervisor → {next_agent}] {handoff}")],
        }
