"""Final-response synthesis.

After all agents in a turn have completed, the supervisor delegates to
:class:`ResponseSynthesizer` to merge their results into a single
user-facing answer. The synthesizer also owns the single-agent fast
path: when exactly one agent ran and produced final text, the LLM call
is skipped and the agent's text is returned directly.

Extracted from ``supervisor.py`` so the supervisor is responsible only
for orchestration, not the prompt-engineering of the synthesis pass.
"""

from __future__ import annotations

import json
import logging
import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from ..config.schema import OrchidSupervisorConfig
from ..core.agent import OrchidAgent
from ._supervisor_helpers import _extract_single_agent_response, _llm_complete
from .state import GraphState

logger = logging.getLogger(__name__)
perf_logger = logging.getLogger("orchid.perf")


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


class ResponseSynthesizer:
    """Combines sub-agent results into a single final response.

    Owns two related responsibilities the supervisor used to inline:
      1. The fast path that returns a lone agent's text verbatim,
         skipping the synthesis LLM call when ``skip_synthesis_when_single_agent``
         is on and exactly one agent produced final text.
      2. The synthesis LLM call that merges multi-agent / sequential output.
    """

    def __init__(
        self,
        *,
        model: str,
        supervisor_config: OrchidSupervisorConfig,
        chat_model: BaseChatModel | None,
    ) -> None:
        self._model = model
        self._supervisor_config = supervisor_config
        self._chat_model = chat_model

    async def synthesise(self, state: GraphState) -> GraphState:
        """Return the final-response state delta, taking the fast path when possible."""
        fast = self._try_single_agent_fast_path(state)
        if fast is not None:
            return fast
        return await self._llm_synthesise(state)

    def _try_single_agent_fast_path(self, state: GraphState) -> GraphState | None:
        """Skip the synthesis LLM call when exactly one agent produced final text.

        The agent's response is already user-ready (the agentic loop's
        last LLM call synthesised it from RAG + tools); a second pass
        at the supervisor level just rewrites it at the cost of 5–15 s
        and ~6–10 k tokens of input on every turn. Multi-agent fan-in
        and sequential pipelines still go through the LLM path so
        their outputs get merged.
        """
        if not self._supervisor_config.skip_synthesis_when_single_agent:
            return None
        single = _extract_single_agent_response(state)
        if single is None:
            return None
        perf_logger.info(
            "[PERF][supervisor] phase=synthesise_skipped (single-agent fast path, %d chars)",
            len(single),
        )
        logger.info("[Supervisor] synthesis skipped — single agent produced final text (%d chars)", len(single))
        return {
            "messages": [AIMessage(content=single)],
            "final_response": single,
            "active_agents": [],
            "pending_agents": [],
        }

    async def _llm_synthesise(self, state: GraphState) -> GraphState:
        """Run the synthesis LLM call and return the resulting state delta."""
        sup = self._supervisor_config
        all_messages = state.get("messages", [])

        last_user_idx = -1
        for i, msg in enumerate(all_messages):
            if isinstance(msg, HumanMessage) or (hasattr(msg, "type") and msg.type == "human"):
                last_user_idx = i

        if last_user_idx > 0:
            current_turn = all_messages[last_user_idx:]
        else:
            current_turn = all_messages

        synthesis_template = sup.synthesis_system_prompt or SYNTHESIS_SYSTEM_PROMPT
        synthesis_prompt = synthesis_template.format(assistant_name=sup.assistant_name)
        llm_messages: list[dict[str, str]] = [
            {"role": "system", "content": synthesis_prompt},
        ]

        history = OrchidAgent.extract_conversation_history(
            state,
            max_turns=sup.history_max_turns,
            max_chars=sup.history_max_chars,
        )

        if history and sup.history_summary_enabled and self._chat_model:
            history = await OrchidAgent.compress_conversation_history(
                history,
                chat_model=self._chat_model,
                recent_turns=sup.history_summary_recent_turns,
            )

        if history:
            visible_history = [m for m in history if not m.get("content", "").startswith("[Conversation summary]")]
            if visible_history:
                llm_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Previous conversation (for context only — do NOT re-answer or "
                            "reproduce any of this):\n"
                            + "\n".join(f"  {m['role']}: {m['content']}" for m in visible_history)
                        ),
                    }
                )

        for msg in current_turn:
            content = str(msg.content) if hasattr(msg, "content") else str(msg)
            if isinstance(msg, HumanMessage) or (hasattr(msg, "type") and msg.type == "human"):
                llm_messages.append({"role": "user", "content": content})
            elif isinstance(msg, AIMessage) or (hasattr(msg, "type") and msg.type == "ai"):
                if content.startswith("[Supervisor") or content.startswith("[Conversation summary]"):
                    continue
                llm_messages.append({"role": "assistant", "content": content})

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
            llm_start = time.perf_counter()
            final = await _llm_complete(self._chat_model, self._model, llm_messages, temperature=0.3)
            llm_elapsed = (time.perf_counter() - llm_start) * 1000
            perf_logger.info(
                "[PERF][supervisor.synth] synthesis LLM call took %.1f ms (out_chars=%d)",
                llm_elapsed,
                len(final),
            )
            logger.info("[Supervisor] synthesis complete (%d chars)", len(final))
        except (ConnectionError, TimeoutError, ValueError, RuntimeError, OSError) as exc:
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
                    f"I encountered an error while synthesizing the response: {error_msg[:200]}. "
                    "Please try again later."
                )

        return {
            "messages": [AIMessage(content=final)],
            "final_response": final,
            "active_agents": [],
            "pending_agents": [],
        }
