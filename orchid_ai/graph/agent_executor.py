"""Shared agent execution path — one implementation for chat and non-chat runs.

:class:`OrchidAgentExecutor` owns what the graph layer used to own around
``OrchidAgent.run()``:

- per-run context binding (auth, chat/message ids, correlation id, memory policy),
- per-agent input/output guardrails,
- model routing selection,
- error translation,
- optional inline mini-agent orchestration (decompose → parallel gather →
  aggregate).

The chat graph's ``_AgentNodeWrapper`` delegates to this class with inline
mini-agents disabled, because the chat graph keeps its own ``Send``-based fork
topology. Callers outside the graph (for example pipeline agent jobs) use the
same class with inline mini-agents enabled, so guardrails, routing and memory
policy are not duplicated.

The executor never imports ``orchid_ai.pipeline`` — it is a graph-layer
component with no knowledge of pipeline sessions or results.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphBubbleUp

from ..config.schema import OrchidAgentConfig
from ..core.agent import OrchidAgent, OrchidAgentRunContext
from ..core.guardrails import (
    OrchidGuardrailAction,
    OrchidGuardrailChain,
    OrchidGuardrailContext,
    OrchidGuardrailDirection,
)
from ..core.run_config import with_auth
from ..core.state import OrchidAgentState
from .model_routing import ModelRoutingSelector, TurnBudgetRegistry

logger = logging.getLogger(__name__)
perf_logger = logging.getLogger("orchid.perf")

__all__ = ["OrchidAgentExecutor", "latest_human_message_id", "turn_key"]


def latest_human_message_id(state: Mapping[str, Any]) -> str | None:
    """Return the id of the most recent ``HumanMessage`` in the state.

    ``None`` means the message list is empty, has no human message, or the
    message carries no id. In-chat progress cards anchor under the message that
    triggered the turn (§LS5), so a missing id stays ``None`` rather than
    falling back to an unrelated message.
    """
    messages = state.get("messages") or []
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            mid = getattr(message, "id", None)
            if isinstance(mid, str) and mid:
                return mid
            return None
    return None


def turn_key(state: Mapping[str, Any]) -> str:
    """Stable key for one user turn across every agent node in a graph run.

    The latest human message and its count do not change while the supervisor
    and agents loop within one request, so the first agent creates the shared
    turn budget and later agents spend from it. The query text is hashed rather
    than stored, and the message id distinguishes repeated identical prompts.
    """
    chat_id = state.get("chat_id") or ""
    human_count = 0
    last_human: Any = None
    for message in state.get("messages") or []:
        if isinstance(message, HumanMessage):
            human_count += 1
            last_human = message

    text = str(getattr(last_human, "content", "")) if last_human is not None else ""
    message_id = (getattr(last_human, "id", "") or "") if last_human is not None else ""
    digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]
    return f"{chat_id}:{human_count}:{message_id}:{digest}"


# ORCHID-MEM: 01a1e9ec-5345-7973-9aa7-f4a0eb44331d — Shared agent executor; chat wrapper and pipeline jobs share guardrails, routing, mini forks and error translation.
class OrchidAgentExecutor:
    """Run one :class:`OrchidAgent` through the shared execution policy.

    Collaborators are injected explicitly so the chat wrapper and any
    non-chat caller can share one instance shape:

    - ``agent`` — the agent instance to run.
    - ``chat_model`` — the model used by the decomposer, mini-agents and
      aggregator. Defaults to ``agent.chat_model``.
    - ``input_guardrails`` / ``output_guardrails`` — per-agent chains.
    - ``agent_config`` — carries the mini-agent options (enable flag, cap,
      allowlist mode, prompts); ``None`` disables mini-agent orchestration.
    - ``model_router`` / ``turn_budgets`` — the routing selection and the
      per-turn budget shared by every agent in a graph turn.
    - ``inline_mini_agents`` — ``True`` for callers outside the chat graph.
      The chat wrapper passes ``False`` and keeps its ``Send`` topology.
    - ``clock`` — monotonic timer used by the perf log lines.

    Model-routing scope differs by design. Inline forks (and the aggregator)
    run inside the parent run's ``bound`` selection, because the executor binds
    it around the whole activation. In the chat graph each ``Send`` branch runs
    after the node returned and the selection was reset, so minis use the
    configured default unless they route themselves. Both paths share the same
    selector and factories; only the scheduling boundary differs.
    """

    def __init__(
        self,
        agent: OrchidAgent,
        *,
        chat_model: Any | None = None,
        input_guardrails: OrchidGuardrailChain | None = None,
        output_guardrails: OrchidGuardrailChain | None = None,
        agent_config: OrchidAgentConfig | None = None,
        model_router: ModelRoutingSelector | None = None,
        turn_budgets: TurnBudgetRegistry | None = None,
        inline_mini_agents: bool = True,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._agent = agent
        self._chat_model = chat_model if chat_model is not None else agent.chat_model
        self._input_guardrails = input_guardrails
        self._output_guardrails = output_guardrails
        self._agent_config = agent_config
        self._model_router = model_router
        self._turn_budgets = turn_budgets
        self._inline_mini_agents = inline_mini_agents
        self._clock = clock

    @property
    def agent(self) -> OrchidAgent:
        return self._agent

    # ORCHID-MEM: 01a1ef10-9e9f-72b9-aadc-2507378fa1f0 — Executor routing trace and inline decomposer fallback.
    async def run(
        self,
        state: OrchidAgentState,
        *,
        auth: Any | None,
        chat_id: str | None = None,
        message_id: str | None = None,
        correlation_id: str | None = None,
        persist_memory: bool = True,
        trace_out: dict[str, Any] | None = None,
    ) -> OrchidAgentState:
        """Execute one agent activation under the shared policy.

        Input guardrails run before model routing, so blocked content never
        reaches the routing decision provider. A redacted user turn is carried
        into the returned state update instead of only into the agent's
        in-memory view. Output guardrails run after the agent only on the normal
        path; the forked mini-agent path has no parent output rail in the chat
        graph either.

        ``trace_out`` is an optional caller-owned mapping. When the agent has a
        model router, the selected model, point and whether it differs from the
        configured default are written under ``"model_routing"`` while the
        selection is bound. The mapping also records whether an input guardrail
        blocked the run, whether the decomposer ran and whether the agent was
        invoked, so non-chat callers can count LLM activations without reading
        private state. The chat wrapper passes ``None``.
        """
        state, redacted_input, blocked = await self._run_input_guardrails(state, auth)
        if blocked is not None:
            if trace_out is not None:
                trace_out["guardrails_blocked"] = True
            return blocked

        if self._model_router is None:
            result = await self._run_after_input_guardrails(
                state,
                auth,
                chat_id=chat_id,
                message_id=message_id,
                correlation_id=correlation_id,
                persist_memory=persist_memory,
                trace_out=trace_out,
            )
        else:
            # Ask the ``model_routing`` point once per agent run and bind the
            # selection for the duration of this coroutine. The routed chat
            # model reads it on every LLM call, so concurrent runs keep their
            # own selection. The budget is shared across every agent node in
            # the same turn (F4).
            query = self._agent.extract_user_query(state)
            budget = self._turn_budgets.for_turn(turn_key(state)) if self._turn_budgets is not None else None
            async with self._model_router.bound(query, budget=budget) as selected:
                if trace_out is not None:
                    default_model = getattr(self._model_router, "default_model", "")
                    trace_out["model_routing"] = {
                        "point": getattr(self._model_router, "point", ""),
                        "model": selected,
                        "default_model": default_model,
                        "routed": selected != default_model,
                    }
                result = await self._run_after_input_guardrails(
                    state,
                    auth,
                    chat_id=chat_id,
                    message_id=message_id,
                    correlation_id=correlation_id,
                    persist_memory=persist_memory,
                    trace_out=trace_out,
                )

        if redacted_input is not None:
            # The replacement must be part of the returned update so the graph
            # state — not just this agent's in-memory view — holds the redacted
            # message.
            result = {**result, "messages": [redacted_input, *result.get("messages", [])]}
        return result

    async def _run_after_input_guardrails(
        self,
        state: OrchidAgentState,
        auth: Any | None,
        *,
        chat_id: str | None,
        message_id: str | None,
        correlation_id: str | None,
        persist_memory: bool,
        trace_out: dict[str, Any] | None = None,
    ) -> OrchidAgentState:
        decomposer_update = await self._run_decomposer(state, auth, trace_out=trace_out)
        if decomposer_update is not None:
            if "mini_agent_decisions" in decomposer_update:
                if self._inline_mini_agents:
                    return await self._run_inline_mini_agents(state, auth, decomposer_update)
                decomposer_update.setdefault("active_agents", [])
                return decomposer_update
            if not self._inline_mini_agents:
                decomposer_update.setdefault("active_agents", [])
                return decomposer_update
            # A rejected decomposition is a short-circuit for the chat graph,
            # where the supervisor can explain it. Non-chat callers have no
            # such surface and must not treat the rejection text as the answer.
            logger.warning(
                "[Graph] Agent '%s' inline decomposer rejected the request; "
                "running the agent directly for this non-chat activation",
                self._agent.name,
            )

        agent_result = await self._run_agent(
            state,
            auth,
            chat_id=chat_id,
            message_id=message_id,
            correlation_id=correlation_id,
            persist_memory=persist_memory,
            trace_out=trace_out,
        )

        await self._run_output_guardrails(state, auth, agent_result)

        agent_result["active_agents"] = []
        return agent_result

    async def _run_input_guardrails(
        self,
        state: OrchidAgentState,
        auth: Any | None,
    ) -> tuple[OrchidAgentState, HumanMessage | None, OrchidAgentState | None]:
        """Apply per-agent input rails.

        Returns ``(state_after_rails, redacted_message, blocked_update)``.
        ``state_after_rails`` carries the redacted message into the agent run;
        ``redacted_message`` is merged into the returned update so the graph
        state keeps the redacted text. A block short-circuits with its update.
        """
        if not self._input_guardrails or self._input_guardrails.empty:
            return state, None, None

        query = self._agent.extract_user_query(state)
        ctx = OrchidGuardrailContext(
            direction=OrchidGuardrailDirection.INPUT,
            agent_name=self._agent.name,
            tenant_key=auth.tenant_key if auth else "default",
            user_id=auth.user_id if auth else "",
            chat_id=state.get("chat_id", ""),
        )
        result = await self._input_guardrails.evaluate(query, ctx)
        if result.blocked:
            logger.warning(
                "[Guardrails] Agent '%s' input blocked by '%s': %s",
                self._agent.name,
                result.guardrail_name,
                result.message,
            )
            return (
                state,
                None,
                {
                    "messages": [AIMessage(content=f"[{self._agent.name.title()} Agent] {result.message}")],
                    "active_agents": [],
                },
            )

        if (
            result.action == OrchidGuardrailAction.REDACT
            and result.redacted_content is not None
            and result.redacted_content != query
        ):
            messages = list(state.get("messages", []))
            # The supervisor appends a dispatch AIMessage (and earlier agents
            # append their replies), so the user turn is not necessarily last.
            index = next(
                (i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)),
                None,
            )
            if index is not None:
                replacement = messages[index].model_copy(update={"content": result.redacted_content})
                messages[index] = replacement
                logger.info("[Guardrails] Agent '%s' input redacted by '%s'", self._agent.name, result.guardrail_name)
                return {**state, "messages": messages}, replacement, None

        return state, None, None

    async def _run_decomposer(
        self,
        state: OrchidAgentState,
        auth: Any | None,
        *,
        trace_out: dict[str, Any] | None = None,
    ) -> OrchidAgentState | None:
        """Run the mini-agent decomposer hook, when configured.

        A state already carrying ``_active_mini_parent`` belongs to a mini run,
        so the non-nesting rule keeps the executor from decomposing again.
        """
        if self._agent_config is None or not self._agent_config.mini_agent.enabled or auth is None:
            return None
        if state.get("_active_mini_parent") or self._chat_model is None:
            return None
        if trace_out is not None:
            trace_out["decomposer_ran"] = True

        from ..agents.mini_agent_decomposer import maybe_decompose

        decomp_start = self._clock()
        update = await maybe_decompose(
            agent_config=self._agent_config,
            chat_model=self._chat_model,
            mcp_clients=self._agent.mcp_clients or [],
            auth=auth,
            state=state,
        )
        decomp_elapsed = (self._clock() - decomp_start) * 1000
        perf_logger.info(
            "[PERF][agent=%s] step=decomposer took %.1f ms (forked=%s)",
            self._agent.name,
            decomp_elapsed,
            bool(update and "mini_agent_decisions" in update),
        )
        return update

    async def _run_inline_mini_agents(
        self,
        state: OrchidAgentState,
        auth: Any | None,
        decomposer_update: OrchidAgentState,
    ) -> OrchidAgentState:
        """Run a fork inline: decompose → parallel gather → aggregate.

        Reuses the existing mini node and aggregator factories; only the
        scheduling differs from the chat graph's ``Send`` fan-out. The
        decomposer already enforced the per-agent ``max_count`` cap and the
        tool allowlist, and :meth:`_run_decomposer` refused to nest.

        Unlike a chat ``Send`` branch, these minis run inside the parent run's
        routed-model binding (see the class docstring); that is intentional
        pipeline semantics.
        """
        from ..agents.mini_agent_aggregator import aggregator_node_factory
        from ..agents.mini_agent_node import mini_agent_node_factory

        decisions = decomposer_update.get("mini_agent_decisions") or {}
        decision = decisions.get(self._agent.name)
        # Mirror the chat graph's fork router: it only fans out when the
        # decision says ``should_fork`` AND carries sub-tasks. ``maybe_decompose``
        # already guarantees that pairing, so this is a defensive guard that
        # also keeps a forward-compatible/malformed decision from diverging the
        # two paths.
        if decision is None or not decision.get("should_fork") or not decision.get("sub_tasks"):
            decomposer_update.setdefault("active_agents", [])
            return decomposer_update
        if self._agent_config is None or self._chat_model is None:
            decomposer_update.setdefault("active_agents", [])
            return decomposer_update

        mini_node = mini_agent_node_factory(
            parent_config=self._agent_config,
            chat_model=self._chat_model,
            mcp_clients=self._agent.mcp_clients or [],
        )
        config = with_auth(auth)
        calls = []
        for index, sub_task in enumerate(decision.get("sub_tasks") or []):
            tool_subset = sub_task.get("resolved_tool_subset") or sub_task.get("allowed_tools") or []
            calls.append(
                mini_node(
                    {
                        **state,
                        "_active_mini_parent": self._agent.name,
                        "_active_mini_id": sub_task.get("id") or f"mini_{index}",
                        "_active_mini_subtask": sub_task,
                        "_active_mini_tool_subset": list(tool_subset),
                    },
                    config,
                )
            )
        updates = await asyncio.gather(*calls)

        messages: list[Any] = list(decomposer_update.get("messages") or [])
        mcp_context: dict[str, Any] = {}
        outcomes: dict[str, Any] = {}
        for update in updates:
            messages.extend(update.get("messages") or [])
            mcp_context.update(update.get("mcp_context") or {})
            outcomes.update(update.get("mini_agent_outcomes") or {})

        aggregator = aggregator_node_factory(parent_config=self._agent_config, chat_model=self._chat_model)
        aggregate_update = await aggregator(
            {
                **state,
                "mini_agent_outcomes": outcomes,
                "mcp_context": mcp_context,
            }
        )
        messages.extend(aggregate_update.get("messages") or [])
        mcp_context.update(aggregate_update.get("mcp_context") or {})

        return {
            "messages": messages,
            "mcp_context": mcp_context,
            "mini_agent_decisions": decisions,
            "mini_agent_outcomes": outcomes,
            "active_agents": [],
        }

    async def _run_agent(
        self,
        state: OrchidAgentState,
        auth: Any | None,
        *,
        chat_id: str | None,
        message_id: str | None,
        correlation_id: str | None,
        persist_memory: bool,
        trace_out: dict[str, Any] | None = None,
    ) -> OrchidAgentState:
        if trace_out is not None:
            trace_out["agent_invoked"] = True
        # Bind per-request state on a ContextVar so two concurrent activations
        # of the same agent (LangGraph ``Send()`` fan-out, mini-agent forks,
        # events runner) stay isolated.
        token = self._agent.set_run_context(
            OrchidAgentRunContext(
                auth=auth,
                correlation_id=correlation_id,
                chat_id=chat_id,
                message_id=message_id,
                persist_memory=persist_memory,
            )
        )

        agent_start = self._clock()
        perf_logger.info("[PERF][agent=%s] >>> START", self._agent.name)
        try:
            result = await self._agent.run(state)
        except GraphBubbleUp:
            # HITL ``interrupt()`` and LangGraph control-flow signals must
            # bubble up to the graph runtime, not be swallowed as agent errors.
            raise
        except Exception as exc:
            logger.exception(
                "[Graph] Agent '%s' raised an unhandled exception",
                self._agent.name,
            )
            result = {
                "messages": [
                    AIMessage(
                        content=(
                            f"[{self._agent.name.title()} Agent] I'm temporarily unable to process "
                            "your request. Please try again in a few moments."
                        )
                    )
                ],
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            self._agent.reset_run_context(token)
        agent_elapsed = (self._clock() - agent_start) * 1000
        perf_logger.info("[PERF][agent=%s] <<< DONE total=%.1f ms", self._agent.name, agent_elapsed)
        return result

    async def _run_output_guardrails(
        self,
        state: OrchidAgentState,
        auth: Any | None,
        agent_result: OrchidAgentState,
    ) -> None:
        if not self._output_guardrails or self._output_guardrails.empty:
            return

        agent_messages = agent_result.get("messages", [])
        if not agent_messages:
            return

        response_text = str(agent_messages[-1].content) if hasattr(agent_messages[-1], "content") else ""
        ctx = OrchidGuardrailContext(
            direction=OrchidGuardrailDirection.OUTPUT,
            agent_name=self._agent.name,
            tenant_key=auth.tenant_key if auth else "default",
            user_id=auth.user_id if auth else "",
            chat_id=state.get("chat_id", ""),
            metadata={"rag_context": agent_result.get("rag_context", {}).get(self._agent.name, [])},
        )
        result = await self._output_guardrails.evaluate(response_text, ctx)
        if result.blocked:
            logger.warning(
                "[Guardrails] Agent '%s' output blocked by '%s': %s",
                self._agent.name,
                result.guardrail_name,
                result.message,
            )
            agent_result["messages"] = [AIMessage(content=f"[{self._agent.name.title()} Agent] {result.message}")]
        elif result.action == OrchidGuardrailAction.REDACT and result.redacted_content is not None:
            logger.info("[Guardrails] Agent '%s' output redacted by '%s'", self._agent.name, result.guardrail_name)
            agent_result["messages"] = [AIMessage(content=result.redacted_content)]
