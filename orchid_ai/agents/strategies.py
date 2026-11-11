"""
Tool call strategies for MCP server interaction — Strategy pattern (OCP).

Each strategy defines *how* tools on a single MCP server are invoked:
  - ``all``          — call every whitelisted tool, collect all results
  - ``sequential``   — call tools in order, chaining context forward
  - ``llm_decides``  — let the ``tool_choice`` decision point choose which
                       shortlisted tools to call, falling back to ``all``

Adding a new strategy:
  1. Subclass ``OrchidToolCallStrategy``
  2. Register it in ``STRATEGY_REGISTRY``
  3. Reference by name in ``agents.yaml`` → ``tool_call_strategy: my_strategy``

.. note:: When strategies are used

   These strategies are executed by :meth:`MCPDispatcher.fetch`, which
   is called during **skill execution** (:class:`SkillExecutor`).  For
   regular (non-skill) queries, :class:`GenericAgent` uses its own
   agentic tool-calling loop (:meth:`GenericAgent._agentic_tool_loop`)
   that always uses "LLM decides" semantics via native ``tool_calls``.
   In that path, the ``tool_call_strategy`` YAML field is not
   consulted — the LLM directly selects which tools to call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from ..config.schema import OrchidMCPServerConfig, OrchidToolConfig
from ..core.decision import (
    OrchidDecisionError,
    OrchidDecisionPointBinding,
    OrchidDecisionQuestion,
    OrchidDecisionResult,
    OrchidDecisionState,
    aggregate_shortlist,
)
from ..core.mcp import OrchidMCPDiscoverable, OrchidMCPToolCaller
from ..core.state import OrchidAuthContext
from ..decision.observability import log_decision

logger = logging.getLogger(__name__)

_POINT = "tool_choice"
_QUESTION_PREFIX = "tool:"
_MAX_QUERY_CHARS = 4000
_MAX_DESCRIPTION_CHARS = 240
#: Hard wall-clock ceiling for one tool-choice decision. Providers also receive
#: this value as their per-call ``timeout``; the outer ``asyncio.timeout``
#: keeps a provider that ignores it from stalling the dispatcher path.
_MAX_DECISION_SECONDS = 2.0

#: Per-run ``tool_choice`` binding for ``llm_decides``.
#:
#: The dispatcher instantiates strategies from the registry, so an agent's
#: per-run binding travels on a context variable — the same execution-context
#: pattern used for auth and skill depth. An explicit constructor or
#: ``execute`` binding always wins over this scope.
_tool_choice_binding_var: ContextVar[OrchidDecisionPointBinding | None] = ContextVar(
    "orchid.tool_choice_binding",
    default=None,
)


@contextmanager
def use_tool_choice_binding(binding: OrchidDecisionPointBinding | None) -> Iterator[None]:
    """Scope ``binding`` to the current async context for ``llm_decides``.

    ``GenericAgent`` enters this scope for one run so any strategy the MCP
    dispatcher instantiates while that run is active sees the agent's
    ``tool_choice`` binding. Nested runs restore the previous binding on exit.
    """
    token = _tool_choice_binding_var.set(binding)
    try:
        yield
    finally:
        _tool_choice_binding_var.reset(token)


class OrchidToolCallStrategy(ABC):
    """Strategy for calling MCP tools on a single server."""

    @abstractmethod
    async def execute(
        self,
        client: OrchidMCPToolCaller,
        tools: list[OrchidToolConfig],
        query: str,
        auth: OrchidAuthContext,
        *,
        agent_name: str = "",
        server_config: OrchidMCPServerConfig | None = None,
        llm_model: str | None = None,
        chat_model: Any | None = None,
    ) -> dict[str, Any]:
        """
        Invoke tools according to this strategy's rules.

        Parameters
        ----------
        client : OrchidMCPClient
            The MCP client to call tools on.
        tools : list[OrchidToolConfig]
            Tools available for this invocation.
        query : str
            The user's query.
        auth : OrchidAuthContext
            Authentication context for tool calls.
        agent_name : str
            Name of the calling agent (for logging).
        server_config : OrchidMCPServerConfig | None
            Full server configuration (for strategies that need it).
        llm_model : str | None
            LLM model identifier (for strategies that use LLM).
        chat_model : Any | None
            LLM provider for strategies that need LLM (e.g. llm_decides).
        """
        ...


class CallAllStrategy(OrchidToolCallStrategy):
    """Call every tool in the list concurrently and collect results."""

    async def execute(
        self,
        client,
        tools,
        query,
        auth,
        *,
        agent_name="",
        **kwargs,
    ) -> dict[str, Any]:
        async def _call_one(tool):
            try:
                args = {"query": query, **tool.arguments}
                result = await client.call_tool(tool.name, args, auth)
                return tool.name, result.text
            except Exception as exc:
                # Fault-isolation boundary: MCP tool calls cross a
                # network/process boundary and can raise arbitrary
                # exceptions (McpError, httpx.HTTPStatusError,
                # AttributeError from buggy implementations, etc.).
                # We catch broadly here to prevent one failing tool
                # from crashing the entire agent.
                logger.error("[%s] Tool '%s' failed: %s", agent_name, tool.name, exc)
                return f"{tool.name}_error", str(exc)

        pairs = await asyncio.gather(*(_call_one(t) for t in tools))
        return dict(pairs)


class SequentialStrategy(OrchidToolCallStrategy):
    """Call tools in order, passing previous results as context."""

    async def execute(
        self,
        client,
        tools,
        query,
        auth,
        *,
        agent_name="",
        **kwargs,
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        previous_results: dict[str, Any] = {}

        for tool in tools:
            try:
                args: dict[str, Any] = {"query": query, **tool.arguments}
                if previous_results:
                    args["previous_results"] = json.dumps(previous_results, default=str)
                result = await client.call_tool(tool.name, args, auth)
                results[tool.name] = result.text
                previous_results[tool.name] = result.text
            except Exception as exc:
                logger.error("[%s] Sequential tool '%s' failed: %s", agent_name, tool.name, exc)
                results[f"{tool.name}_error"] = str(exc)

        return results


class LLMDecidesStrategy(OrchidToolCallStrategy):
    """Choose tools through the ``tool_choice`` decision point.

    The point answers one bounded boolean question per tool that is already
    shortlisted for this call (the ``tools`` argument). Only selected tools are
    executed, so an answer can never widen the allowlist. Every shortlisted
    tool question is required: a partial answer is treated as uncertain and
    falls back to ``all`` rather than executing an inferred subset.

    Decision answers carry names, never arguments. Arguments are derived
    deterministically as ``{"query": query, **tool.arguments}`` — the same
    values the ``all`` strategy sends — and a tool whose input schema requires
    an argument that cannot be derived this way is skipped instead of being
    called with an invented value. When no selected tool is safe to call, the
    strategy falls back to ``all`` so the historical behaviour is preserved.

    Any uncertainty or failure — no binding, no candidates, a missing answer,
    low confidence, refusal, an invalid answer, a timeout, a provider error, or
    an unexpected exception — falls back to the configured ``all`` strategy. A
    complete, confident "call no tools" answer is a valid decision and executes
    nothing. Every evaluated decision emits exactly one structured line that
    carries no tool payloads, arguments, or user text.
    """

    def __init__(
        self,
        *,
        binding: OrchidDecisionPointBinding | None = None,
        timeout: float | None = _MAX_DECISION_SECONDS,
    ) -> None:
        if timeout is not None and timeout <= 0:
            raise ValueError("tool choice timeout must be positive or None")
        self._binding = binding
        self._timeout = timeout

    async def execute(
        self,
        client,
        tools,
        query,
        auth,
        *,
        agent_name="",
        server_config=None,
        llm_model=None,
        chat_model=None,
        tool_choice_binding: OrchidDecisionPointBinding | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        binding = tool_choice_binding or self._binding or _tool_choice_binding_var.get()
        if binding is None:
            logger.info("[%s] tool_choice is not configured; using the all strategy", agent_name)
            return await self._call_all(client, tools, query, auth, agent_name)

        candidates = self._candidates(tools)
        if not candidates:
            return await self._fallback(
                client,
                tools,
                query,
                auth,
                agent_name=agent_name,
                binding=binding,
                reason="no_candidates",
            )

        cap = binding.max_candidates
        if cap is not None and len(candidates) > cap:
            return await self._fallback(
                client,
                tools,
                query,
                auth,
                agent_name=agent_name,
                binding=binding,
                reason="too_many_candidates",
            )

        # Schemas are needed to prove a selected tool can be called with the
        # deterministic argument set. Without them the strategy cannot honour
        # the "never invent arguments" rule, so it falls back to today's path.
        listed = await self._discover_tools(client, auth, agent_name)
        if listed is None:
            return await self._fallback(
                client,
                tools,
                query,
                auth,
                agent_name=agent_name,
                binding=binding,
                reason="schema_unavailable",
            )

        questions: list[OrchidDecisionQuestion] = []
        tool_by_question: dict[str, OrchidToolConfig] = {}
        for tool in candidates:
            question_name = f"{_QUESTION_PREFIX}{tool.name}"
            tool_by_question[question_name] = tool
            instructions = f"Should the {tool.name} tool be used for this request?"
            description = str(listed.get(tool.name, {}).get("description") or "").strip()
            if description:
                instructions = f"{instructions} {description[:_MAX_DESCRIPTION_CHARS]}"
            questions.append(OrchidDecisionQuestion(name=question_name, kind="boolean", instructions=instructions))

        state_text = query[:_MAX_QUERY_CHARS] if isinstance(query, str) else str(query)[:_MAX_QUERY_CHARS]
        state = OrchidDecisionState(
            text=state_text,
            payload={"agent": agent_name, "tools": [tool.name for tool in candidates]},
        )

        started = time.perf_counter()
        try:
            async with asyncio.timeout(self._timeout):
                result = await binding.decide(state, questions, timeout=self._timeout)
        except OrchidDecisionError as exc:
            return await self._fallback(
                client,
                tools,
                query,
                auth,
                agent_name=agent_name,
                binding=binding,
                reason=exc.code or "provider_error",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                provider=exc.provider,
            )
        except TimeoutError:
            return await self._fallback(
                client,
                tools,
                query,
                auth,
                agent_name=agent_name,
                binding=binding,
                reason="timeout",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:
            # Only the exception type crosses this seam: a provider message can
            # contain user state and must not reach the logs.
            logger.warning("[%s] tool_choice provider raised %s", agent_name, type(exc).__name__)
            return await self._fallback(
                client,
                tools,
                query,
                auth,
                agent_name=agent_name,
                binding=binding,
                reason="provider_error",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        provider = result.provider or binding.provider
        question_names = list(tool_by_question)
        # Every shortlisted tool question is required. A partial answer cannot
        # be trusted to mean "do not call the omitted tool", so validate the
        # whole answer set before accepting any selection.
        reason, confidence = self._answer_problem(result, question_names, binding.confidence_threshold)
        if reason is not None:
            return await self._fallback(
                client,
                tools,
                query,
                auth,
                agent_name=agent_name,
                binding=binding,
                reason=reason,
                latency_ms=latency_ms,
                confidence=confidence,
                provider=provider,
            )

        selected = aggregate_shortlist(result, question_names, threshold=binding.confidence_threshold)
        if selected is None:
            # Every shortlisted tool was answered with a confident "no": that
            # is a valid direct answer, not a fallback. Execute no tools and
            # let the agent answer from its own context.
            log_decision(
                point=_POINT,
                path="decision",
                provider=provider,
                latency_ms=latency_ms,
                confidence=confidence,
                outcome="accepted",
            )
            logger.info("[%s] tool_choice selected no tools (confident decision)", agent_name)
            return {}

        executable: list[tuple[str, dict[str, Any]]] = []
        for question_name in selected:
            tool = tool_by_question[question_name]
            arguments = {"query": query, **tool.arguments}
            entry = listed.get(tool.name)
            if entry is None:
                logger.warning("[%s] skipping tool %r: its input schema is unavailable", agent_name, tool.name)
                continue
            missing = self._missing_required(entry.get("schema"), arguments)
            if missing:
                logger.warning(
                    "[%s] skipping tool %r: required argument(s) cannot be derived deterministically (%s)",
                    agent_name,
                    tool.name,
                    ", ".join(missing),
                )
                continue
            executable.append((tool.name, arguments))

        if not executable:
            return await self._fallback(
                client,
                tools,
                query,
                auth,
                agent_name=agent_name,
                binding=binding,
                reason="unsafe_arguments",
                latency_ms=latency_ms,
                confidence=confidence,
                provider=provider,
            )

        log_decision(
            point=_POINT,
            path="decision",
            provider=provider,
            latency_ms=latency_ms,
            confidence=confidence,
            outcome="accepted",
        )
        logger.info(
            "[%s] tool_choice selected %d of %d shortlisted tool(s)",
            agent_name,
            len(executable),
            len(candidates),
        )
        if len(executable) < len(selected):
            logger.info(
                "[%s] tool_choice skipped %d selected tool(s) with underivable required arguments",
                agent_name,
                len(selected) - len(executable),
            )

        results: dict[str, Any] = {}
        for tool_name, arguments in executable:
            try:
                result_value = await client.call_tool(tool_name, arguments, auth)
                results[tool_name] = result_value.text
            except Exception as exc:
                # Fault-isolation boundary: one failing tool must not abort the
                # other selected tools or the agent run.
                logger.error("[%s] tool_choice tool '%s' failed: %s", agent_name, tool_name, exc)
                results[f"{tool_name}_error"] = str(exc)
        return results

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _candidates(tools: list[OrchidToolConfig]) -> list[OrchidToolConfig]:
        """Return the shortlist in order, dropping unnamed and duplicate tools."""
        candidates: list[OrchidToolConfig] = []
        seen: set[str] = set()
        for tool in tools:
            if not tool.name or tool.name in seen:
                continue
            seen.add(tool.name)
            candidates.append(tool)
        return candidates

    @staticmethod
    async def _discover_tools(
        client: OrchidMCPDiscoverable,
        auth: OrchidAuthContext,
        agent_name: str,
    ) -> dict[str, Mapping[str, Any]] | None:
        """Return tool metadata keyed by name, or ``None`` when discovery fails."""
        try:
            listed = await client.list_tools(auth)
        except Exception as exc:
            logger.warning("[%s] tool_choice could not list tool schemas: %s", agent_name, type(exc).__name__)
            return None
        by_name: dict[str, Mapping[str, Any]] = {}
        for entry in listed or []:
            if isinstance(entry, Mapping) and isinstance(entry.get("name"), str):
                by_name[entry["name"]] = entry
        return by_name

    @staticmethod
    def _missing_required(schema: Any, arguments: Mapping[str, Any]) -> list[str]:
        """Return required schema arguments the deterministic set cannot supply."""
        if not isinstance(schema, Mapping):
            return []
        required = schema.get("required")
        if not isinstance(required, list):
            return []
        return [name for name in required if isinstance(name, str) and name and name not in arguments]

    @staticmethod
    def _answer_problem(
        result: OrchidDecisionResult,
        question_names: list[str],
        threshold: float,
    ) -> tuple[str | None, float | None]:
        """Return why the answer set is unusable, or ``None`` when it is complete.

        Every shortlisted tool question is required. A missing/partial answer,
        a refusal, a non-boolean value, or any answer below the confidence
        threshold makes the whole result unusable and triggers the ``all``
        fallback. Only a complete, confident answer set may select a subset —
        or confidently select no tools.
        """
        answers = [result.answers.get(name) for name in question_names]
        present = [answer for answer in answers if answer is not None]
        confidence = min((answer.confidence for answer in present), default=None)
        if not present:
            return "invalid_response", None
        if any(answer.refusal is not None for answer in present):
            return "refusal", confidence
        if any(answer.kind != "boolean" for answer in present):
            return "invalid_response", confidence
        if len(present) != len(answers):
            # A missing answer cannot be read as a confident "do not call".
            return "invalid_response", confidence
        if any(answer.confidence < threshold for answer in present):
            return "low_confidence", confidence
        return None, confidence

    async def _fallback(
        self,
        client: OrchidMCPToolCaller,
        tools: list[OrchidToolConfig],
        query: str,
        auth: OrchidAuthContext,
        *,
        agent_name: str,
        binding: OrchidDecisionPointBinding,
        reason: str,
        latency_ms: float | None = None,
        confidence: float | None = None,
        provider: str = "",
    ) -> dict[str, Any]:
        """Emit one structured fallback line and run the ``all`` strategy."""
        logger.warning("[%s] tool_choice fell back to the all strategy (reason=%s)", agent_name, reason)
        log_decision(
            point=_POINT,
            path="llm_fallback",
            provider=provider or binding.provider,
            latency_ms=latency_ms,
            confidence=confidence,
            outcome="fallback",
            fallback_reason=reason,
        )
        return await self._call_all(client, tools, query, auth, agent_name)

    @staticmethod
    async def _call_all(
        client: OrchidMCPToolCaller,
        tools: list[OrchidToolConfig],
        query: str,
        auth: OrchidAuthContext,
        agent_name: str,
    ) -> dict[str, Any]:
        return await get_strategy("all").execute(client, tools, query, auth, agent_name=agent_name)

    @staticmethod
    async def _llm_complete(
        chat_model: Any | None,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        response_format: dict[str, str] | None = None,
    ) -> str:
        """Call LLM via the injected BaseChatModel.

        Retained for integrations that subclass this strategy; the
        decision-backed path does not call a free-form LLM prompt anymore.
        """
        if not chat_model:
            raise RuntimeError("LLMDecidesStrategy requires a BaseChatModel. Pass chat_model= when building the graph.")
        kwargs: dict[str, Any] = {"temperature": temperature}
        if response_format:
            kwargs["response_format"] = response_format
        result = await chat_model.ainvoke(messages, **kwargs)
        return result.content or ""


# ── Strategy Registry ──────────────────────────────────────────

STRATEGY_REGISTRY: dict[str, type[OrchidToolCallStrategy]] = {
    "all": CallAllStrategy,
    "sequential": SequentialStrategy,
    "llm_decides": LLMDecidesStrategy,
}


def register_strategy(name: str, cls: type[OrchidToolCallStrategy]) -> None:
    """Register a custom tool call strategy by name."""
    STRATEGY_REGISTRY[name] = cls
    logger.info("[Strategies] Registered '%s' → %s", name, cls.__name__)


def clear_strategies() -> None:
    """Reset to built-in strategies (useful for test isolation)."""
    STRATEGY_REGISTRY.clear()
    STRATEGY_REGISTRY.update(
        {
            "all": CallAllStrategy,
            "sequential": SequentialStrategy,
            "llm_decides": LLMDecidesStrategy,
        }
    )


def get_strategy(name: str, **kwargs: Any) -> OrchidToolCallStrategy:
    """Look up and instantiate a tool call strategy by name.

    Falls back to whatever is registered under ``"all"`` when the name
    is unknown — this lets integrators override the default by
    re-registering ``"all"`` without editing this function.

    ``kwargs`` are forwarded to the strategy constructor; built-in
    strategies take none, while ``llm_decides`` accepts an optional
    ``binding`` and ``timeout`` for direct integrations.
    """
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        logger.warning("Unknown tool call strategy '%s', falling back to 'all'", name)
        cls = STRATEGY_REGISTRY.get("all", CallAllStrategy)
    return cls(**kwargs) if kwargs else cls()
