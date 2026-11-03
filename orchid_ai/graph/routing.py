"""Routing strategies for the supervisor.

A routing strategy returns an :class:`OrchidRoutingDecision` or ``None``; the
supervisor owns validation and expansion. The decision strategy uses a
configured decision provider on a cheap, filtered conversation projection, and
the existing LLM path remains the fallback when no confident decision is
available.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from ..config.schema import OrchidAgentsConfig
from ..core.agent import OrchidAgent
from ..core.decision import (
    OrchidDecisionError,
    OrchidDecisionModel,
    OrchidDecisionQuestion,
    OrchidDecisionResult,
    OrchidDecisionState,
    aggregate_shortlist,
    require_confidence,
)
from ..core.state import OrchidAuthContext
from .supervisor import OrchidRoutingDecision

if TYPE_CHECKING:
    from ..runtime import OrchidRuntime

__all__ = [
    "DecisionRoutingStrategy",
    "LLMRoutingStrategy",
    "OrchidRoutingStrategy",
    "build_routing_strategy",
]

logger = logging.getLogger(__name__)
decision_logger = logging.getLogger("orchid.decision")

_AGENT_PREFIX = "agent:"
_EXECUTION_QUESTION = "execution"
_SKILL_QUESTION = "skill"
_SKILL_NONE = "__none__"
_MAX_AGENTS = 32
_MAX_QUERY_CHARS = 4000


class OrchidRoutingStrategy(ABC):
    """Resolve a routing decision, or return ``None`` for the fallback path."""

    @property
    def metadata(self) -> dict[str, Any]:
        """Per-call metadata for ``GraphState.routing_metadata``."""
        return {}

    @property
    def on_low_confidence(self) -> str:
        """Declared fallback: ``llm``, ``default``, or ``escalate``."""
        return "llm"

    @abstractmethod
    async def route(
        self,
        *,
        state: dict[str, Any],
        agent_descriptions: dict[str, str],
        orchestrator_skills: dict[str, Any] | None = None,
        auth: OrchidAuthContext | None = None,
        context: dict[str, Any] | None = None,
    ) -> OrchidRoutingDecision | None:
        """Return a validated-ready decision, or ``None`` to fall back."""
        raise NotImplementedError


class LLMRoutingStrategy(OrchidRoutingStrategy):
    """Adapter that exposes an existing LLM routing call as a strategy.

    ``decide`` receives the same keyword arguments as :meth:`route` and returns
    an ``OrchidRoutingDecision``. The supervisor's inline LLM path remains the
    default implementation; this class is the composition seam for callers that
    want the LLM strategy to be selectable like any other.
    """

    def __init__(
        self,
        decide: Callable[..., Awaitable[OrchidRoutingDecision | None]],
    ) -> None:
        self._decide = decide

    async def route(
        self,
        *,
        state: dict[str, Any],
        agent_descriptions: dict[str, str],
        orchestrator_skills: dict[str, Any] | None = None,
        auth: OrchidAuthContext | None = None,
        context: dict[str, Any] | None = None,
    ) -> OrchidRoutingDecision | None:
        return await self._decide(
            state=state,
            agent_descriptions=agent_descriptions,
            orchestrator_skills=orchestrator_skills,
            auth=auth,
            context=context,
        )


class DecisionRoutingStrategy(OrchidRoutingStrategy):
    """Route with a decision provider over a filtered conversation projection.

    # ORCHID-MEM: 01a1a1c3-74a9-7845-9a95-c351bf0ebf9a — Decision-model routing with required answers and metadata.

    One boolean question per candidate agent, one required execution-mode
    choice, and a required skill choice when orchestrator skills exist. Missing,
    refused, or low-confidence answers return ``None`` (the supervisor's LLM
    fallback) and record a stable fallback reason. Sequential multi-agent order
    is not expressible with independent boolean questions, so that case also
    falls back.
    """

    def __init__(
        self,
        *,
        provider: OrchidDecisionModel,
        confidence_threshold: float = 0.75,
        provider_name: str = "",
        on_low_confidence: str = "llm",
    ) -> None:
        self._provider = provider
        self._threshold = max(0.0, min(1.0, confidence_threshold))
        self._provider_name = provider_name
        self._on_low_confidence = on_low_confidence
        self._last: dict[str, Any] = {}

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._last)

    @property
    def confidence_threshold(self) -> float:
        return self._threshold

    @property
    def on_low_confidence(self) -> str:
        return self._on_low_confidence

    async def route(
        self,
        *,
        state: dict[str, Any],
        agent_descriptions: dict[str, str],
        orchestrator_skills: dict[str, Any] | None = None,
        auth: OrchidAuthContext | None = None,
        context: dict[str, Any] | None = None,
    ) -> OrchidRoutingDecision | None:
        self._last = {
            "provider": self._provider_name,
            "threshold": self._threshold,
            "outcome": "error",
            "fallback_reason": None,
        }
        if len(agent_descriptions) > _MAX_AGENTS:
            return self._fallback("too_many_candidates")

        skills = orchestrator_skills or {}
        questions = [
            OrchidDecisionQuestion(
                name=f"{_AGENT_PREFIX}{name}",
                kind="boolean",
                instructions=f"Should the {name} agent handle this request? {description}",
            )
            for name, description in agent_descriptions.items()
        ]
        questions.append(
            OrchidDecisionQuestion(
                name=_EXECUTION_QUESTION,
                kind="choice",
                options=("parallel", "sequential"),
                instructions="Run the selected agents in parallel or one after another?",
            )
        )
        if skills:
            skill_lines = "\n".join(f"- {name}: {getattr(skill, 'description', '')}" for name, skill in skills.items())
            questions.append(
                OrchidDecisionQuestion(
                    name=_SKILL_QUESTION,
                    kind="choice",
                    options=(_SKILL_NONE, *skills.keys()),
                    instructions=(
                        "Is there a pre-defined workflow that matches this request? "
                        f"Choose {_SKILL_NONE!r} when no workflow applies.\n{skill_lines}"
                    ),
                )
            )

        auth_hint = (context or {}).get("auth_hint", "")
        started = time.perf_counter()
        try:
            result = await self._provider.decide(
                _project_state(state, auth_hint=auth_hint),
                questions,
            )
        except OrchidDecisionError as exc:
            logger.warning("[Decision] routing provider failed code=%s", exc.code)
            return self._fallback(exc.code, latency_ms=(time.perf_counter() - started) * 1000.0)
        except Exception:
            logger.warning("[Decision] routing provider raised an unexpected error", exc_info=True)
            return self._fallback("provider_error", latency_ms=(time.perf_counter() - started) * 1000.0)

        latency_ms = (time.perf_counter() - started) * 1000.0
        self._last["latency_ms"] = latency_ms
        return self._decision_from_result(result, agent_descriptions, skills)

    def _decision_from_result(
        self,
        result: OrchidDecisionResult,
        agent_descriptions: dict[str, str],
        skills: dict[str, Any],
    ) -> OrchidRoutingDecision | None:
        if skills:
            skill_answer = require_confidence(result, _SKILL_QUESTION, self._threshold)
            if skill_answer is None:
                return self._fallback("low_confidence")
            if skill_answer.value != _SKILL_NONE:
                self._last.update(
                    {
                        "outcome": "accepted",
                        "confidence": skill_answer.confidence,
                        "skill": skill_answer.value,
                        "execution_mode": "skill",
                    }
                )
                return OrchidRoutingDecision(
                    reasoning="decision-model",
                    execution="skill",
                    agents=[],
                    skill=str(skill_answer.value),
                )

        selected = aggregate_shortlist(
            result,
            [f"{_AGENT_PREFIX}{name}" for name in agent_descriptions],
            threshold=self._threshold,
        )
        if not selected:
            return self._fallback("no_agent_selected")

        execution_answer = require_confidence(result, _EXECUTION_QUESTION, self._threshold)
        if execution_answer is None:
            return self._fallback("low_confidence")
        execution = str(execution_answer.value)
        if execution == "sequential" and len(selected) > 1:
            # Independent boolean questions cannot express dependency order;
            # let the LLM path choose and order the pipeline.
            return self._fallback("sequential_order_unsupported")

        confidence = min(result.answers[name].confidence for name in selected if name in result.answers)
        self._last.update(
            {
                "outcome": "accepted",
                "confidence": confidence,
                "agents": [name[len(_AGENT_PREFIX) :] for name in selected],
                "execution_mode": execution,
            }
        )
        return OrchidRoutingDecision(
            reasoning="decision-model",
            execution=execution,
            agents=[name[len(_AGENT_PREFIX) :] for name in selected],
        )

    def _fallback(self, reason: str, *, latency_ms: float | None = None) -> None:
        self._last.update(
            {
                "outcome": "fallback",
                "fallback_reason": reason,
            }
        )
        if latency_ms is not None:
            self._last["latency_ms"] = latency_ms


def build_routing_strategy(
    config: OrchidAgentsConfig,
    runtime: OrchidRuntime,
) -> OrchidRoutingStrategy | None:
    """Build the routing strategy from ``decisions.routing``, if configured.

    # ORCHID-MEM: 01a1a1c3-74e5-77f9-8eae-fe0a1a08092d — Routing strategy composition from config and runtime providers.
    """
    binding = config.decisions.resolve("routing")
    if binding is None or not binding.model:
        return None
    provider = runtime.get_decision_model(binding.model)
    if provider is None:
        logger.warning("[Decision] routing provider %r is configured but missing from the runtime", binding.model)
        return None
    return DecisionRoutingStrategy(
        provider=provider,
        confidence_threshold=binding.confidence_threshold,
        provider_name=binding.model,
        on_low_confidence=binding.on_low_confidence,
    )


def _project_state(state: dict[str, Any], *, auth_hint: str = "") -> OrchidDecisionState:
    query = OrchidAgent.extract_user_query(state)[:_MAX_QUERY_CHARS]
    excerpt = OrchidAgent.extract_conversation_history(state, max_turns=3, max_chars=400)
    payload: dict[str, Any] = {}
    if excerpt:
        payload["conversation"] = excerpt
    if auth_hint:
        payload["auth_hint"] = auth_hint[:1000]
    return OrchidDecisionState(text=query, payload=payload)
