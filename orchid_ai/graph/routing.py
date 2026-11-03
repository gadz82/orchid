"""Routing strategies for the supervisor.

A routing strategy returns an :class:`OrchidRoutingDecision` or ``None``; the
supervisor owns validation and expansion. The decision strategy uses a
configured decision provider on a cheap state projection, and the existing LLM
path remains the fallback when no confident decision is available.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from langchain_core.messages import BaseMessage

from ..config.schema import OrchidAgentsConfig
from ..core.agent import OrchidAgent
from ..core.decision import (
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
    "OrchidRoutingStrategy",
    "build_routing_strategy",
]

_AGENT_PREFIX = "agent:"
_EXECUTION_QUESTION = "execution"
_SKILL_QUESTION = "skill"
_MAX_EXCERPT_CHARS = 1200


class OrchidRoutingStrategy(ABC):
    """Resolve a routing decision, or return ``None`` for the fallback path."""

    @abstractmethod
    async def route(
        self,
        *,
        state: dict[str, Any],
        agent_descriptions: dict[str, str],
        orchestrator_skills: dict[str, Any] | None = None,
        auth: OrchidAuthContext | None = None,
    ) -> OrchidRoutingDecision | None:
        """Return a validated-ready decision, or ``None`` to fall back."""
        raise NotImplementedError


class DecisionRoutingStrategy(OrchidRoutingStrategy):
    """Route with a decision provider over a cheap conversation projection.

    One boolean question per candidate agent, one execution-mode choice, and an
    optional skill choice. Low confidence on the agent set returns ``None`` so
    the supervisor's LLM routing runs; a confident skill selection short-circuits
    agent routing.
    """

    def __init__(
        self,
        *,
        provider: OrchidDecisionModel,
        confidence_threshold: float = 0.75,
        provider_name: str = "",
    ) -> None:
        self._provider = provider
        self._threshold = max(0.0, min(1.0, confidence_threshold))
        self._provider_name = provider_name

    @property
    def metadata(self) -> dict[str, Any]:
        """Metadata recorded in ``GraphState.routing_metadata``."""
        return {"provider": self._provider_name, "threshold": self._threshold}

    async def route(
        self,
        *,
        state: dict[str, Any],
        agent_descriptions: dict[str, str],
        orchestrator_skills: dict[str, Any] | None = None,
        auth: OrchidAuthContext | None = None,
    ) -> OrchidRoutingDecision | None:
        questions = [
            OrchidDecisionQuestion(
                name=f"{_AGENT_PREFIX}{name}",
                kind="boolean",
                instructions=f"Should the {name} agent handle this request? {description}",
                required=False,
            )
            for name, description in agent_descriptions.items()
        ]
        questions.append(
            OrchidDecisionQuestion(
                name=_EXECUTION_QUESTION,
                kind="choice",
                options=("parallel", "sequential"),
                instructions="Run the selected agents in parallel or one after another?",
                required=False,
            )
        )
        skills = orchestrator_skills or {}
        if skills:
            questions.append(
                OrchidDecisionQuestion(
                    name=_SKILL_QUESTION,
                    kind="choice",
                    options=("none", *skills.keys()),
                    instructions="Is there a pre-defined workflow that matches this request?",
                    required=False,
                )
            )

        result = await self._provider.decide(_project_state(state), questions)
        return self._decision_from_result(result, agent_descriptions, skills)

    def _decision_from_result(
        self,
        result: OrchidDecisionResult,
        agent_descriptions: dict[str, str],
        skills: dict[str, Any],
    ) -> OrchidRoutingDecision | None:
        if skills:
            skill_answer = require_confidence(result, _SKILL_QUESTION, self._threshold)
            if skill_answer is not None and skill_answer.value != "none":
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
            return None

        execution_answer = require_confidence(result, _EXECUTION_QUESTION, self._threshold)
        execution = "parallel"
        if execution_answer is not None and execution_answer.value in ("parallel", "sequential"):
            execution = str(execution_answer.value)

        return OrchidRoutingDecision(
            reasoning="decision-model",
            execution=execution,
            agents=[name[len(_AGENT_PREFIX) :] for name in selected],
        )


def build_routing_strategy(
    config: OrchidAgentsConfig,
    runtime: OrchidRuntime,
) -> OrchidRoutingStrategy | None:
    """Build the routing strategy from ``decisions.routing``, if configured."""
    binding = config.decisions.resolve("routing")
    if binding is None or not binding.model:
        return None
    provider = runtime.get_decision_model(binding.model)
    if provider is None:
        return None
    return DecisionRoutingStrategy(
        provider=provider,
        confidence_threshold=binding.confidence_threshold,
        provider_name=binding.model,
    )


def _project_state(state: dict[str, Any]) -> OrchidDecisionState:
    query = OrchidAgent.extract_user_query(state)
    excerpt: list[dict[str, str]] = []
    for message in (state.get("messages") or [])[-6:]:
        content = _content_text(message)
        if not content:
            continue
        role = "assistant" if getattr(message, "type", "") == "ai" else "user"
        excerpt.append({"role": role, "content": content[:400]})
    payload = {"conversation": excerpt} if excerpt else {}
    return OrchidDecisionState(text=query[:4000], payload=payload)


def _content_text(message: Any) -> str:
    if isinstance(message, BaseMessage):
        content = message.content
    else:
        content = getattr(message, "content", "")
    if isinstance(content, list):
        return " ".join(str(part) for part in content)
    return str(content)
