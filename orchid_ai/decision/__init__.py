"""Decision-model providers, transport, factory, and shared validation.

The public extension surface is :class:`~orchid_ai.core.decision.OrchidDecisionModel`;
everything here is a built-in implementation or plumbing. The top-level
``orchid_ai`` package intentionally does not re-export the adapters.
"""

from __future__ import annotations

from .answers import MISSING, ErrorFactory, finite_number
from .factory import (
    DecisionBuildContext,
    build_decision_model,
    build_decision_models,
    register_decision_protocol,
)
from .http import DecisionBreaker, DecisionHttpClient
from .llm import LLMDecisionModel
from .openai_compatible import OpenAICompatibleDecisionModel
from .state_questions import StateQuestionsDecisionModel
from .stub import StubDecisionModel

__all__ = [
    "MISSING",
    "DecisionBreaker",
    "DecisionBuildContext",
    "DecisionHttpClient",
    "ErrorFactory",
    "LLMDecisionModel",
    "OpenAICompatibleDecisionModel",
    "StateQuestionsDecisionModel",
    "StubDecisionModel",
    "build_decision_model",
    "build_decision_models",
    "finite_number",
    "register_decision_protocol",
]
