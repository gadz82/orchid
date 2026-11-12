"""Decision-model providers, transport, factory, registry, and observability.

The public extension surface is :class:`~orchid_ai.core.decision.OrchidDecisionModel`
plus the decision-point contracts (:class:`~orchid_ai.core.decision.OrchidDecisionPointSpec`,
:func:`~orchid_ai.decision.registry.register_decision_point`). Everything else
here is a built-in implementation or plumbing. The top-level ``orchid_ai``
package intentionally does not re-export the adapters.
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
from .observability import (
    DECISION_LOG_FIELDS,
    DECISION_LOGGER_NAME,
    build_decision_record,
    log_decision,
    safe_log_token,
)
from .openai_compatible import OpenAICompatibleDecisionModel
from .registry import (
    DECISION_POINT_REGISTRY,
    get_decision_point,
    iter_decision_points,
    list_decision_points,
    register_decision_point,
)
from .state_questions import StateQuestionsDecisionModel
from .stub import StubDecisionModel

__all__ = [
    "DECISION_LOGGER_NAME",
    "DECISION_LOG_FIELDS",
    "DECISION_POINT_REGISTRY",
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
    "build_decision_record",
    "finite_number",
    "get_decision_point",
    "iter_decision_points",
    "list_decision_points",
    "log_decision",
    "register_decision_point",
    "register_decision_protocol",
    "safe_log_token",
]
