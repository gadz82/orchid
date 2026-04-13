"""
Guardrail abstractions — input/output firewalls for agents and orchestrator.

This module defines the **core contracts** for the guardrail system.
All classes here use ONLY Python stdlib types (zero external dependencies),
following the same rule as the rest of ``core/``.

Architecture (3-tier):

  User Input
       │
       ▼
  ┌─────────────────────┐
  │ Global Input Rails   │  content safety, prompt injection, PII, max length
  └──────────┬──────────┘
             ▼
  ┌─────────────────────┐
  │ Supervisor (routing) │
  └──────────┬──────────┘
       ┌─────┼─────┐
       ▼     ▼     ▼
    Agent A  B  C         Per-agent input/output rails
       │     │     │
       ▼     ▼     ▼
  ┌─────────────────────┐
  │ Global Output Rails  │  PII redaction, content safety, groundedness
  └──────────┬──────────┘
             ▼
       User Response

Guardrails are configured via YAML (``guardrails:`` key in orchid.yml
and per-agent in agents.yaml) and instantiated via a registry, following
the same Open/Closed pattern as ``ToolCallStrategy``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Enums ────────────────────────────────────────────────────


class GuardrailAction(Enum):
    """Action to take when a guardrail check fails."""

    ALLOW = "allow"
    BLOCK = "block"
    REDACT = "redact"
    WARN = "warn"
    LOG = "log"


class GuardrailDirection(Enum):
    """Whether this guardrail runs on input or output."""

    INPUT = "input"
    OUTPUT = "output"


# ── Data classes ─────────────────────────────────────────────


@dataclass(frozen=True)
class GuardrailContext:
    """
    Contextual information passed to every guardrail check.

    Guardrails receive this so they can make scope-aware decisions
    (e.g. different rules per tenant, per agent, per direction).
    """

    direction: GuardrailDirection
    agent_name: str = ""  # empty for global rails
    tenant_key: str = "default"
    user_id: str = ""
    chat_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardrailResult:
    """
    Outcome of a single guardrail check.

    When ``triggered`` is True, the guardrail matched and the ``action``
    field indicates what should happen.  ``message`` is the user-facing
    explanation (shown when blocking).  ``redacted_content`` replaces
    the original content when ``action == REDACT``.
    """

    triggered: bool
    action: GuardrailAction = GuardrailAction.ALLOW
    guardrail_name: str = ""
    message: str = ""
    redacted_content: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        """Convenience: True when the guardrail blocked the content."""
        return self.triggered and self.action == GuardrailAction.BLOCK

    @staticmethod
    def passed(guardrail_name: str = "") -> GuardrailResult:
        """Factory for a clean pass result."""
        return GuardrailResult(triggered=False, guardrail_name=guardrail_name)


# ── ABCs ─────────────────────────────────────────────────────


class Guardrail(ABC):
    """
    Abstract base for all guardrails.

    A guardrail inspects content (input or output) and returns a
    ``GuardrailResult`` indicating whether to allow, block, redact, or warn.

    Implementations must be stateless — all context comes via parameters.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this guardrail type (e.g. 'content_safety')."""
        ...

    @abstractmethod
    async def check(
        self,
        content: str,
        context: GuardrailContext,
    ) -> GuardrailResult:
        """
        Evaluate content against this guardrail's rules.

        Parameters
        ----------
        content : str
            The text to evaluate (user message or agent response).
        context : GuardrailContext
            Scope information (direction, agent, tenant, etc.).

        Returns
        -------
        GuardrailResult
            The outcome — ``triggered=False`` means the content is clean.
        """
        ...


class GuardrailChain:
    """
    Ordered sequence of guardrails with short-circuit-on-block semantics.

    Runs each guardrail in order.  If any guardrail returns ``BLOCK``,
    evaluation stops immediately.  ``REDACT`` actions are accumulated
    (the redacted content is passed to subsequent guardrails).
    ``WARN`` and ``LOG`` are collected but don't stop evaluation.
    """

    def __init__(self, guardrails: list[Guardrail] | None = None) -> None:
        self._guardrails: list[Guardrail] = guardrails or []

    def add(self, guardrail: Guardrail) -> None:
        """Append a guardrail to the chain."""
        self._guardrails.append(guardrail)

    @property
    def guardrails(self) -> list[Guardrail]:
        """Read-only access to the guardrail list."""
        return list(self._guardrails)

    @property
    def empty(self) -> bool:
        """True when the chain has no guardrails."""
        return len(self._guardrails) == 0

    async def evaluate(
        self,
        content: str,
        context: GuardrailContext,
    ) -> GuardrailResult:
        """
        Run all guardrails in sequence.

        Returns the first ``BLOCK`` result, the last ``REDACT`` result,
        or a ``passed`` result if nothing was triggered.
        """
        current_content = content
        last_redact: GuardrailResult | None = None
        warnings: list[GuardrailResult] = []

        for guardrail in self._guardrails:
            result = await guardrail.check(current_content, context)

            if not result.triggered:
                continue

            if result.action == GuardrailAction.BLOCK:
                return result

            if result.action == GuardrailAction.REDACT and result.redacted_content is not None:
                current_content = result.redacted_content
                last_redact = result

            if result.action in (GuardrailAction.WARN, GuardrailAction.LOG):
                warnings.append(result)

        # Return the last redaction if any, otherwise clean pass
        if last_redact is not None:
            return last_redact

        if warnings:
            # Return the first warning (caller can decide what to do)
            return warnings[0]

        return GuardrailResult.passed()

    def __len__(self) -> int:
        return len(self._guardrails)

    def __repr__(self) -> str:
        names = [g.name for g in self._guardrails]
        return f"GuardrailChain({names})"
