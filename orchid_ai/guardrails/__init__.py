"""
Built-in guardrail implementations and registry.

Guardrails are the firewall layer for agent input and output.
They are configured via YAML and instantiated via the registry,
following the same Open/Closed pattern as tool call strategies.

Usage::

    from orchid_ai.guardrails import get_guardrail, build_guardrail_chain
    from orchid_ai.core.guardrails import GuardrailContext, GuardrailDirection

    chain = build_guardrail_chain([
        {"type": "max_length", "fail_action": "block", "config": {"max_characters": 5000}},
        {"type": "content_safety", "fail_action": "block"},
    ])
    result = await chain.evaluate("user message", context)
"""

from __future__ import annotations

from .registry import build_guardrail_chain, get_guardrail, register_guardrail

__all__ = [
    "build_guardrail_chain",
    "get_guardrail",
    "register_guardrail",
]
