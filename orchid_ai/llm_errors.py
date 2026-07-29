"""
Shared LLM error handling — maps exceptions to user-friendly messages.

Used by the agentic loop and supervisor to produce consistent error
responses for transient LLM failures (503, rate limits, timeouts).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def format_llm_error(exc: Exception, *, context: str = "") -> str:
    """Convert an LLM API exception to a user-friendly error message.

    Parameters
    ----------
    exc : Exception
        The exception raised by the LLM call.
    context : str
        Optional context prefix for logging (e.g. agent name).

    Returns
    -------
    str
        A user-friendly error message.
    """
    error_msg = str(exc)
    error_lower = error_msg.lower()

    if context:
        logger.error("[%s] LLM API error: %s", context, error_msg)
    else:
        logger.error("LLM API error: %s", error_msg)

    if "503" in error_msg or "high demand" in error_lower:
        return "Currently experiencing high demand. Please try again shortly."

    if "rate limit" in error_lower:
        return "Rate limit reached. Please try again in a few moments."

    return f"Error processing request: {error_msg[:200]}. Please try again later."
