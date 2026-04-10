"""
LLM provider abstraction — keeps ``core/`` free of litellm imports.

The concrete ``LiteLLMProvider`` lives in ``src/llm.py``.
Supervisor, GenericAgent, and BaseAgent should eventually depend on this
ABC rather than calling ``litellm.acompletion()`` directly.

This module uses ONLY stdlib types — safe for ``core/``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """
    Abstract LLM completion provider.

    Wraps the details of API key resolution, model routing,
    and response format so callers don't depend on any specific
    LLM SDK.
    """

    @abstractmethod
    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        response_format: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Run an LLM completion and return the response text.

        Parameters
        ----------
        model : str
            LiteLLM-format model identifier (e.g. ``gemini/gemini-2.5-flash``).
        messages : list[dict]
            Chat messages in ``[{role, content}]`` format.
        temperature : float
            Sampling temperature (0 = deterministic).
        response_format : dict | None
            Optional response format constraint (e.g. ``{"type": "json_object"}``).
        **kwargs
            Additional provider-specific parameters (e.g. ``tools``, ``tool_choice``).

        Returns
        -------
        str
            The generated text content.
        """
        ...
