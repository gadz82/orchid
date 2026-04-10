"""
Concrete LLMProvider implementation using LiteLLM.

This is the ONLY file that should import ``litellm`` for completion calls.
All other modules should depend on the ``LLMProvider`` ABC from ``core/llm_provider.py``.
"""

from __future__ import annotations

import logging
from typing import Any

from .core.llm_provider import LLMProvider
from .llm import get_llm_kwargs

logger = logging.getLogger(__name__)


class LiteLLMProvider(LLMProvider):
    """
    LLM provider backed by LiteLLM — delegates to any LLM service via a
    single unified API.

    Handles API key resolution, provider routing, and error handling.
    """

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        response_format: dict[str, str] | None = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ) -> str:
        import asyncio

        import litellm

        call_kwargs = get_llm_kwargs(model)
        call_kwargs.update(kwargs)
        if response_format:
            call_kwargs["response_format"] = response_format

        async with asyncio.timeout(timeout):
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                **call_kwargs,
            )
        return response.choices[0].message.content or ""
