from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class PromptNormalizer(ABC):
    """Strategy: reformulates a prompt before it is sent to an external CLI."""

    @abstractmethod
    async def normalize(self, prompt: str, *, context: dict[str, Any] | None = None) -> str: ...


class PassthroughNormalizer(PromptNormalizer):
    """Returns the prompt unchanged."""

    async def normalize(self, prompt: str, *, context: dict[str, Any] | None = None) -> str:
        return prompt


class LLMNormalizer(PromptNormalizer):
    """Reformulates the prompt via an LLM call (BaseChatModel — DIP)."""

    def __init__(self, chat_model: BaseChatModel, *, instruction: str) -> None:
        self._chat_model = chat_model
        self._instruction = instruction

    async def normalize(self, prompt: str, *, context: dict[str, Any] | None = None) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=self._instruction),
            HumanMessage(content=prompt),
        ]
        result = await self._chat_model.ainvoke(messages)
        return str(result.content or "")


_NORMALIZER_REGISTRY: dict[str, type[PromptNormalizer]] = {
    "passthrough": PassthroughNormalizer,
}


def register_normalizer(name: str, cls: type[PromptNormalizer]) -> None:
    _NORMALIZER_REGISTRY[name] = cls


def get_normalizer(name: str) -> PromptNormalizer:
    cls = _NORMALIZER_REGISTRY.get(name)
    if cls is not None:
        try:
            return cls()
        except TypeError:
            logger.warning("[Normalizer] '%s' cannot be instantiated without args; falling back to passthrough", name)
            return PassthroughNormalizer()
    logger.warning("[Normalizer] unknown name '%s'; falling back to passthrough", name)
    return PassthroughNormalizer()
