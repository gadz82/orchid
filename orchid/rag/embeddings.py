"""
Embedding abstraction — decouples agents from any specific embedding provider.

Uses LiteLLM's embedding API, which supports multiple providers:
  - PoC:  ``text-embedding-3-small`` via OpenAI-compatible endpoint (or Groq)
  - Prod: ``amazon.titan-embed-text-v2:0`` via Bedrock

The concrete model is chosen via ``Settings.embedding_model``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import litellm

from ..llm import get_llm_kwargs

logger = logging.getLogger(__name__)


class Embedder(ABC):
    """Abstract embedder — produces float vectors from text."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string (convenience wrapper)."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension (needed for collection creation)."""
        ...


class LiteLLMEmbedder(Embedder):
    """
    Embedding via LiteLLM — delegates to any provider behind a single API.

    Supported model strings (examples):
      - ``text-embedding-3-small``        → OpenAI
      - ``text-embedding-3-large``        → OpenAI (3072-d)
      - ``bedrock/amazon.titan-embed-text-v2:0`` → AWS Bedrock
      - ``ollama/nomic-embed-text``       → Local Ollama
    """

    # Known dimensions per model family — extend as needed
    _KNOWN_DIMS: dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        "bedrock/amazon.titan-embed-text-v2:0": 1024,
        "ollama/nomic-embed-text": 768,
        "gemini/gemini-embedding-001": 3072,
        "gemini/gemini-embedding-2-preview": 3072,
    }

    def __init__(self, model: str, *, dimension: int | None = None):
        self._model = model
        self._dimension = dimension or self._KNOWN_DIMS.get(model, 1536)

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await litellm.aembedding(
            model=self._model,
            input=texts,
            **get_llm_kwargs(self._model),
        )
        return [item["embedding"] for item in response.data]

    async def embed_query(self, query: str) -> list[float]:
        vectors = await self.embed([query])
        return vectors[0]
