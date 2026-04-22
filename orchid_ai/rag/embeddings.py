"""
Embedding factory — resolves a LiteLLM-style model string to a LangChain Embeddings instance.

Strategy: provider-specific package first, LiteLLM fallback.

1. Parse the model string prefix (``openai/text-embedding-3-small`` → provider=openai)
2. Try the specific LangChain provider package (e.g. ``langchain-openai``)
3. Fall back to litellm-based embedding (wraps ``litellm.aembedding``)

Embeddings returned from :func:`build_embeddings` are automatically wrapped
with :class:`BatchLimitingEmbeddings` when the model belongs to a provider
with a known server-side batch cap (e.g. Gemini's
``BatchEmbedContentsRequest`` rejects > 100 inputs per call). Callers pass
arbitrarily large document lists to ``aembed_documents`` / ``embed_documents``
without needing to know the cap.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# Known dimensions per model family — used by QdrantRepository for collection creation
KNOWN_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "bedrock/amazon.titan-embed-text-v2:0": 1024,
    "ollama/nomic-embed-text": 768,
    "gemini/gemini-embedding-001": 3072,
    "gemini/gemini-embedding-2-preview": 3072,
}

# Per-provider safe batch size for ``embed_documents`` / ``aembed_documents``.
#
# Providers whose embedding APIs reject large batches have an entry here;
# callers using :func:`build_embeddings` get transparent chunking below
# the declared size.  Unlisted providers tolerate arbitrarily large
# batches and are returned unwrapped.
#
# Keys match via :py:meth:`str.startswith`, so prefix entries end in
# ``"/"`` or ``"."`` and bare OpenAI model names (no provider prefix)
# are listed verbatim.  Values leave headroom below the documented
# hard limit so transient retries don't bump the ceiling.
#
# Known caps:
#   * Gemini ``BatchEmbedContentsRequest`` — 100 inputs per call.
#   * Cohere ``embed`` v3 (direct + Bedrock-proxied) — 96 inputs per call.
#   * Voyage ``voyage-2`` / ``voyage-3`` — 128 inputs per call.
#   * OpenAI ``text-embedding-3-*`` / ``ada-002`` — 2048 inputs per call.
#
# Not listed (intentional):
#   * ``ollama/*`` — local, no documented server-side batch cap.
#   * ``bedrock/amazon.titan-*`` — single-input API; ``langchain-aws``
#     already loops one text at a time, so wrapping adds no value.
_PROVIDER_BATCH_LIMITS: dict[str, int] = {
    # Prefix-matched providers.
    "gemini/": 80,
    "google/": 80,
    "cohere/": 80,
    "voyage/": 100,
    "bedrock/cohere.": 80,
    # OpenAI bare model names (the SDK keys off the model, not a prefix).
    "text-embedding-3-small": 2000,
    "text-embedding-3-large": 2000,
    "text-embedding-ada-002": 2000,
}

# ── Provider prefix → (package, class_name, strip_prefix) ──

_PROVIDER_MAP: list[tuple[str, str, str, str | None]] = [
    ("ollama/", "langchain_ollama", "OllamaEmbeddings", "ollama/"),
    ("gemini/", "langchain_google_genai", "GoogleGenerativeAIEmbeddings", "gemini/"),
    ("google/", "langchain_google_genai", "GoogleGenerativeAIEmbeddings", "google/"),
]

_PROVIDER_API_KEY_ENV: dict[str, str] = {
    "gemini/": "GEMINI_API_KEY",
    "google/": "GEMINI_API_KEY",
}

_PROVIDER_API_BASE_ENV: dict[str, str] = {
    "ollama/": "OLLAMA_API_BASE",
}


def get_embedding_dimension(model: str) -> int:
    """Return the known dimension for a model, or 1536 as default."""
    return KNOWN_DIMS.get(model, 1536)


def get_embedding_batch_size(model: str) -> int | None:
    """Return the safe batch size for a model, or ``None`` if unbounded.

    Used by :func:`build_embeddings` to detect providers whose embedding
    APIs reject requests above a server-side cap so large indexing jobs
    are transparently chunked inside :class:`BatchLimitingEmbeddings`.
    """
    for prefix, limit in _PROVIDER_BATCH_LIMITS.items():
        if model.startswith(prefix):
            return limit
    return None


def build_embeddings(model: str, **kwargs: Any) -> Embeddings:
    """
    Build a LangChain ``Embeddings`` instance from a model string.

    Tries provider-specific packages first, falls back to a LiteLLM-based
    wrapper.  When the model belongs to a provider with a known batch
    cap (see :data:`_PROVIDER_BATCH_LIMITS`) the result is wrapped in
    :class:`BatchLimitingEmbeddings` so oversize document batches are
    split transparently.

    Parameters
    ----------
    model : str
        LiteLLM-format model identifier (e.g. ``"text-embedding-3-small"``,
        ``"ollama/nomic-embed-text"``, ``"gemini/gemini-embedding-001"``).

    Returns
    -------
    Embeddings
        A ready-to-use LangChain embeddings instance.
    """
    # Try provider-specific package first
    for prefix, module_path, class_name, strip_prefix in _PROVIDER_MAP:
        if not model.startswith(prefix):
            continue

        try:
            import importlib

            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)

            model_name = model[len(strip_prefix) :] if strip_prefix else model
            provider_kwargs = _resolve_provider_kwargs(prefix, model_name, **kwargs)

            instance = cls(**provider_kwargs)
            logger.info("[Embeddings] Using %s.%s for model '%s'", module_path, class_name, model)
            return _maybe_wrap_with_batch_limit(instance, model)
        except ImportError:
            logger.debug("[Embeddings] %s not installed, trying fallback for '%s'", module_path, model)
            continue
        except Exception as exc:
            logger.debug("[Embeddings] Failed to create %s for '%s': %s", class_name, model, exc)
            continue

    # Fallback: OpenAI-compatible embeddings (covers text-embedding-3-small etc.)
    # or LiteLLM-based wrapper
    return _maybe_wrap_with_batch_limit(_build_fallback_embeddings(model, **kwargs), model)


def _maybe_wrap_with_batch_limit(embeddings: Embeddings, model: str) -> Embeddings:
    """Wrap ``embeddings`` in :class:`BatchLimitingEmbeddings` when the model has a known cap."""
    batch_size = get_embedding_batch_size(model)
    if batch_size is None:
        return embeddings
    logger.debug(
        "[Embeddings] Wrapping '%s' with BatchLimitingEmbeddings(batch_size=%d)",
        model,
        batch_size,
    )
    return BatchLimitingEmbeddings(embeddings, batch_size)


def _resolve_provider_kwargs(prefix: str, model_name: str, **extra: Any) -> dict[str, Any]:
    """Build constructor kwargs for a provider-specific Embeddings class."""
    provider_kwargs: dict[str, Any] = {"model": model_name}

    env_key = _PROVIDER_API_KEY_ENV.get(prefix)
    if env_key:
        api_key = os.environ.get(env_key, "")
        if api_key:
            if prefix in ("gemini/", "google/"):
                provider_kwargs["google_api_key"] = api_key
            else:
                provider_kwargs["api_key"] = api_key

    env_base = _PROVIDER_API_BASE_ENV.get(prefix)
    if env_base:
        api_base = os.environ.get(env_base, "")
        if api_base:
            provider_kwargs["base_url"] = api_base

    provider_kwargs.update(extra)
    return provider_kwargs


def _build_fallback_embeddings(model: str, **kwargs: Any) -> Embeddings:
    """Build embeddings using OpenAI-compatible or LiteLLM fallback."""
    # Try langchain-openai for bare model names (text-embedding-3-small etc.)
    if "/" not in model:
        try:
            from langchain_openai import OpenAIEmbeddings

            api_key = os.environ.get("OPENAI_API_KEY", "")
            instance = OpenAIEmbeddings(model=model, **({"api_key": api_key} if api_key else {}), **kwargs)
            logger.info("[Embeddings] Using OpenAIEmbeddings for model '%s'", model)
            return instance
        except ImportError:
            pass

    # Final fallback: LiteLLM-based async wrapper
    from ..llm import get_llm_kwargs

    litellm_kwargs = get_llm_kwargs(model)

    class _LiteLLMEmbeddings(Embeddings):
        """Thin wrapper around litellm.aembedding for LangChain compatibility."""

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            import asyncio

            return asyncio.get_event_loop().run_until_complete(self.aembed_documents(texts))

        def embed_query(self, text: str) -> list[float]:
            import asyncio

            return asyncio.get_event_loop().run_until_complete(self.aembed_query(text))

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            import litellm

            response = await litellm.aembedding(model=model, input=texts, **litellm_kwargs)
            return [item["embedding"] for item in response.data]

        async def aembed_query(self, text: str) -> list[float]:
            result = await self.aembed_documents([text])
            return result[0]

    logger.info("[Embeddings] Using LiteLLM fallback for model '%s'", model)
    return _LiteLLMEmbeddings()


class BatchLimitingEmbeddings(Embeddings):
    """Embeddings wrapper that splits large document batches into sub-batches.

    Some providers (notably Gemini's ``BatchEmbedContentsRequest``) reject
    requests exceeding a server-side cap.  This wrapper splits the input
    text list into chunks of at most ``batch_size`` before delegating to
    the inner provider and concatenates the results, preserving order.

    Query-embedding methods (:meth:`embed_query`, :meth:`aembed_query`)
    pass through unchanged — single-text requests are never over cap.
    Batches smaller than ``batch_size`` are forwarded without copying.
    """

    def __init__(self, inner: Embeddings, batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._inner = inner
        self._batch_size = batch_size

    @property
    def inner(self) -> Embeddings:
        return self._inner

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if len(texts) <= self._batch_size:
            return self._inner.embed_documents(texts)
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            out.extend(self._inner.embed_documents(texts[i : i + self._batch_size]))
        return out

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if len(texts) <= self._batch_size:
            return await self._inner.aembed_documents(texts)
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            out.extend(await self._inner.aembed_documents(texts[i : i + self._batch_size]))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)

    async def aembed_query(self, text: str) -> list[float]:
        return await self._inner.aembed_query(text)
