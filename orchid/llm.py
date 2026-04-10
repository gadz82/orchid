"""
LLM utilities — provider-agnostic API key resolution for LiteLLM.

Every ``litellm.acompletion()`` / ``litellm.aembedding()`` call in the
project should include ``**get_llm_kwargs(model)`` so the correct API
key and base URL are always forwarded, regardless of the provider.

Supported providers:
  - ``gemini/``      → Google AI Studio  (``GEMINI_API_KEY``)
  - ``groq/``        → Groq              (``GROQ_API_KEY``)
  - ``anthropic/``   → Anthropic         (``ANTHROPIC_API_KEY``)
  - ``claude-``      → Anthropic         (``ANTHROPIC_API_KEY``)
  - ``openai/``      → OpenAI            (``OPENAI_API_KEY``)
  - ``ollama/``      → Local Ollama      (``OLLAMA_API_BASE``)
  - ``bedrock/``     → AWS Bedrock       (uses AWS credentials)
  - ``vertex_ai/``   → GCP Vertex AI     (uses GCP credentials)

Adding a new provider is one line in ``_PREFIX_TO_API_KEY_ENV``.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# ── Provider → API key env var ───────────────────────────────
# Order matters: longer prefixes first to avoid partial matches.

_PREFIX_TO_API_KEY_ENV: list[tuple[str, str]] = [
    ("gemini/", "GEMINI_API_KEY"),
    ("groq/", "GROQ_API_KEY"),
    ("anthropic/", "ANTHROPIC_API_KEY"),
    ("claude-", "ANTHROPIC_API_KEY"),
    ("openai/", "OPENAI_API_KEY"),
    ("deepseek/", "DEEPSEEK_API_KEY"),
    ("mistral/", "MISTRAL_API_KEY"),
    ("cohere/", "COHERE_API_KEY"),
    ("together_ai/", "TOGETHERAI_API_KEY"),
]

# ── Provider → API base URL env var ──────────────────────────

_PREFIX_TO_API_BASE_ENV: list[tuple[str, str]] = [
    ("ollama/", "OLLAMA_API_BASE"),
    ("ollama_chat/", "OLLAMA_API_BASE"),
]


def get_llm_kwargs(model: str) -> dict[str, str]:
    """
    Resolve provider-specific kwargs for a LiteLLM call.

    Returns a dict with optional ``api_key`` and/or ``api_base`` that
    should be spread into ``litellm.acompletion()`` / ``aembedding()``:

        await litellm.acompletion(model=model, messages=msgs, **get_llm_kwargs(model))

    If the env var isn't set, the key is omitted — LiteLLM's own fallback
    logic (env vars, litellm.api_key, etc.) still applies.
    """
    kwargs: dict[str, str] = {}

    for prefix, env_var in _PREFIX_TO_API_KEY_ENV:
        if model.startswith(prefix):
            api_key = os.environ.get(env_var, "")
            if api_key:
                kwargs["api_key"] = api_key
            break

    for prefix, env_var in _PREFIX_TO_API_BASE_ENV:
        if model.startswith(prefix):
            api_base = os.environ.get(env_var, "")
            if api_base:
                kwargs["api_base"] = api_base
            break

    return kwargs


# ── Bare model name handling ─────────────────────────────────
# Models without a prefix (e.g. "text-embedding-3-small") are assumed
# to be OpenAI.  LiteLLM handles this by default, but if there's no
# OPENAI_API_KEY the call will fail with a confusing Vertex AI error
# because google-generativeai is installed.  We detect this early.

def _resolve_bare_model_api_key() -> str:
    """Return the OpenAI API key for bare model names."""
    return os.environ.get("OPENAI_API_KEY", "")


def get_embedding_kwargs(model: str) -> dict[str, str]:
    """
    Like ``get_llm_kwargs`` but specifically for embedding calls.

    Handles the edge case where bare model names (e.g.
    ``text-embedding-3-small``) need an explicit OpenAI key.
    """
    kwargs = get_llm_kwargs(model)

    # Bare model names (no prefix) → OpenAI
    if "/" not in model and "-" not in model[:4] and "api_key" not in kwargs:
        api_key = _resolve_bare_model_api_key()
        if api_key:
            kwargs["api_key"] = api_key

    return kwargs
