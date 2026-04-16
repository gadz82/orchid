"""
LLM factory — resolves a LiteLLM-style model string to a LangChain BaseChatModel.

Strategy: provider-specific package first, ChatLiteLLM fallback.

1. Parse the model string prefix (``openai/gpt-4o`` → provider=openai)
2. Try the specific LangChain provider package (e.g. ``langchain-openai``)
3. Fall back to ``ChatLiteLLM`` from ``langchain-litellm`` (wraps litellm)

This replaces the old ``LiteLLMProvider`` and ``LLMProvider`` ABC.
All callers now receive a standard ``BaseChatModel`` that supports
``invoke()``, ``ainvoke()``, ``bind_tools()``, ``with_structured_output()``,
and the full LangChain Runnable interface.

Example::

    from orchid_ai.llm_factory import build_chat_model

    model = build_chat_model("gemini/gemini-2.5-flash", temperature=0.2)
    result = await model.ainvoke([HumanMessage(content="Hello")])
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# ── Provider prefix → (package, class_name, model_key_strip) ──
# Tried in order; first match wins.  If the import fails (package not
# installed), we fall through to ChatLiteLLM.

_PROVIDER_MAP: list[tuple[str, str, str, str | None]] = [
    # (prefix, module_path, class_name, strip_prefix_for_model_name)
    # strip_prefix=None means pass the full model string as-is
    ("openai/", "langchain_openai", "ChatOpenAI", "openai/"),
    ("anthropic/", "langchain_anthropic", "ChatAnthropic", "anthropic/"),
    ("claude-", "langchain_anthropic", "ChatAnthropic", None),  # keep full name
    ("gemini/", "langchain_google_genai", "ChatGoogleGenerativeAI", "gemini/"),
    ("google/", "langchain_google_genai", "ChatGoogleGenerativeAI", "google/"),
    ("ollama/", "langchain_ollama", "ChatOllama", "ollama/"),
    ("ollama_chat/", "langchain_ollama", "ChatOllama", "ollama_chat/"),
    ("groq/", "langchain_groq", "ChatGroq", "groq/"),
    ("mistral/", "langchain_mistralai", "ChatMistralAI", "mistral/"),
    ("bedrock/", "langchain_aws", "ChatBedrock", "bedrock/"),
]

# ── API key env var mapping (for provider-specific classes) ──

_PROVIDER_API_KEY_ENV: dict[str, str] = {
    "openai/": "OPENAI_API_KEY",
    "anthropic/": "ANTHROPIC_API_KEY",
    "claude-": "ANTHROPIC_API_KEY",
    "gemini/": "GEMINI_API_KEY",
    "google/": "GEMINI_API_KEY",
    "groq/": "GROQ_API_KEY",
    "mistral/": "MISTRAL_API_KEY",
}

_PROVIDER_API_BASE_ENV: dict[str, str] = {
    "ollama/": "OLLAMA_API_BASE",
    "ollama_chat/": "OLLAMA_API_BASE",
}


def build_chat_model(
    model: str,
    *,
    temperature: float = 0.2,
    fallback_model: str | None = None,
    retry_attempts: int = 0,
    **kwargs: Any,
) -> BaseChatModel:
    """
    Build a LangChain ``BaseChatModel`` from a LiteLLM-style model string.

    Tries provider-specific packages first for best native support,
    then falls back to ``ChatLiteLLM`` (which wraps litellm and supports
    all providers).

    When ``fallback_model`` is provided, the returned model automatically
    tries the fallback if the primary fails (503, rate limit, timeout).

    When ``retry_attempts`` > 0, each model (primary and fallback) is
    wrapped with ``.with_retry()`` using exponential backoff with jitter.
    Retries are applied per-model so that transient errors on the primary
    are retried before falling through to the fallback.

    Parameters
    ----------
    model : str
        LiteLLM-format model identifier (e.g. ``"gemini/gemini-2.5-flash"``,
        ``"openai/gpt-4o"``, ``"ollama/llama3.2"``).
    temperature : float
        Default sampling temperature.
    fallback_model : str | None
        Optional fallback model string. When the primary model fails,
        the fallback is tried automatically. Disabled by default.
    retry_attempts : int
        Max retry attempts on transient errors (rate limits, 503s, timeouts).
        0 = disabled (default). When > 0, retries use exponential backoff
        with jitter.
    **kwargs
        Additional keyword arguments passed to the model constructor.

    Returns
    -------
    BaseChatModel
        A ready-to-use LangChain chat model (with retry and/or fallback
        if configured).
    """
    primary = _build_single_model(model, temperature=temperature, **kwargs)

    if retry_attempts > 0:
        primary = primary.with_retry(
            stop_after_attempt=retry_attempts,
            wait_exponential_jitter=True,
        )
        logger.info("[LLM] Retry configured for '%s': %d attempts", model, retry_attempts)

    if fallback_model:
        fallback = _build_single_model(fallback_model, temperature=temperature, **kwargs)
        if retry_attempts > 0:
            fallback = fallback.with_retry(
                stop_after_attempt=retry_attempts,
                wait_exponential_jitter=True,
            )
        logger.info("[LLM] Fallback configured: '%s' → '%s'", model, fallback_model)
        return primary.with_fallbacks([fallback])

    return primary


def _build_single_model(
    model: str,
    *,
    temperature: float = 0.2,
    **kwargs: Any,
) -> BaseChatModel:
    """Build a single BaseChatModel without fallbacks."""
    # Try provider-specific package first
    for prefix, module_path, class_name, strip_prefix in _PROVIDER_MAP:
        if not model.startswith(prefix):
            continue

        try:
            import importlib

            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)

            model_name = model[len(strip_prefix) :] if strip_prefix else model
            provider_kwargs = _resolve_provider_kwargs(prefix, model_name, temperature, **kwargs)

            instance = cls(**provider_kwargs)
            logger.info(
                "[LLM] Using %s.%s for model '%s'",
                module_path,
                class_name,
                model,
            )
            return instance
        except ImportError:
            logger.debug(
                "[LLM] %s not installed, trying next provider for '%s'",
                module_path,
                model,
            )
            continue
        except Exception as exc:
            logger.debug(
                "[LLM] Failed to create %s.%s for '%s': %s — trying ChatLiteLLM",
                module_path,
                class_name,
                model,
                exc,
            )
            continue

    # Fallback: ChatLiteLLM (wraps litellm, supports everything)
    return _build_litellm_fallback(model, temperature=temperature, **kwargs)


def _resolve_provider_kwargs(
    prefix: str,
    model_name: str,
    temperature: float,
    **extra: Any,
) -> dict[str, Any]:
    """Build constructor kwargs for a provider-specific ChatModel."""
    provider_kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
    }

    # Inject API key from environment
    env_key = _PROVIDER_API_KEY_ENV.get(prefix)
    if env_key:
        api_key = os.environ.get(env_key, "")
        if api_key:
            # Different providers use different kwarg names
            if prefix in ("openai/", "groq/", "mistral/"):
                provider_kwargs["api_key"] = api_key
            elif prefix in ("anthropic/", "claude-"):
                provider_kwargs["api_key"] = api_key
            elif prefix in ("gemini/", "google/"):
                provider_kwargs["google_api_key"] = api_key

    # Inject API base URL
    env_base = _PROVIDER_API_BASE_ENV.get(prefix)
    if env_base:
        api_base = os.environ.get(env_base, "")
        if api_base:
            if prefix.startswith("ollama"):
                provider_kwargs["base_url"] = api_base
            else:
                provider_kwargs["base_url"] = api_base

    provider_kwargs.update(extra)
    return provider_kwargs


def _build_litellm_fallback(
    model: str,
    *,
    temperature: float = 0.2,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a ChatLiteLLM instance as fallback."""
    try:
        from langchain_litellm import ChatLiteLLM
    except ImportError:
        try:
            from langchain_community.chat_models.litellm import ChatLiteLLM
        except ImportError as exc:
            raise ImportError(
                "Cannot create LLM: no provider-specific package found for "
                f"'{model}' and langchain-litellm is not installed. "
                "Install it with: pip install langchain-litellm"
            ) from exc

    from .llm import get_llm_kwargs

    litellm_kwargs = get_llm_kwargs(model)

    instance = ChatLiteLLM(
        model=model,
        temperature=temperature,
        **litellm_kwargs,
        **kwargs,
    )
    logger.info("[LLM] Using ChatLiteLLM fallback for model '%s'", model)
    return instance
