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
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel

from .llm import _PREFIX_TO_API_BASE_ENV as _LLM_BASE_MAP
from .llm import _PREFIX_TO_API_KEY_ENV as _LLM_KEY_MAP

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderEntry:
    """A registered LangChain chat-model provider.

    Attributes
    ----------
    prefix : str
        Model string prefix (e.g. ``"openai/"``, ``"claude-"``).
    module_path, class_name : str
        LangChain class to import when the prefix matches.
    strip_prefix : str | None
        Prefix to strip from the model name before passing to the
        constructor.  ``None`` keeps the full string.
    api_key_kwarg : str
        Constructor kwarg name to use for the API key (e.g. ``api_key``,
        ``google_api_key``). Lets new providers register without editing
        :func:`_resolve_provider_kwargs`.
    """

    prefix: str
    module_path: str
    class_name: str
    strip_prefix: str | None
    api_key_kwarg: str = "api_key"


# Provider prefix → entry.  Looked up by prefix when ``build_chat_model``
# needs to resolve a model string.  A dict prevents duplicate registrations
# from silently stacking and makes override semantics explicit.
_PROVIDER_MAP: dict[str, ProviderEntry] = {
    "openai/": ProviderEntry("openai/", "langchain_openai", "ChatOpenAI", "openai/"),
    "anthropic/": ProviderEntry("anthropic/", "langchain_anthropic", "ChatAnthropic", "anthropic/"),
    "claude-": ProviderEntry("claude-", "langchain_anthropic", "ChatAnthropic", None),
    "gemini/": ProviderEntry(
        "gemini/", "langchain_google_genai", "ChatGoogleGenerativeAI", "gemini/", api_key_kwarg="google_api_key"
    ),
    "google/": ProviderEntry(
        "google/", "langchain_google_genai", "ChatGoogleGenerativeAI", "google/", api_key_kwarg="google_api_key"
    ),
    "ollama/": ProviderEntry("ollama/", "langchain_ollama", "ChatOllama", "ollama/"),
    "ollama_chat/": ProviderEntry("ollama_chat/", "langchain_ollama", "ChatOllama", "ollama_chat/"),
    "groq/": ProviderEntry("groq/", "langchain_groq", "ChatGroq", "groq/"),
    "mistral/": ProviderEntry("mistral/", "langchain_mistralai", "ChatMistralAI", "mistral/"),
    "bedrock/": ProviderEntry("bedrock/", "langchain_aws", "ChatBedrock", "bedrock/"),
}

# ── API key / base URL env var mapping ──
# Built from llm.py's litellm maps + extra aliases for LangChain providers.

_PROVIDER_API_KEY_ENV: dict[str, str] = {prefix: env for prefix, env in _LLM_KEY_MAP}
_PROVIDER_API_KEY_ENV["google/"] = "GEMINI_API_KEY"  # alias not in litellm map

_PROVIDER_API_BASE_ENV: dict[str, str] = {prefix: env for prefix, env in _LLM_BASE_MAP}


def register_provider(
    prefix: str,
    module_path: str,
    class_name: str,
    strip_prefix: str | None = None,
    *,
    api_key_env: str = "",
    api_base_env: str = "",
    api_key_kwarg: str = "api_key",
) -> None:
    """Register a custom LLM provider for ``build_chat_model()``.

    Allows integrators to add new providers without modifying the framework.

    Parameters
    ----------
    prefix : str
        Model string prefix (e.g. ``"cohere/"``).
    module_path : str
        Python module containing the LangChain chat model class.
    class_name : str
        Class name to import from the module.
    strip_prefix : str | None
        Prefix to strip from the model string before passing to the
        constructor. ``None`` means keep the full string.
    api_key_env : str
        Environment variable name for the API key (optional).
    api_base_env : str
        Environment variable name for the API base URL (optional).
    api_key_kwarg : str
        Constructor kwarg name to pass the API key as. Defaults to
        ``"api_key"``; some providers use a different name (e.g.
        ``"google_api_key"``).
    """
    if prefix in _PROVIDER_MAP:
        existing = _PROVIDER_MAP[prefix]
        logger.warning(
            "[LLM] Replacing provider for prefix %r: %s.%s → %s.%s",
            prefix,
            existing.module_path,
            existing.class_name,
            module_path,
            class_name,
        )
    _PROVIDER_MAP[prefix] = ProviderEntry(
        prefix=prefix,
        module_path=module_path,
        class_name=class_name,
        strip_prefix=strip_prefix,
        api_key_kwarg=api_key_kwarg,
    )
    if api_key_env:
        _PROVIDER_API_KEY_ENV[prefix] = api_key_env
    if api_base_env:
        _PROVIDER_API_BASE_ENV[prefix] = api_base_env
    logger.info("[LLM] Registered custom provider: %s → %s.%s", prefix, module_path, class_name)


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
    # Try provider-specific package first.  Longest matching prefix wins so
    # ``"openai/"`` takes priority over a bare ``"gpt-"`` were someone to add it.
    matching_prefixes = sorted(
        (p for p in _PROVIDER_MAP if model.startswith(p)),
        key=len,
        reverse=True,
    )
    for prefix in matching_prefixes:
        entry = _PROVIDER_MAP[prefix]
        try:
            import importlib

            mod = importlib.import_module(entry.module_path)
            cls = getattr(mod, entry.class_name)

            model_name = model[len(entry.strip_prefix) :] if entry.strip_prefix else model
            provider_kwargs = _resolve_provider_kwargs(entry, model_name, temperature, **kwargs)

            instance = cls(**provider_kwargs)
            logger.info(
                "[LLM] Using %s.%s for model '%s'",
                entry.module_path,
                entry.class_name,
                model,
            )
            return instance
        except ImportError:
            logger.debug(
                "[LLM] %s not installed, trying next provider for '%s'",
                entry.module_path,
                model,
            )
            continue
        except Exception as exc:
            logger.warning(
                "[LLM] Failed to create %s.%s for '%s': %s — trying next provider",
                entry.module_path,
                entry.class_name,
                model,
                exc,
            )
            continue

    # Fallback: ChatLiteLLM (wraps litellm, supports everything)
    return _build_litellm_fallback(model, temperature=temperature, **kwargs)


def _resolve_provider_kwargs(
    entry: ProviderEntry,
    model_name: str,
    temperature: float,
    **extra: Any,
) -> dict[str, Any]:
    """Build constructor kwargs for a provider-specific ChatModel.

    The provider-specific kwarg name for the API key (``api_key`` vs
    ``google_api_key`` vs whatever a future provider needs) lives on
    :class:`ProviderEntry` so adding a new provider only requires a
    registry entry — never an edit here.
    """
    provider_kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
    }

    env_key = _PROVIDER_API_KEY_ENV.get(entry.prefix)
    if env_key:
        api_key = os.environ.get(env_key, "")
        if api_key:
            provider_kwargs[entry.api_key_kwarg] = api_key

    env_base = _PROVIDER_API_BASE_ENV.get(entry.prefix)
    if env_base:
        api_base = os.environ.get(env_base, "")
        if api_base:
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
