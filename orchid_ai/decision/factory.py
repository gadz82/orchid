"""Composition root for decision providers.

The factory maps validated YAML configuration to a provider instance. It is the
only place that reads an API-key environment variable or constructs an HTTP
transport, so tests and in-process integrators can replace either without
touching the adapters.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..config.errors import OrchidConfigError
from ..config.schema_decision import OrchidDecisionModelConfig, iter_agents
from ..core.decision import OrchidDecisionError, OrchidDecisionModel, OrchidDecisionState
from ..utils import import_class
from .http import DecisionHttpClient
from .llm import LLMDecisionModel
from .openai_compatible import OpenAICompatibleDecisionModel
from .state_questions import StateQuestionsDecisionModel
from .stub import StubDecisionModel

__all__ = [
    "DecisionBuildContext",
    "build_decision_model",
    "build_decision_models",
    "register_decision_protocol",
]

logger = logging.getLogger(__name__)

DecisionBuilder = Callable[[str, OrchidDecisionModelConfig, "DecisionBuildContext"], OrchidDecisionModel]


@dataclass(frozen=True)
class DecisionBuildContext:
    """Dependencies the factory may need, resolved once at the composition root."""

    allowed_hosts: list[str]
    failure_threshold: int = 3
    cooldown_s: float = 30.0
    api_key_resolver: Callable[[str], str] = os.environ.get
    chat_model_factory: Callable[[str], Any] | None = None
    client_factory: Callable[[OrchidDecisionModelConfig, str], DecisionHttpClient] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def _http_builder(adapter_cls: type[OrchidDecisionModel]) -> DecisionBuilder:
    def builder(name: str, config: OrchidDecisionModelConfig, context: DecisionBuildContext) -> OrchidDecisionModel:
        api_key = ""
        if config.api_key_env:
            api_key = (context.api_key_resolver(config.api_key_env) or "").strip()
            if not api_key:
                raise OrchidConfigError(
                    f"decision provider {name!r}: environment variable {config.api_key_env} is not set"
                )
        if context.client_factory is not None:
            client = context.client_factory(config, api_key)
        else:
            try:
                client = DecisionHttpClient(
                    base_url=config.base_url or "",
                    allowed_hosts=context.allowed_hosts,
                    provider=name,
                    api_key=api_key,
                    timeout=config.timeout_seconds,
                    max_retries=config.max_retries,
                    max_response_bytes=config.max_response_bytes,
                    failure_threshold=context.failure_threshold,
                    cooldown_s=context.cooldown_s,
                )
            except OrchidDecisionError as exc:
                raise OrchidConfigError(f"decision provider {name!r}: {exc}") from None
        adapter_kwargs: dict[str, Any] = {"client": client, "model": config.model, "provider": name}
        if config.decide_path:
            adapter_kwargs["decide_path"] = config.decide_path
        return adapter_cls(**adapter_kwargs)  # type: ignore[call-arg]

    return builder


def _build_llm(name: str, config: OrchidDecisionModelConfig, context: DecisionBuildContext) -> OrchidDecisionModel:
    if context.chat_model_factory is None:
        raise OrchidConfigError(
            f"decision provider {name!r} uses protocol 'llm' but no chat model factory was supplied"
        )
    logger.warning(
        "[Decision] provider %r uses protocol 'llm'; egress is bounded by the chat model provider, "
        "not allowed_decision_hosts",
        name,
    )
    chat_model = context.chat_model_factory(config.model)
    timeout = config.timeout_seconds if "timeout_seconds" in config.model_fields_set else 10.0
    return LLMDecisionModel(
        chat_model=chat_model,
        model=config.model,
        provider=name,
        timeout=timeout,
    )


def _build_stub(name: str, config: OrchidDecisionModelConfig, context: DecisionBuildContext) -> OrchidDecisionModel:
    stub = config.stub
    return StubDecisionModel(
        provider=name,
        answers=(stub.answers if stub is not None else context.extras.get("stub_answers")),
        confidence=(stub.confidence if stub is not None else 0.9),
        mode=(stub.mode if stub is not None else "answer"),
    )


_PROTOCOL_BUILDERS: dict[str, DecisionBuilder] = {
    "state-questions": _http_builder(StateQuestionsDecisionModel),
    "openai-compatible": _http_builder(OpenAICompatibleDecisionModel),
    "llm": _build_llm,
    "stub": _build_stub,
}


def register_decision_protocol(name: str, builder: DecisionBuilder) -> None:
    """Register a custom protocol builder for the decision factory."""
    if not name or not name.strip():
        raise ValueError("protocol name must be non-empty")
    if not callable(builder):
        raise TypeError("protocol builder must be callable")
    if name in _PROTOCOL_BUILDERS:
        logger.warning("[Decision] Replacing protocol builder %r", name)
    _PROTOCOL_BUILDERS[name] = builder


def build_decision_model(
    name: str,
    config: OrchidDecisionModelConfig,
    *,
    context: DecisionBuildContext,
) -> OrchidDecisionModel:
    """Build one provider and apply its optional state transform."""
    builder = _PROTOCOL_BUILDERS.get(config.protocol)
    if builder is None:
        raise OrchidConfigError(
            f"decision provider {name!r} uses unknown protocol {config.protocol!r}; "
            f"known protocols: {sorted(_PROTOCOL_BUILDERS)}"
        )
    transform = _resolve_transform(name, config.state_transform)
    provider = builder(name, config, context)
    if transform is None:
        return provider
    return _TransformDecisionModel(inner=provider, transform=transform, provider=name)


def build_decision_models(
    config: Any,
    *,
    api_key_resolver: Callable[[str], str] | None = None,
    chat_model_factory: Callable[[str], Any] | None = None,
    client_factory: Callable[[OrchidDecisionModelConfig, str], DecisionHttpClient] | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, OrchidDecisionModel]:
    """Build every declared provider. An empty declaration builds nothing.

    # ORCHID-MEM: 01a1a186-10ab-72b6-93a9-c02a115162f8 — Decision provider composition root reading YAML, env keys, and transforms.
    """
    providers = getattr(config, "decision_models", {})
    if not providers:
        return {}

    budget = config.decisions.budget
    context = DecisionBuildContext(
        allowed_hosts=list(config.allowed_decision_hosts),
        failure_threshold=budget.failure_threshold,
        cooldown_s=budget.breaker_cooldown_s,
        api_key_resolver=api_key_resolver or os.environ.get,
        chat_model_factory=chat_model_factory,
        client_factory=client_factory,
        extras=dict(extras or {}),
    )
    referenced = _referenced_providers(config)
    for name in providers:
        if name not in referenced:
            logger.warning("[Decision] provider %r is declared but no decision point references it", name)
    return {
        name: build_decision_model(name, provider, context=context)
        for name, provider in providers.items()
        if name in referenced
    }


def _referenced_providers(config: Any) -> set[str]:
    referenced: set[str] = set()
    for point in ("routing", "skills", "tools", "rag"):
        resolved = config.decisions.resolve(point)
        if resolved is not None and resolved.model:
            referenced.add(resolved.model)
    for _path, agent in iter_agents(config.agents):
        agent_model = getattr(agent, "decision_model", None)
        if agent_model:
            referenced.add(agent_model)
        decisions = getattr(agent, "decisions", None)
        if decisions is None:
            continue
        for point in ("skills", "tools", "rag"):
            resolved = config.decisions.resolve(point, decisions)
            if resolved is not None and resolved.model:
                referenced.add(resolved.model)
    return referenced


class _TransformDecisionModel(OrchidDecisionModel):
    """Applies a configured state transform before delegating to a provider."""

    def __init__(self, *, inner: OrchidDecisionModel, transform: Callable[..., Any], provider: str) -> None:
        self._inner = inner
        self._transform = transform
        self._provider = provider

    async def decide(
        self,
        state: OrchidDecisionState,
        questions: Any,
        *,
        timeout: float | None = None,
    ) -> Any:
        try:
            if timeout is not None:
                async with asyncio.timeout(max(0.1, float(timeout))):
                    transformed = await self._run_transform(state)
            else:
                transformed = await self._run_transform(state)
        except TimeoutError:
            raise OrchidDecisionError(
                "state transform timed out",
                code="timeout",
                provider=self._provider,
                retryable=True,
            ) from None
        except Exception:
            logger.exception("[Decision] state transform failed provider=%s", self._provider)
            raise OrchidDecisionError(
                "state transform failed",
                code="invalid_response",
                provider=self._provider,
                retryable=False,
            ) from None
        if not isinstance(transformed, OrchidDecisionState):
            raise OrchidDecisionError(
                "state transform must return OrchidDecisionState",
                code="invalid_response",
                provider=self._provider,
                retryable=False,
            )
        return await self._inner.decide(transformed, questions, timeout=timeout)

    async def _run_transform(self, state: OrchidDecisionState) -> Any:
        if inspect.iscoroutinefunction(self._transform):
            return await self._transform(state)
        return await asyncio.to_thread(self._transform, state)

    async def aclose(self) -> None:
        await self._inner.aclose()


def _resolve_transform(name: str, state_transform: str | None) -> Callable[..., Any] | None:
    if not state_transform:
        return None
    try:
        transform = import_class(state_transform)
    except Exception as exc:
        raise OrchidConfigError(
            f"decision provider {name!r}: cannot resolve state_transform {state_transform!r}: {exc}"
        ) from exc
    if not callable(transform):
        raise OrchidConfigError(f"decision provider {name!r}: state_transform {state_transform!r} is not callable")
    return transform
