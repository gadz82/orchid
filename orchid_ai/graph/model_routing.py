"""Decision-backed chat-model routing for graph agents.

The ``model_routing`` point chooses one chat model from a *configured catalog*
for a request. This module owns the narrow pieces the graph needs:

- :class:`ModelRoutingSelector` asks the bound decision provider once and
  resolves every outcome to a model id, with the configured default as the
  fallback on any uncertainty or failure.
- :class:`RoutedChatModel` is a transparent ``BaseChatModel`` delegate that
  sends each call to the model currently selected for the request.
- :class:`TurnBudgetRegistry` shares one :class:`ModelRoutingBudget` across
  every agent in a graph turn, so the per-turn call and wall-clock ceilings
  span the whole turn rather than a single agent run.

Neither piece imports a concrete provider: selected models are built through
:func:`orchid_ai.llm_factory.build_chat_model`, so routing only chooses among
models the operator already configured. The graph node wrapper binds the
selection for the duration of one agent run; code that uses the model outside
the graph keeps the default behaviour because the selection defaults to the
agent's configured model.

Structured output and tool binding deliberately resolve against the selected
model's raw ``BaseChatModel`` primary, not just the fallback-wrapped invocation
chain: ``RunnableWithFallbacks`` (and a retry ``RunnableBinding``) do not expose
the provider-specific ``with_structured_output``/``bind_tools`` methods.

Observability note (F6, accepted): the shared ``orchid.decision`` line keeps the
P-001 field set and does not carry the selected model id. The selection is
logged on the separate ``[Model] routing selected <id>`` INFO line; neither line
contains user text or credentials.

Fail-safe note (F8, accepted): a malformed ``options.models`` value is treated
as an empty catalog, so the point falls back without a provider call. A
selected model that cannot be constructed (for example a typo'd provider
prefix) falls back to the configured default and logs only the exception type.
Both paths are deliberately fail-open; the trade-off is one warning line at
selector construction (malformed catalog) or on first use (build failure), not
per request.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import PrivateAttr

from ..core.decision import (
    OrchidDecisionError,
    OrchidDecisionPointBinding,
    OrchidDecisionQuestion,
    OrchidDecisionState,
)
from ..decision.observability import log_decision
from ..llm_factory import build_chat_model

__all__ = [
    "ModelRoutingBudget",
    "ModelRoutingSelector",
    "RoutedChatModel",
    "TurnBudgetRegistry",
    "build_routed_chat_model",
]

logger = logging.getLogger(__name__)

_POINT = "model_routing"
_QUESTION = "model"
_MAX_QUERY_CHARS = 4000
_MIN_CANDIDATES = 2

_QUESTION_INSTRUCTIONS = (
    "Choose the chat model that best fits the user's request. "
    "Only the offered models are configured. Prefer the default model unless "
    "another model is clearly a better fit."
)


def _model_catalog(
    options: Mapping[str, Any],
    *,
    default_model: str,
    fallback_model: str,
) -> tuple[str, ...]:
    """Return the ordered, de-duplicated model catalog for one binding.

    ``options.models`` is authoritative when set. Without it the catalog is the
    agent's configured model plus its fallback, which preserves today's static
    choice as the only candidates. A malformed allowlist is treated as empty so
    the caller falls back instead of guessing.
    """
    configured = options.get("models")
    if configured is None:
        raw: Sequence[Any] = (default_model, fallback_model)
    elif isinstance(configured, (list, tuple)):
        raw = configured
    else:
        logger.warning("[Model] options.models must be a list of model ids; ignoring it")
        raw = ()

    catalog: list[str] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, str):
            continue
        model = entry.strip()
        if model and model not in seen:
            seen.add(model)
            catalog.append(model)
    return tuple(catalog)


class ModelRoutingBudget:
    """A per-turn call and wall-clock allowance shared with the selector.

    The decision layer declares one budget per turn; this object is the narrow
    consumer-facing part of it. ``max_calls`` bounds provider calls and
    ``max_total_ms`` bounds the turn wall clock. Both are checked before a call
    is made, so an exhausted budget never adds provider traffic.
    """

    __slots__ = ("_clock", "_deadline", "_remaining")

    def __init__(
        self,
        max_calls: int,
        *,
        max_total_ms: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(max_calls, bool) or not isinstance(max_calls, int):
            raise TypeError("model routing budget max_calls must be an integer")
        if max_calls < 1:
            raise ValueError("model routing budget max_calls must be at least 1")
        if max_total_ms is not None:
            if isinstance(max_total_ms, bool) or not isinstance(max_total_ms, int):
                raise TypeError("model routing budget max_total_ms must be an integer or None")
            if max_total_ms < 1:
                raise ValueError("model routing budget max_total_ms must be at least 1")
        self._remaining = max_calls
        self._clock = clock
        self._deadline = clock() + (max_total_ms / 1000.0) if max_total_ms is not None else None

    @property
    def remaining(self) -> int:
        return self._remaining

    @property
    def remaining_seconds(self) -> float | None:
        """Seconds left before the wall-clock ceiling, or ``None`` when unbounded."""
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - self._clock())

    @property
    def expired(self) -> bool:
        """``True`` when the wall-clock ceiling has been reached."""
        return self._deadline is not None and self._clock() >= self._deadline

    def try_acquire(self) -> bool:
        """Reserve one call, returning ``False`` when the budget is exhausted."""
        if self._remaining <= 0 or self.expired:
            return False
        self._remaining -= 1
        return True


class TurnBudgetRegistry:
    """One :class:`ModelRoutingBudget` per graph turn, shared by every agent.

    The graph keys a turn by chat + latest human message, so the first agent
    that runs creates the budget and every later agent in the same invocation
    spends from it. Entries are tiny and bounded: the oldest is evicted when the
    registry reaches ``max_entries``.
    """

    __slots__ = ("_budgets", "_max_calls", "_max_entries", "_max_total_ms")

    def __init__(
        self,
        *,
        max_calls_per_turn: int = 1,
        max_total_ms: int | None = None,
        max_entries: int = 256,
    ) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("turn budget registry max_entries must be a positive integer")
        self._max_calls = max_calls_per_turn
        self._max_total_ms = max_total_ms
        self._max_entries = max_entries
        self._budgets: dict[str, ModelRoutingBudget] = {}

    def for_turn(self, key: str) -> ModelRoutingBudget:
        """Return the budget for ``key``, creating it on first use."""
        if not isinstance(key, str) or not key:
            raise ValueError("a turn budget key must be a non-empty string")
        budget = self._budgets.get(key)
        if budget is not None:
            return budget
        if len(self._budgets) >= self._max_entries:
            self._budgets.pop(next(iter(self._budgets)))
        budget = ModelRoutingBudget(self._max_calls, max_total_ms=self._max_total_ms)
        self._budgets[key] = budget
        return budget

    def clear(self) -> None:
        self._budgets.clear()

    def __len__(self) -> int:
        return len(self._budgets)


@dataclass(frozen=True)
class _ModelBundle:
    """One selected model: the raw primary, its raw fallbacks, and invocation.

    ``primary``/``fallbacks`` are concrete models used for structured output and
    tool binding. ``invocation`` is the model actually invoked; it may be a
    ``RunnableWithFallbacks`` or a retry ``RunnableBinding``.
    """

    primary: Any
    fallbacks: tuple[Any, ...]
    invocation: Any


def _raw_chat_model(runnable: Any) -> Any | None:
    """Return the concrete ``BaseChatModel`` behind a LangChain wrapper."""
    if isinstance(runnable, BaseChatModel):
        return runnable
    bound = getattr(runnable, "bound", None)
    if isinstance(bound, BaseChatModel):
        return bound
    return None


def _bundle_from_model(model: Any) -> _ModelBundle | None:
    """Split a built model into its raw primary/fallbacks and invocation chain."""
    primary = _raw_chat_model(model)
    if primary is not None:
        return _ModelBundle(primary=primary, fallbacks=(), invocation=model)

    inner = getattr(model, "runnable", None)
    if inner is not None:
        primary = _raw_chat_model(inner)
        if primary is not None:
            fallbacks = tuple(
                raw for fb in (getattr(model, "fallbacks", ()) or ()) if (raw := _raw_chat_model(fb)) is not None
            )
            return _ModelBundle(primary=primary, fallbacks=fallbacks, invocation=model)
    return None


def _fallback_model_ids(model_id: str, *, default_model: str, fallback_model: str) -> tuple[str, ...]:
    """Return the fallback model ids in the same order ``build_chat_model`` uses."""
    ordered: list[str] = []
    for candidate in (fallback_model, default_model):
        if candidate and candidate != model_id and candidate not in ordered:
            ordered.append(candidate)
    return tuple(ordered)


class ModelRoutingSelector:
    """Choose one model id from a configured catalog through a bound point.

    The selector is safe to share across concurrent requests: the active
    selection lives on a :class:`ContextVar` and is set by :meth:`bound` for
    exactly one agent run. ``select`` never raises for an operational failure —
    every path returns a model from the catalog, with the configured default as
    the fallback.
    """

    def __init__(
        self,
        binding: OrchidDecisionPointBinding,
        *,
        default_model: str,
        fallback_model: str | None = None,
        agent_name: str = "",
        max_calls_per_turn: int = 1,
        max_total_ms: int | None = None,
    ) -> None:
        if not default_model:
            raise ValueError("model routing requires a non-empty default model")
        self._binding = binding
        self._default_model = default_model
        self._fallback_model = (fallback_model or "").strip()
        self._agent_name = agent_name
        self._max_calls_per_turn = max(1, int(max_calls_per_turn))
        self._max_total_ms = max_total_ms
        self._catalog = _model_catalog(
            binding.options,
            default_model=default_model,
            fallback_model=self._fallback_model,
        )
        self._selection: ContextVar[str] = ContextVar(
            f"orchid.model_routing.{agent_name or id(self)}",
            default=default_model,
        )

    @property
    def point(self) -> str:
        return self._binding.point

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def fallback_model(self) -> str:
        return self._fallback_model

    @property
    def catalog(self) -> tuple[str, ...]:
        """Models offered to the provider; empty when routing cannot choose."""
        return self._catalog

    @property
    def max_calls_per_turn(self) -> int:
        return self._max_calls_per_turn

    @property
    def max_total_ms(self) -> int | None:
        return self._max_total_ms

    @property
    def current_model(self) -> str:
        """The model selected for the current request context."""
        return self._selection.get()

    def set_selection(self, model: str) -> Token[str]:
        """Bind ``model`` for the current context; return the reset token.

        Only a catalog member or the configured default may be bound, so a
        consumer cannot bypass the decision result and route to an unconfigured
        model. The default stays valid even when the catalog is empty, because
        it is the point's declared fallback.
        """
        if model != self._default_model and model not in self._catalog:
            raise ValueError(f"model {model!r} is not in the configured routing catalog")
        return self._selection.set(model)

    def reset_selection(self, token: Token[str]) -> None:
        self._selection.reset(token)

    @asynccontextmanager
    async def bound(
        self,
        query: str,
        *,
        budget: ModelRoutingBudget | None = None,
    ) -> AsyncIterator[str]:
        """Select a model for ``query`` and bind it for one agent run.

        ``budget`` is normally the shared per-turn object from
        :class:`TurnBudgetRegistry`; without one the selector creates a
        single-run budget from its configured limits.
        """
        effective = budget or ModelRoutingBudget(
            self._max_calls_per_turn,
            max_total_ms=self._max_total_ms,
        )
        selected = await self.select(query, budget=effective)
        token = self.set_selection(selected)
        try:
            yield selected
        finally:
            self.reset_selection(token)

    async def select(
        self,
        query: str,
        *,
        budget: ModelRoutingBudget | None = None,
    ) -> str:
        """Ask the bound point which model to use, falling back to the default.

        Every failure mode returns :attr:`default_model`; the reason is carried
        on exactly one structured decision line. The provider is not called when
        the catalog cannot offer a choice or the caller's budget is exhausted.
        """
        if len(self._catalog) < _MIN_CANDIDATES:
            return self._fallback("no_candidates")
        if budget is not None and not budget.try_acquire():
            return self._fallback("budget_exhausted")

        state = OrchidDecisionState(
            text=query[:_MAX_QUERY_CHARS] if isinstance(query, str) else str(query)[:_MAX_QUERY_CHARS],
            payload={"agent": self._agent_name} if self._agent_name else {},
        )
        question = OrchidDecisionQuestion(
            name=_QUESTION,
            kind="choice",
            options=self._catalog,
            instructions=_QUESTION_INSTRUCTIONS,
        )

        started = time.perf_counter()
        try:
            # The turn ceiling is an outer cancellation boundary; the provider
            # keeps its own configured per-call timeout when none is supplied.
            timeout = budget.remaining_seconds if budget is not None else None
            async with asyncio.timeout(timeout):
                result = await self._binding.decide(state, [question])
        except OrchidDecisionError as exc:
            return self._fallback(
                exc.code or "provider_error",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                provider=exc.provider,
            )
        except TimeoutError:
            return self._fallback(
                "timeout",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:
            # Only the exception type crosses this seam: a provider message can
            # contain user state, and a traceback would duplicate it in logs.
            logger.warning("[Model] model routing provider raised %s", type(exc).__name__)
            return self._fallback(
                "provider_error",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        provider = result.provider or self._binding.provider
        answer = result.answers.get(_QUESTION)
        if answer is None:
            return self._fallback("invalid_response", latency_ms=latency_ms, provider=provider)
        if answer.refusal is not None:
            return self._fallback(
                "refusal",
                latency_ms=latency_ms,
                confidence=answer.confidence,
                provider=provider,
            )
        if answer.kind != "choice":
            return self._fallback(
                "invalid_response",
                latency_ms=latency_ms,
                confidence=answer.confidence,
                provider=provider,
            )
        if answer.confidence < self._binding.confidence_threshold:
            return self._fallback(
                "low_confidence",
                latency_ms=latency_ms,
                confidence=answer.confidence,
                provider=provider,
            )

        selected = str(answer.value)
        if selected not in self._catalog:
            return self._fallback(
                "unknown_model",
                latency_ms=latency_ms,
                confidence=answer.confidence,
                provider=provider,
            )

        log_decision(
            point=_POINT,
            path="decision",
            provider=provider,
            latency_ms=latency_ms,
            confidence=answer.confidence,
            outcome="accepted",
        )
        logger.info("[Model] routing selected %r", selected)
        return selected

    def _fallback(
        self,
        reason: str,
        *,
        latency_ms: float | None = None,
        confidence: float | None = None,
        provider: str = "",
    ) -> str:
        """Emit one structured fallback line and return the configured default."""
        logger.warning("[Model] model routing fell back to %r (reason=%s)", self._default_model, reason)
        log_decision(
            point=_POINT,
            path="llm_fallback",
            provider=provider or self._binding.provider,
            latency_ms=latency_ms,
            confidence=confidence,
            outcome="fallback",
            fallback_reason=reason,
        )
        return self._default_model


class RoutedChatModel(BaseChatModel):
    """A ``BaseChatModel`` that delegates to the request's selected model.

    Invocation, tool binding, structured output and streaming all resolve the
    selected bundle at call time. Structured output and tool binding use the
    bundle's raw ``BaseChatModel`` primary (with raw fallbacks when available),
    so a fallback-wrapped invocation chain never hides the provider methods.
    """

    _selector: ModelRoutingSelector = PrivateAttr()
    _default_bundle: _ModelBundle | None = PrivateAttr(default=None)
    _bundle_factory: Callable[[str], _ModelBundle] = PrivateAttr()
    _bundles: dict[str, _ModelBundle] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        *,
        selector: ModelRoutingSelector,
        default_chat_model: Any | None = None,
        bundle_factory: Callable[[str], _ModelBundle] | None = None,
    ) -> None:
        super().__init__()
        self._selector = selector
        self._default_bundle = _bundle_from_model(default_chat_model) if default_chat_model is not None else None
        if default_chat_model is not None and self._default_bundle is None:
            # Duck-typed/injected model (tests, custom in-process integrations).
            self._default_bundle = _ModelBundle(
                primary=default_chat_model,
                fallbacks=(),
                invocation=default_chat_model,
            )
        self._bundle_factory = bundle_factory or self._build_default_bundle
        self._bundles = {}

    @property
    def selector(self) -> ModelRoutingSelector:
        """Routing configuration backing this model (for diagnostics/tests)."""
        return self._selector

    @property
    def _llm_type(self) -> str:
        return "orchid-routed"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self._selector.current_model, "routed": True}

    def _build_default_bundle(self, model_id: str) -> _ModelBundle:
        fallback_ids = _fallback_model_ids(
            model_id,
            default_model=self._selector.default_model,
            fallback_model=self._selector.fallback_model,
        )
        # ``selected_model=model_id`` would make ``build_chat_model`` append the
        # default as a fallback, returning a ``RunnableWithFallbacks`` instead of
        # a raw ``BaseChatModel``. Build each raw model without a selection.
        primary = build_chat_model(model_id, fallback_model=None, retry_attempts=0)
        fallbacks = tuple(
            build_chat_model(fallback_id, fallback_model=None, retry_attempts=0) for fallback_id in fallback_ids
        )
        invocation = build_chat_model(
            self._selector.default_model,
            selected_model=model_id,
            fallback_model=self._selector.fallback_model or None,
        )
        return _ModelBundle(primary=primary, fallbacks=fallbacks, invocation=invocation)

    def _bundle_for_current_request(self) -> _ModelBundle:
        """Resolve and cache the selected model bundle.

        A selected model that cannot be constructed (for example a malformed
        catalog entry) falls back to the configured default bundle instead of
        failing the request. Only the exception type is logged.
        """
        model_id = self._selector.current_model
        bundle = self._bundles.get(model_id)
        if bundle is not None:
            return bundle

        if model_id == self._selector.default_model and self._default_bundle is not None:
            bundle = self._default_bundle
        else:
            try:
                bundle = self._bundle_factory(model_id)
            except Exception as exc:
                logger.warning(
                    "[Model] selected model %r could not be built (%s); using the configured default",
                    model_id,
                    type(exc).__name__,
                )
                if self._default_bundle is not None:
                    bundle = self._default_bundle
                else:
                    bundle = self._bundle_factory(self._selector.default_model)

        self._bundles[model_id] = bundle
        return bundle

    # ── Runnable surface ────────────────────────────────────────

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return self._bundle_for_current_request().invocation.invoke(input, config=config, **kwargs)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return await self._bundle_for_current_request().invocation.ainvoke(input, config=config, **kwargs)

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Bind tools on the selected model, resolving it at invocation time.

        Each raw fallback is bound with the same tools so tool calls retain the
        normal fallback semantics. ``tool_choice`` is omitted entirely when the
        caller did not set one, matching the provider ``bind_tools`` contract.
        """

        def _bound() -> Any:
            bundle = self._bundle_for_current_request()
            bound_kwargs = dict(kwargs)
            if tool_choice is not None:
                bound_kwargs["tool_choice"] = tool_choice
            primary = bundle.primary.bind_tools(tools, **bound_kwargs)
            fallbacks = _bind_tool_fallbacks(bundle.fallbacks, tools, bound_kwargs)
            if not fallbacks:
                return primary
            return primary.with_fallbacks(fallbacks)

        def _invoke(input: Any, config: Any = None, **extra: Any) -> Any:
            return _bound().invoke(input, config=config, **extra)

        async def _ainvoke(input: Any, config: Any = None, **extra: Any) -> Any:
            return await _bound().ainvoke(input, config=config, **extra)

        return RunnableLambda(_invoke, afunc=_ainvoke)

    def with_structured_output(
        self,
        schema: Any,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Return a structured-output runnable over the selected raw primary.

        This is the F1 contract: ``RunnableWithFallbacks`` has no
        ``with_structured_output``, so resolve it on the concrete primary and
        add structured fallbacks where the provider supports them.
        """

        def _structured() -> Any:
            bundle = self._bundle_for_current_request()
            primary = bundle.primary.with_structured_output(schema, include_raw=include_raw, **kwargs)
            fallbacks = _structured_fallbacks(
                bundle.fallbacks,
                schema,
                include_raw=include_raw,
                kwargs=kwargs,
            )
            if not fallbacks:
                return primary
            return primary.with_fallbacks(fallbacks)

        def _invoke(input: Any, config: Any = None, **extra: Any) -> Any:
            return _structured().invoke(input, config=config, **extra)

        async def _ainvoke(input: Any, config: Any = None, **extra: Any) -> Any:
            return await _structured().ainvoke(input, config=config, **extra)

        return RunnableLambda(_invoke, afunc=_ainvoke)

    # ── BaseChatModel surface ───────────────────────────────────

    def _generate(
        self,
        messages: Any,
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = self._bundle_for_current_request().invocation.invoke(messages, stop=stop, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: Any,
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = await self._bundle_for_current_request().invocation.ainvoke(messages, stop=stop, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: Any,
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for chunk in self._bundle_for_current_request().invocation.stream(messages, stop=stop, **kwargs):
            yield ChatGenerationChunk(message=chunk)

    async def _astream(
        self,
        messages: Any,
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        async for chunk in self._bundle_for_current_request().invocation.astream(messages, stop=stop, **kwargs):
            yield ChatGenerationChunk(message=chunk)


def _bind_tool_fallbacks(
    fallbacks: Sequence[Any],
    tools: Sequence[Any],
    bound_kwargs: Mapping[str, Any],
) -> list[Any]:
    """Bind the same tools on each raw fallback, skipping unsupported models."""
    bound: list[Any] = []
    for fallback in fallbacks:
        try:
            bound.append(fallback.bind_tools(tools, **dict(bound_kwargs)))
        except Exception as exc:
            logger.debug("[Model] tool fallback skipped (%s)", type(exc).__name__)
            continue
    return bound


def _structured_fallbacks(
    fallbacks: Sequence[Any],
    schema: Any,
    *,
    include_raw: bool,
    kwargs: Mapping[str, Any],
) -> list[Any]:
    """Build structured-output runnables for raw fallbacks that support them."""
    structured: list[Any] = []
    for fallback in fallbacks:
        try:
            structured.append(
                fallback.with_structured_output(schema, include_raw=include_raw, **dict(kwargs)),
            )
        except Exception as exc:
            logger.debug("[Model] structured-output fallback skipped (%s)", type(exc).__name__)
            continue
    return structured


def build_routed_chat_model(
    selector: ModelRoutingSelector,
    *,
    default_chat_model: Any | None = None,
    temperature: float = 0.2,
    fallback_model: str | None = None,
    retry_attempts: int = 0,
    model_factory: Callable[[str], Any] | None = None,
) -> RoutedChatModel:
    """Build the transparent routed model for one agent.

    Selected models are built with the agent's configured fallback and retry
    policy for invocation, and separately as raw primaries for structured
    output and tool binding. ``model_factory`` is an injection seam for
    integrators and tests; the default uses
    :func:`orchid_ai.llm_factory.build_chat_model`.
    """
    if model_factory is not None:

        def bundle_factory(model_id: str) -> _ModelBundle:
            model = model_factory(model_id)
            return _bundle_from_model(model) or _ModelBundle(
                primary=model,
                fallbacks=(),
                invocation=model,
            )

    else:

        def bundle_factory(model_id: str) -> _ModelBundle:
            # Raw primaries for structured output and tool binding. Do NOT pass
            # ``selected_model`` here: ``build_chat_model`` would append the
            # default fallback and return a ``RunnableWithFallbacks``.
            primary = build_chat_model(
                model_id,
                temperature=temperature,
                fallback_model=None,
                retry_attempts=0,
            )
            fallback_ids = _fallback_model_ids(
                model_id,
                default_model=selector.default_model,
                fallback_model=fallback_model or "",
            )
            fallbacks = tuple(
                build_chat_model(
                    fallback_id,
                    temperature=temperature,
                    fallback_model=None,
                    retry_attempts=0,
                )
                for fallback_id in fallback_ids
            )
            invocation = build_chat_model(
                selector.default_model,
                selected_model=model_id,
                temperature=temperature,
                fallback_model=fallback_model,
                retry_attempts=retry_attempts,
            )
            return _ModelBundle(primary=primary, fallbacks=fallbacks, invocation=invocation)

    return RoutedChatModel(
        selector=selector,
        default_chat_model=default_chat_model,
        bundle_factory=bundle_factory,
    )
