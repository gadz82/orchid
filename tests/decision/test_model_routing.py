"""Tests for decision-backed chat-model routing.

The ``model_routing`` point chooses one model from a configured catalog for a
request. Every uncertain or failed outcome must keep the agent's configured
model, and the routed chat model must stay transparent to the rest of the
framework (no network, no provider-specific import).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

import orchid_ai.graph.graph as graph_module
from orchid_ai.config.schema import OrchidAgentsConfig
from orchid_ai.config.schema_decision import OrchidDecisionPointConfig
from orchid_ai.core.agent import OrchidAgent
from orchid_ai.core.decision import (
    OrchidDecisionAnswer,
    OrchidDecisionError,
    OrchidDecisionQuestion,
    OrchidDecisionResult,
    OrchidDecisionState,
)
from orchid_ai.core.guardrails import (
    OrchidGuardrail,
    OrchidGuardrailAction,
    OrchidGuardrailChain,
    OrchidGuardrailContext,
    OrchidGuardrailResult,
)
from orchid_ai.decision.registry import get_decision_point
from orchid_ai.decision.stub import StubDecisionModel
from orchid_ai.graph.decision_bindings import DecisionBinding, resolve_decision_binding
from orchid_ai.graph.graph import _create_agent_node, build_graph
from orchid_ai.graph.model_routing import (
    ModelRoutingBudget,
    ModelRoutingSelector,
    RoutedChatModel,
    TurnBudgetRegistry,
    build_routed_chat_model,
)
from orchid_ai.llm_factory import build_chat_model
from orchid_ai.runtime import OrchidRuntime

DEFAULT_MODEL = "test/default"
FALLBACK_MODEL = "test/fallback"
ALT_MODEL = "test/alt"
SENTINEL = "SENTINEL-USER-TEXT-DO-NOT-LOG"


# ── Helpers ─────────────────────────────────────────────────


def _binding(
    provider: Any,
    *,
    threshold: float = 0.75,
    options: dict[str, Any] | None = None,
    provider_key: str = "stub",
) -> DecisionBinding:
    return DecisionBinding(
        spec=get_decision_point("model_routing"),
        config=OrchidDecisionPointConfig(
            model=provider_key,
            confidence_threshold=threshold,
            options=options or {},
        ),
        model=provider,
    )


def _selector(
    provider: Any,
    *,
    options: dict[str, Any] | None = None,
    threshold: float = 0.75,
    default_model: str = DEFAULT_MODEL,
    fallback_model: str | None = FALLBACK_MODEL,
    max_calls_per_turn: int = 4,
) -> ModelRoutingSelector:
    return ModelRoutingSelector(
        _binding(provider, threshold=threshold, options=options),
        default_model=default_model,
        fallback_model=fallback_model,
        agent_name="a",
        max_calls_per_turn=max_calls_per_turn,
    )


class _ChoiceProvider:
    """Return a fixed choice, even when it is outside the question options."""

    def __init__(self, value: Any, *, confidence: float = 0.95, kind: str = "choice") -> None:
        self.value = value
        self.confidence = confidence
        self.kind = kind
        self.questions: list[OrchidDecisionQuestion] = []
        self.states: list[OrchidDecisionState] = []

    async def decide(self, state, questions, *, timeout=None):
        self.states.append(state)
        self.questions.extend(questions)
        return OrchidDecisionResult(
            answers={
                "model": OrchidDecisionAnswer(
                    name="model",
                    kind=self.kind,
                    value=self.value,
                    confidence=self.confidence,
                )
            },
            provider="recorder",
            model="recorder",
        )


class _ErrorProvider:
    def __init__(self, code: str) -> None:
        self.code = code
        self.calls = 0

    async def decide(self, state, questions, *, timeout=None):
        self.calls += 1
        raise OrchidDecisionError("provider failed", code=self.code, provider="error-provider")


class _RaisingProvider:
    async def decide(self, state, questions, *, timeout=None):
        raise RuntimeError("unexpected")


class _TimeoutProvider:
    async def decide(self, state, questions, *, timeout=None):
        raise TimeoutError


class _EmptyProvider:
    async def decide(self, state, questions, *, timeout=None):
        return OrchidDecisionResult(answers={}, provider="empty", model="empty")


class _CountingProvider:
    def __init__(self, value: str) -> None:
        self.value = value
        self.calls = 0
        self.timeouts: list[float | None] = []
        self.questions: list[OrchidDecisionQuestion] = []

    async def decide(self, state, questions, *, timeout=None):
        self.calls += 1
        self.timeouts.append(timeout)
        self.questions.extend(questions)
        return OrchidDecisionResult(
            answers={
                "model": OrchidDecisionAnswer(
                    name="model",
                    kind="choice",
                    value=self.value,
                    confidence=0.95,
                )
            },
            provider="counter",
            model="counter",
        )


class _FakeModel:
    """Minimal delegate used by the routed model; no LangChain or network."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.calls: list[dict[str, Any]] = []
        self.structured: list[Any] = []
        self.fallbacks: list[Any] = []

    def with_fallbacks(self, fallbacks):
        self.fallbacks = list(fallbacks)
        return self

    def with_retry(self, **kwargs):
        return self

    def invoke(self, input, config=None, **kwargs):
        self.calls.append(kwargs)
        return AIMessage(content=self.tag)

    async def ainvoke(self, input, config=None, **kwargs):
        # A real suspension point so concurrency tests interleave tasks.
        await asyncio.sleep(0)
        self.calls.append(kwargs)
        return AIMessage(content=self.tag)

    def with_structured_output(self, schema, *, include_raw=False, **kwargs):
        self.structured.append(schema)
        return _FakeStructured(self)

    def bind_tools(self, tools, **kwargs):
        self.calls.append({"tools": tools})
        return self


class _ToolRecordingModel(_FakeModel):
    def __init__(self, tag: str) -> None:
        super().__init__(tag)
        self.bind_calls: list[tuple[list[Any], dict[str, Any]]] = []

    def bind_tools(self, tools, **kwargs):
        self.bind_calls.append((list(tools), dict(kwargs)))
        return self


class _StreamingModel(_FakeModel):
    def stream(self, input, config=None, **kwargs):
        self.calls.append(kwargs)
        for part in ("a", "l", "t"):
            yield AIMessageChunk(content=part)

    async def astream(self, input, config=None, **kwargs):
        self.calls.append(kwargs)
        for part in ("a", "l", "t"):
            yield AIMessageChunk(content=part)


class _StructuredChatModel(BaseChatModel):
    """A real ``BaseChatModel`` so the fallback chain wraps actual provider API."""

    tag: str = "model"

    @property
    def _llm_type(self) -> str:
        return "structured-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.tag))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def with_structured_output(self, schema, *, include_raw=False, **kwargs):
        def _invoke(input, config=None, **extra):
            return {"model": self.tag}

        async def _ainvoke(input, config=None, **extra):
            return {"model": self.tag}

        return RunnableLambda(_invoke, afunc=_ainvoke)


class _CallRecorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def add(self, value: str) -> None:
        self.calls.append(value)


class _OrderedChatModel(BaseChatModel):
    """A real ``BaseChatModel`` that records calls and can fail per operation."""

    tag: str = "model"
    fail_invoke: bool = False
    fail_structured: bool = False
    fail_tools: bool = False
    recorder: Any = None

    @property
    def _llm_type(self) -> str:
        return "ordered-fake"

    def _record(self, value: str) -> None:
        if self.recorder is not None:
            self.recorder.add(value)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self._record(f"invoke:{self.tag}")
        if self.fail_invoke:
            raise RuntimeError(f"{self.tag} invocation failed")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.tag))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def with_structured_output(self, schema, *, include_raw=False, **kwargs):
        def _invoke(input, config=None, **extra):
            self._record(f"structured:{self.tag}")
            if self.fail_structured:
                raise RuntimeError(f"{self.tag} structured output failed")
            return {"tag": self.tag}

        async def _ainvoke(input, config=None, **extra):
            self._record(f"structured:{self.tag}")
            if self.fail_structured:
                raise RuntimeError(f"{self.tag} structured output failed")
            return {"tag": self.tag}

        return RunnableLambda(_invoke, afunc=_ainvoke)

    def bind_tools(self, tools, **kwargs):
        def _invoke(input, config=None, **extra):
            self._record(f"tools:{self.tag}")
            if self.fail_tools:
                raise RuntimeError(f"{self.tag} tool call failed")
            return AIMessage(content=self.tag)

        async def _ainvoke(input, config=None, **extra):
            self._record(f"tools:{self.tag}")
            if self.fail_tools:
                raise RuntimeError(f"{self.tag} tool call failed")
            return AIMessage(content=self.tag)

        return RunnableLambda(_invoke, afunc=_ainvoke)


class _LeakyError(RuntimeError):
    pass


class _LeakyProvider:
    async def decide(self, state, questions, *, timeout=None):
        raise _LeakyError(SENTINEL)


class _SlowProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, state, questions, *, timeout=None):
        self.calls += 1
        await asyncio.sleep(0.05)
        raise AssertionError("the turn timeout should have cancelled this call")


class _BlockingRail(OrchidGuardrail):
    @property
    def name(self) -> str:
        return "blocker"

    async def check(self, content: str, context: OrchidGuardrailContext) -> OrchidGuardrailResult:
        return OrchidGuardrailResult(
            triggered=True,
            action=OrchidGuardrailAction.BLOCK,
            guardrail_name="blocker",
            message="blocked",
        )


class _CountingAgent(OrchidAgent):
    def __init__(self, name: str) -> None:
        super().__init__(reader=None)  # type: ignore[arg-type]
        self._name = name
        self.ran = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"{self._name} agent"

    async def run(self, state):
        self.ran = True
        return {"messages": [AIMessage(content=f"{self._name} ran")]}


class _ChatAgent(OrchidAgent):
    def __init__(self, name: str, chat_model: Any) -> None:
        super().__init__(reader=None, chat_model=chat_model)  # type: ignore[arg-type]
        self._name = name
        self.answer = ""

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"{self._name} agent"

    async def run(self, state):
        message = await self.chat_model.ainvoke([HumanMessage(content="q")])
        self.answer = message.content
        return {"messages": [AIMessage(content=self.answer)]}


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _FakeStructured:
    def __init__(self, model: _FakeModel) -> None:
        self._model = model

    async def ainvoke(self, input, config=None, **kwargs):
        self._model.calls.append({"structured": True})
        return {"tag": self._model.tag}

    def invoke(self, input, config=None, **kwargs):
        self._model.calls.append({"structured": True})
        return {"tag": self._model.tag}


def _decision_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == "orchid.decision"]


def _patch_single_model_builder(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    """Replace provider construction with an offline fake that records call order."""

    def fake_build(name: str, **kwargs: Any):
        calls.append(name)
        return _FakeModel(name)

    monkeypatch.setattr("orchid_ai.llm_factory._build_single_model", fake_build)


def _patch_ordered_models(monkeypatch: pytest.MonkeyPatch, models: dict[str, Any]) -> None:
    """Return one concrete fake per model id through the real ``build_chat_model``."""

    def fake_single(name: str, **kwargs: Any):
        return models[name]

    monkeypatch.setattr("orchid_ai.llm_factory._build_single_model", fake_single)


def _agent_entry(description: str, *, decisions: dict[str, Any] | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "description": description,
        "prompt": "you are a test agent",
        "llm": {"model": "test/model"},
        "rag": {"namespace": "kb", "retrieval": {"strategy": "simple"}},
    }
    if decisions is not None:
        entry["decisions"] = {"points": {"model_routing": decisions}}
    return entry


def _graph_config(
    *,
    global_decisions: dict[str, Any] | None = None,
    agent_decisions: dict[str, Any] | None = None,
    budget: dict[str, Any] | None = None,
    include_provider: bool = True,
) -> OrchidAgentsConfig:
    data: dict[str, Any] = {
        "version": "1",
        "agents": {"a": _agent_entry("test agent", decisions=agent_decisions)},
    }
    if include_provider:
        data["decision_models"] = {"stub": {"protocol": "stub"}}
    decisions: dict[str, Any] = {}
    if global_decisions is not None:
        decisions["points"] = {"model_routing": global_decisions}
    if budget is not None:
        decisions["budget"] = budget
    if decisions:
        data["decisions"] = decisions
    return OrchidAgentsConfig.model_validate(data)


def _graph_runtime(*, answer: str | None = None, with_provider: bool = True) -> OrchidRuntime:
    chat = MagicMock(spec=BaseChatModel)
    chat.bind_tools = MagicMock(return_value=chat)
    providers: dict[str, Any] = {}
    if with_provider:
        providers["stub"] = StubDecisionModel(answers={"model": {"choice": answer, "confidence": 0.95}})
    return OrchidRuntime(default_model="test/model", chat_model=chat, decision_models=providers)


def _routed(
    selector: ModelRoutingSelector,
    *,
    default_chat_model: _FakeModel | None = None,
    models: dict[str, _FakeModel] | None = None,
) -> tuple[RoutedChatModel, _FakeModel, dict[str, _FakeModel]]:
    default = default_chat_model or _FakeModel("default")
    catalog = models or {ALT_MODEL: _FakeModel("alt")}

    def factory(model_id: str):
        return catalog[model_id]

    return (
        build_routed_chat_model(selector, default_chat_model=default, model_factory=factory),  # type: ignore[arg-type]
        default,
        catalog,
    )


# ── Selector: accepted and fallback paths ───────────────────


async def test_accepted_model_is_selected_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    provider = StubDecisionModel(answers={"model": {"choice": FALLBACK_MODEL, "confidence": 0.95}})
    selector = _selector(provider)

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        selected = await selector.select("which model should answer?")

    assert selected == FALLBACK_MODEL
    lines = _decision_lines(caplog)
    assert len(lines) == 1
    for fragment in (
        "decision point=model_routing",
        "path=decision",
        "provider=stub",
        "outcome=accepted",
        "fallback_reason=null",
    ):
        assert fragment in lines[0]
    # The request text never reaches the structured line.
    assert "which model" not in lines[0]


async def test_default_model_is_used_on_low_confidence(caplog: pytest.LogCaptureFixture) -> None:
    provider = StubDecisionModel(mode="low_confidence", answers={"model": {"choice": FALLBACK_MODEL}})
    selector = _selector(provider, threshold=0.8)

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        selected = await selector.select("q")

    assert selected == DEFAULT_MODEL
    lines = _decision_lines(caplog)
    assert len(lines) == 1
    assert "path=llm_fallback" in lines[0]
    assert "outcome=fallback" in lines[0]
    assert "fallback_reason=low_confidence" in lines[0]


async def test_refusal_falls_back_to_the_default_model(caplog: pytest.LogCaptureFixture) -> None:
    provider = StubDecisionModel(answers={"model": {"refusal": "not enough context"}})
    selector = _selector(provider)

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        selected = await selector.select("q")

    assert selected == DEFAULT_MODEL
    assert "fallback_reason=refusal" in _decision_lines(caplog)[0]


async def test_unknown_model_falls_back_to_the_default_model(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ChoiceProvider("telepathy/model")
    selector = _selector(provider)

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        selected = await selector.select("q")

    assert selected == DEFAULT_MODEL
    assert "fallback_reason=unknown_model" in _decision_lines(caplog)[0]


async def test_provider_error_falls_back_to_the_default_model(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ErrorProvider("provider_unavailable")
    selector = _selector(provider)

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        selected = await selector.select("q")

    assert selected == DEFAULT_MODEL
    assert "fallback_reason=provider_unavailable" in _decision_lines(caplog)[0]


async def test_timeout_falls_back_to_the_default_model(caplog: pytest.LogCaptureFixture) -> None:
    selector = _selector(_ErrorProvider("timeout"))

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        selected = await selector.select("q")

    assert selected == DEFAULT_MODEL
    assert "fallback_reason=timeout" in _decision_lines(caplog)[0]


async def test_builtin_timeout_falls_back_to_the_default_model(caplog: pytest.LogCaptureFixture) -> None:
    selector = _selector(_TimeoutProvider())

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        selected = await selector.select("q")

    assert selected == DEFAULT_MODEL
    assert "fallback_reason=timeout" in _decision_lines(caplog)[0]


async def test_unexpected_provider_exception_falls_back(caplog: pytest.LogCaptureFixture) -> None:
    selector = _selector(_RaisingProvider())

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        selected = await selector.select("q")

    assert selected == DEFAULT_MODEL
    assert "fallback_reason=provider_error" in _decision_lines(caplog)[0]


async def test_missing_answer_falls_back(caplog: pytest.LogCaptureFixture) -> None:
    selector = _selector(_EmptyProvider())

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        selected = await selector.select("q")

    assert selected == DEFAULT_MODEL
    assert "fallback_reason=invalid_response" in _decision_lines(caplog)[0]


async def test_non_choice_answer_falls_back(caplog: pytest.LogCaptureFixture) -> None:
    selector = _selector(_ChoiceProvider(True, kind="boolean"))

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        selected = await selector.select("q")

    assert selected == DEFAULT_MODEL
    assert "fallback_reason=invalid_response" in _decision_lines(caplog)[0]


async def test_budget_exhaustion_does_not_add_a_provider_call(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ChoiceProvider(ALT_MODEL)
    selector = _selector(provider)
    budget = ModelRoutingBudget(1)
    assert budget.try_acquire() is True
    assert budget.try_acquire() is False

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        selected = await selector.select("q", budget=budget)

    assert selected == DEFAULT_MODEL
    assert provider.questions == []
    assert budget.remaining == 0
    assert "fallback_reason=budget_exhausted" in _decision_lines(caplog)[0]


async def test_empty_catalog_falls_back_without_a_call(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ChoiceProvider(ALT_MODEL)
    selector = _selector(provider, options={"models": []})

    assert selector.catalog == ()

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        selected = await selector.select("q")

    assert selected == DEFAULT_MODEL
    assert provider.questions == []
    assert "fallback_reason=no_candidates" in _decision_lines(caplog)[0]


async def test_single_candidate_catalog_falls_back_without_a_call(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ChoiceProvider(ALT_MODEL)
    selector = _selector(provider, options={"models": [ALT_MODEL]})

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        selected = await selector.select("q")

    assert selected == DEFAULT_MODEL
    assert provider.questions == []
    assert "fallback_reason=no_candidates" in _decision_lines(caplog)[0]


async def test_bound_with_empty_catalog_binds_the_default_without_a_call() -> None:
    provider = _ChoiceProvider(ALT_MODEL)
    selector = _selector(provider, options={"models": []})

    async with selector.bound("q") as selected:
        assert selected == DEFAULT_MODEL
        assert selector.current_model == DEFAULT_MODEL

    assert provider.questions == []
    assert selector.current_model == DEFAULT_MODEL


async def test_catalog_defaults_to_the_configured_model_and_fallback() -> None:
    selector = _selector(_ChoiceProvider(DEFAULT_MODEL))

    assert selector.catalog == (DEFAULT_MODEL, FALLBACK_MODEL)


async def test_option_catalog_replaces_the_default_pair() -> None:
    provider = _ChoiceProvider(ALT_MODEL)
    selector = _selector(provider, options={"models": [DEFAULT_MODEL, ALT_MODEL]})

    assert selector.catalog == (DEFAULT_MODEL, ALT_MODEL)

    selected = await selector.select("q")
    assert selected == ALT_MODEL
    assert provider.questions[0].options == (DEFAULT_MODEL, ALT_MODEL)
    assert provider.questions[0].kind == "choice"
    assert provider.states[0].text == "q"
    assert dict(provider.states[0].payload) == {"agent": "a"}


def test_malformed_catalog_is_treated_as_empty() -> None:
    selector = _selector(_ChoiceProvider(ALT_MODEL), options={"models": "not-a-list"})

    assert selector.catalog == ()


def test_catalog_ignores_non_string_blank_and_duplicate_entries() -> None:
    selector = _selector(
        _ChoiceProvider(ALT_MODEL),
        options={"models": [DEFAULT_MODEL, 3, "", " test/default ", ALT_MODEL, ALT_MODEL]},
    )

    assert selector.catalog == (DEFAULT_MODEL, ALT_MODEL)


def test_budget_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError):
        ModelRoutingBudget(0)
    with pytest.raises(TypeError):
        ModelRoutingBudget(True)  # type: ignore[arg-type]


def test_set_selection_rejects_models_outside_the_catalog() -> None:
    selector = _selector(_ChoiceProvider(ALT_MODEL))

    with pytest.raises(ValueError, match="not in the configured routing catalog"):
        selector.set_selection(ALT_MODEL)


# ── Routed chat model ───────────────────────────────────────


async def test_routed_model_uses_the_selected_model_for_one_run() -> None:
    provider = StubDecisionModel(answers={"model": {"choice": ALT_MODEL, "confidence": 0.95}})
    selector = _selector(provider, options={"models": [DEFAULT_MODEL, ALT_MODEL]})
    router, default, models = _routed(selector)

    async with selector.bound("q"):
        assert selector.current_model == ALT_MODEL
        result = await router.ainvoke([HumanMessage(content="q")])

    assert result.content == "alt"
    assert default.calls == []
    assert len(models[ALT_MODEL].calls) == 1
    # Outside the binding the configured model is used again.
    assert selector.current_model == DEFAULT_MODEL
    await router.ainvoke([HumanMessage(content="q")])
    assert len(default.calls) == 1


async def test_routed_model_selection_is_isolated_between_concurrent_runs() -> None:
    class _PerQuery:
        async def decide(self, state, questions, *, timeout=None):
            raise AssertionError("not used")

    selector = _selector(_PerQuery(), options={"models": [DEFAULT_MODEL, ALT_MODEL]})
    router, default, models = _routed(selector)

    async def _run(model: str) -> str:
        token = selector.set_selection(model)
        try:
            result = await router.ainvoke([HumanMessage(content="q")])
        finally:
            selector.reset_selection(token)
        return result.content

    results = await asyncio.gather(_run(ALT_MODEL), _run(DEFAULT_MODEL))

    assert results == ["alt", "default"]
    assert len(models[ALT_MODEL].calls) == 1
    assert len(default.calls) == 1


async def test_routed_model_bind_tools_reaches_the_selected_model() -> None:
    selector = _selector(
        StubDecisionModel(answers={"model": {"choice": ALT_MODEL, "confidence": 0.95}}),
        options={"models": [DEFAULT_MODEL, ALT_MODEL]},
    )
    router, default, models = _routed(selector)

    async with selector.bound("q"):
        bound = router.bind_tools([{"type": "function", "function": {"name": "lookup"}}])
        await bound.ainvoke([HumanMessage(content="q")])

    assert default.calls == []
    assert models[ALT_MODEL].calls[0]["tools"] == [{"type": "function", "function": {"name": "lookup"}}]


async def test_routed_model_structured_output_reaches_the_selected_model() -> None:
    selector = _selector(
        StubDecisionModel(answers={"model": {"choice": ALT_MODEL, "confidence": 0.95}}),
        options={"models": [DEFAULT_MODEL, ALT_MODEL]},
    )
    router, default, models = _routed(selector)

    async with selector.bound("q"):
        structured = router.with_structured_output(dict)
        result = await structured.ainvoke([HumanMessage(content="q")])

    assert result == {"tag": "alt"}
    assert default.structured == []
    assert models[ALT_MODEL].structured == [dict]


async def test_agent_node_asks_once_and_binds_the_selection_for_the_run() -> None:
    provider = _CountingProvider(ALT_MODEL)
    selector = _selector(provider, options={"models": [DEFAULT_MODEL, ALT_MODEL]})
    default = _FakeModel("default")
    alt = _FakeModel("alt")
    router = build_routed_chat_model(
        selector,
        default_chat_model=default,  # type: ignore[arg-type]
        model_factory=lambda model_id: alt,  # type: ignore[arg-type]
    )
    seen: dict[str, str] = {}

    class _Agent(OrchidAgent):
        @property
        def name(self) -> str:
            return "a"

        @property
        def description(self) -> str:
            return "test agent"

        async def run(self, state):
            seen["selection"] = selector.current_model
            first = await self.chat_model.ainvoke([HumanMessage(content="q")])
            # A second model call in the same run must not ask the point again.
            await self.chat_model.ainvoke([HumanMessage(content="q")])
            return {"messages": [AIMessage(content=first.content)]}

    agent = _Agent(reader=None, chat_model=router)  # type: ignore[arg-type]
    node = _create_agent_node(agent, model_router=selector)

    result = await node({"messages": [HumanMessage(content="q")]})  # type: ignore[arg-type]

    assert provider.calls == 1
    assert seen["selection"] == ALT_MODEL
    assert result["messages"][0].content == "alt"
    assert default.calls == []
    assert len(alt.calls) == 2
    # The selection is scoped to the run and restored afterwards.
    assert selector.current_model == DEFAULT_MODEL


async def test_routed_model_sync_invoke_uses_the_bound_selection() -> None:
    selector = _selector(_ChoiceProvider(ALT_MODEL), options={"models": [DEFAULT_MODEL, ALT_MODEL]})
    router, default, models = _routed(selector)

    token = selector.set_selection(ALT_MODEL)
    try:
        result = router.invoke([HumanMessage(content="q")])
    finally:
        selector.reset_selection(token)

    assert result.content == "alt"
    assert default.calls == []
    assert len(models[ALT_MODEL].calls) == 1


async def test_routed_model_generate_paths_use_the_selected_model() -> None:
    selector = _selector(_ChoiceProvider(ALT_MODEL), options={"models": [DEFAULT_MODEL, ALT_MODEL]})
    router, default, models = _routed(selector)

    token = selector.set_selection(ALT_MODEL)
    try:
        sync_result = router.generate([[HumanMessage(content="q")]])
        async_result = await router.agenerate([[HumanMessage(content="q")]])
    finally:
        selector.reset_selection(token)

    assert sync_result.generations[0][0].message.content == "alt"
    assert async_result.generations[0][0].message.content == "alt"
    assert default.calls == []
    assert len(models[ALT_MODEL].calls) == 2


# ── F1/F3/F5/F7/F9 regressions ──────────────────────────────


async def test_structured_output_resolves_on_the_real_primary_of_a_fallback_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F1: ``RunnableWithFallbacks`` has no ``with_structured_output``."""
    selector = _selector(
        StubDecisionModel(answers={"model": {"choice": ALT_MODEL, "confidence": 0.95}}),
        options={"models": [DEFAULT_MODEL, ALT_MODEL]},
    )
    built: list[str] = []

    def fake_single(name: str, *, temperature: float = 0.2, **kwargs: Any):
        built.append(name)
        return _StructuredChatModel(tag=name)

    monkeypatch.setattr("orchid_ai.llm_factory._build_single_model", fake_single)
    router = build_routed_chat_model(selector)

    async with selector.bound("q"):
        structured = router.with_structured_output(dict)
        sync_result = structured.invoke([HumanMessage(content="q")])
        async_result = await structured.ainvoke([HumanMessage(content="q")])
        invocation = await router.ainvoke([HumanMessage(content="q")])
        # Invocation still goes through the fallback chain; structured output
        # used the raw primary inside it rather than calling a method the
        # chain does not have.
        assert hasattr(router._bundle_for_current_request().invocation, "fallbacks")

    assert sync_result == {"model": ALT_MODEL}
    assert async_result == {"model": ALT_MODEL}
    assert invocation.content == ALT_MODEL
    assert ALT_MODEL in built and DEFAULT_MODEL in built


async def test_bind_tools_delegates_to_the_selected_model_and_omits_none_tool_choice() -> None:
    """F3: bind_tools must reach the selected model's own implementation."""
    provider = _CountingProvider(ALT_MODEL)
    selector = _selector(provider, options={"models": [DEFAULT_MODEL, ALT_MODEL]})
    default = _ToolRecordingModel("default")
    alt = _ToolRecordingModel("alt")
    router = build_routed_chat_model(
        selector,
        default_chat_model=default,  # type: ignore[arg-type]
        model_factory=lambda model_id: alt,  # type: ignore[arg-type]
    )
    tools = [{"type": "function", "function": {"name": "lookup"}}]

    async with selector.bound("q"):
        bound = router.bind_tools(tools)
        # Resolution is deferred to invocation time.
        assert alt.bind_calls == []
        await bound.ainvoke([HumanMessage(content="q")])
        chosen = router.bind_tools(tools, tool_choice="auto")
        await chosen.ainvoke([HumanMessage(content="q")])

    assert alt.bind_calls == [(tools, {}), (tools, {"tool_choice": "auto"})]
    assert default.bind_calls == []


def test_fallback_wrapped_default_model_exposes_its_raw_primary() -> None:
    """F1: a default model may itself be a fallback chain built by the graph."""
    selector = _selector(_ChoiceProvider(DEFAULT_MODEL))
    primary = _StructuredChatModel(tag=DEFAULT_MODEL)
    fallback = _StructuredChatModel(tag=ALT_MODEL)
    wrapped = primary.with_fallbacks([fallback])

    router = build_routed_chat_model(selector, default_chat_model=wrapped)
    bundle = router._bundle_for_current_request()

    assert bundle.primary is primary
    assert bundle.fallbacks == (fallback,)
    assert bundle.invocation is wrapped


async def test_real_factory_bundle_uses_raw_primary_and_ordered_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F1: the real factory path exposes a raw primary and ordered fallbacks."""
    selector = _selector(
        _ChoiceProvider(ALT_MODEL),
        options={"models": [DEFAULT_MODEL, FALLBACK_MODEL, ALT_MODEL]},
    )
    models = {name: _OrderedChatModel(tag=name) for name in (DEFAULT_MODEL, FALLBACK_MODEL, ALT_MODEL)}
    _patch_ordered_models(monkeypatch, models)

    router = build_routed_chat_model(selector, fallback_model=FALLBACK_MODEL)
    async with selector.bound("q"):
        bundle = router._bundle_for_current_request()

    assert isinstance(bundle.primary, BaseChatModel)
    assert all(isinstance(model, BaseChatModel) for model in bundle.fallbacks)
    assert [model.tag for model in bundle.fallbacks] == [FALLBACK_MODEL, DEFAULT_MODEL]

    # The default path must also expose a real BaseChatModel primary.
    default_selector = _selector(
        _ChoiceProvider(DEFAULT_MODEL),
        options={"models": [DEFAULT_MODEL, FALLBACK_MODEL, ALT_MODEL]},
    )
    default_router = build_routed_chat_model(default_selector, fallback_model=FALLBACK_MODEL)
    default_bundle = default_router._bundle_for_current_request()

    assert isinstance(default_bundle.primary, BaseChatModel)
    assert [model.tag for model in default_bundle.fallbacks] == [FALLBACK_MODEL]


async def test_structured_output_fallback_order_with_real_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F1: selected -> configured fallback -> default, on the real factory."""
    recorder = _CallRecorder()
    selector = _selector(
        _ChoiceProvider(ALT_MODEL),
        options={"models": [DEFAULT_MODEL, FALLBACK_MODEL, ALT_MODEL]},
    )
    models = {
        ALT_MODEL: _OrderedChatModel(tag=ALT_MODEL, fail_structured=True, recorder=recorder),
        FALLBACK_MODEL: _OrderedChatModel(tag=FALLBACK_MODEL, fail_structured=True, recorder=recorder),
        DEFAULT_MODEL: _OrderedChatModel(tag=DEFAULT_MODEL, recorder=recorder),
    }
    _patch_ordered_models(monkeypatch, models)

    router = build_routed_chat_model(selector, fallback_model=FALLBACK_MODEL)
    async with selector.bound("q"):
        result = await router.with_structured_output(dict).ainvoke([HumanMessage(content="q")])

    assert result == {"tag": DEFAULT_MODEL}
    assert recorder.calls == [
        f"structured:{ALT_MODEL}",
        f"structured:{FALLBACK_MODEL}",
        f"structured:{DEFAULT_MODEL}",
    ]


async def test_tool_call_fallback_order_with_real_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F1/F3: selected -> configured fallback -> default for tool calls."""
    recorder = _CallRecorder()
    selector = _selector(
        _ChoiceProvider(ALT_MODEL),
        options={"models": [DEFAULT_MODEL, FALLBACK_MODEL, ALT_MODEL]},
    )
    models = {
        ALT_MODEL: _OrderedChatModel(tag=ALT_MODEL, fail_tools=True, recorder=recorder),
        FALLBACK_MODEL: _OrderedChatModel(tag=FALLBACK_MODEL, fail_tools=True, recorder=recorder),
        DEFAULT_MODEL: _OrderedChatModel(tag=DEFAULT_MODEL, recorder=recorder),
    }
    _patch_ordered_models(monkeypatch, models)

    router = build_routed_chat_model(selector, fallback_model=FALLBACK_MODEL)
    async with selector.bound("q"):
        result = await router.bind_tools(
            [{"type": "function", "function": {"name": "lookup"}}],
        ).ainvoke([HumanMessage(content="q")])

    assert result.content == DEFAULT_MODEL
    assert recorder.calls == [
        f"tools:{ALT_MODEL}",
        f"tools:{FALLBACK_MODEL}",
        f"tools:{DEFAULT_MODEL}",
    ]


async def test_direct_default_bundle_uses_a_raw_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    """F1: the direct ``RoutedChatModel`` factory path is raw too."""
    selector = _selector(
        _ChoiceProvider(ALT_MODEL),
        options={"models": [DEFAULT_MODEL, FALLBACK_MODEL, ALT_MODEL]},
    )
    models = {name: _OrderedChatModel(tag=name) for name in (DEFAULT_MODEL, FALLBACK_MODEL, ALT_MODEL)}
    _patch_ordered_models(monkeypatch, models)

    router = RoutedChatModel(selector=selector)
    async with selector.bound("q"):
        bundle = router._bundle_for_current_request()

    assert isinstance(bundle.primary, BaseChatModel)
    assert [model.tag for model in bundle.fallbacks] == [FALLBACK_MODEL, DEFAULT_MODEL]


async def test_routed_model_streams_through_the_selected_model() -> None:
    """F5: sync and async streaming delegate to the selected model."""
    selector = _selector(_ChoiceProvider(ALT_MODEL), options={"models": [DEFAULT_MODEL, ALT_MODEL]})
    default = _StreamingModel("default")
    alt = _StreamingModel("alt")
    router = build_routed_chat_model(
        selector,
        default_chat_model=default,  # type: ignore[arg-type]
        model_factory=lambda model_id: alt,  # type: ignore[arg-type]
    )

    async with selector.bound("q"):
        sync_chunks = list(router.stream([HumanMessage(content="q")]))
        async_chunks = [chunk async for chunk in router.astream([HumanMessage(content="q")])]

    assert "".join(chunk.content for chunk in sync_chunks) == "alt"
    assert "".join(chunk.content for chunk in async_chunks) == "alt"
    assert all(isinstance(chunk, AIMessageChunk) for chunk in sync_chunks + async_chunks)
    assert default.calls == []
    assert len(alt.calls) == 2


async def test_selected_model_build_failure_falls_back_to_the_configured_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """F7: a malformed catalog entry must not fail the request."""
    selector = _selector(_ChoiceProvider(ALT_MODEL), options={"models": [DEFAULT_MODEL, ALT_MODEL]})
    default = _FakeModel("default")

    def broken_factory(model_id: str):
        raise RuntimeError(f"boom {SENTINEL}")

    router = build_routed_chat_model(
        selector,
        default_chat_model=default,  # type: ignore[arg-type]
        model_factory=broken_factory,
    )

    with caplog.at_level(logging.WARNING, logger="orchid_ai.graph.model_routing"):
        async with selector.bound("q"):
            result = await router.ainvoke([HumanMessage(content="q")])

    assert result.content == "default"
    assert len(default.calls) == 1
    assert "could not be built" in caplog.text
    assert "RuntimeError" in caplog.text
    assert SENTINEL not in caplog.text


async def test_unexpected_provider_error_logs_only_the_exception_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """F9: no traceback and no provider message reaches the logs."""
    selector = _selector(_LeakyProvider())

    with (
        caplog.at_level(logging.WARNING, logger="orchid_ai.graph.model_routing"),
        caplog.at_level(logging.INFO, logger="orchid.decision"),
    ):
        selected = await selector.select("q")

    assert selected == DEFAULT_MODEL
    assert SENTINEL not in caplog.text
    assert "_LeakyError" in caplog.text
    lines = _decision_lines(caplog)
    assert len(lines) == 1
    assert "fallback_reason=provider_error" in lines[0]


# ── llm_factory selection ───────────────────────────────────


def test_build_chat_model_without_selection_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _patch_single_model_builder(monkeypatch, calls)

    model = build_chat_model(DEFAULT_MODEL)

    assert calls == [DEFAULT_MODEL]
    assert isinstance(model, _FakeModel)


def test_build_chat_model_selected_model_keeps_the_default_as_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _patch_single_model_builder(monkeypatch, calls)

    model = build_chat_model(DEFAULT_MODEL, selected_model=ALT_MODEL)

    assert calls == [ALT_MODEL, DEFAULT_MODEL]
    assert model.tag == ALT_MODEL


def test_build_chat_model_selected_model_uses_the_configured_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _patch_single_model_builder(monkeypatch, calls)

    build_chat_model(DEFAULT_MODEL, selected_model=ALT_MODEL, fallback_model=FALLBACK_MODEL)

    assert calls == [ALT_MODEL, FALLBACK_MODEL, DEFAULT_MODEL]


def test_build_chat_model_selected_equal_to_default_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _patch_single_model_builder(monkeypatch, calls)

    build_chat_model(DEFAULT_MODEL, selected_model=DEFAULT_MODEL)

    assert calls == [DEFAULT_MODEL]


# ── Graph wiring ────────────────────────────────────────────


def test_disabled_point_keeps_the_shared_default_model() -> None:
    config = _graph_config(global_decisions={"model": "stub", "enabled": False})
    runtime = _graph_runtime(answer=ALT_MODEL)
    agents: dict[str, Any] = {}

    build_graph(config=config, runtime=runtime, agents_out=agents)

    agent = agents["a"]
    assert agent.chat_model is runtime.chat_model
    assert not isinstance(agent.chat_model, RoutedChatModel)
    assert resolve_decision_binding(config, runtime, "model_routing", agent_name="a") is None


def test_missing_runtime_provider_keeps_the_default_model(caplog: pytest.LogCaptureFixture) -> None:
    config = _graph_config(global_decisions={"model": "stub"})
    runtime = _graph_runtime(with_provider=False)
    agents: dict[str, Any] = {}

    with caplog.at_level(logging.WARNING, logger="orchid_ai.graph.decision_bindings"):
        build_graph(config=config, runtime=runtime, agents_out=agents)

    assert agents["a"].chat_model is runtime.chat_model
    assert "missing from the runtime" in caplog.text


def test_graph_top_level_agent_receives_the_model_router() -> None:
    config = _graph_config(
        agent_decisions={"model": "stub", "options": {"models": ["test/model", ALT_MODEL]}},
        budget={"max_calls_per_turn": 2, "max_total_ms": 5000},
    )
    runtime = _graph_runtime(answer=ALT_MODEL)
    agents: dict[str, Any] = {}

    build_graph(config=config, runtime=runtime, agents_out=agents)

    chat_model = agents["a"].chat_model
    assert isinstance(chat_model, RoutedChatModel)
    selector = chat_model.selector
    assert selector.point == "model_routing"
    assert selector.default_model == "test/model"
    assert selector.catalog == ("test/model", ALT_MODEL)
    assert selector.max_calls_per_turn == 2
    assert selector.max_total_ms == 5000


def test_graph_child_agent_receives_its_own_model_router(monkeypatch: pytest.MonkeyPatch) -> None:
    child = _agent_entry("child", decisions={"model": "stub"})
    sibling = _agent_entry("sibling")
    parent = _agent_entry("parent")
    parent["children"] = {"kid": child, "sibling": sibling}
    data = {
        "version": "1",
        "agents": {"parent": parent},
        "decision_models": {"stub": {"protocol": "stub"}},
    }
    config = OrchidAgentsConfig.model_validate(data)
    runtime = _graph_runtime(answer=ALT_MODEL)

    created: dict[str, Any] = {}
    real_instantiate = graph_module._instantiate_agent

    def _capture(name, agent_config, *args, **kwargs):
        agent = real_instantiate(name, agent_config, *args, **kwargs)
        created[name] = agent
        return agent

    monkeypatch.setattr(graph_module, "_instantiate_agent", _capture)
    build_graph(config=config, runtime=runtime)

    child_agent = created["kid"]
    sibling_agent = created["sibling"]
    child_model = child_agent.chat_model
    assert isinstance(child_model, RoutedChatModel)
    assert child_model.selector.catalog == ("test/model",)
    # The sibling has no binding and keeps the shared default model.
    assert not isinstance(sibling_agent.chat_model, RoutedChatModel)
    assert sibling_agent.chat_model is runtime.chat_model


async def test_graph_child_router_uses_the_selected_model(monkeypatch: pytest.MonkeyPatch) -> None:
    child = _agent_entry(
        "child",
        decisions={"model": "stub", "options": {"models": ["test/model", ALT_MODEL]}},
    )
    parent = _agent_entry("parent")
    parent["children"] = {"kid": child}
    config = OrchidAgentsConfig.model_validate(
        {
            "version": "1",
            "agents": {"parent": parent},
            "decision_models": {"stub": {"protocol": "stub"}},
        }
    )
    runtime = _graph_runtime(answer=ALT_MODEL)

    created: dict[str, Any] = {}
    real_instantiate = graph_module._instantiate_agent

    def _capture(name, agent_config, *args, **kwargs):
        agent = real_instantiate(name, agent_config, *args, **kwargs)
        created[name] = agent
        return agent

    recorder = _CallRecorder()
    alt = _OrderedChatModel(tag=ALT_MODEL, recorder=recorder)
    models = {"test/model": _OrderedChatModel(tag="test/model"), ALT_MODEL: alt}
    _patch_ordered_models(monkeypatch, models)

    monkeypatch.setattr(graph_module, "_instantiate_agent", _capture)
    build_graph(config=config, runtime=runtime)
    child_model = created["kid"].chat_model
    assert isinstance(child_model, RoutedChatModel)
    async with child_model.selector.bound("q"):
        result = await child_model.ainvoke([HumanMessage(content="q")])

    assert result.content == ALT_MODEL
    assert recorder.calls == [f"invoke:{ALT_MODEL}"]


def test_retrieval_binding_wiring_is_untouched() -> None:
    config = OrchidAgentsConfig.model_validate(
        {
            "version": "1",
            "agents": {"a": _agent_entry("test agent")},
            "decision_models": {"stub": {"protocol": "stub"}},
            "decisions": {
                "points": {
                    "retrieval_strategy": {"model": "stub"},
                    "model_routing": {"model": "stub"},
                }
            },
        }
    )
    runtime = _graph_runtime(answer=ALT_MODEL)
    agents: dict[str, Any] = {}

    build_graph(config=config, runtime=runtime, agents_out=agents)

    triangle = agents["a"]
    assert triangle._rag_pipeline._retrieval_strategy_binding is not None
    assert triangle._rag_pipeline._retrieval_strategy_binding.point == "retrieval_strategy"
    assert isinstance(triangle.chat_model, RoutedChatModel)


# ── F2/F4 guardrail ordering and shared turn budget ─────────


async def test_input_guardrails_block_before_model_routing(caplog: pytest.LogCaptureFixture) -> None:
    """F2: blocked input must never reach the routing decision provider."""
    provider = _CountingProvider(ALT_MODEL)
    selector = _selector(provider, options={"models": [DEFAULT_MODEL, ALT_MODEL]})
    agent = _CountingAgent("a")
    node = _create_agent_node(
        agent,
        input_guardrails=OrchidGuardrailChain([_BlockingRail()]),
        model_router=selector,
        turn_budgets=TurnBudgetRegistry(max_calls_per_turn=4),
    )

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        result = await node({"messages": [HumanMessage(content="secret query")], "chat_id": "c1"})  # type: ignore[arg-type]

    assert provider.calls == 0
    assert provider.questions == []
    assert agent.ran is False
    assert result["messages"][0].content == "[A Agent] blocked"
    assert _decision_lines(caplog) == []


async def test_budget_wall_clock_expiry_blocks_the_provider_call() -> None:
    """F4: ``max_total_ms`` bounds the call itself and later calls do not run."""
    provider = _CountingProvider(ALT_MODEL)
    selector = _selector(provider, options={"models": [DEFAULT_MODEL, ALT_MODEL]})
    clock = _Clock()
    budget = ModelRoutingBudget(4, max_total_ms=1000, clock=clock)

    selected = await selector.select("q", budget=budget)
    assert selected == ALT_MODEL
    # The provider keeps its own configured timeout; the turn ceiling is an
    # outer cancellation boundary.
    assert provider.timeouts == [None]

    clock.now = 2.0  # past the deadline
    selected = await selector.select("q", budget=budget)

    assert selected == DEFAULT_MODEL
    assert provider.calls == 1
    assert len(provider.questions) == 1
    assert budget.expired is True


async def test_wall_clock_timeout_cancels_a_slow_provider(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """F4: ``max_total_ms`` bounds a call that is already in flight."""
    provider = _SlowProvider()
    selector = _selector(provider, options={"models": [DEFAULT_MODEL, ALT_MODEL]})
    budget = ModelRoutingBudget(4, max_total_ms=10)

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        selected = await selector.select("q", budget=budget)

    assert selected == DEFAULT_MODEL
    assert provider.calls == 1
    assert "fallback_reason=timeout" in _decision_lines(caplog)[0]


async def test_shared_turn_budget_exhausts_on_the_second_agent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """F4: the second agent sees the first agent's spent turn budget."""
    provider = _CountingProvider(ALT_MODEL)
    options = {"models": [DEFAULT_MODEL, ALT_MODEL]}
    selector_a = _selector(provider, options=options)
    selector_b = _selector(provider, options=options)

    default_a = _FakeModel("default-a")
    alt_a = _FakeModel("alt-a")
    default_b = _FakeModel("default-b")
    alt_b = _FakeModel("alt-b")
    router_a = build_routed_chat_model(
        selector_a,
        default_chat_model=default_a,  # type: ignore[arg-type]
        model_factory=lambda model_id: alt_a,  # type: ignore[arg-type]
    )
    router_b = build_routed_chat_model(
        selector_b,
        default_chat_model=default_b,  # type: ignore[arg-type]
        model_factory=lambda model_id: alt_b,  # type: ignore[arg-type]
    )

    registry = TurnBudgetRegistry(max_calls_per_turn=1)
    node_a = _create_agent_node(_ChatAgent("a", router_a), model_router=selector_a, turn_budgets=registry)
    node_b = _create_agent_node(_ChatAgent("b", router_b), model_router=selector_b, turn_budgets=registry)
    state = {"messages": [HumanMessage(content="same turn", id="m-1")], "chat_id": "c1"}

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        await node_a(state)  # type: ignore[arg-type]
        await node_b(state)  # type: ignore[arg-type]

    assert provider.calls == 1
    assert len(provider.questions) == 1
    assert default_a.calls == []
    assert len(alt_a.calls) == 1
    assert alt_b.calls == []
    assert len(default_b.calls) == 1
    lines = _decision_lines(caplog)
    assert len(lines) == 2
    assert "fallback_reason=null" in lines[0]
    assert "fallback_reason=budget_exhausted" in lines[1]
