"""Tests for decision-backed retrieval-strategy selection.

The ``retrieval_strategy`` point selects the strategy *before* retrieval. It is
independent from the legacy ``rag`` chunk gate: a provider failure, refusal,
low confidence, timeout, unknown name, or a disabled point must leave the
configured strategy untouched, and the gate must keep filtering passages.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel

import orchid_ai.graph.graph as graph_module
from orchid_ai.agents.rag_pipeline import RagPipeline
from orchid_ai.config.schema import OrchidAgentsConfig
from orchid_ai.config.schema_decision import OrchidDecisionPointConfig
from orchid_ai.config.schema_rag import OrchidGraphRetrievalConfig, OrchidRetrievalConfig
from orchid_ai.core.decision import (
    OrchidDecisionAnswer,
    OrchidDecisionError,
    OrchidDecisionQuestion,
    OrchidDecisionResult,
    OrchidDecisionState,
)
from orchid_ai.core.graph_store import OrchidEdge, OrchidEntity
from orchid_ai.core.repository import OrchidDocument, OrchidSearchResult
from orchid_ai.decision.registry import get_decision_point
from orchid_ai.decision.stub import StubDecisionModel
from orchid_ai.graph.decision_bindings import DecisionBinding, resolve_decision_binding
from orchid_ai.graph.graph import build_graph
from orchid_ai.rag.backends.in_memory_graph import InMemoryGraphStore
from orchid_ai.rag.backends.null import NullVectorReader
from orchid_ai.rag.scopes import OrchidRAGScope
from orchid_ai.rag.strategies import RETRIEVAL_REGISTRY
from orchid_ai.runtime import OrchidRuntime


def _scope() -> OrchidRAGScope:
    return OrchidRAGScope(tenant_id="t", user_id="u", chat_id="c", agent_id="a")


def _binding(provider: Any, *, threshold: float = 0.75, options: dict[str, Any] | None = None) -> DecisionBinding:
    return DecisionBinding(
        spec=get_decision_point("retrieval_strategy"),
        config=OrchidDecisionPointConfig(model="stub", confidence_threshold=threshold, options=options or {}),
        model=provider,
    )


def _pipeline(
    *,
    binding: Any | None = None,
    gate_model: Any | None = None,
    graph_store: Any | None = None,
    strategies: list[str] | None = None,
) -> RagPipeline:
    return RagPipeline(
        NullVectorReader(),
        graph_store=graph_store,
        decision_model=gate_model,
        strategies=strategies,
        retrieval_strategy_binding=binding,
    )


class _FakeStrategy:
    def __init__(self, results: list[OrchidSearchResult] | None = None) -> None:
        self.results = results or []
        self.calls: list[dict[str, Any]] = []

    async def retrieve(self, **kwargs: Any) -> list[OrchidSearchResult]:
        self.calls.append(kwargs)
        if kwargs["namespace"] == "uploads":
            return []
        return list(self.results)


class _ChoiceProvider:
    """Return one fixed choice even when it is outside the question options."""

    def __init__(self, value: str, *, confidence: float = 0.95) -> None:
        self.value = value
        self.confidence = confidence
        self.questions: list[OrchidDecisionQuestion] = []
        self.states: list[OrchidDecisionState] = []

    async def decide(self, state, questions, *, timeout=None):
        self.states.append(state)
        self.questions.extend(questions)
        return OrchidDecisionResult(
            answers={
                "strategy": OrchidDecisionAnswer(
                    name="strategy",
                    kind="choice",
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

    async def decide(self, state, questions, *, timeout=None):
        raise OrchidDecisionError("provider failed", code=self.code, provider="error-provider")


class _GateRecorder:
    def __init__(self, selected: set[int]) -> None:
        self.selected = selected
        self.questions: list[OrchidDecisionQuestion] = []
        self.state: OrchidDecisionState | None = None

    async def decide(self, state, questions, *, timeout=None):
        self.state = state
        self.questions = list(questions)
        answers = {}
        for index, question in enumerate(questions):
            keep = index in self.selected
            answers[question.name] = OrchidDecisionAnswer(
                name=question.name,
                kind="boolean",
                value=keep,
                confidence=0.95 if keep else 0.9,
            )
        return OrchidDecisionResult(answers=answers, provider="gate", model="gate")


def _decision_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == "orchid.decision"]


def _graph_config(*, agent_level: bool = False) -> OrchidAgentsConfig:
    agent: dict[str, Any] = {
        "description": "test agent",
        "prompt": "you are a test agent",
        # Match the runtime default so ``build_graph`` reuses the injected
        # chat model instead of importing a provider SDK.
        "llm": {"model": "test/model"},
        "rag": {"namespace": "kb", "retrieval": {"strategy": "simple"}},
    }
    data: dict[str, Any] = {
        "version": "1",
        "agents": {"a": agent},
        "decision_models": {"stub": {"protocol": "stub"}},
    }
    if agent_level:
        agent["decisions"] = {"points": {"retrieval_strategy": {"model": "stub"}}}
    else:
        data["decisions"] = {"points": {"retrieval_strategy": {"model": "stub"}}}
    return OrchidAgentsConfig.model_validate(data)


def _agent_entry(description: str) -> dict[str, Any]:
    return {
        "description": description,
        "prompt": "you are a test agent",
        "llm": {"model": "test/model"},
        "rag": {"namespace": "kb", "retrieval": {"strategy": "simple"}},
    }


def _parent_child_config(
    *,
    child_decision: bool = False,
    top_level_kid: bool = False,
) -> OrchidAgentsConfig:
    child = _agent_entry("child")
    if child_decision:
        child["decisions"] = {"points": {"retrieval_strategy": {"model": "stub"}}}

    parent = _agent_entry("parent")
    parent["children"] = {"kid": child, "sibling": _agent_entry("sibling")}

    agents: dict[str, Any] = {"parent": parent}
    if top_level_kid:
        agents["kid"] = _agent_entry("top-level kid")

    data: dict[str, Any] = {"version": "1", "agents": agents}
    if child_decision:
        data["decision_models"] = {"stub": {"protocol": "stub"}}
    return OrchidAgentsConfig.model_validate(data)


def _graph_runtime(*, decision_answer: str | None = None) -> OrchidRuntime:
    chat = MagicMock(spec=BaseChatModel)
    chat.bind_tools = MagicMock(return_value=chat)
    providers: dict[str, Any] = {}
    if decision_answer is not None:
        providers["stub"] = StubDecisionModel(answers={"strategy": {"choice": decision_answer, "confidence": 0.95}})
    return OrchidRuntime(default_model="test/model", chat_model=chat, decision_models=providers)


async def test_accepted_strategy_is_used(caplog: pytest.LogCaptureFixture) -> None:
    provider = StubDecisionModel(answers={"strategy": {"choice": "hyde", "confidence": 0.95}})
    fake = _FakeStrategy()

    with (
        caplog.at_level(logging.INFO, logger="orchid.decision"),
        patch("orchid_ai.agents.rag_pipeline.get_retrieval_strategy", return_value=fake) as get,
    ):
        result = await _pipeline(binding=_binding(provider)).retrieve(
            "billing question",
            _scope(),
            rag_namespace="kb",
            k=3,
            enabled=True,
            retrieval_strategy="simple",
        )

    assert result == []
    assert get.call_args.args[0] == "hyde"
    lines = _decision_lines(caplog)
    assert len(lines) == 1
    for fragment in (
        "decision point=retrieval_strategy",
        "path=decision",
        "provider=stub",
        "outcome=accepted",
        "fallback_reason=null",
    ):
        assert fragment in lines[0]


async def test_low_confidence_falls_back_to_the_configured_strategy(caplog: pytest.LogCaptureFixture) -> None:
    provider = StubDecisionModel(mode="low_confidence")
    fake = _FakeStrategy()

    with (
        caplog.at_level(logging.INFO, logger="orchid.decision"),
        patch("orchid_ai.agents.rag_pipeline.get_retrieval_strategy", return_value=fake) as get,
    ):
        await _pipeline(binding=_binding(provider, threshold=0.8)).retrieve(
            "q",
            _scope(),
            rag_namespace="kb",
            k=3,
            enabled=True,
            retrieval_strategy="simple",
        )

    assert get.call_args.args[0] == "simple"
    lines = _decision_lines(caplog)
    assert len(lines) == 1
    assert "path=llm_fallback" in lines[0]
    assert "outcome=fallback" in lines[0]
    assert "fallback_reason=low_confidence" in lines[0]


async def test_refusal_falls_back_to_the_configured_strategy(caplog: pytest.LogCaptureFixture) -> None:
    provider = StubDecisionModel(answers={"strategy": {"refusal": "not enough context"}})
    fake = _FakeStrategy()

    with (
        caplog.at_level(logging.INFO, logger="orchid.decision"),
        patch("orchid_ai.agents.rag_pipeline.get_retrieval_strategy", return_value=fake) as get,
    ):
        await _pipeline(binding=_binding(provider)).retrieve(
            "q",
            _scope(),
            rag_namespace="kb",
            k=3,
            enabled=True,
            retrieval_strategy="simple",
        )

    assert get.call_args.args[0] == "simple"
    lines = _decision_lines(caplog)
    assert len(lines) == 1
    assert "fallback_reason=refusal" in lines[0]


async def test_custom_registered_strategy_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    class _CustomStrategy:
        @classmethod
        def from_config(cls, config):
            return cls()

        async def retrieve(self, **kwargs):
            return []

    monkeypatch.setitem(RETRIEVAL_REGISTRY, "custom", _CustomStrategy)
    provider = _ChoiceProvider("custom")
    fake = _FakeStrategy()

    with patch("orchid_ai.agents.rag_pipeline.get_retrieval_strategy", return_value=fake) as get:
        await _pipeline(binding=_binding(provider)).retrieve(
            "q",
            _scope(),
            rag_namespace="kb",
            k=3,
            enabled=True,
            retrieval_strategy="simple",
        )

    assert get.call_args.args[0] == "custom"
    assert "custom" in provider.questions[0].options


async def test_registered_but_excluded_strategy_falls_back_out_of_allowlist(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _ChoiceProvider("graph_rag")
    fake = _FakeStrategy()

    with (
        caplog.at_level(logging.INFO, logger="orchid.decision"),
        patch("orchid_ai.agents.rag_pipeline.get_retrieval_strategy", return_value=fake) as get,
    ):
        await _pipeline(binding=_binding(provider, options={"strategies": ["simple", "hyde"]})).retrieve(
            "q",
            _scope(),
            rag_namespace="kb",
            k=3,
            enabled=True,
            retrieval_strategy="simple",
        )

    assert get.call_args.args[0] == "simple"
    assert "fallback_reason=out_of_allowlist" in _decision_lines(caplog)[0]


async def test_unknown_strategy_falls_back_to_the_configured_strategy(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ChoiceProvider("telepathy")
    fake = _FakeStrategy()

    with (
        caplog.at_level(logging.INFO, logger="orchid.decision"),
        patch("orchid_ai.agents.rag_pipeline.get_retrieval_strategy", return_value=fake) as get,
    ):
        await _pipeline(binding=_binding(provider)).retrieve(
            "q",
            _scope(),
            rag_namespace="kb",
            k=3,
            enabled=True,
            retrieval_strategy="simple",
        )

    assert get.call_args.args[0] == "simple"
    assert "fallback_reason=unknown_strategy" in _decision_lines(caplog)[0]


async def test_provider_error_falls_back_to_the_configured_strategy(caplog: pytest.LogCaptureFixture) -> None:
    fake = _FakeStrategy()

    with (
        caplog.at_level(logging.INFO, logger="orchid.decision"),
        patch("orchid_ai.agents.rag_pipeline.get_retrieval_strategy", return_value=fake) as get,
    ):
        await _pipeline(binding=_binding(StubDecisionModel(mode="fail"))).retrieve(
            "q",
            _scope(),
            rag_namespace="kb",
            k=3,
            enabled=True,
            retrieval_strategy="simple",
        )

    assert get.call_args.args[0] == "simple"
    assert "fallback_reason=provider_unavailable" in _decision_lines(caplog)[0]


async def test_unexpected_provider_error_falls_back_to_the_configured_strategy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _RaisingProvider:
        async def decide(self, state, questions, *, timeout=None):
            raise RuntimeError("unexpected")

    fake = _FakeStrategy()

    with (
        caplog.at_level(logging.INFO, logger="orchid.decision"),
        patch("orchid_ai.agents.rag_pipeline.get_retrieval_strategy", return_value=fake) as get,
    ):
        await _pipeline(binding=_binding(_RaisingProvider())).retrieve(
            "q",
            _scope(),
            rag_namespace="kb",
            k=3,
            enabled=True,
            retrieval_strategy="simple",
        )

    assert get.call_args.args[0] == "simple"
    assert "fallback_reason=provider_error" in _decision_lines(caplog)[0]


async def test_missing_answer_falls_back_to_the_configured_strategy(caplog: pytest.LogCaptureFixture) -> None:
    class _EmptyProvider:
        async def decide(self, state, questions, *, timeout=None):
            return OrchidDecisionResult(answers={}, provider="empty", model="empty")

    fake = _FakeStrategy()

    with (
        caplog.at_level(logging.INFO, logger="orchid.decision"),
        patch("orchid_ai.agents.rag_pipeline.get_retrieval_strategy", return_value=fake) as get,
    ):
        await _pipeline(binding=_binding(_EmptyProvider())).retrieve(
            "q",
            _scope(),
            rag_namespace="kb",
            k=3,
            enabled=True,
            retrieval_strategy="simple",
        )

    assert get.call_args.args[0] == "simple"
    assert "fallback_reason=invalid_response" in _decision_lines(caplog)[0]


async def test_too_few_registered_strategies_falls_back_without_a_call(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _ChoiceProvider("simple")
    fake = _FakeStrategy()
    monkeypatch.setattr("orchid_ai.agents.rag_pipeline.RETRIEVAL_REGISTRY", {"only": object()})

    with (
        caplog.at_level(logging.INFO, logger="orchid.decision"),
        patch("orchid_ai.agents.rag_pipeline.get_retrieval_strategy", return_value=fake) as get,
    ):
        await _pipeline(binding=_binding(provider)).retrieve(
            "q",
            _scope(),
            rag_namespace="kb",
            k=3,
            enabled=True,
            retrieval_strategy="simple",
        )

    assert get.call_args.args[0] == "simple"
    assert provider.questions == []
    assert "fallback_reason=no_candidates" in _decision_lines(caplog)[0]


async def test_timeout_falls_back_to_the_configured_strategy(caplog: pytest.LogCaptureFixture) -> None:
    fake = _FakeStrategy()

    with (
        caplog.at_level(logging.INFO, logger="orchid.decision"),
        patch("orchid_ai.agents.rag_pipeline.get_retrieval_strategy", return_value=fake) as get,
    ):
        await _pipeline(binding=_binding(_ErrorProvider("timeout"))).retrieve(
            "q",
            _scope(),
            rag_namespace="kb",
            k=3,
            enabled=True,
            retrieval_strategy="simple",
        )

    assert get.call_args.args[0] == "simple"
    assert "fallback_reason=timeout" in _decision_lines(caplog)[0]


async def test_disabled_point_uses_the_configured_strategy_without_a_decision_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = _FakeStrategy()

    with (
        caplog.at_level(logging.INFO, logger="orchid.decision"),
        patch("orchid_ai.agents.rag_pipeline.get_retrieval_strategy", return_value=fake) as get,
    ):
        await _pipeline(binding=None).retrieve(
            "q",
            _scope(),
            rag_namespace="kb",
            k=3,
            enabled=True,
            retrieval_strategy="hyde",
        )

    assert get.call_args.args[0] == "hyde"
    assert _decision_lines(caplog) == []


async def test_missing_runtime_provider_yields_no_binding_and_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _graph_config()
    runtime = OrchidRuntime(default_model="test/model")
    with caplog.at_level(logging.WARNING, logger="orchid_ai.graph.decision_bindings"):
        binding = resolve_decision_binding(config, runtime, "retrieval_strategy", agent_name="a")
    assert binding is None
    assert "missing from the runtime" in caplog.text

    fake = _FakeStrategy()
    with patch("orchid_ai.agents.rag_pipeline.get_retrieval_strategy", return_value=fake) as get:
        await _pipeline(binding=binding).retrieve(
            "q",
            _scope(),
            rag_namespace="kb",
            k=3,
            enabled=True,
            retrieval_strategy="simple",
        )
    assert get.call_args.args[0] == "simple"


async def test_option_allowlist_narrows_the_candidates() -> None:
    provider = _ChoiceProvider("hyde")
    fake = _FakeStrategy()

    with patch("orchid_ai.agents.rag_pipeline.get_retrieval_strategy", return_value=fake) as get:
        await _pipeline(binding=_binding(provider, options={"strategies": ["simple", "hyde"]})).retrieve(
            "q",
            _scope(),
            rag_namespace="kb",
            k=3,
            enabled=True,
            retrieval_strategy="simple",
        )

    assert get.call_args.args[0] == "hyde"
    assert provider.questions[0].options == ("hyde", "simple")


async def test_option_allowlist_with_too_few_candidates_falls_back(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ChoiceProvider("simple")
    fake = _FakeStrategy()

    with (
        caplog.at_level(logging.INFO, logger="orchid.decision"),
        patch("orchid_ai.agents.rag_pipeline.get_retrieval_strategy", return_value=fake) as get,
    ):
        await _pipeline(binding=_binding(provider, options={"strategies": ["simple"]})).retrieve(
            "q",
            _scope(),
            rag_namespace="kb",
            k=3,
            enabled=True,
            retrieval_strategy="hyde",
        )

    assert get.call_args.args[0] == "hyde"
    assert provider.questions == []
    assert "fallback_reason=no_candidates" in _decision_lines(caplog)[0]


async def test_question_is_bounded_to_the_registered_strategies() -> None:
    provider = _ChoiceProvider("simple")
    fake = _FakeStrategy()

    with patch("orchid_ai.agents.rag_pipeline.get_retrieval_strategy", return_value=fake):
        await _pipeline(binding=_binding(provider)).retrieve(
            "what is the refund policy?",
            _scope(),
            rag_namespace="kb",
            k=3,
            enabled=True,
            retrieval_strategy="hyde",
        )

    assert len(provider.questions) == 1
    question = provider.questions[0]
    assert question.kind == "choice"
    assert set(question.options) == set(RETRIEVAL_REGISTRY)
    assert provider.states[0].text == "what is the refund policy?"
    assert provider.states[0].payload == {}


async def test_graph_resolves_the_binding_per_agent_and_uses_it() -> None:
    config = _graph_config(agent_level=True)
    chat = MagicMock(spec=BaseChatModel)
    chat.bind_tools = MagicMock(return_value=chat)
    runtime = OrchidRuntime(
        default_model="test/model",
        chat_model=chat,
        decision_models={
            "stub": StubDecisionModel(answers={"strategy": {"choice": "hyde", "confidence": 0.95}}),
        },
    )

    agents: dict[str, Any] = {}
    build_graph(config=config, runtime=runtime, agents_out=agents)

    agent = agents["a"]
    binding = agent._rag_pipeline._retrieval_strategy_binding
    assert binding is not None
    assert binding.point == "retrieval_strategy"
    assert binding.provider == "stub"

    fake = _FakeStrategy()
    with patch("orchid_ai.agents.rag_pipeline.get_retrieval_strategy", return_value=fake) as get:
        await agent._step_rag_retrieval("q", _scope())
    assert get.call_args.args[0] == "hyde"


def test_graph_build_tolerates_duplicate_short_agent_names() -> None:
    config = _parent_child_config(top_level_kid=True)
    agents: dict[str, Any] = {}

    # The top-level ``kid`` and the nested ``parent.kid`` both exist. With no
    # decision configured, resolution must neither raise nor select a binding.
    build_graph(config=config, runtime=_graph_runtime(), agents_out=agents)

    assert set(agents) == {"kid"}
    assert config.resolve_decision("retrieval_strategy", "kid") is None
    assert config.resolve_decision("retrieval_strategy", "parent.kid") is None


def test_ambiguous_short_agent_name_still_raises() -> None:
    parent_a = _agent_entry("parent a")
    parent_a["children"] = {"kid": _agent_entry("a kid")}
    parent_b = _agent_entry("parent b")
    parent_b["children"] = {"kid": _agent_entry("b kid")}
    config = OrchidAgentsConfig.model_validate({"version": "1", "agents": {"parent_a": parent_a, "parent_b": parent_b}})

    with pytest.raises(ValueError, match="ambiguous agent name 'kid'"):
        config.resolve_decision("retrieval_strategy", "kid")


def test_exact_agent_path_wins_over_short_name_suffix() -> None:
    config = _parent_child_config(child_decision=True, top_level_kid=True)

    # ``kid`` is an exact path; it must not be confused with ``parent.kid``.
    assert config.resolve_decision("retrieval_strategy", "kid") is None
    child = config.resolve_decision("retrieval_strategy", "parent.kid")
    assert child is not None
    assert child.model == "stub"


async def test_child_agents_receive_their_own_retrieval_strategy_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _parent_child_config(child_decision=True)
    created: dict[str, Any] = {}
    real_instantiate = graph_module._instantiate_agent

    def _capture(name, agent_config, *args, **kwargs):
        agent = real_instantiate(name, agent_config, *args, **kwargs)
        created[name] = agent
        return agent

    monkeypatch.setattr(graph_module, "_instantiate_agent", _capture)
    build_graph(config=config, runtime=_graph_runtime(decision_answer="hyde"))

    child = created["kid"]
    sibling = created["sibling"]
    child_binding = child._rag_pipeline._retrieval_strategy_binding
    assert child_binding is not None
    assert child_binding.point == "retrieval_strategy"
    assert child_binding.provider == "stub"
    assert sibling._rag_pipeline._retrieval_strategy_binding is None

    fake = _FakeStrategy()
    with patch("orchid_ai.agents.rag_pipeline.get_retrieval_strategy", return_value=fake) as get:
        await child._step_rag_retrieval("q", _scope())
    assert get.call_args.args[0] == "hyde"


async def test_chunk_gate_is_kept_separate_and_still_applies() -> None:
    strategy_provider = _ChoiceProvider("simple")
    gate = _GateRecorder(selected={0})
    results = [
        OrchidSearchResult(document=OrchidDocument(page_content="billing policy", id="d1"), score=0.9),
        OrchidSearchResult(document=OrchidDocument(page_content="pizza recipe", id="d2"), score=0.8),
    ]
    fake = _FakeStrategy(results=results)

    with patch("orchid_ai.agents.rag_pipeline.get_retrieval_strategy", return_value=fake):
        combined = await _pipeline(binding=_binding(strategy_provider), gate_model=gate).retrieve(
            "refund",
            _scope(),
            rag_namespace="kb",
            k=5,
            enabled=True,
            retrieval_strategy="simple",
        )

    assert [passage["content"] for passage in combined] == ["billing policy"]
    assert [question.name for question in gate.questions] == ["chunk:0", "chunk:1"]
    assert gate.state is not None
    assert gate.state.text == "refund"


async def test_chunk_gate_allowlist_still_uses_the_configured_strategy(caplog: pytest.LogCaptureFixture) -> None:
    strategy_provider = _ChoiceProvider("hyde")
    gate = _GateRecorder(selected={0})
    results = [
        OrchidSearchResult(document=OrchidDocument(page_content="billing policy", id="d1"), score=0.9),
        OrchidSearchResult(document=OrchidDocument(page_content="pizza recipe", id="d2"), score=0.8),
    ]
    fake = _FakeStrategy(results=results)

    with (
        caplog.at_level(logging.INFO, logger="orchid_ai.agents.rag_pipeline"),
        patch("orchid_ai.agents.rag_pipeline.get_retrieval_strategy", return_value=fake) as get,
    ):
        combined = await _pipeline(
            binding=_binding(strategy_provider),
            gate_model=gate,
            strategies=["simple"],
        ).retrieve(
            "refund",
            _scope(),
            rag_namespace="kb",
            k=5,
            enabled=True,
            retrieval_strategy="simple",
        )

    assert get.call_args.args[0] == "hyde"
    assert [passage["content"] for passage in combined] == ["billing policy"]
    assert any("strategy=simple" in record.getMessage() for record in caplog.records)


async def test_decision_can_select_graph_rag() -> None:
    scope = _scope()
    store = InMemoryGraphStore()
    await store.upsert_entities(
        [
            OrchidEntity(id="e1", type="topic", name="Billing"),
            OrchidEntity(id="e2", type="topic", name="Refunds"),
        ],
        scope,
    )
    await store.upsert_edges([OrchidEdge(source_id="e1", target_id="e2", relation="related_to")], scope)

    provider = StubDecisionModel(answers={"strategy": {"choice": "graph_rag", "confidence": 0.95}})
    retrieval_config = OrchidRetrievalConfig(graph=OrchidGraphRetrievalConfig(fuse_with_vectors=False))

    combined = await _pipeline(binding=_binding(provider), graph_store=store).retrieve(
        "billing",
        scope,
        rag_namespace="kb",
        k=5,
        enabled=True,
        retrieval_strategy="simple",
        retrieval_config=retrieval_config,
    )

    assert combined
    assert combined[0]["metadata"].get("source") == "graph_rag"
    assert "Billing" in combined[0]["content"]
