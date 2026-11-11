"""Tests for decision-backed tool choice in the ``llm_decides`` strategy.

The ``tool_choice`` point answers bounded boolean questions about the tools
already shortlisted for one call. Only selected tools may run, a confident
"call no tools" answer executes nothing, arguments are derived deterministically
(never composed by the provider), and every uncertain, partial, or failed
outcome falls back to the existing ``all`` strategy.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

import orchid_ai.graph.graph as graph_module
from orchid_ai.agents import strategies as strategies_module
from orchid_ai.agents.generic_agent import GenericAgent
from orchid_ai.agents.strategies import (
    LLMDecidesStrategy,
    get_strategy,
    use_tool_choice_binding,
)
from orchid_ai.config.schema import OrchidAgentConfig, OrchidAgentsConfig, OrchidToolConfig
from orchid_ai.config.schema_decision import OrchidDecisionPointConfig
from orchid_ai.core.agent import OrchidAgentRunContext
from orchid_ai.core.decision import (
    OrchidDecisionAnswer,
    OrchidDecisionError,
    OrchidDecisionQuestion,
    OrchidDecisionResult,
    OrchidDecisionState,
)
from orchid_ai.core.mcp import OrchidMCPClient, OrchidMCPToolResult
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.decision.registry import get_decision_point
from orchid_ai.decision.stub import StubDecisionModel
from orchid_ai.graph.decision_bindings import DecisionBinding
from orchid_ai.graph.graph import build_graph
from orchid_ai.rag.backends.null import NullVectorReader
from orchid_ai.runtime import OrchidRuntime

SENTINEL = "SENTINEL-USER-TEXT-DO-NOT-LOG"


# ── Helpers ─────────────────────────────────────────────────


def _auth() -> OrchidAuthContext:
    return OrchidAuthContext(access_token="test-token")


def _binding(provider: Any, *, threshold: float = 0.75, max_candidates: int | None = None) -> DecisionBinding:
    return DecisionBinding(
        spec=get_decision_point("tool_choice"),
        config=OrchidDecisionPointConfig(
            model="scripted",
            confidence_threshold=threshold,
            max_candidates=max_candidates,
        ),
        model=provider,
    )


def _tools(*names: str, arguments: dict[str, dict[str, Any]] | None = None) -> list[OrchidToolConfig]:
    configured = arguments or {}
    return [OrchidToolConfig(name=name, arguments=configured.get(name, {})) for name in names]


class _MockMCPClient(OrchidMCPClient):
    """Offline MCP client with scripted schemas and call recording."""

    def __init__(
        self,
        *,
        schemas: dict[str, dict[str, Any]] | None = None,
        list_error: BaseException | None = None,
        raise_for: set[str] | None = None,
    ) -> None:
        self._schemas = schemas or {}
        self._list_error = list_error
        self._raise_for = raise_for or set()
        self.calls: list[dict[str, Any]] = []
        self.list_calls = 0

    async def call_tool(self, tool_name: str, arguments: dict[str, Any], auth: OrchidAuthContext):
        self.calls.append({"tool": tool_name, "args": dict(arguments)})
        if tool_name in self._raise_for:
            raise RuntimeError(f"tool {tool_name} failed")
        return OrchidMCPToolResult(content=[{"type": "text", "text": f"result_{tool_name}"}])

    async def list_tools(self, auth: OrchidAuthContext) -> list[dict[str, Any]]:
        self.list_calls += 1
        if self._list_error is not None:
            raise self._list_error
        return [
            {"name": name, "description": f"{name} tool", "schema": schema} for name, schema in self._schemas.items()
        ]

    async def list_prompts(self, auth: OrchidAuthContext) -> list[dict[str, Any]]:
        return []

    async def list_resources(self, auth: OrchidAuthContext) -> list[dict[str, Any]]:
        return []

    async def get_prompt(self, name: str, arguments: dict[str, str], auth: OrchidAuthContext) -> list[dict[str, Any]]:
        return []

    async def read_resource(self, uri: str, auth: OrchidAuthContext) -> str:
        return ""

    @property
    def server_url(self) -> str:
        return "http://mock"


class _ScriptedProvider:
    """Decision provider returning explicit per-question answers."""

    def __init__(
        self,
        answers: dict[str, Any],
        *,
        confidence: float = 0.95,
        error: BaseException | None = None,
        delay: float = 0.0,
    ) -> None:
        self._answers = answers
        self._confidence = confidence
        self._error = error
        self._delay = delay
        self.questions: list[OrchidDecisionQuestion] = []
        self.states: list[OrchidDecisionState] = []
        self.timeouts: list[float | None] = []

    async def decide(
        self,
        state: OrchidDecisionState,
        questions: list[OrchidDecisionQuestion],
        *,
        timeout: float | None = None,
    ) -> OrchidDecisionResult:
        self.states.append(state)
        self.questions.extend(questions)
        self.timeouts.append(timeout)
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error

        answers: dict[str, OrchidDecisionAnswer] = {}
        for question in questions:
            spec = self._answers.get(question.name)
            if spec is None:
                continue
            if isinstance(spec, OrchidDecisionAnswer):
                answers[question.name] = spec
                continue
            if spec.get("refusal"):
                answers[question.name] = OrchidDecisionAnswer(
                    name=question.name,
                    kind="boolean",
                    value=None,
                    confidence=spec.get("confidence", self._confidence),
                    refusal=spec["refusal"],
                )
                continue
            answers[question.name] = OrchidDecisionAnswer(
                name=question.name,
                kind=spec.get("kind", "boolean"),
                value=spec.get("value", True),
                confidence=spec.get("confidence", self._confidence),
            )
        return OrchidDecisionResult(answers=answers, provider="scripted", model="scripted")


class _UnconditionalProvider:
    """Return every configured answer, whether or not the question was asked.

    Real adapters only answer questions they receive; this fake deliberately
    emits an unexpected name so the allowlist assertion can actually fail if
    the strategy ever trusts answer names instead of the shortlist.
    """

    def __init__(self, answers: dict[str, bool], *, confidence: float = 0.95) -> None:
        self._answers = answers
        self._confidence = confidence
        self.questions: list[OrchidDecisionQuestion] = []
        self.emitted: list[str] = []

    async def decide(
        self,
        state: OrchidDecisionState,
        questions: list[OrchidDecisionQuestion],
        *,
        timeout: float | None = None,
    ) -> OrchidDecisionResult:
        self.questions.extend(questions)
        answers = {
            name: OrchidDecisionAnswer(
                name=name,
                kind="boolean",
                value=value,
                confidence=self._confidence,
            )
            for name, value in self._answers.items()
        }
        self.emitted.extend(answers)
        return OrchidDecisionResult(answers=answers, provider="unconditional", model="unconditional")


def _decision_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == "orchid.decision"]


async def _execute(
    strategy: LLMDecidesStrategy,
    client: _MockMCPClient,
    tools: list[OrchidToolConfig],
    *,
    query: str = "find docs",
) -> dict[str, Any]:
    return await strategy.execute(client, tools, query, _auth(), agent_name="a")


# ── Accepted selections ─────────────────────────────────────


async def test_selected_tools_execute_with_deterministic_arguments(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ScriptedProvider({"tool:search": {"value": True}, "tool:delete": {"value": False}})
    client = _MockMCPClient(
        schemas={
            "search": {"type": "object", "required": ["query"]},
            "delete": {"type": "object"},
        },
    )
    strategy = LLMDecidesStrategy(binding=_binding(provider))

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        results = await _execute(
            strategy,
            client,
            _tools("search", "delete", arguments={"search": {"limit": 5}}),
            query=SENTINEL,
        )

    # The user text is the query argument but must never reach the log line.
    assert client.calls == [{"tool": "search", "args": {"query": SENTINEL, "limit": 5}}]
    assert results == {"search": "result_search"}
    # Questions are bounded to the shortlist; argument values never reach the
    # provider because the decision layer answers names only.
    assert [question.name for question in provider.questions] == ["tool:search", "tool:delete"]
    assert provider.states[0].text == SENTINEL
    assert "limit" not in str(dict(provider.states[0].payload))

    lines = _decision_lines(caplog)
    assert len(lines) == 1
    for fragment in (
        "decision point=tool_choice",
        "path=decision",
        "provider=scripted",
        "outcome=accepted",
        "fallback_reason=null",
    ):
        assert fragment in lines[0]
    assert SENTINEL not in lines[0]


async def test_decision_cannot_execute_a_tool_outside_the_shortlist() -> None:
    provider = _UnconditionalProvider({"tool:search": True, "tool:ghost": True})
    client = _MockMCPClient(schemas={"search": {"type": "object"}})
    strategy = LLMDecidesStrategy(binding=_binding(provider))

    results = await _execute(strategy, client, _tools("search"))

    # The provider really did emit the unasked answer; the strategy still
    # keys execution off the shortlist questions, so the tool never runs.
    assert "tool:ghost" in provider.emitted
    assert [question.name for question in provider.questions] == ["tool:search"]
    assert [call["tool"] for call in client.calls] == ["search"]
    assert "ghost" not in results


async def test_only_the_shortlisted_tools_become_questions() -> None:
    provider = _ScriptedProvider({"tool:search": {"value": True}})
    client = _MockMCPClient(schemas={"search": {"type": "object"}, "unlisted": {"type": "object"}})
    strategy = LLMDecidesStrategy(binding=_binding(provider))

    await _execute(strategy, client, _tools("search"))

    assert [question.name for question in provider.questions] == ["tool:search"]


async def test_duplicate_and_unnamed_tools_are_deduplicated() -> None:
    provider = _ScriptedProvider({"tool:search": {"value": True}})
    client = _MockMCPClient(schemas={"search": {"type": "object"}})
    strategy = LLMDecidesStrategy(binding=_binding(provider))

    await _execute(
        strategy,
        client,
        [OrchidToolConfig(name=""), OrchidToolConfig(name="search"), OrchidToolConfig(name="search")],
    )

    assert [question.name for question in provider.questions] == ["tool:search"]


async def test_confident_no_selection_executes_no_tools(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ScriptedProvider({"tool:search": {"value": False}, "tool:delete": {"value": False}})
    client = _MockMCPClient(schemas={"search": {"type": "object"}, "delete": {"type": "object"}})
    strategy = LLMDecidesStrategy(binding=_binding(provider))

    with caplog.at_level(logging.INFO):
        results = await _execute(strategy, client, _tools("search", "delete"))

    # A confident all-False decision is a direct answer: no tool runs and the
    # strategy does not widen the selection to the all fallback.
    assert results == {}
    assert client.calls == []
    line = _decision_lines(caplog)[0]
    assert "outcome=accepted" in line
    assert "fallback_reason=null" in line


async def test_partial_answer_is_not_a_confident_no_selection(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ScriptedProvider({"tool:search": {"value": False}})
    client = _MockMCPClient(schemas={"search": {"type": "object"}, "delete": {"type": "object"}})
    strategy = LLMDecidesStrategy(binding=_binding(provider))

    with caplog.at_level(logging.INFO):
        results = await _execute(strategy, client, _tools("search", "delete"))

    # A missing answer cannot be read as a confident "do not call".
    assert set(results) == {"search", "delete"}
    assert "fallback_reason=invalid_response" in _decision_lines(caplog)[0]


async def test_partial_answer_with_selection_falls_back_to_all(caplog: pytest.LogCaptureFixture) -> None:
    # One tool is confidently selected but the other question is omitted: the
    # result is incomplete, so no subset may run.
    provider = _ScriptedProvider({"tool:search": {"value": True}})
    client = _MockMCPClient(schemas={"search": {"type": "object"}, "delete": {"type": "object"}})
    strategy = LLMDecidesStrategy(binding=_binding(provider))

    with caplog.at_level(logging.INFO):
        results = await _execute(strategy, client, _tools("search", "delete"))

    assert set(results) == {"search", "delete"}
    assert [call["tool"] for call in client.calls] == ["search", "delete"]
    assert "fallback_reason=invalid_response" in _decision_lines(caplog)[0]


async def test_low_confidence_false_is_not_a_confident_no_selection(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ScriptedProvider(
        {
            "tool:search": {"value": False, "confidence": 0.4},
            "tool:delete": {"value": False, "confidence": 0.4},
        }
    )
    client = _MockMCPClient(schemas={"search": {"type": "object"}, "delete": {"type": "object"}})
    strategy = LLMDecidesStrategy(binding=_binding(provider, threshold=0.75))

    with caplog.at_level(logging.INFO):
        results = await _execute(strategy, client, _tools("search", "delete"))

    assert set(results) == {"search", "delete"}
    assert "fallback_reason=low_confidence" in _decision_lines(caplog)[0]


# ── Argument safety ─────────────────────────────────────────


async def test_required_argument_that_cannot_be_derived_is_skipped(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ScriptedProvider({"tool:search": {"value": True}, "tool:city": {"value": True}})
    client = _MockMCPClient(
        schemas={
            "search": {"type": "object", "required": ["query"]},
            "city": {"type": "object", "required": ["city"]},
        },
    )
    strategy = LLMDecidesStrategy(binding=_binding(provider))

    with caplog.at_level(logging.WARNING, logger="orchid_ai.agents.strategies"):
        results = await _execute(strategy, client, _tools("search", "city"))

    # The unsafe tool is skipped, never called with an invented argument.
    assert [call["tool"] for call in client.calls] == ["search"]
    assert results == {"search": "result_search"}
    assert not any("city" in call["args"] for call in client.calls)
    assert "required argument(s) cannot be derived" in caplog.text


async def test_configured_arguments_satisfy_required_schema_fields() -> None:
    provider = _ScriptedProvider({"tool:city": {"value": True}})
    client = _MockMCPClient(schemas={"city": {"type": "object", "required": ["city", "query"]}})
    strategy = LLMDecidesStrategy(binding=_binding(provider))

    await _execute(strategy, client, _tools("city", arguments={"city": {"city": "Rome"}}))

    assert client.calls == [{"tool": "city", "args": {"query": "find docs", "city": "Rome"}}]


async def test_no_safe_tool_falls_back_to_all(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ScriptedProvider({"tool:city": {"value": True}})
    client = _MockMCPClient(schemas={"city": {"type": "object", "required": ["city"]}})
    strategy = LLMDecidesStrategy(binding=_binding(provider))

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        results = await _execute(strategy, client, _tools("city"))

    assert [call["tool"] for call in client.calls] == ["city"]
    assert results == {"city": "result_city"}
    assert "fallback_reason=unsafe_arguments" in _decision_lines(caplog)[0]


async def test_selected_tool_without_a_listed_schema_is_skipped(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ScriptedProvider({"tool:search": {"value": True}, "tool:unknown": {"value": True}})
    client = _MockMCPClient(schemas={"search": {"type": "object"}})
    strategy = LLMDecidesStrategy(binding=_binding(provider))

    with caplog.at_level(logging.WARNING, logger="orchid_ai.agents.strategies"):
        results = await _execute(strategy, client, _tools("search", "unknown"))

    assert [call["tool"] for call in client.calls] == ["search"]
    assert results == {"search": "result_search"}
    assert "input schema is unavailable" in caplog.text


async def test_one_failing_tool_does_not_abort_the_other_selected_tools() -> None:
    provider = _ScriptedProvider({"tool:search": {"value": True}, "tool:delete": {"value": True}})
    client = _MockMCPClient(
        schemas={"search": {"type": "object"}, "delete": {"type": "object"}},
        raise_for={"delete"},
    )
    strategy = LLMDecidesStrategy(binding=_binding(provider))

    results = await _execute(strategy, client, _tools("search", "delete"))

    assert results["search"] == "result_search"
    assert "delete_error" in results


async def test_schema_discovery_failure_falls_back_to_all(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ScriptedProvider({"tool:search": {"value": True}})
    client = _MockMCPClient(list_error=RuntimeError("discovery unavailable"))
    strategy = LLMDecidesStrategy(binding=_binding(provider))

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        results = await _execute(strategy, client, _tools("search"))

    assert results == {"search": "result_search"}
    assert provider.questions == []
    assert "fallback_reason=schema_unavailable" in _decision_lines(caplog)[0]


# ── Fallbacks ───────────────────────────────────────────────


async def test_low_confidence_falls_back_to_all(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ScriptedProvider(
        {
            "tool:search": {"value": True, "confidence": 0.4},
            "tool:delete": {"value": False},
        }
    )
    client = _MockMCPClient(schemas={"search": {"type": "object"}, "delete": {"type": "object"}})
    strategy = LLMDecidesStrategy(binding=_binding(provider, threshold=0.75))

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        results = await _execute(strategy, client, _tools("search", "delete"))

    assert set(results) == {"search", "delete"}
    line = _decision_lines(caplog)[0]
    assert "outcome=fallback" in line
    assert "fallback_reason=low_confidence" in line


async def test_refusal_falls_back_to_all(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ScriptedProvider({"tool:search": {"refusal": "not enough context"}})
    client = _MockMCPClient(schemas={"search": {"type": "object"}, "delete": {"type": "object"}})
    strategy = LLMDecidesStrategy(binding=_binding(provider))

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        results = await _execute(strategy, client, _tools("search", "delete"))

    assert set(results) == {"search", "delete"}
    assert "fallback_reason=refusal" in _decision_lines(caplog)[0]


async def test_missing_answer_falls_back_to_all(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ScriptedProvider({})
    client = _MockMCPClient(schemas={"search": {"type": "object"}, "delete": {"type": "object"}})
    strategy = LLMDecidesStrategy(binding=_binding(provider))

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        results = await _execute(strategy, client, _tools("search", "delete"))

    assert set(results) == {"search", "delete"}
    assert "fallback_reason=invalid_response" in _decision_lines(caplog)[0]


async def test_non_boolean_answer_falls_back_to_all(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ScriptedProvider(
        {"tool:search": OrchidDecisionAnswer(name="tool:search", kind="choice", value="yes", confidence=0.95)}
    )
    client = _MockMCPClient(schemas={"search": {"type": "object"}, "delete": {"type": "object"}})
    strategy = LLMDecidesStrategy(binding=_binding(provider))

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        results = await _execute(strategy, client, _tools("search", "delete"))

    assert set(results) == {"search", "delete"}
    assert "fallback_reason=invalid_response" in _decision_lines(caplog)[0]


async def test_stalled_provider_falls_back_to_all(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ScriptedProvider({"tool:search": {"value": True}}, delay=0.05)
    client = _MockMCPClient(schemas={"search": {"type": "object"}, "delete": {"type": "object"}})
    strategy = LLMDecidesStrategy(binding=_binding(provider), timeout=0.01)

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        results = await _execute(strategy, client, _tools("search", "delete"))

    assert set(results) == {"search", "delete"}
    assert provider.timeouts == [0.01]
    assert "fallback_reason=timeout" in _decision_lines(caplog)[0]


async def test_default_decision_timeout_bounds_the_dispatcher_path() -> None:
    # Constructed exactly as ``MCPDispatcher.fetch`` does: no explicit timeout.
    provider = _ScriptedProvider({"tool:search": {"value": True}, "tool:delete": {"value": False}})
    binding = _binding(provider)
    client = _MockMCPClient(schemas={"search": {"type": "object"}, "delete": {"type": "object"}})

    strategy = get_strategy("llm_decides", binding=binding)
    results = await _execute(strategy, client, _tools("search", "delete"))

    assert results == {"search": "result_search"}
    assert provider.timeouts == [2.0]


async def test_provider_error_falls_back_to_all(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ScriptedProvider(
        {},
        error=OrchidDecisionError("provider unavailable", code="provider_unavailable", provider="err"),
    )
    client = _MockMCPClient(schemas={"search": {"type": "object"}, "delete": {"type": "object"}})
    strategy = LLMDecidesStrategy(binding=_binding(provider))

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        results = await _execute(strategy, client, _tools("search", "delete"))

    assert set(results) == {"search", "delete"}
    line = _decision_lines(caplog)[0]
    assert "fallback_reason=provider_unavailable" in line
    assert "provider=err" in line


async def test_unexpected_exception_falls_back_to_all(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ScriptedProvider({}, error=RuntimeError(SENTINEL))
    client = _MockMCPClient(schemas={"search": {"type": "object"}, "delete": {"type": "object"}})
    strategy = LLMDecidesStrategy(binding=_binding(provider))

    with caplog.at_level(logging.INFO):
        results = await _execute(strategy, client, _tools("search", "delete"))

    assert set(results) == {"search", "delete"}
    assert SENTINEL not in caplog.text
    assert "fallback_reason=provider_error" in _decision_lines(caplog)[0]


# ── Candidates and disabled point ───────────────────────────


async def test_no_candidates_falls_back_without_a_provider_call(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ScriptedProvider({})
    client = _MockMCPClient()
    strategy = LLMDecidesStrategy(binding=_binding(provider))

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        results = await _execute(strategy, client, [])

    assert results == {}
    assert provider.questions == []
    assert "fallback_reason=no_candidates" in _decision_lines(caplog)[0]


async def test_candidate_cap_falls_back_without_a_provider_call(caplog: pytest.LogCaptureFixture) -> None:
    provider = _ScriptedProvider({})
    client = _MockMCPClient(schemas={"a": {}, "b": {}, "c": {}})
    strategy = LLMDecidesStrategy(binding=_binding(provider, max_candidates=2))

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        results = await _execute(strategy, client, _tools("a", "b", "c"))

    assert set(results) == {"a", "b", "c"}
    assert provider.questions == []
    assert "fallback_reason=too_many_candidates" in _decision_lines(caplog)[0]


async def test_disabled_point_uses_all_without_a_decision_line(caplog: pytest.LogCaptureFixture) -> None:
    client = _MockMCPClient(schemas={"search": {"type": "object"}, "delete": {"type": "object"}})
    strategy = LLMDecidesStrategy()

    with caplog.at_level(logging.INFO, logger="orchid.decision"):
        results = await _execute(strategy, client, _tools("search", "delete"))

    assert set(results) == {"search", "delete"}
    assert _decision_lines(caplog) == []


# ── Binding plumbing ────────────────────────────────────────


async def test_constructor_binding_is_used_and_get_strategy_forwards_it() -> None:
    provider = _ScriptedProvider({"tool:search": {"value": True}, "tool:delete": {"value": False}})
    binding = _binding(provider)
    client = _MockMCPClient(schemas={"search": {"type": "object"}, "delete": {"type": "object"}})

    strategy = get_strategy("llm_decides", binding=binding)
    assert isinstance(strategy, LLMDecidesStrategy)

    results = await _execute(strategy, client, _tools("search", "delete"))

    assert results == {"search": "result_search"}


async def test_execute_binding_overrides_the_constructor_binding() -> None:
    unused = _ScriptedProvider({})
    provider = _ScriptedProvider({"tool:search": {"value": True}, "tool:delete": {"value": False}})
    client = _MockMCPClient(schemas={"search": {"type": "object"}, "delete": {"type": "object"}})
    strategy = LLMDecidesStrategy(binding=_binding(unused))

    results = await strategy.execute(
        client,
        _tools("search", "delete"),
        "find docs",
        _auth(),
        agent_name="a",
        tool_choice_binding=_binding(provider),
    )

    assert results == {"search": "result_search"}
    assert unused.questions == []


async def test_run_scoped_binding_is_used_by_the_strategy() -> None:
    provider = _ScriptedProvider({"tool:search": {"value": True}, "tool:delete": {"value": False}})
    client = _MockMCPClient(schemas={"search": {"type": "object"}, "delete": {"type": "object"}})

    with use_tool_choice_binding(_binding(provider)):
        results = await _execute(LLMDecidesStrategy(), client, _tools("search", "delete"))

    assert results == {"search": "result_search"}


def test_run_scoped_binding_is_restored_after_the_scope() -> None:
    binding = _binding(_ScriptedProvider({}))
    with use_tool_choice_binding(binding):
        assert strategies_module._tool_choice_binding_var.get() is binding
    assert strategies_module._tool_choice_binding_var.get() is None


def test_timeout_must_be_positive() -> None:
    with pytest.raises(ValueError, match="timeout must be positive"):
        LLMDecidesStrategy(timeout=0)


# ── Run-scoped binding through the dispatcher seam ──────────


class _FetchDispatcher:
    """Fake dispatcher seam: ``fetch`` builds the strategy from the registry.

    This mirrors :meth:`MCPDispatcher.fetch`, which is the only production
    caller of ``get_strategy(...).execute(...)``. The strategy is constructed
    without a binding, so it can only see the agent's run-scoped decision.
    """

    def __init__(
        self,
        client: _MockMCPClient,
        tools: list[OrchidToolConfig],
        seen_bindings: list[Any],
    ) -> None:
        self._client = client
        self._tools = tools
        self._seen = seen_bindings

    async def fetch(
        self, query: str, auth: OrchidAuthContext, *, agent_name: str = "", **kwargs: Any
    ) -> dict[str, Any]:
        # Yield so concurrent agent runs interleave inside the strategy seam.
        await asyncio.sleep(0)
        self._seen.append(strategies_module._tool_choice_binding_var.get())
        strategy = get_strategy("llm_decides")
        return await strategy.execute(self._client, self._tools, query, auth, agent_name=agent_name)


def _pipeline_agent(binding: Any, dispatcher: Any) -> GenericAgent:
    """GenericAgent whose tool loop runs through the dispatcher's ``fetch``."""
    agent = GenericAgent(
        config=OrchidAgentConfig(description="d", prompt="p", name="a"),
        reader=NullVectorReader(),
        chat_model=MagicMock(spec=BaseChatModel),
        tool_choice_binding=binding,
    )
    agent._mcp_dispatcher = dispatcher

    async def _no_rag(query: str, scope: Any) -> list[dict[str, Any]]:
        return []

    async def _no_cache(scope: Any) -> dict[str, Any]:
        return {}

    async def _no_injection(mcp_data: dict[str, Any], scope: Any) -> None:
        return None

    async def _loop(query: str, auth: OrchidAuthContext, state: Any, rag_data: list[dict[str, Any]], **kwargs: Any):
        data = await agent._mcp_dispatcher.fetch(query, auth, agent_name=agent.name)
        return "final", data, []

    agent._step_rag_retrieval = _no_rag  # type: ignore[method-assign]
    agent._step_cache_check = _no_cache  # type: ignore[method-assign]
    agent._step_dynamic_injection = _no_injection  # type: ignore[method-assign]
    agent._agentic_tool_loop = _loop  # type: ignore[method-assign]
    return agent


async def _run_agent(agent: GenericAgent, query: str = SENTINEL) -> dict[str, Any]:
    token = agent.set_run_context(OrchidAgentRunContext(auth=_auth()))
    try:
        return await agent.run({"messages": [HumanMessage(content=query)]})
    finally:
        agent.reset_run_context(token)


async def test_generic_agent_run_scopes_the_binding_through_dispatcher_fetch() -> None:
    provider = _ScriptedProvider({"tool:search": {"value": True}, "tool:delete": {"value": False}})
    binding = _binding(provider)
    client = _MockMCPClient(schemas={"search": {"type": "object"}, "delete": {"type": "object"}})
    seen: list[Any] = []
    agent = _pipeline_agent(binding, _FetchDispatcher(client, _tools("search", "delete"), seen))

    result = await _run_agent(agent)

    # The registry-built strategy saw the agent's binding during the run and
    # executed only the selected tool.
    assert seen == [binding]
    assert client.calls == [{"tool": "search", "args": {"query": SENTINEL}}]
    assert result["mcp_context"]["a"] == {"search": "result_search"}
    # The binding is restored when ``run`` returns.
    assert strategies_module._tool_choice_binding_var.get() is None


async def test_run_scoped_binding_is_isolated_between_concurrent_agents() -> None:
    async def _one(selected: str, other: str) -> tuple[Any, _MockMCPClient, dict[str, Any], list[Any]]:
        provider = _ScriptedProvider(
            {
                f"tool:{selected}": {"value": True},
                f"tool:{other}": {"value": False},
            }
        )
        binding = _binding(provider)
        client = _MockMCPClient(schemas={selected: {"type": "object"}, other: {"type": "object"}})
        seen: list[Any] = []
        agent = _pipeline_agent(binding, _FetchDispatcher(client, _tools(selected, other), seen))
        result = await _run_agent(agent)
        return binding, client, result, seen

    first, second = await asyncio.gather(
        _one("search", "delete"),
        _one("delete", "search"),
    )

    for binding, client, _result, seen in (first, second):
        assert seen == [binding]
    assert first[1].calls == [{"tool": "search", "args": {"query": SENTINEL}}]
    assert second[1].calls == [{"tool": "delete", "args": {"query": SENTINEL}}]
    assert strategies_module._tool_choice_binding_var.get() is None


# ── Agent and graph wiring ──────────────────────────────────


def _agent_entry(description: str = "test agent") -> dict[str, Any]:
    return {
        "description": description,
        "prompt": "you are a test agent",
        "llm": {"model": "test/model"},
        "rag": {"enabled": False},
    }


def _graph_runtime() -> OrchidRuntime:
    chat = MagicMock(spec=BaseChatModel)
    chat.bind_tools = MagicMock(return_value=chat)
    return OrchidRuntime(
        default_model="test/model",
        chat_model=chat,
        decision_models={"stub": StubDecisionModel()},
    )


def test_generic_agent_accepts_both_tool_flow_bindings() -> None:
    tool_binding = _binding(_ScriptedProvider({}))
    loop_binding = DecisionBinding(
        spec=get_decision_point("agentic_loop"),
        config=OrchidDecisionPointConfig(model="stub"),
        model=StubDecisionModel(),
    )
    agent = GenericAgent(
        config=OrchidAgentConfig(description="d", prompt="p", name="a"),
        reader=NullVectorReader(),
        tool_choice_binding=tool_binding,
        agentic_loop_binding=loop_binding,
    )

    assert agent._tool_choice_binding is tool_binding
    assert agent._agentic_loop_binding is loop_binding


def test_graph_resolves_tool_flow_bindings_per_top_level_agent() -> None:
    config = OrchidAgentsConfig.model_validate(
        {
            "version": "1",
            "agents": {"a": _agent_entry()},
            "decision_models": {"stub": {"protocol": "stub"}},
            "decisions": {"points": {"tool_choice": {"model": "stub"}, "agentic_loop": {"model": "stub"}}},
        }
    )
    agents: dict[str, Any] = {}

    build_graph(config=config, runtime=_graph_runtime(), agents_out=agents)

    agent = agents["a"]
    assert isinstance(agent._tool_choice_binding, DecisionBinding)
    assert agent._tool_choice_binding.point == "tool_choice"
    assert isinstance(agent._agentic_loop_binding, DecisionBinding)
    assert agent._agentic_loop_binding.point == "agentic_loop"


def test_graph_child_agents_receive_their_own_tool_flow_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    child = _agent_entry("child")
    child["decisions"] = {"points": {"tool_choice": {"model": "stub"}, "agentic_loop": {"model": "stub"}}}
    sibling = _agent_entry("sibling")
    parent = _agent_entry("parent")
    parent["children"] = {"kid": child, "sibling": sibling}
    config = OrchidAgentsConfig.model_validate(
        {
            "version": "1",
            "agents": {"parent": parent},
            "decision_models": {"stub": {"protocol": "stub"}},
        }
    )

    created: dict[str, Any] = {}
    real_instantiate = graph_module._instantiate_agent

    def _capture(name, agent_config, *args, **kwargs):
        agent = real_instantiate(name, agent_config, *args, **kwargs)
        created[name] = agent
        return agent

    monkeypatch.setattr(graph_module, "_instantiate_agent", _capture)
    build_graph(config=config, runtime=_graph_runtime())

    assert created["kid"]._tool_choice_binding.point == "tool_choice"
    assert created["kid"]._agentic_loop_binding.point == "agentic_loop"
    assert created["sibling"]._tool_choice_binding is None
    assert created["sibling"]._agentic_loop_binding is None


def test_disabled_tool_flow_points_resolve_to_none() -> None:
    config = OrchidAgentsConfig.model_validate(
        {
            "version": "1",
            "agents": {"a": _agent_entry()},
            "decision_models": {"stub": {"protocol": "stub"}},
        }
    )
    agents: dict[str, Any] = {}

    build_graph(config=config, runtime=_graph_runtime(), agents_out=agents)

    assert agents["a"]._tool_choice_binding is None
    assert agents["a"]._agentic_loop_binding is None
