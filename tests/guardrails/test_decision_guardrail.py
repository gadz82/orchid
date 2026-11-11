"""Tests for the decision-backed guardrail.

The rail is opt-in: without a resolved ``guardrails`` binding it never calls a
provider and applies its own ``on_failure`` policy. Content sent to the provider
is bounded and redacted, the provider's text is never reused as a user message
or as the redacted replacement, and every check emits exactly one structured
decision line without the payload.

No test reaches the network: providers are stubs or scripted fakes, and the
autouse fixture rejects construction of a real ``httpx.AsyncClient``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import ValidationError

from orchid_ai.config.schema import OrchidAgentsConfig, OrchidGuardrailRuleConfig, OrchidGuardrailsConfig
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
    OrchidGuardrailDirection,
    OrchidGuardrailFailurePolicy,
    OrchidGuardrailResult,
)
from orchid_ai.decision.observability import DECISION_LOG_FIELDS
from orchid_ai.decision.registry import get_decision_point
from orchid_ai.decision.stub import StubDecisionModel
from orchid_ai.graph import graph as graph_module
from orchid_ai.graph.decision_bindings import DecisionBinding
from orchid_ai.graph.graph import _build_subgraph, _create_agent_node
from orchid_ai.graph.guardrail_wiring import _GuardrailWiring
from orchid_ai.graph.state import GraphState
from orchid_ai.guardrails.decision_guardrail import DecisionGuardrail
from orchid_ai.guardrails.registry import GUARDRAIL_REGISTRY, build_guardrail_chain, register_guardrail
from orchid_ai.rag.backends.null import NullVectorReader
from orchid_ai.runtime import OrchidRuntime

SENTINEL = "SENTINEL-RAW-PAYLOAD-DO-NOT-LOG"
RAW_EMAIL = "leak@example.com"
RAW_API_KEY = "sk-abcdefghijklmnop1234"
JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.c2lnbmF0dXJlMTIzNDU2"
AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
GOOGLE_KEY = "AIzaSyD-1234567890abcdefghijklmnopqrstu"


@pytest.fixture(autouse=True)
def _no_live_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if a test actually sends an HTTP request.

    Constructing a client is allowed: importing LiteLLM may do so with a
    non-network transport. Only a real ``send`` would reach the network.
    """

    async def _blocked_send(self: Any, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("decision guardrail tests must not send an HTTP request")

    monkeypatch.setattr(httpx.AsyncClient, "send", _blocked_send)


@pytest.fixture
def input_ctx() -> OrchidGuardrailContext:
    return OrchidGuardrailContext(direction=OrchidGuardrailDirection.INPUT, agent_name="a")


@pytest.fixture
def output_ctx() -> OrchidGuardrailContext:
    return OrchidGuardrailContext(direction=OrchidGuardrailDirection.OUTPUT, agent_name="a")


# ── Helpers ─────────────────────────────────────────────────


def _binding(provider: Any, *, threshold: float = 0.75) -> DecisionBinding:
    return DecisionBinding(
        spec=get_decision_point("guardrails"),
        config=OrchidDecisionPointConfig(model="scripted", confidence_threshold=threshold),
        model=provider,
    )


def _stub(choice: str = "allow", *, confidence: float = 0.9) -> StubDecisionModel:
    return StubDecisionModel(answers={"verdict": {"choice": choice, "confidence": confidence}})


class _ScriptedProvider:
    """Return a fixed answer, raise a fixed error, or stall past the timeout."""

    def __init__(
        self,
        answer: Any = None,
        *,
        error: BaseException | None = None,
        delay: float = 0.0,
        provider: str = "scripted",
    ) -> None:
        self._answer = answer
        self._error = error
        self._delay = delay
        self._provider = provider
        self.states: list[OrchidDecisionState] = []
        self.questions: list[OrchidDecisionQuestion] = []
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
        answer = self._answer
        answers = {} if answer is None else {answer.name: answer}
        return OrchidDecisionResult(answers=answers, provider=self._provider, model="scripted")


def _choice_answer(
    value: Any,
    *,
    confidence: float = 0.9,
    kind: str = "choice",
    refusal: str | None = None,
) -> OrchidDecisionAnswer:
    return OrchidDecisionAnswer(
        name="verdict",
        kind=kind,  # type: ignore[arg-type]
        value=value,
        confidence=confidence,
        refusal=refusal,
    )


def _decision_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == "orchid.decision"]


class _SecretDetectionGuardrail(OrchidGuardrail):
    """Stand-in for a custom rail registered by a startup hook."""

    def __init__(self, *, fail_action: str = "block", marker: str = "", **kwargs: Any) -> None:
        self._fail_action = fail_action
        self.marker = marker

    @property
    def name(self) -> str:
        return "secret_detection"

    async def check(self, content: str, context: OrchidGuardrailContext) -> OrchidGuardrailResult:
        return OrchidGuardrailResult.passed(self.name)


# ── Registry and config ─────────────────────────────────────


class TestRegistryAndConfig:
    def test_decision_type_is_registered(self) -> None:
        assert GUARDRAIL_REGISTRY["decision"] is DecisionGuardrail

    def test_decision_requires_an_explicit_on_failure(self) -> None:
        with pytest.raises(ValidationError, match="requires an explicit on_failure"):
            OrchidGuardrailRuleConfig(type="decision")

    def test_decision_accepts_an_explicit_on_failure(self) -> None:
        rule = OrchidGuardrailRuleConfig(type="decision", on_failure="closed")
        assert rule.on_failure == "closed"

    def test_deterministic_rails_reject_on_failure(self) -> None:
        with pytest.raises(ValidationError, match="on_failure is not supported"):
            OrchidGuardrailRuleConfig(type="content_safety", on_failure="closed")

    def test_extra_rule_keys_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OrchidGuardrailRuleConfig(type="decision", on_failure="open", typo=True)  # type: ignore[call-arg]

    def test_unknown_guardrail_type_is_deferred_to_build(self) -> None:
        # Custom rails may be registered by a startup hook after config load.
        assert "secret_detection" not in GUARDRAIL_REGISTRY
        rule = OrchidGuardrailRuleConfig(type="secret_detection")
        assert rule.type == "secret_detection"

        with pytest.raises(ValueError, match="unknown guardrail type 'not_real'"):
            build_guardrail_chain([{"type": "not_real"}])

    def test_custom_type_registered_after_config_load_builds(self) -> None:
        assert "secret_detection" not in GUARDRAIL_REGISTRY
        config = OrchidGuardrailsConfig(input=[{"type": "secret_detection", "config": {"marker": "from-config"}}])
        assert config.input[0].type == "secret_detection"

        register_guardrail("secret_detection", _SecretDetectionGuardrail)
        try:
            input_chain, _ = _GuardrailWiring.build_chains(config)
            assert len(input_chain) == 1
            assert input_chain.guardrails[0].name == "secret_detection"
        finally:
            GUARDRAIL_REGISTRY.pop("secret_detection", None)

    def test_wiring_unknown_type_fails_at_build(self) -> None:
        config = OrchidGuardrailsConfig(input=[{"type": "not_real"}])
        with pytest.raises(ValueError, match="unknown guardrail type 'not_real'"):
            _GuardrailWiring.build_chains(config)

    def test_container_typos_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OrchidGuardrailsConfig(inputs=[{"type": "content_safety"}])  # type: ignore[call-arg]

    @pytest.mark.parametrize(
        "reserved",
        ["on_failure", "type", "fail_action", "binding", "point", "configuration_reason"],
    )
    def test_reserved_nested_config_keys_are_rejected(self, reserved: str) -> None:
        with pytest.raises(ValidationError, match="reserved keys"):
            OrchidGuardrailRuleConfig(
                type="decision",
                on_failure="open",
                config={reserved: "closed"},
            )

    @pytest.mark.parametrize(
        "config",
        [
            {"timeout_seconds": 0},
            {"max_state_chars": "many"},
            {"block_message": ""},
            {"not_a_rail_option": True},
        ],
    )
    def test_invalid_decision_config_fails_config_load(self, config: dict[str, Any]) -> None:
        with pytest.raises(ValidationError, match="invalid 'decision' guardrail config"):
            OrchidGuardrailsConfig(input=[{"type": "decision", "on_failure": "open", "config": config}])

    def test_invalid_decision_config_fails_the_builder(self) -> None:
        with pytest.raises(ValueError, match="invalid guardrail config for 'decision'"):
            build_guardrail_chain([{"type": "decision", "on_failure": "open", "config": {"timeout_seconds": 0}}])

    def test_direct_builder_rejects_nested_on_failure(self) -> None:
        with pytest.raises(ValueError, match="on_failure must be declared at the rule level"):
            build_guardrail_chain([{"type": "decision", "config": {"on_failure": "closed"}}])

    def test_builder_requires_an_explicit_on_failure_for_decision(self) -> None:
        with pytest.raises(ValueError, match="requires an explicit on_failure"):
            build_guardrail_chain([{"type": "decision"}])

    @pytest.mark.parametrize("value", ["maybe", "fail-open", "OPEN", ""])
    def test_unknown_on_failure_fails_validation(self, value: str) -> None:
        with pytest.raises(ValidationError):
            OrchidGuardrailRuleConfig(type="decision", on_failure=value)  # type: ignore[arg-type]

    def test_unknown_on_failure_fails_rail_construction(self) -> None:
        with pytest.raises(ValueError):
            DecisionGuardrail(on_failure="sometimes")

    def test_wrong_point_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            DecisionGuardrail(point="tools")

    def test_allow_is_not_a_valid_failure_action(self) -> None:
        with pytest.raises(ValueError):
            DecisionGuardrail(fail_action="allow", on_failure="closed")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"timeout_seconds": 0},
            {"timeout_seconds": 11},
            {"timeout_seconds": True},
            {"timeout_seconds": "2"},
            {"max_state_chars": 0},
            {"max_state_chars": True},
            {"block_message": ""},
            {"warn_message": 3},
            {"redaction_placeholder": " "},
        ],
    )
    def test_invalid_constructor_values_fail(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises((TypeError, ValueError)):
            DecisionGuardrail(**kwargs)  # type: ignore[arg-type]

    def test_the_rail_exposes_its_point(self) -> None:
        assert DecisionGuardrail().point == "guardrails"

    def test_builder_forwards_on_failure_and_binding(self) -> None:
        provider = _stub("block")
        chain = build_guardrail_chain(
            [
                {
                    "type": "decision",
                    "fail_action": "block",
                    "on_failure": "closed",
                    "config": {"binding": _binding(provider)},
                }
            ]
        )
        assert len(chain) == 1
        rail = chain.guardrails[0]
        assert isinstance(rail, DecisionGuardrail)
        assert rail.on_failure is OrchidGuardrailFailurePolicy.CLOSED

    def test_builder_accepts_top_level_on_failure(self) -> None:
        chain = build_guardrail_chain([{"type": "decision", "on_failure": "closed"}])
        assert len(chain) == 1
        assert chain.guardrails[0].on_failure is OrchidGuardrailFailurePolicy.CLOSED  # type: ignore[attr-defined]

    def test_deterministic_rails_warn_and_ignore_on_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            chain = build_guardrail_chain(
                [
                    {
                        "type": "max_length",
                        "fail_action": "block",
                        "on_failure": "closed",
                        "config": {"max_characters": 5},
                    },
                    {"type": "content_safety", "fail_action": "block", "on_failure": "closed"},
                ]
            )
        assert len(chain) == 2
        assert caplog.text.count("on_failure is ignored") == 2

    def test_deterministic_rail_matches_are_unchanged(self) -> None:
        async def run() -> Any:
            chain = build_guardrail_chain(
                [{"type": "content_safety", "fail_action": "block", "config": {"blocklist": ["forbidden"]}}]
            )
            return await chain.evaluate(
                "this is forbidden",
                OrchidGuardrailContext(direction=OrchidGuardrailDirection.INPUT),
            )

        result = asyncio.run(run())
        assert result.blocked

    def test_agent_config_parses_on_failure(self) -> None:
        config = OrchidAgentsConfig.model_validate(
            {
                "agents": {
                    "a": {
                        "description": "A",
                        "prompt": "p",
                        "guardrails": {"input": [{"type": "decision", "on_failure": "closed"}]},
                    }
                }
            }
        )
        assert config.agents["a"].guardrails.input[0].on_failure == "closed"


# ── Accepted verdicts ───────────────────────────────────────


class TestAcceptedVerdicts:
    async def test_allow_passes(self, input_ctx: OrchidGuardrailContext) -> None:
        rail = DecisionGuardrail(binding=_binding(_stub("allow")))
        result = await rail.check("clean content", input_ctx)
        assert not result.triggered
        assert result.action is OrchidGuardrailAction.ALLOW
        assert result.details["verdict"] == "allow"

    async def test_block_uses_a_safe_message(self, input_ctx: OrchidGuardrailContext) -> None:
        rail = DecisionGuardrail(binding=_binding(_stub("block")), block_message="Blocked by policy.")
        result = await rail.check(f"harmful {SENTINEL}", input_ctx)
        assert result.blocked
        assert result.message == "Blocked by policy."
        assert SENTINEL not in result.message
        assert SENTINEL not in str(result.details)
        assert result.redacted_content is None

    async def test_redact_replaces_the_whole_payload(self, input_ctx: OrchidGuardrailContext) -> None:
        rail = DecisionGuardrail(binding=_binding(_stub("redact")))
        result = await rail.check(f"write to {RAW_EMAIL}", input_ctx)
        assert result.triggered
        assert result.action is OrchidGuardrailAction.REDACT
        assert result.redacted_content == "[REDACTED]"
        assert RAW_EMAIL not in (result.redacted_content or "")
        assert RAW_EMAIL not in str(result.details)

    async def test_redact_uses_a_configurable_placeholder(self, input_ctx: OrchidGuardrailContext) -> None:
        rail = DecisionGuardrail(binding=_binding(_stub("redact")), redaction_placeholder="[REMOVED]")
        result = await rail.check("nothing locally recognisable", input_ctx)
        assert result.action is OrchidGuardrailAction.REDACT
        assert result.redacted_content == "[REMOVED]"

    async def test_warn_maps_to_warn(self, input_ctx: OrchidGuardrailContext) -> None:
        rail = DecisionGuardrail(binding=_binding(_stub("warn")), warn_message="Review this.")
        result = await rail.check("borderline content", input_ctx)
        assert result.triggered
        assert result.action is OrchidGuardrailAction.WARN
        assert result.message == "Review this."

    async def test_log_maps_to_log(self, input_ctx: OrchidGuardrailContext) -> None:
        rail = DecisionGuardrail(binding=_binding(_stub("log")))
        result = await rail.check("content worth noting", input_ctx)
        assert result.triggered
        assert result.action is OrchidGuardrailAction.LOG

    async def test_unknown_choice_is_invalid_response(self, input_ctx: OrchidGuardrailContext) -> None:
        provider = _ScriptedProvider(_choice_answer("explode"))
        rail = DecisionGuardrail(binding=_binding(provider), on_failure="closed")
        result = await rail.check("content", input_ctx)
        assert result.blocked
        assert result.details["fallback_reason"] == "invalid_response"

    async def test_wrong_answer_kind_is_invalid_response(self, input_ctx: OrchidGuardrailContext) -> None:
        provider = _ScriptedProvider(_choice_answer(True, kind="boolean"))
        rail = DecisionGuardrail(binding=_binding(provider))
        result = await rail.check("content", input_ctx)
        assert not result.triggered
        assert result.details["fallback_reason"] == "invalid_response"

    async def test_a_missing_answer_is_invalid_response(self, input_ctx: OrchidGuardrailContext) -> None:
        provider = _ScriptedProvider()
        rail = DecisionGuardrail(binding=_binding(provider), on_failure="closed")
        result = await rail.check("content", input_ctx)
        assert result.blocked
        assert result.details["fallback_reason"] == "invalid_response"


# ── Failure policy ──────────────────────────────────────────


class TestFailurePolicy:
    async def test_low_confidence_open_allows(self, input_ctx: OrchidGuardrailContext) -> None:
        rail = DecisionGuardrail(binding=_binding(_stub("block", confidence=0.4)), on_failure="open")
        result = await rail.check("content", input_ctx)
        assert not result.triggered
        assert result.details["fallback_reason"] == "low_confidence"

    async def test_a_low_confidence_block_does_not_block_when_open(self, input_ctx: OrchidGuardrailContext) -> None:
        rail = DecisionGuardrail(binding=_binding(_stub("block", confidence=0.4)), on_failure="open")
        result = await rail.check("content", input_ctx)
        assert not result.triggered

    async def test_low_confidence_closed_blocks(self, input_ctx: OrchidGuardrailContext) -> None:
        rail = DecisionGuardrail(binding=_binding(_stub("allow", confidence=0.4)), on_failure="closed")
        result = await rail.check("content", input_ctx)
        assert result.blocked
        assert result.details["on_failure"] == "closed"

    async def test_refusal_open_allows(self, input_ctx: OrchidGuardrailContext) -> None:
        provider = StubDecisionModel(answers={"verdict": {"refusal": "cannot judge"}})
        rail = DecisionGuardrail(binding=_binding(provider), on_failure="open")
        result = await rail.check("content", input_ctx)
        assert not result.triggered
        assert result.details["fallback_reason"] == "refusal"

    async def test_refusal_closed_blocks(self, input_ctx: OrchidGuardrailContext) -> None:
        provider = StubDecisionModel(answers={"verdict": {"refusal": "cannot judge"}})
        rail = DecisionGuardrail(binding=_binding(provider), on_failure="closed")
        result = await rail.check("content", input_ctx)
        assert result.blocked

    async def test_timeout_open_allows(self, input_ctx: OrchidGuardrailContext) -> None:
        provider = _ScriptedProvider(_choice_answer("block"), delay=0.05)
        rail = DecisionGuardrail(binding=_binding(provider), timeout_seconds=0.01, on_failure="open")
        result = await rail.check("content", input_ctx)
        assert not result.triggered
        assert result.details["fallback_reason"] == "timeout"

    async def test_timeout_closed_blocks(self, input_ctx: OrchidGuardrailContext) -> None:
        provider = _ScriptedProvider(_choice_answer("allow"), delay=0.05)
        rail = DecisionGuardrail(binding=_binding(provider), timeout_seconds=0.01, on_failure="closed")
        result = await rail.check("content", input_ctx)
        assert result.blocked
        assert result.details["fallback_reason"] == "timeout"

    async def test_provider_error_open_allows(self, input_ctx: OrchidGuardrailContext) -> None:
        provider = _ScriptedProvider(error=RuntimeError(f"boom {SENTINEL}"))
        rail = DecisionGuardrail(binding=_binding(provider), on_failure="open")
        result = await rail.check("content", input_ctx)
        assert not result.triggered
        assert result.details["fallback_reason"] == "provider_error"

    async def test_provider_error_closed_blocks(self, input_ctx: OrchidGuardrailContext) -> None:
        provider = _ScriptedProvider(error=RuntimeError(f"boom {SENTINEL}"))
        rail = DecisionGuardrail(binding=_binding(provider), on_failure="closed")
        result = await rail.check("content", input_ctx)
        assert result.blocked
        assert result.details["fallback_reason"] == "provider_error"

    async def test_decision_error_code_is_preserved(self, input_ctx: OrchidGuardrailContext) -> None:
        provider = _ScriptedProvider(
            error=OrchidDecisionError(
                "provider is unavailable",
                code="provider_unavailable",
                provider="scripted",
            )
        )
        rail = DecisionGuardrail(binding=_binding(provider), on_failure="closed")
        result = await rail.check("content", input_ctx)
        assert result.blocked

    async def test_decision_error_code_appears_in_the_decision_line(
        self, input_ctx: OrchidGuardrailContext, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = _ScriptedProvider(
            error=OrchidDecisionError(
                "provider is unavailable",
                code="provider_unavailable",
                provider="scripted",
            )
        )
        rail = DecisionGuardrail(binding=_binding(provider), on_failure="open")

        with caplog.at_level(logging.INFO):
            await rail.check("content", input_ctx)

        lines = _decision_lines(caplog)
        assert len(lines) == 1
        assert "fallback_reason=provider_unavailable" in lines[0]

    async def test_no_binding_open_allows(self, input_ctx: OrchidGuardrailContext) -> None:
        rail = DecisionGuardrail(binding=None, on_failure="open")
        result = await rail.check("content", input_ctx)
        assert not result.triggered
        assert result.details["fallback_reason"] == "not_configured"

    async def test_no_binding_closed_blocks(self, input_ctx: OrchidGuardrailContext) -> None:
        rail = DecisionGuardrail(binding=None, on_failure="closed")
        result = await rail.check("content", input_ctx)
        assert result.blocked
        assert result.details["fallback_reason"] == "not_configured"

    async def test_closed_redact_policy_replaces_the_whole_payload(self, input_ctx: OrchidGuardrailContext) -> None:
        provider = _ScriptedProvider(error=RuntimeError("down"))
        rail = DecisionGuardrail(
            binding=_binding(provider),
            on_failure="closed",
            fail_action="redact",
        )
        result = await rail.check(f"contact {RAW_EMAIL}", input_ctx)
        assert result.action is OrchidGuardrailAction.REDACT
        assert result.redacted_content == "[REDACTED]"
        assert RAW_EMAIL not in (result.redacted_content or "")

    async def test_closed_warn_policy_warns(self, input_ctx: OrchidGuardrailContext) -> None:
        rail = DecisionGuardrail(binding=None, on_failure="closed", fail_action="warn")
        result = await rail.check("content", input_ctx)
        assert result.action is OrchidGuardrailAction.WARN
        assert result.triggered

    async def test_closed_log_policy_logs(self, input_ctx: OrchidGuardrailContext) -> None:
        rail = DecisionGuardrail(binding=None, on_failure="closed", fail_action="log")
        result = await rail.check("content", input_ctx)
        assert result.action is OrchidGuardrailAction.LOG
        assert result.triggered

    async def test_timeout_records_the_bound_provider(
        self, input_ctx: OrchidGuardrailContext, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = _ScriptedProvider(_choice_answer("allow"), delay=0.05, provider="scripted")
        rail = DecisionGuardrail(binding=_binding(provider), timeout_seconds=0.01, on_failure="open")

        with caplog.at_level(logging.INFO):
            await rail.check("content", input_ctx)

        assert "provider=scripted" in _decision_lines(caplog)[0]

    async def test_generic_error_records_the_bound_provider(
        self, input_ctx: OrchidGuardrailContext, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = _ScriptedProvider(error=RuntimeError("boom"), provider="scripted")
        rail = DecisionGuardrail(binding=_binding(provider), on_failure="open")

        with caplog.at_level(logging.INFO):
            await rail.check("content", input_ctx)

        assert "provider=scripted" in _decision_lines(caplog)[0]

    async def test_fallback_warning_is_emitted_once_per_scope(
        self, input_ctx: OrchidGuardrailContext, caplog: pytest.LogCaptureFixture
    ) -> None:
        rail = DecisionGuardrail(binding=_binding(_stub("allow", confidence=0.4)), on_failure="closed")

        with caplog.at_level(logging.DEBUG):
            await rail.check("first", input_ctx)
            await rail.check("second", input_ctx)

        warnings = [
            record
            for record in caplog.records
            if record.name == "orchid_ai.guardrails.decision_guardrail" and record.levelno == logging.WARNING
        ]
        assert len(warnings) == 1
        assert len(_decision_lines(caplog)) == 2

    async def test_no_provider_call_when_binding_is_none(self, input_ctx: OrchidGuardrailContext) -> None:
        provider = _ScriptedProvider(_choice_answer("block"))
        rail = DecisionGuardrail(binding=None, on_failure="closed")

        result = await rail.check("content", input_ctx)

        assert result.blocked
        assert provider.states == []
        assert provider.questions == []


# ── Payload safety and observability ────────────────────────


class TestPayloadSafety:
    async def test_provider_receives_redacted_text_under_the_limit(self, input_ctx: OrchidGuardrailContext) -> None:
        provider = _ScriptedProvider(_choice_answer("allow"))
        rail = DecisionGuardrail(binding=_binding(provider), max_state_chars=500)
        content = f"{SENTINEL} {RAW_EMAIL} {RAW_API_KEY}"

        await rail.check(content, input_ctx)

        state = provider.states[0]
        assert len(state.text) <= 500
        assert RAW_EMAIL not in state.text
        assert RAW_API_KEY not in state.text
        assert "[REDACTED_EMAIL]" in state.text
        assert "[REDACTED_API_KEY]" in state.text
        assert state.payload["truncated"] is False
        assert state.payload["redactions"] >= 2
        assert state.payload["content_chars"] == len(content)

    @pytest.mark.parametrize(
        ("content", "raw", "placeholder"),
        [
            (f"email {RAW_EMAIL}", RAW_EMAIL, "[REDACTED_EMAIL]"),
            ("phone +1 (555) 123-4567", "+1 (555) 123-4567", "[REDACTED_PHONE]"),
            ("card 4111 1111 1111 1111", "4111 1111 1111 1111", "[REDACTED_CARD]"),
            ("ssn 123-45-6789", "123-45-6789", "[REDACTED_SSN]"),
            ("ssn 123 45 6789", "123 45 6789", "[REDACTED_SSN]"),
            ("server 192.168.1.10", "192.168.1.10", "[REDACTED_IP]"),
            (f"jwt {JWT}", JWT, "[REDACTED_JWT]"),
            (f"aws {AWS_KEY}", AWS_KEY, "[REDACTED_AWS_ACCESS_KEY]"),
            (f"google {GOOGLE_KEY}", GOOGLE_KEY, "[REDACTED_GOOGLE_API_KEY]"),
            ("Authorization: Basic dXNlcjpwYXNz", "dXNlcjpwYXNz", "[REDACTED_AUTH_HEADER]"),
            ("Authorization: Digest dXNlcjpwYXNz", "dXNlcjpwYXNz", "[REDACTED_AUTH_HEADER]"),
            ("Authorization: Bearer abcdefghijklmnop", "abcdefghijklmnop", "[REDACTED_BEARER_TOKEN]"),
            ("api_key=supersecretvalue", "supersecretvalue", "[REDACTED_SECRET]"),
            ("secret_key: abcdef123456", "abcdef123456", "[REDACTED_SECRET]"),
            ("token=abcdef123456", "abcdef123456", "[REDACTED_SECRET]"),
            ("the password is hunter2", "hunter2", "[REDACTED_SECRET]"),
            (f"key {RAW_API_KEY}", RAW_API_KEY, "[REDACTED_API_KEY]"),
            (
                "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n-----END RSA PRIVATE KEY-----",
                "MIIEowIBAAKCAQEA",
                "[REDACTED_PRIVATE_KEY]",
            ),
        ],
    )
    async def test_redaction_categories(
        self,
        content: str,
        raw: str,
        placeholder: str,
        input_ctx: OrchidGuardrailContext,
    ) -> None:
        provider = _ScriptedProvider(_choice_answer("allow"))
        rail = DecisionGuardrail(binding=_binding(provider))

        await rail.check(content, input_ctx)

        state = provider.states[0]
        assert raw not in state.text
        assert placeholder in state.text

    async def test_truncated_input_skips_the_provider_and_closed_blocks(
        self, input_ctx: OrchidGuardrailContext
    ) -> None:
        provider = _ScriptedProvider(_choice_answer("allow"))
        rail = DecisionGuardrail(binding=_binding(provider), max_state_chars=20, on_failure="closed")
        content = ("safe padding " * 10) + "harmful suffix"

        result = await rail.check(content, input_ctx)

        assert result.blocked
        assert result.details["fallback_reason"] == "truncated"
        assert provider.states == []

    async def test_truncated_input_open_passes_without_a_provider_call(self, input_ctx: OrchidGuardrailContext) -> None:
        provider = _ScriptedProvider(_choice_answer("block"))
        rail = DecisionGuardrail(binding=_binding(provider), max_state_chars=20, on_failure="open")

        result = await rail.check(("x" * 40) + " harmful", input_ctx)

        assert not result.triggered
        assert result.details["fallback_reason"] == "truncated"
        assert provider.states == []

    async def test_direction_is_sent_for_input_and_output(
        self, input_ctx: OrchidGuardrailContext, output_ctx: OrchidGuardrailContext
    ) -> None:
        provider = _ScriptedProvider(_choice_answer("allow"))
        rail = DecisionGuardrail(binding=_binding(provider))

        await rail.check("in", input_ctx)
        await rail.check("out", output_ctx)

        assert provider.states[0].payload["direction"] == "input"
        assert provider.states[1].payload["direction"] == "output"
        assert "input content" in provider.questions[0].instructions
        assert "output content" in provider.questions[1].instructions

    async def test_the_provider_timeout_is_bounded(self, input_ctx: OrchidGuardrailContext) -> None:
        provider = _ScriptedProvider(_choice_answer("allow"))
        rail = DecisionGuardrail(binding=_binding(provider), timeout_seconds=1.5)

        await rail.check("content", input_ctx)

        assert provider.timeouts == [1.5]

    @pytest.mark.parametrize("outcome", ["accepted", "fallback"])
    async def test_one_structured_decision_line_per_check(
        self, input_ctx: OrchidGuardrailContext, caplog: pytest.LogCaptureFixture, outcome: str
    ) -> None:
        if outcome == "accepted":
            rail = DecisionGuardrail(binding=_binding(_stub("allow")))
        else:
            rail = DecisionGuardrail(binding=None)

        with caplog.at_level(logging.INFO):
            await rail.check("content", input_ctx)

        lines = _decision_lines(caplog)
        assert len(lines) == 1
        for field in DECISION_LOG_FIELDS:
            assert f"{field}=" in lines[0]
        assert "point=guardrails" in lines[0]
        assert f"outcome={outcome}" in lines[0]

    async def test_logs_never_contain_the_raw_payload(
        self, input_ctx: OrchidGuardrailContext, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = _ScriptedProvider(error=RuntimeError(f"boom {SENTINEL} {RAW_EMAIL}"))
        rail = DecisionGuardrail(binding=_binding(provider), on_failure="closed")

        with caplog.at_level(logging.DEBUG):
            await rail.check(f"{SENTINEL} {RAW_EMAIL} {RAW_API_KEY}", input_ctx)

        log_text = "\n".join(record.getMessage() for record in caplog.records)
        assert SENTINEL not in log_text
        assert RAW_EMAIL not in log_text
        assert RAW_API_KEY not in log_text
        assert "boom" not in log_text

    async def test_check_works_inside_the_existing_chain(
        self, input_ctx: OrchidGuardrailContext, caplog: pytest.LogCaptureFixture
    ) -> None:
        chain = OrchidGuardrailChain([DecisionGuardrail(binding=_binding(_stub("block")))])

        with caplog.at_level(logging.INFO):
            result = await chain.evaluate("content", input_ctx)

        assert result.blocked
        assert len(_decision_lines(caplog)) == 1


# ── Graph-level input rails ─────────────────────────────────


class _GraphAgent(OrchidAgent):
    """Minimal graph-node agent that records the state it was given."""

    def __init__(self, name: str = "a") -> None:
        super().__init__(reader=None)  # type: ignore[arg-type]
        self._name = name
        self.ran = False
        self.seen_state: Any = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "graph test agent"

    async def run(self, state: Any) -> Any:
        self.ran = True
        self.seen_state = state
        return {"messages": [AIMessage(content="ok")]}


def _dispatch_state(content: str) -> dict[str, Any]:
    """State shaped exactly like a supervisor dispatch to an agent."""
    return {
        "messages": [
            HumanMessage(
                content=content,
                id="m-1",
                name="alice",
                additional_kwargs={"trace": "t-1"},
            ),
            AIMessage(content="dispatch to a", id="dispatch-1"),
            AIMessage(content="previous agent reply", id="reply-1"),
        ],
        "chat_id": "c-1",
    }


def _last_human(messages: list[Any]) -> HumanMessage:
    return next(message for message in reversed(messages) if isinstance(message, HumanMessage))


def _merge(state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    return {**state, "messages": add_messages(state["messages"], update.get("messages", []))}


async def _run_supervisor_then_agent(node: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Compile supervisor → agent so LangGraph's reducer merges the update."""
    graph = StateGraph(GraphState)

    async def supervisor(state: dict[str, Any], config=None) -> dict[str, Any]:
        return {
            "messages": [AIMessage(content="dispatch", id="dispatch-1")],
            "active_agents": ["a"],
        }

    graph.add_node("supervisor", supervisor)
    graph.add_node("a_agent", node)
    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "a_agent")
    graph.add_edge("a_agent", END)
    return await graph.compile().ainvoke(state)


class TestGraphInputGuardrails:
    async def test_compiled_supervisor_to_agent_applies_accepted_redact(self) -> None:
        agent = _GraphAgent()
        chain = OrchidGuardrailChain([DecisionGuardrail(binding=_binding(_stub("redact")))])
        node = _create_agent_node(agent, input_guardrails=chain)

        result = await _run_supervisor_then_agent(
            node,
            {"messages": [HumanMessage(content=f"contact {RAW_EMAIL}", id="m-1")], "chat_id": "c-1"},
        )

        assert agent.ran
        assert _last_human(agent.seen_state["messages"]).content == "[REDACTED]"
        assert all(RAW_EMAIL not in str(message.content) for message in result["messages"])
        replacement = _last_human(result["messages"])
        assert replacement.content == "[REDACTED]"
        assert replacement.id == "m-1"

    async def test_compiled_supervisor_to_agent_applies_closed_fallback_redact(self) -> None:
        agent = _GraphAgent()
        provider = _ScriptedProvider(error=RuntimeError("down"))
        chain = OrchidGuardrailChain(
            [
                DecisionGuardrail(
                    binding=_binding(provider),
                    on_failure="closed",
                    fail_action="redact",
                )
            ]
        )
        node = _create_agent_node(agent, input_guardrails=chain)

        result = await _run_supervisor_then_agent(
            node,
            {"messages": [HumanMessage(content=f"contact {RAW_EMAIL}", id="m-1")], "chat_id": "c-1"},
        )

        assert agent.ran
        assert _last_human(agent.seen_state["messages"]).content == "[REDACTED]"
        assert all(RAW_EMAIL not in str(message.content) for message in result["messages"])

    async def test_accepted_redact_preserves_message_metadata(self) -> None:
        agent = _GraphAgent()
        chain = OrchidGuardrailChain([DecisionGuardrail(binding=_binding(_stub("redact")))])
        node = _create_agent_node(agent, input_guardrails=chain)
        state = _dispatch_state(f"contact {RAW_EMAIL}")

        result = await node(state)  # type: ignore[arg-type]
        merged = _merge(state, result)

        assert agent.ran
        assert _last_human(agent.seen_state["messages"]).content == "[REDACTED]"
        assert all(RAW_EMAIL not in str(message.content) for message in merged["messages"])
        replacement = _last_human(result["messages"])
        assert replacement.id == "m-1"
        assert replacement.name == "alice"
        assert replacement.additional_kwargs == {"trace": "t-1"}

    async def test_closed_fallback_redact_uses_the_last_human_not_the_dispatch(self) -> None:
        agent = _GraphAgent()
        provider = _ScriptedProvider(error=RuntimeError("down"))
        chain = OrchidGuardrailChain(
            [DecisionGuardrail(binding=_binding(provider), on_failure="closed", fail_action="redact")]
        )
        node = _create_agent_node(agent, input_guardrails=chain)
        state = _dispatch_state(f"contact {RAW_EMAIL}")

        result = await node(state)  # type: ignore[arg-type]
        merged = _merge(state, result)

        assert agent.ran
        assert _last_human(agent.seen_state["messages"]).content == "[REDACTED]"
        assert all(RAW_EMAIL not in str(message.content) for message in merged["messages"])
        assert _last_human(result["messages"]).id == "m-1"

    async def test_child_agent_redacts_the_query_before_its_last_human(self) -> None:
        config = OrchidAgentsConfig.model_validate(
            {
                "agents": {
                    "parent": {
                        "description": "Parent",
                        "prompt": "p",
                        "children": {
                            "kid": {
                                "description": "Kid",
                                "prompt": "p",
                                "guardrails": {"input": [{"type": "decision", "on_failure": "closed"}]},
                            }
                        },
                    }
                },
                "decision_models": {"redactor": {"protocol": "stub"}},
                "decisions": {"points": {"guardrails": {"model": "redactor"}}},
            }
        )
        runtime = OrchidRuntime(
            default_model="test/model",
            decision_models={"redactor": _stub("redact")},
        )
        child_config = config.agents["parent"].children["kid"]
        chain, _ = _GuardrailWiring.build_chains(
            child_config.guardrails,
            config=config,
            runtime=runtime,
            agent_name="parent.kid",
        )
        agent = _GraphAgent("kid")
        node = _create_agent_node(agent, input_guardrails=chain)
        state = _dispatch_state(f"contact {RAW_EMAIL}")

        result = await node(state)  # type: ignore[arg-type]
        merged = _merge(state, result)

        assert agent.ran
        assert _last_human(agent.seen_state["messages"]).content == "[REDACTED]"
        assert all(RAW_EMAIL not in str(message.content) for message in merged["messages"])
        assert result["messages"][0].id == "m-1"

    async def test_deterministic_pii_redact_now_reaches_the_agent(self) -> None:
        from orchid_ai.guardrails.pii import PIIDetectionGuardrail

        agent = _GraphAgent()
        chain = OrchidGuardrailChain([PIIDetectionGuardrail(fail_action="redact", entities=["email"])])
        node = _create_agent_node(agent, input_guardrails=chain)

        result = await node(_dispatch_state(f"mail {RAW_EMAIL}"))  # type: ignore[arg-type]

        assert _last_human(agent.seen_state["messages"]).content == "mail [REDACTED_EMAIL]"
        assert result["messages"][0].content == "mail [REDACTED_EMAIL]"

    async def test_global_input_node_redacts_with_a_dispatch_message_last(self) -> None:
        chain = OrchidGuardrailChain([DecisionGuardrail(binding=_binding(_stub("redact")))])
        node = _GuardrailWiring.create_global_input_node(chain, agent_name="parent")
        state = _dispatch_state(f"contact {RAW_EMAIL}")

        update = await node(state)  # type: ignore[arg-type]
        merged = _merge(state, update)

        assert all(RAW_EMAIL not in str(message.content) for message in merged["messages"])
        replacement = _last_human(update["messages"])
        assert replacement.content == "[REDACTED]"
        assert replacement.id == "m-1"
        assert replacement.name == "alice"
        assert replacement.additional_kwargs == {"trace": "t-1"}

    async def test_decision_block_short_circuits_the_agent(self) -> None:
        agent = _GraphAgent()
        chain = OrchidGuardrailChain([DecisionGuardrail(binding=_binding(_stub("block")))])
        node = _create_agent_node(agent, input_guardrails=chain)

        result = await node(_dispatch_state("harmful content"))  # type: ignore[arg-type]

        assert agent.ran is False
        assert result["messages"][0].content == "[A Agent] This content was blocked by a policy check."
        assert result["active_agents"] == []

    async def test_the_guardrail_contract_is_exported(self) -> None:
        import orchid_ai
        from orchid_ai.core.guardrails import OrchidGuardrailFailurePolicyAware

        assert orchid_ai.OrchidGuardrailFailurePolicy is OrchidGuardrailFailurePolicy
        assert orchid_ai.OrchidGuardrailFailurePolicyAware is OrchidGuardrailFailurePolicyAware


# ── Nested-agent wiring ─────────────────────────────────────


class _NestedFakeAgent:
    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description


def _nested_config() -> OrchidAgentsConfig:
    return OrchidAgentsConfig.model_validate(
        {
            "agents": {
                "parent": {
                    "description": "Parent",
                    "prompt": "p",
                    "children": {
                        "kid": {
                            "description": "Kid",
                            "prompt": "p",
                            "guardrails": {"input": [{"type": "decision", "on_failure": "closed"}]},
                        }
                    },
                }
            },
            "decision_models": {"stubber": {"protocol": "stub"}},
            "decisions": {"points": {"guardrails": {"model": "stubber"}}},
        }
    )


async def test_nested_agent_guardrails_are_built(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _nested_config()
    runtime = OrchidRuntime(
        default_model="test/model",
        decision_models={"stubber": _stub("block")},
    )
    captured: dict[str, Any] = {}

    def _fake_instantiate(name: str, cfg: Any, *args: Any, **kwargs: Any) -> _NestedFakeAgent:
        return _NestedFakeAgent(name, cfg.description)

    def _fake_create_node(
        agent: Any,
        input_guardrails: Any = None,
        output_guardrails: Any = None,
        **kwargs: Any,
    ) -> Any:
        captured["agent"] = agent
        captured["input"] = input_guardrails
        captured["output"] = output_guardrails

        async def _node(state, config=None):
            return state

        return _node

    monkeypatch.setattr(graph_module, "_instantiate_agent", _fake_instantiate)
    monkeypatch.setattr(graph_module, "_create_agent_node", _fake_create_node)

    _build_subgraph(
        "parent",
        config.agents["parent"],
        "test/model",
        NullVectorReader(),
        default_chat_model=FakeListChatModel(responses=["ok"]),
        root_config=config,
        runtime=runtime,
    )

    assert captured["agent"].name == "kid"
    chain = captured["input"]
    assert isinstance(chain, OrchidGuardrailChain)
    result = await chain.evaluate(
        "content",
        OrchidGuardrailContext(direction=OrchidGuardrailDirection.INPUT, agent_name="parent.kid"),
    )
    assert result.blocked
    assert result.details["verdict"] == "block"


async def test_parent_with_children_runs_its_own_guardrails() -> None:
    config = OrchidAgentsConfig.model_validate(
        {
            "agents": {
                "parent": {
                    "description": "Parent",
                    "prompt": "p",
                    "guardrails": {
                        "input": [{"type": "decision", "on_failure": "closed"}],
                        "output": [{"type": "decision", "on_failure": "closed"}],
                    },
                    "children": {
                        "kid": {
                            "description": "Kid",
                            "prompt": "p",
                            "llm": {"model": "test/model"},
                        }
                    },
                }
            },
            "decision_models": {"blocker": {"protocol": "stub"}},
            "decisions": {"points": {"guardrails": {"model": "blocker"}}},
        }
    )
    runtime = OrchidRuntime(
        default_model="test/model",
        decision_models={"blocker": _stub("block")},
    )

    subgraph = _build_subgraph(
        "parent",
        config.agents["parent"],
        "test/model",
        NullVectorReader(),
        default_chat_model=FakeListChatModel(responses=["should not run"]),
        root_config=config,
        runtime=runtime,
    )

    nodes = set(subgraph.get_graph().nodes)
    assert {"input_guardrails", "output_guardrails"} <= nodes

    result = await subgraph.ainvoke({"messages": [HumanMessage(content="harmful content")], "chat_id": "c-1"})

    assert result["final_response"] == "This content was blocked by a policy check."
    assert result["messages"][-1].content == "This content was blocked by a policy check."


async def test_parent_output_guardrail_blocks_the_synthesised_answer() -> None:
    config = OrchidAgentsConfig.model_validate(
        {
            "agents": {
                "parent": {
                    "description": "Parent",
                    "prompt": "p",
                    "guardrails": {"output": [{"type": "decision", "on_failure": "closed"}]},
                    "children": {"kid": {"description": "Kid", "prompt": "p"}},
                }
            },
            "decision_models": {"blocker": {"protocol": "stub"}},
            "decisions": {"points": {"guardrails": {"model": "blocker"}}},
        }
    )
    runtime = OrchidRuntime(default_model="test/model", decision_models={"blocker": _stub("block")})

    _, chain = _GuardrailWiring.build_chains(
        config.agents["parent"].guardrails,
        config=config,
        runtime=runtime,
        agent_name="parent",
    )
    node = _GuardrailWiring.create_global_output_node(chain, agent_name="parent")

    result = await node(  # type: ignore[arg-type]
        {
            "messages": [AIMessage(content="child answer")],
            "final_response": "child answer",
            "chat_id": "c-1",
        }
    )

    assert result["final_response"] == "This content was blocked by a policy check."
    assert result["messages"][-1].content == "This content was blocked by a policy check."


def _wiring_config() -> OrchidAgentsConfig:
    return OrchidAgentsConfig.model_validate(
        {
            "agents": {
                "a": {
                    "description": "A",
                    "prompt": "p",
                    "guardrails": {"input": [{"type": "decision", "on_failure": "closed"}]},
                    "decisions": {"points": {"guardrails": {"model": "blocker"}}},
                },
                "b": {
                    "description": "B",
                    "prompt": "p",
                    "guardrails": {"input": [{"type": "decision", "on_failure": "open"}]},
                },
                "c": {
                    "description": "C",
                    "prompt": "p",
                    "guardrails": {"input": [{"type": "decision", "on_failure": "closed"}]},
                    "decisions": {"points": {"guardrails": {"enabled": False}}},
                },
            },
            "guardrails": {"input": [{"type": "decision", "on_failure": "open"}]},
            "decision_models": {
                "allow_all": {"protocol": "stub"},
                "blocker": {"protocol": "stub"},
            },
            "decisions": {"points": {"guardrails": {"model": "allow_all"}}},
        }
    )


def _wiring_runtime() -> OrchidRuntime:
    return OrchidRuntime(
        default_model="test/model",
        decision_models={
            "allow_all": _stub("allow"),
            "blocker": _stub("block"),
        },
    )


def _chain_for(
    config: OrchidAgentsConfig,
    runtime: OrchidRuntime,
    *,
    agent_name: str | None = None,
) -> OrchidGuardrailChain:
    guardrails_config = config.guardrails if agent_name is None else config.agents[agent_name].guardrails
    chain, _ = _GuardrailWiring.build_chains(
        guardrails_config,
        config=config,
        runtime=runtime,
        agent_name=agent_name,
    )
    return chain


async def _first_rail_check(
    chain: OrchidGuardrailChain,
    context: OrchidGuardrailContext,
) -> Any:
    """Inspect the rail directly: the chain drops details for a clean pass."""
    return await chain.guardrails[0].check("content", context)


class TestWiring:
    async def test_global_chain_uses_the_fleet_level_binding(self) -> None:
        chain = _chain_for(_wiring_config(), _wiring_runtime())
        result = await _first_rail_check(
            chain,
            OrchidGuardrailContext(direction=OrchidGuardrailDirection.INPUT),
        )
        assert not result.triggered
        assert result.details["verdict"] == "allow"

    async def test_per_agent_override_wins(self) -> None:
        chain = _chain_for(_wiring_config(), _wiring_runtime(), agent_name="a")
        result = await chain.evaluate(
            "content",
            OrchidGuardrailContext(direction=OrchidGuardrailDirection.INPUT, agent_name="a"),
        )
        assert result.blocked
        assert result.details["verdict"] == "block"

    async def test_agent_without_an_override_inherits_the_global_binding(self) -> None:
        chain = _chain_for(_wiring_config(), _wiring_runtime(), agent_name="b")
        result = await _first_rail_check(
            chain,
            OrchidGuardrailContext(direction=OrchidGuardrailDirection.INPUT, agent_name="b"),
        )
        assert not result.triggered
        assert result.details["verdict"] == "allow"

    async def test_disabled_binding_applies_the_rail_failure_policy(self) -> None:
        chain = _chain_for(_wiring_config(), _wiring_runtime(), agent_name="c")
        result = await chain.evaluate(
            "content",
            OrchidGuardrailContext(direction=OrchidGuardrailDirection.INPUT, agent_name="c"),
        )
        assert result.blocked
        assert result.details["fallback_reason"] == "disabled"

    async def test_missing_provider_is_distinguished_from_not_configured(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        config = OrchidAgentsConfig.model_validate(
            {
                "agents": {
                    "a": {
                        "description": "A",
                        "prompt": "p",
                        "guardrails": {"input": [{"type": "decision", "on_failure": "closed"}]},
                    }
                },
                "decision_models": {"ghost": {"protocol": "stub"}},
                "decisions": {"points": {"guardrails": {"model": "ghost"}}},
            }
        )
        runtime = OrchidRuntime(default_model="test/model")  # ghost provider not built

        with caplog.at_level(logging.WARNING):
            chain = _chain_for(config, runtime, agent_name="a")

        result = await chain.evaluate(
            "content",
            OrchidGuardrailContext(direction=OrchidGuardrailDirection.INPUT, agent_name="a"),
        )

        assert result.blocked
        assert result.details["fallback_reason"] == "provider_missing"
        guardrail_warnings = [
            record
            for record in caplog.records
            if record.name == "orchid_ai.graph.guardrail_wiring" and "provider_missing" in record.getMessage()
        ]
        assert len(guardrail_warnings) == 1
        assert "[Decision]" not in caplog.text

    async def test_agent_rule_keeps_its_on_failure_policy(self) -> None:
        config = _wiring_config()
        chain = _chain_for(config, _wiring_runtime(), agent_name="a")
        rail = chain.guardrails[0]
        assert isinstance(rail, DecisionGuardrail)
        assert rail.on_failure is OrchidGuardrailFailurePolicy.CLOSED

    async def test_a_decision_rail_without_config_warns_and_uses_the_policy(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        config = OrchidAgentsConfig.model_validate(
            {
                "agents": {
                    "a": {
                        "description": "A",
                        "prompt": "p",
                        "guardrails": {"input": [{"type": "decision", "on_failure": "closed"}]},
                    }
                }
            }
        )

        with caplog.at_level(logging.WARNING):
            chain, _ = _GuardrailWiring.build_chains(config.agents["a"].guardrails, agent_name="a")

        assert "no config/runtime" in caplog.text
        result = await chain.evaluate(
            "content",
            OrchidGuardrailContext(direction=OrchidGuardrailDirection.INPUT, agent_name="a"),
        )
        assert result.blocked

    async def test_deterministic_chains_are_not_rewired(self) -> None:
        config = OrchidAgentsConfig.model_validate(
            {
                "agents": {
                    "a": {
                        "description": "A",
                        "prompt": "p",
                        "guardrails": {"input": [{"type": "content_safety", "fail_action": "block"}]},
                    }
                }
            }
        )
        runtime = OrchidRuntime(default_model="test/model")
        chain = _chain_for(config, runtime, agent_name="a")
        result = await chain.evaluate(
            "this contains no harmful pattern",
            OrchidGuardrailContext(direction=OrchidGuardrailDirection.INPUT, agent_name="a"),
        )
        assert not result.triggered
