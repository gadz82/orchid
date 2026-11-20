"""Tests for the decision pipeline step."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest

from orchid_ai.config.schema_pipeline import (
    OrchidPipelineDecisionJobConfig,
    OrchidPipelineDefinitionConfig,
    OrchidPipelineDeterministicJobConfig,
    OrchidPipelineRetryConfig,
)
from orchid_ai.core.decision import (
    OrchidDecisionAnswer,
    OrchidDecisionError,
    OrchidDecisionModel,
    OrchidDecisionQuestion,
    OrchidDecisionResult,
    OrchidDecisionState,
)
from orchid_ai.core.pipeline import (
    OrchidPipelineConfigError,
    OrchidPipelineJobContext,
    OrchidPipelineLimitError,
    OrchidPipelineRunContext,
    OrchidPipelineStepError,
    OrchidPipelineTimeoutError,
)
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.pipeline.builder import PipelineExecution
from orchid_ai.pipeline.definition import CompiledJob, CompiledPipeline, PipelineDependencies, PipelineLimits
from orchid_ai.pipeline.graph import DataGraph
from orchid_ai.pipeline.mapping import build_mapping_context
from orchid_ai.pipeline.runner import OrchidPipelineRunner
from orchid_ai.pipeline.session import PipelineStepContext
from orchid_ai.pipeline.steps.decision import DecisionStep

START = datetime(2026, 11, 18, 10, 0, tzinfo=UTC)


class _Model(OrchidDecisionModel):
    def __init__(
        self,
        *,
        value: str | None = "retain",
        confidence: float = 0.9,
        refusal: str | None = None,
        answers: dict[str, OrchidDecisionAnswer] | None = None,
        error: Exception | None = None,
        latency_ms: float = 5.0,
    ) -> None:
        self._value = value
        self._confidence = confidence
        self._refusal = refusal
        self._answers = answers
        self._error = error
        self._latency_ms = latency_ms
        self.calls: list[tuple[OrchidDecisionState, list[OrchidDecisionQuestion], float | None]] = []

    async def decide(
        self,
        state: OrchidDecisionState,
        questions: Any,
        *,
        timeout: float | None = None,
    ) -> OrchidDecisionResult:
        self.calls.append((state, list(questions), timeout))
        if self._error is not None:
            raise self._error
        if self._answers is not None:
            answers = self._answers
        else:
            answers = {
                "decision": OrchidDecisionAnswer(
                    name="decision",
                    kind="choice",
                    value=None if self._refusal else self._value,
                    confidence=self._confidence,
                    refusal=self._refusal,
                )
            }
        return OrchidDecisionResult(
            answers=answers,
            provider="stub",
            model="stub",
            latency_ms=self._latency_ms,
            usage={"prompt_tokens": 3, "completion_tokens": 2},
        )


class _Chat:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    async def ainvoke(self, messages: Any) -> Any:
        self.calls += 1
        return _Message(self.responses.pop(0))


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


def _config(**overrides: Any) -> OrchidPipelineDecisionJobConfig:
    values: dict[str, Any] = {
        "kind": "decision",
        "decision_model": "triage",
        "question": "Which action for {{ input.account_id }}?",
        "labels": ["retain", "expand"],
    }
    values.update(overrides)
    return OrchidPipelineDecisionJobConfig.model_validate(values)


def _job(config: OrchidPipelineDecisionJobConfig | None = None, *, job_id: str = "choose") -> CompiledJob:
    resolved = config or _config()
    return CompiledJob(
        id=job_id,
        kind="decision",
        config=resolved,
        depends_on=tuple(resolved.depends_on),
        branches=dict(resolved.branches),
        fallback=resolved.fallback,
    )


def _pipeline(*, limits: PipelineLimits | None = None, defaults: dict[str, Any] | None = None) -> CompiledPipeline:
    return CompiledPipeline(
        name="demo",
        version="1",
        description="",
        definition=OrchidPipelineDefinitionConfig(),
        jobs={},
        limits=limits or PipelineLimits.from_config(),
        defaults=defaults or {},
    )


def _ctx(
    *,
    job: CompiledJob | None = None,
    model: Any = None,
    chat_model: Any = None,
    pipeline: CompiledPipeline | None = None,
    limits: PipelineLimits | None = None,
    payload: dict[str, Any] | None = None,
) -> PipelineStepContext:
    job = job or _job()
    pipeline = pipeline or _pipeline(limits=limits)
    if limits is not None:
        pipeline.limits = limits
    run = OrchidPipelineRunContext(
        run_id="run-1",
        pipeline="demo",
        pipeline_version="1",
        mode="direct",
        tenant_key="t1",
        user_id="u1",
        started_at=START,
    )
    inputs = payload or {"account_id": "a1"}
    return PipelineStepContext(
        run=run,
        job=OrchidPipelineJobContext(job_id=job.id, kind="decision"),
        input=inputs,
        graph=DataGraph().view(),
        auth=OrchidAuthContext(access_token="token", tenant_key="t1", user_id="u1"),
        chat_model=chat_model,
        reader=None,
        logger=logging.getLogger("decision-step-test"),
        idempotency_key="key",
        pipeline=pipeline,
        mapping_context=build_mapping_context(input=inputs, run=run),
    )


def _run(step: DecisionStep, ctx: Any) -> Any:
    return asyncio.run(step.execute(ctx))


# ── Accepted decisions ─────────────────────────────────────


def test_accepted_label_is_returned_with_confidence_and_answers() -> None:
    model = _Model(value="retain", confidence=0.91)
    result = _run(DecisionStep(_job(), models={"triage": model}), _ctx(model=model))

    assert result.output["label"] == "retain"
    assert result.output["confidence"] == 0.91
    assert result.output["answers"]["decision"]["value"] == "retain"
    assert "decision model" in result.output["rationale"]
    assert "retain" in result.output["rationale"]
    assert len(result.output["rationale"]) <= 300
    assert result.usage["decision_calls"] == 1


def test_question_is_rendered_and_context_is_passed() -> None:
    model = _Model(value="expand")
    job = _job(_config(context_from="input"))
    result = _run(DecisionStep(job, models={"triage": model}), _ctx(model=model))

    state, questions, _timeout = model.calls[0]
    assert state.text == "Which action for a1?"
    assert state.payload == {"account_id": "a1"}
    assert questions[0].kind == "choice"
    assert questions[0].options == ("retain", "expand")
    assert result.output["label"] == "expand"


def test_non_mapping_context_is_wrapped() -> None:
    model = _Model()
    job = _job(_config(context_from="input.account_id"))
    _run(DecisionStep(job, models={"triage": model}), _ctx(model=model))

    state, _, _ = model.calls[0]
    assert state.payload == {"context": "a1"}


def test_provider_usage_is_added_to_the_step_usage() -> None:
    model = _Model()
    result = _run(DecisionStep(_job(), models={"triage": model}), _ctx(model=model))

    assert result.usage["prompt_tokens"] == 3
    assert result.usage["completion_tokens"] == 2


def test_missing_decision_model_is_a_config_error() -> None:
    with pytest.raises(OrchidPipelineConfigError, match="unknown decision model 'triage'"):
        _run(DecisionStep(_job(), models={}), _ctx())


def test_fewer_than_two_labels_is_a_config_error() -> None:
    with pytest.raises(OrchidPipelineConfigError, match="at least two labels"):
        DecisionStep(_job(_config(labels=["only"])), models={"triage": _Model()})


def test_non_decision_job_is_rejected() -> None:
    job = CompiledJob(
        id="x",
        kind="deterministic",
        config=OrchidPipelineDeterministicJobConfig.model_validate({"kind": "deterministic", "uses": "pkg.step"}),
    )
    with pytest.raises(TypeError, match="OrchidPipelineDecisionJobConfig"):
        DecisionStep(job, models={})


# ── Low confidence / fallback / fail ───────────────────────


def test_low_confidence_with_fallback_uses_the_fallback_label() -> None:
    model = _Model(value="retain", confidence=0.2)
    job = _job(_config(fallback="expand", on_low_confidence="fallback"))

    result = _run(DecisionStep(job, models={"triage": model}), _ctx(model=model))

    assert result.output["label"] == "expand"
    assert result.output["confidence"] == 0.0
    assert "low_confidence" in result.output["rationale"]


def test_low_confidence_without_fallback_fails() -> None:
    model = _Model(value="retain", confidence=0.2)
    job = _job(_config(on_low_confidence="fallback"))

    with pytest.raises(OrchidPipelineStepError, match="no fallback label"):
        _run(DecisionStep(job, models={"triage": model}), _ctx(model=model))


def test_low_confidence_with_fail_policy_fails() -> None:
    model = _Model(value="retain", confidence=0.2)
    job = _job(_config(on_low_confidence="fail"))

    with pytest.raises(OrchidPipelineStepError, match="low_confidence"):
        _run(DecisionStep(job, models={"triage": model}), _ctx(model=model))


def test_refusal_follows_the_policy() -> None:
    model = _Model(refusal="cannot judge", confidence=0.99)
    job = _job(_config(fallback="retain", on_low_confidence="fallback"))

    result = _run(DecisionStep(job, models={"triage": model}), _ctx(model=model))

    assert result.output["label"] == "retain"
    assert "refusal" in result.output["rationale"]


def test_missing_answer_follows_the_policy() -> None:
    model = _Model(answers={})
    job = _job(_config(fallback="retain", on_low_confidence="fallback"))

    result = _run(DecisionStep(job, models={"triage": model}), _ctx(model=model))

    assert result.output["label"] == "retain"


def test_unknown_label_with_fallback_uses_the_fallback() -> None:
    model = _Model(value="not-a-label", confidence=0.99)
    job = _job(_config(fallback="expand", on_low_confidence="fallback"))

    result = _run(DecisionStep(job, models={"triage": model}), _ctx(model=model))

    assert result.output["label"] == "expand"


def test_unknown_label_without_fallback_fails() -> None:
    model = _Model(value="not-a-label", confidence=0.99)
    job = _job(_config(on_low_confidence="fallback"))

    with pytest.raises(OrchidPipelineStepError, match="unknown_label"):
        _run(DecisionStep(job, models={"triage": model}), _ctx(model=model))


def test_provider_error_follows_the_policy() -> None:
    model = _Model(error=OrchidDecisionError("timed out", code="timeout", provider="stub"))
    job = _job(_config(fallback="retain", on_low_confidence="fallback"))

    result = _run(DecisionStep(job, models={"triage": model}), _ctx(model=model))

    assert result.output["label"] == "retain"
    assert "timeout" in result.output["rationale"]


def test_provider_error_with_fail_policy_fails() -> None:
    model = _Model(error=OrchidDecisionError("timed out", code="timeout", provider="stub"))
    job = _job(_config(on_low_confidence="fail"))

    with pytest.raises(OrchidPipelineStepError, match="timeout"):
        _run(DecisionStep(job, models={"triage": model}), _ctx(model=model))


def test_unexpected_provider_exception_is_sanitized() -> None:
    model = _Model(error=RuntimeError("secret sk-provider"))
    job = _job(_config(fallback="retain", on_low_confidence="fallback"))

    result = _run(DecisionStep(job, models={"triage": model}), _ctx(model=model))

    assert result.output["label"] == "retain"
    assert "secret" not in result.output["rationale"]


# ── LLM fallback ───────────────────────────────────────────


def test_llm_fallback_chooses_a_label() -> None:
    model = _Model(value="retain", confidence=0.1)
    chat = _Chat(['{"label": "expand", "confidence": 0.8}'])
    job = _job(_config(on_low_confidence="llm"))

    result = _run(DecisionStep(job, models={"triage": model}), _ctx(model=model, chat_model=chat))

    assert result.output["label"] == "expand"
    assert result.usage["llm_calls"] == 1


def test_llm_fallback_repairs_once() -> None:
    model = _Model(value="retain", confidence=0.1)
    chat = _Chat(["not json", '{"label": "expand", "confidence": 0.7}'])
    job = _job(_config(on_low_confidence="llm"))

    result = _run(DecisionStep(job, models={"triage": model}), _ctx(model=model, chat_model=chat))

    assert result.output["label"] == "expand"
    assert result.usage["llm_calls"] == 2


def test_llm_fallback_failure_uses_the_fallback_label() -> None:
    model = _Model(value="retain", confidence=0.1)
    chat = _Chat(["not json", "still not json"])
    job = _job(_config(on_low_confidence="llm", fallback="retain"))

    result = _run(DecisionStep(job, models={"triage": model}), _ctx(model=model, chat_model=chat))

    assert result.output["label"] == "retain"
    assert "llm_failed" in result.output["rationale"]


def test_llm_fallback_failure_without_fallback_fails() -> None:
    model = _Model(value="retain", confidence=0.1)
    chat = _Chat(["not json", "still not json"])
    job = _job(_config(on_low_confidence="llm"))

    with pytest.raises(OrchidPipelineStepError, match="llm fallback failed"):
        _run(DecisionStep(job, models={"triage": model}), _ctx(model=model, chat_model=chat))


def test_llm_fallback_uses_the_defaults_model() -> None:
    built: list[str] = []
    chat = _Chat(['{"label": "expand", "confidence": 0.8}'])

    def factory(ref: str) -> Any:
        built.append(ref)
        return chat

    model = _Model(value="retain", confidence=0.1)
    job = _job(_config(on_low_confidence="llm"))
    _run(
        DecisionStep(job, models={"triage": model}, model_factory=factory),
        _ctx(model=model, chat_model=None, pipeline=_pipeline(defaults={"model": "default/model"})),
    )

    assert built == ["default/model"]


def test_llm_fallback_without_any_model_fails() -> None:
    model = _Model(value="retain", confidence=0.1)
    job = _job(_config(on_low_confidence="llm"))

    with pytest.raises(OrchidPipelineStepError, match="llm fallback failed"):
        _run(DecisionStep(job, models={"triage": model}), _ctx(model=model, chat_model=None))


# ── Limits, timeouts and logs ─────────────────────────────


def test_context_payload_over_the_cap_is_a_limit_error() -> None:
    model = _Model()
    limits = replace(PipelineLimits.from_config(), max_context_chars=5)
    job = _job(_config(context_from="input"))

    with pytest.raises(OrchidPipelineLimitError, match="max_context_chars=5"):
        _run(
            DecisionStep(job, models={"triage": model}),
            _ctx(model=model, limits=limits, pipeline=_pipeline(limits=limits), payload={"account_id": "a1"}),
        )


def test_job_timeout_is_enforced() -> None:
    class _SlowModel(OrchidDecisionModel):
        async def decide(self, state: Any, questions: Any, *, timeout: float | None = None) -> Any:
            await asyncio.sleep(0.3)
            return OrchidDecisionResult(answers={}, provider="stub", model="stub")

    job = replace(_job(), timeout_seconds=0.01)

    with pytest.raises(OrchidPipelineTimeoutError, match="timed out after 0.010 seconds"):
        _run(DecisionStep(job, models={"triage": _SlowModel()}), _ctx())


def test_accepted_decision_logs_safe_fields(caplog: pytest.LogCaptureFixture) -> None:
    model = _Model(value="retain", confidence=0.91)
    job = _job(_config(fallback="expand", on_low_confidence="fallback"))
    caplog.set_level(logging.INFO)

    _run(DecisionStep(job, models={"triage": model}), _ctx(model=model))

    line = caplog.records[-1].getMessage()
    assert "decision point=pipeline:demo:choose" in line
    assert "path=decision" in line
    assert "provider=stub" in line
    assert "outcome=accepted" in line
    assert "confidence=0.91" in line
    assert "a1" not in line


def test_fallback_decision_logs_the_reason(caplog: pytest.LogCaptureFixture) -> None:
    model = _Model(value="retain", confidence=0.2)
    job = _job(_config(fallback="expand", on_low_confidence="fallback"))
    caplog.set_level(logging.INFO)

    _run(DecisionStep(job, models={"triage": model}), _ctx(model=model))

    assert "outcome=fallback" in caplog.records[-1].getMessage()
    assert "fallback_reason=low_confidence" in caplog.records[-1].getMessage()


# ── Branch routing integration ────────────────────────────


def _deterministic_branch(job_id: str, result: str) -> CompiledJob:
    config = OrchidPipelineDeterministicJobConfig.model_validate({"kind": "deterministic", "uses": "pkg.branch"})
    return CompiledJob(
        id=job_id,
        kind="deterministic",
        config=config,
        handler=lambda ctx: {"branch": result},
        retry=OrchidPipelineRetryConfig(),
    )


def test_decision_selects_the_branch_in_the_compiled_graph() -> None:
    decision = _job(_config(labels=["retain", "expand"], branches={"retain": "left", "expand": "right"}))
    pipeline = _pipeline()
    pipeline.jobs = {
        "choose": decision,
        "left": _deterministic_branch("left", "left"),
        "right": _deterministic_branch("right", "right"),
    }
    model = _Model(value="expand")

    result = asyncio.run(
        OrchidPipelineRunner(
            pipeline,
            deps=PipelineDependencies(decision_models={"triage": model}),
            execution=PipelineExecution(),
        ).run({"account_id": "a1"}, user_id="u1")
    )

    assert result.status == "succeeded"
    assert result.jobs["choose"]["output"]["label"] == "expand"
    assert result.jobs["right"]["status"] == "succeeded"
    assert "left" not in result.jobs


def test_decision_routes_the_fallback_label_to_its_branch() -> None:
    decision = _job(
        _config(
            labels=["retain", "expand"],
            branches={"retain": "left", "expand": "right"},
            fallback="retain",
            on_low_confidence="fallback",
        )
    )
    pipeline = _pipeline()
    pipeline.jobs = {
        "choose": decision,
        "left": _deterministic_branch("left", "left"),
        "right": _deterministic_branch("right", "right"),
    }
    model = _Model(value="retain", confidence=0.1)

    result = asyncio.run(
        OrchidPipelineRunner(
            pipeline,
            deps=PipelineDependencies(decision_models={"triage": model}),
            execution=PipelineExecution(),
        ).run({"account_id": "a1"}, user_id="u1")
    )

    assert result.jobs["choose"]["output"]["label"] == "retain"
    assert result.jobs["left"]["status"] == "succeeded"
    assert "right" not in result.jobs


# ── Registration ──────────────────────────────────────────


def test_decision_kind_is_registered() -> None:
    from orchid_ai.pipeline.registry import PIPELINE_STEP_REGISTRY, resolve_pipeline_step
    from orchid_ai.pipeline.steps import register_builtin_steps

    before = dict(PIPELINE_STEP_REGISTRY)
    try:
        PIPELINE_STEP_REGISTRY.pop("decision", None)
        register_builtin_steps()
        step = resolve_pipeline_step(
            "decision",
            _job(),
            PipelineDependencies(decision_models={"triage": _Model()}),
        )
        assert isinstance(step, DecisionStep)
    finally:
        PIPELINE_STEP_REGISTRY.clear()
        PIPELINE_STEP_REGISTRY.update(before)


# ── Review regressions (P-009 CR round 1) ──────────────────


def test_provider_timeout_uses_the_policy_and_does_not_leak(caplog: pytest.LogCaptureFixture) -> None:
    class _TimingOut(OrchidDecisionModel):
        async def decide(self, state: Any, questions: Any, *, timeout: float | None = None) -> Any:
            raise TimeoutError("provider deadline at https://host/?token=SECRET")

    job = _job(_config(fallback="retain", on_low_confidence="fallback"))
    caplog.set_level(logging.INFO)
    result = _run(DecisionStep(job, models={"triage": _TimingOut()}), _ctx())

    assert result.output["label"] == "retain"
    assert "SECRET" not in str(result.output)
    assert "SECRET" not in caplog.text


def test_llm_fallback_rationale_is_preserved() -> None:
    model = _Model(value="retain", confidence=0.1)
    chat = _Chat(['{"label": "expand", "confidence": 0.8, "rationale": "customer grew 40%"}'])
    job = _job(_config(on_low_confidence="llm"))

    result = _run(DecisionStep(job, models={"triage": model}), _ctx(model=model, chat_model=chat))

    assert result.output["rationale"] == "customer grew 40%"


def test_llm_unknown_label_is_repaired() -> None:
    model = _Model(value="retain", confidence=0.1)
    chat = _Chat(
        [
            '{"label": "not-a-label", "confidence": 0.8}',
            '{"label": "expand", "confidence": 0.8}',
        ]
    )
    job = _job(_config(on_low_confidence="llm"))

    result = _run(DecisionStep(job, models={"triage": model}), _ctx(model=model, chat_model=chat))

    assert result.output["label"] == "expand"
    assert result.usage["llm_calls"] == 2


def test_llm_out_of_range_confidence_is_repaired() -> None:
    model = _Model(value="retain", confidence=0.1)
    chat = _Chat(['{"label": "expand", "confidence": 3}', '{"label": "expand", "confidence": 0.8}'])
    job = _job(_config(on_low_confidence="llm"))

    result = _run(DecisionStep(job, models={"triage": model}), _ctx(model=model, chat_model=chat))

    assert result.output["label"] == "expand"
    assert result.output["confidence"] == 0.8
    assert result.usage["llm_calls"] == 2


def test_llm_fallback_logs_the_path(caplog: pytest.LogCaptureFixture) -> None:
    model = _Model(value="retain", confidence=0.1)
    chat = _Chat(['{"label": "expand", "confidence": 0.8}'])
    job = _job(_config(on_low_confidence="llm"))
    caplog.set_level(logging.INFO)

    _run(DecisionStep(job, models={"triage": model}), _ctx(model=model, chat_model=chat))

    line = caplog.records[-1].getMessage()
    assert "path=llm_fallback" in line
    assert "provider=llm" in line
    assert "outcome=accepted" in line
    assert "a1" not in line


def test_effective_timeout_is_passed_to_the_provider() -> None:
    from datetime import timedelta

    model = _Model()
    ctx = _ctx(model=model)
    ctx.run = replace(ctx.run, deadline=datetime.now(UTC) + timedelta(seconds=0.15))

    _run(DecisionStep(_job(), models={"triage": model}), ctx)

    _state, _questions, timeout = model.calls[0]
    assert timeout is not None
    assert 0.0 < timeout <= 0.2


def test_run_deadline_caps_the_provider_call() -> None:
    from datetime import timedelta

    class _SlowModel(OrchidDecisionModel):
        async def decide(self, state: Any, questions: Any, *, timeout: float | None = None) -> Any:
            await asyncio.sleep(0.3)
            return OrchidDecisionResult(answers={}, provider="stub", model="stub")

    ctx = _ctx()
    ctx.run = replace(ctx.run, deadline=datetime.now(UTC) + timedelta(seconds=0.05))

    with pytest.raises(OrchidPipelineTimeoutError):
        _run(DecisionStep(_job(), models={"triage": _SlowModel()}), ctx)


def test_fallback_prompt_keeps_the_json_instruction() -> None:
    from orchid_ai.pipeline.steps.decision import _fallback_prompt

    prompt = _fallback_prompt(["retain", "expand"], "Q" * 5000, "C" * 5000, 300)

    assert len(prompt) <= 300
    assert "Return JSON only" in prompt


def test_provider_graph_interrupt_is_re_raised() -> None:
    from langgraph.errors import GraphInterrupt

    class _Interrupting(OrchidDecisionModel):
        async def decide(self, state: Any, questions: Any, *, timeout: float | None = None) -> Any:
            raise GraphInterrupt(())

    with pytest.raises(GraphInterrupt):
        _run(DecisionStep(_job(), models={"triage": _Interrupting()}), _ctx())


def test_llm_fallback_graph_interrupt_is_re_raised() -> None:
    from langgraph.errors import GraphInterrupt

    class _Chat:
        async def ainvoke(self, messages: Any) -> Any:
            raise GraphInterrupt(())

    model = _Model(value="retain", confidence=0.1)
    job = _job(_config(on_low_confidence="llm"))

    with pytest.raises(GraphInterrupt):
        _run(DecisionStep(job, models={"triage": model}), _ctx(model=model, chat_model=_Chat()))


def test_builtin_registration_keeps_an_explicit_factory() -> None:
    from orchid_ai.pipeline.registry import PIPELINE_STEP_REGISTRY, register_pipeline_step
    from orchid_ai.pipeline.steps import register_builtin_steps

    def factory(job: Any, deps: Any) -> Any:
        return object()

    before = dict(PIPELINE_STEP_REGISTRY)
    try:
        register_pipeline_step("decision", factory, override=True)
        register_builtin_steps()
        assert PIPELINE_STEP_REGISTRY["decision"] is factory
    finally:
        PIPELINE_STEP_REGISTRY.clear()
        PIPELINE_STEP_REGISTRY.update(before)


def test_llm_repair_sends_the_correction_with_the_previous_reply() -> None:
    class _RecordingChat:
        def __init__(self) -> None:
            self.messages: list[list[dict[str, Any]]] = []

        async def ainvoke(self, messages: Any) -> Any:
            self.messages.append([dict(message) for message in messages])
            if len(self.messages) == 1:
                return _Message("not json")
            return _Message('{"label": "expand", "confidence": 0.8}')

    model = _Model(value="retain", confidence=0.1)
    chat = _RecordingChat()
    job = _job(_config(on_low_confidence="llm"))

    result = _run(DecisionStep(job, models={"triage": model}), _ctx(model=model, chat_model=chat))

    assert result.output["label"] == "expand"
    assert len(chat.messages) == 2
    repair = chat.messages[1]
    assert len(repair) == 3
    assert repair[1]["role"] == "assistant"
    assert "not json" in repair[1]["content"]
    assert "did not validate" in repair[2]["content"]
