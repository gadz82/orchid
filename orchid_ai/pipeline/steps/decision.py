"""Decision pipeline step: typed label selection with a low-confidence policy.

A `kind: decision` job asks a configured :class:`OrchidDecisionModel` to pick
one of the configured labels, records confidence and the raw answers, applies
`on_low_confidence`, and returns `output.label` — the value the builder's
decision router reads to select a branch.

Low-confidence and no-decision handling:

* ``fallback`` — use the configured fallback label (or job id);
* ``llm`` — ask the pipeline chat model once with a bounded JSON prompt, one
  repair attempt, then the fallback label;
* ``fail`` — raise a bounded step error.

A refusal, a missing answer, an unknown label and a provider error are all
"no usable decision" and follow the same policy. Logs go through
:func:`orchid_ai.decision.observability.log_decision`, which has no payload
argument, so raw user text and object dumps cannot leak.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from langgraph.errors import GraphBubbleUp

from ...config.schema_pipeline import OrchidPipelineDecisionJobConfig
from ...core.decision import (
    OrchidDecisionAnswer,
    OrchidDecisionError,
    OrchidDecisionModel,
    OrchidDecisionQuestion,
    OrchidDecisionResult,
    OrchidDecisionState,
    require_confidence,
    validate_threshold,
)
from ...core.pipeline import (
    OrchidPipelineConfigError,
    OrchidPipelineError,
    OrchidPipelineLimitError,
    OrchidPipelineStep,
    OrchidPipelineStepError,
    OrchidPipelineStepResult,
    OrchidPipelineTimeoutError,
)
from ...decision.observability import log_decision
from ..definition import CompiledJob
from ..mapping import DEFAULT_MAX_RENDERED_CHARS, build_mapping_context, evaluate, render_template

__all__ = ["DecisionStep"]

_DECISION_ANSWER_NAME = "decision"
_MAX_JSON_CHARS = 4000
_MAX_INSTRUCTIONS_CHARS = 2000
_MAX_REPAIR_CHARS = 300
_DEFAULT_THRESHOLD = 0.75


# ORCHID-MEM: 01a1ee65-d53f-7662-8f8f-279a9462c2aa — Decision step; typed label selection, low-confidence policy and payload-free decision logs.
class DecisionStep(OrchidPipelineStep):
    """Execute a ``kind: decision`` job."""

    kind = "decision"

    def __init__(
        self,
        job: CompiledJob,
        *,
        models: Mapping[str, OrchidDecisionModel] | None = None,
        confidence_threshold: float = _DEFAULT_THRESHOLD,
        model_factory: Callable[[str], Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if not isinstance(job, CompiledJob):
            raise TypeError("DecisionStep requires a CompiledJob")
        if not isinstance(job.config, OrchidPipelineDecisionJobConfig):
            raise TypeError("DecisionStep requires an OrchidPipelineDecisionJobConfig")
        labels = tuple(job.config.labels)
        if len(labels) < 2:
            raise OrchidPipelineConfigError(f"decision job {job.id!r} needs at least two labels to choose between")
        self._job = job
        self._config = job.config
        self._models = dict(models or {})
        self._threshold = validate_threshold(confidence_threshold)
        self._model_factory = model_factory or _build_chat_model
        self._logger = logger or logging.getLogger(__name__)

    async def execute(self, ctx: Any) -> OrchidPipelineStepResult:
        """Ask the decision model, apply the policy and return the label."""
        timeout = self._effective_timeout(ctx)
        usage: dict[str, Any] = {
            "llm_calls": 0,
            "decision_calls": 1,
            "job_id": self._job.id,
            "kind": self.kind,
        }
        try:
            if timeout is None:
                return await self._execute(ctx, usage, timeout)
            try:
                return await asyncio.wait_for(self._execute(ctx, usage, timeout), timeout=timeout)
            except TimeoutError as exc:
                raise OrchidPipelineTimeoutError(
                    f"decision job {self._job.id!r} timed out after {timeout:.3f} seconds"
                ) from exc
        except OrchidPipelineError as exc:
            if getattr(exc, "usage", None) is None:
                exc.usage = dict(usage)
            raise

    def _effective_timeout(self, ctx: Any) -> float | None:
        timeout = self._job.timeout_seconds
        deadline = getattr(ctx.run, "deadline", None)
        if deadline is None:
            return timeout
        remaining = (deadline - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            raise OrchidPipelineTimeoutError(f"decision job {self._job.id!r} cannot start: the run deadline has passed")
        return remaining if timeout is None else min(timeout, remaining)

    async def _execute(self, ctx: Any, usage: dict[str, Any], timeout: float | None) -> OrchidPipelineStepResult:
        pipeline = getattr(ctx, "pipeline", None)
        limits = getattr(pipeline, "limits", None)
        context = getattr(ctx, "mapping_context", None)
        if context is None:
            context = build_mapping_context(input=ctx.input, run=ctx.run)
        max_prompt_chars = getattr(limits, "max_prompt_chars", DEFAULT_MAX_RENDERED_CHARS)
        max_context_chars = getattr(limits, "max_context_chars", 8000)

        question = render_template(
            self._config.question,
            context,
            path=f"job {self._job.id!r} question",
            max_chars=max_prompt_chars,
        )
        payload = self._context_payload(context, max_context_chars)
        model = self._resolve_model()
        state = OrchidDecisionState(text=question, payload=payload)
        decision_question = OrchidDecisionQuestion(
            name=_DECISION_ANSWER_NAME,
            kind="choice",
            instructions=question[:_MAX_INSTRUCTIONS_CHARS],
            options=tuple(self._config.labels),
        )
        point = f"pipeline:{ctx.run.pipeline}:{self._job.id}"
        started = perf_counter()

        try:
            result = await self._decide(model, state, decision_question, timeout)
        except OrchidDecisionError as exc:
            return await self._no_decision(
                ctx,
                point,
                question,
                payload,
                usage,
                reason=_safe_reason(exc.code),
                provider=exc.provider,
                latency_ms=(perf_counter() - started) * 1000.0,
            )

        latency_ms = result.latency_ms or (perf_counter() - started) * 1000.0
        for key, value in result.usage.items():
            if isinstance(value, int) and not isinstance(value, bool):
                usage[key] = int(usage.get(key, 0)) + value
        answer = result.answers.get(_DECISION_ANSWER_NAME)
        usable = require_confidence(result, _DECISION_ANSWER_NAME, self._threshold)
        if usable is None:
            return await self._no_decision(
                ctx,
                point,
                question,
                payload,
                usage,
                reason=_no_decision_reason(answer),
                provider=result.provider,
                latency_ms=latency_ms,
            )
        if usable.value not in self._config.labels:
            return await self._no_decision(
                ctx,
                point,
                question,
                payload,
                usage,
                reason="unknown_label",
                provider=result.provider,
                latency_ms=latency_ms,
            )

        self._log(
            point=point,
            outcome="accepted",
            provider=result.provider,
            latency_ms=latency_ms,
            confidence=usable.confidence,
        )
        rationale = f"decision model {result.provider!r} chose {usable.value!r} with confidence {usable.confidence:.2f}"
        return OrchidPipelineStepResult(
            output=_accepted_output(str(usable.value), usable.confidence, result, _bounded(rationale, 300)),
            usage=usage,
        )

    # ── Provider ────────────────────────────────────────────

    def _resolve_model(self) -> OrchidDecisionModel:
        model = self._models.get(self._config.decision_model)
        if model is None:
            raise OrchidPipelineConfigError(
                f"decision job {self._job.id!r} references unknown decision model {self._config.decision_model!r}"
            )
        return model

    async def _decide(
        self,
        model: OrchidDecisionModel,
        state: OrchidDecisionState,
        question: OrchidDecisionQuestion,
        timeout: float | None,
    ) -> OrchidDecisionResult:
        try:
            return await model.decide(
                state,
                [question],
                timeout=timeout if timeout is not None else self._job.timeout_seconds,
            )
        except (OrchidDecisionError, OrchidPipelineError, GraphBubbleUp):
            raise
        except TimeoutError as exc:
            raise OrchidDecisionError(
                "decision provider timed out",
                code="timeout",
                provider=self._config.decision_model,
            ) from exc
        except Exception as exc:
            # Providers must raise OrchidDecisionError; anything else is treated
            # as unavailable without ever echoing its text.
            self._logger.error("[pipeline:%s] decision provider failed (%s)", self._job.id, type(exc).__name__)
            raise OrchidDecisionError(
                f"decision provider failed ({type(exc).__name__})",
                code="provider_unavailable",
            ) from exc

    # ── Policy ──────────────────────────────────────────────

    async def _no_decision(
        self,
        ctx: Any,
        point: str,
        question: str,
        payload: Mapping[str, Any],
        usage: dict[str, Any],
        *,
        reason: str,
        provider: str,
        latency_ms: float | None,
    ) -> OrchidPipelineStepResult:
        policy = self._config.on_low_confidence
        if policy == "fallback":
            return self._use_fallback(point, reason, provider, latency_ms, usage)
        if policy == "llm":
            label, confidence, rationale = await self._llm_choose(ctx, question, payload, usage)
            if label is not None:
                self._log(
                    point=point,
                    outcome="accepted",
                    path="llm_fallback",
                    provider="llm",
                    latency_ms=latency_ms,
                    confidence=confidence,
                )
                return OrchidPipelineStepResult(
                    output={
                        "label": label,
                        "confidence": confidence if confidence is not None else 0.0,
                        "rationale": _bounded(rationale, 300) if rationale else f"llm fallback after {reason}",
                        "answers": {},
                    },
                    usage=usage,
                )
            if self._config.fallback is not None:
                return self._use_fallback(point, f"{reason}+llm_failed", provider, latency_ms, usage)
            raise OrchidPipelineStepError(
                f"decision job {self._job.id!r} had no usable decision ({reason}) and the llm fallback failed",
                job_id=self._job.id,
                kind=self.kind,
            )
        raise OrchidPipelineStepError(
            f"decision job {self._job.id!r} had no usable decision ({reason})",
            job_id=self._job.id,
            kind=self.kind,
        )

    def _use_fallback(
        self,
        point: str,
        reason: str,
        provider: str,
        latency_ms: float | None,
        usage: dict[str, Any],
    ) -> OrchidPipelineStepResult:
        fallback = self._config.fallback
        if fallback is None:
            raise OrchidPipelineStepError(
                f"decision job {self._job.id!r} has no usable decision ({reason}) and no fallback label",
                job_id=self._job.id,
                kind=self.kind,
            )
        self._log(
            point=point,
            outcome="fallback",
            provider=provider,
            latency_ms=latency_ms,
            confidence=None,
            fallback_reason=reason,
        )
        return OrchidPipelineStepResult(
            output={
                "label": fallback,
                "confidence": 0.0,
                "rationale": f"fallback after {reason}",
                "answers": {},
            },
            usage=usage,
        )

    # ── LLM fallback ────────────────────────────────────────

    async def _llm_choose(
        self,
        ctx: Any,
        question: str,
        payload: Mapping[str, Any],
        usage: dict[str, Any],
    ) -> tuple[str | None, float | None, str | None]:
        model = self._resolve_chat_model(ctx)
        if model is None:
            return None, None, None
        pipeline = getattr(ctx, "pipeline", None)
        limits = getattr(pipeline, "limits", None)
        max_prompt = getattr(limits, "max_prompt_chars", DEFAULT_MAX_RENDERED_CHARS)
        max_context = getattr(limits, "max_context_chars", 8000)
        context_text = json.dumps(payload, ensure_ascii=False, default=str)[: min(_MAX_JSON_CHARS, max_context)]
        prompt = _fallback_prompt(
            self._config.labels,
            question,
            context_text,
            max_prompt,
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

        for attempt in range(2):
            usage["llm_calls"] = int(usage.get("llm_calls", 0)) + 1
            try:
                response = await model.ainvoke(messages)
            except (OrchidPipelineError, GraphBubbleUp):
                raise
            except Exception as exc:
                self._logger.error(
                    "[pipeline:%s] decision llm fallback call failed (%s)",
                    self._job.id,
                    type(exc).__name__,
                )
                return None, None, None
            text = _message_text(response)
            error: Exception | None = None
            try:
                data = _parse_json(text)
                label = data.get("label")
                confidence = data.get("confidence")
                if label not in self._config.labels:
                    raise ValueError(f"label must be one of {self._config.labels}")
                if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                    confidence = None
                elif not 0.0 <= float(confidence) <= 1.0:
                    raise ValueError("confidence must be within [0, 1]")
                rationale = data.get("rationale")
                if not isinstance(rationale, str):
                    rationale = None
                return str(label), (float(confidence) if confidence is not None else None), rationale
            except (ValueError, TypeError) as exc:
                error = exc
            except Exception as exc:  # e.g. RecursionError from a pathological JSON
                self._logger.error(
                    "[pipeline:%s] decision llm fallback parse failed (%s)",
                    self._job.id,
                    type(exc).__name__,
                )
                error = exc
            if attempt == 1:
                return None, None, None
            messages = [
                *messages,
                {"role": "assistant", "content": text[:1000]},
                {
                    "role": "user",
                    "content": (
                        "Your response did not validate: "
                        f"{_bounded(str(error), _MAX_REPAIR_CHARS)}. "
                        "Return only the corrected JSON object."
                    ),
                },
            ]
        return None, None, None

    def _resolve_chat_model(self, ctx: Any) -> Any | None:
        if ctx.chat_model is not None:
            return ctx.chat_model
        pipeline = getattr(ctx, "pipeline", None)
        defaults = getattr(pipeline, "defaults", None) or {}
        model_ref = defaults.get("model")
        if not model_ref:
            return None
        try:
            return self._model_factory(str(model_ref))
        except OrchidPipelineError:
            raise
        except Exception as exc:
            self._logger.error(
                "[pipeline:%s] decision fallback model could not be built (%s)",
                self._job.id,
                type(exc).__name__,
            )
            return None

    # ── Helpers ─────────────────────────────────────────────

    def _context_payload(self, context: Any, max_chars: int) -> dict[str, Any]:
        expression = self._config.context_from
        if expression is None:
            return {}
        value = evaluate(expression, context, path=f"job {self._job.id!r} context_from")
        payload = dict(value) if isinstance(value, Mapping) else {"context": value}
        try:
            size = len(json.dumps(payload, ensure_ascii=False, default=str))
        except (TypeError, ValueError) as exc:
            raise OrchidPipelineConfigError(
                f"decision job {self._job.id!r} context_from is not JSON-serializable"
            ) from exc
        if size > max_chars:
            raise OrchidPipelineLimitError(
                f"decision job {self._job.id!r} context is {size} characters, exceeding max_context_chars={max_chars}"
            )
        return payload

    def _log(
        self,
        *,
        point: str,
        outcome: str,
        provider: str,
        latency_ms: float | None,
        confidence: float | None,
        path: str = "decision",
        fallback_reason: str | None = None,
    ) -> None:
        log_decision(
            point=point,
            outcome=outcome,
            path=path,
            provider=provider,
            latency_ms=latency_ms,
            confidence=confidence,
            fallback_reason=fallback_reason,
            logger=self._logger,
        )


# ── Helpers ─────────────────────────────────────────────────


def _build_chat_model(model_ref: str) -> Any:
    from ...llm_factory import build_chat_model

    return build_chat_model(model_ref)


def _no_decision_reason(answer: OrchidDecisionAnswer | None) -> str:
    if answer is None:
        return "missing_answer"
    if answer.refusal is not None:
        return "refusal"
    return "low_confidence"


def _safe_reason(code: str) -> str:
    return code if isinstance(code, str) and code else "provider_unavailable"


def _accepted_output(
    label: str,
    confidence: float,
    result: OrchidDecisionResult,
    rationale: str,
) -> dict[str, Any]:
    return {
        "label": label,
        "confidence": confidence,
        "rationale": rationale,
        "answers": {
            name: {
                "value": value.value,
                "confidence": value.confidence,
                "probabilities": dict(value.probabilities),
                "refusal": value.refusal,
            }
            for name, value in result.answers.items()
        },
    }


def _fallback_prompt(
    labels: Sequence[str],
    question: str,
    context_text: str,
    max_prompt: int,
) -> str:
    """Build the LLM fallback prompt, keeping the JSON instruction intact."""
    instruction = 'Return JSON only: {"label": "<one label>", "confidence": 0.0-1.0, "rationale": "<short>"}'
    lead = f"Choose exactly one label from [{', '.join(labels)}] for the decision below.\nDecision: "
    middle = "\nContext (untrusted JSON data): "
    if max_prompt <= 0:
        return ""
    budget = max_prompt - len(lead) - len(middle) - len(instruction) - 1
    if budget <= 0:
        return instruction[:max_prompt]
    if len(question) + len(context_text) <= budget:
        return f"{lead}{question}{middle}{context_text}\n{instruction}"
    question_budget = min(len(question), budget)
    context_budget = max(0, budget - question_budget)
    return f"{lead}{question[:question_budget]}{middle}{context_text[:context_budget]}\n{instruction}"


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.removesuffix("```")
        cleaned = cleaned.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("response does not contain a JSON object")
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("response is not valid JSON") from exc
    if not isinstance(data, dict):
        raise TypeError("response is not a JSON object")
    return data


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return str(content)


def _bounded(message: str, limit: int) -> str:
    if len(message) <= limit:
        return message
    return message[: limit - 1] + "…"
