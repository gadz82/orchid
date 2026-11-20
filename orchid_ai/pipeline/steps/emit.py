"""Emit pipeline step: turn a run result into one identity-only Pollen signal.

An ``kind: emit`` job renders its declared payload and dedupe key from the
documented JMESPath context, builds a :class:`SignalEnvelope` with the pipeline
source, run correlation id, tenant/user identifiers and an identity claim, and
hands it to the injected emitter. It never persists anything itself and never
puts the bearer token in the envelope.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

from ...config.schema_pipeline import OrchidPipelineEmitJobConfig
from ...core.events.signal import SignalEnvelope
from ...core.pipeline import (
    OrchidPipelineEventsDisabledError,
    OrchidPipelineSerializationError,
    OrchidPipelineStep,
    OrchidPipelineStepError,
    OrchidPipelineStepResult,
    OrchidPipelineTimeoutError,
)
from ..definition import CompiledJob
from ..mapping import build_mapping_context, render_template, render_value

__all__ = ["EmitStep"]

_MAX_DEDUPE_KEY_CHARS = 1024
_MAX_ERROR_CHARS = 500


# ORCHID-MEM: 01a1f2c5-f4cf-744a-b1f3-9d8c6b829d13 — Emit step; renders an identity-only SignalEnvelope and calls the injected emitter, never persisting or leaking the token.
class EmitStep(OrchidPipelineStep):
    """Execute a ``kind: emit`` job."""

    kind = "emit"

    def __init__(self, job: CompiledJob) -> None:
        if not isinstance(job, CompiledJob):
            raise TypeError("EmitStep requires a CompiledJob")
        if not isinstance(job.config, OrchidPipelineEmitJobConfig):
            raise TypeError("EmitStep requires an OrchidPipelineEmitJobConfig")
        self._job = job
        self._config = job.config

    async def execute(self, ctx: Any) -> OrchidPipelineStepResult:
        """Render the declared payload, build the envelope and emit it."""
        context = getattr(ctx, "mapping_context", None)
        if context is None:
            context = build_mapping_context(input=ctx.input, run=ctx.run)
        payload = render_value(
            self._config.payload,
            context,
            path=f"job {self._job.id!r} payload",
            strict=True,
        )
        if not isinstance(payload, dict):
            raise OrchidPipelineStepError(
                f"emit job {self._job.id!r} payload must render to an object",
                job_id=self._job.id,
                kind=self.kind,
            )
        _require_json_payload(payload)

        dedupe_key: str | None = None
        if self._config.dedupe_key is not None:
            dedupe_key = render_template(
                self._config.dedupe_key,
                context,
                path=f"job {self._job.id!r} dedupe_key",
                max_chars=_MAX_DEDUPE_KEY_CHARS,
            )
            if not dedupe_key.strip():
                raise OrchidPipelineStepError(
                    f"emit job {self._job.id!r} dedupe_key rendered empty",
                    job_id=self._job.id,
                    kind=self.kind,
                )

        auth = ctx.auth
        tenant_key = getattr(auth, "tenant_key", "default") if auth is not None else "default"
        user_id = getattr(auth, "user_id", None) if auth is not None else None
        identity = (
            self._config.identity.model_dump()
            if self._config.identity is not None
            else {"mode": "act_as_user", "user_id": user_id or ""}
        )
        chat_binding = self._config.chat_binding.model_dump() if self._config.chat_binding is not None else None
        envelope = SignalEnvelope(
            type=self._config.signal,
            payload=payload,
            source=f"pipeline:{ctx.run.pipeline}",
            occurred_at=datetime.now(UTC),
            tenant_key=tenant_key,
            user_id=user_id,
            correlation_id=ctx.run.run_id,
            dedupe_key=dedupe_key,
            identity_claim=identity,
            chat_binding=chat_binding,
        )
        timeout = self._effective_timeout(ctx)
        try:
            if timeout is None:
                result = await ctx.emit_signal(envelope)
            else:
                try:
                    result = await asyncio.wait_for(ctx.emit_signal(envelope), timeout=timeout)
                except TimeoutError as exc:
                    raise OrchidPipelineTimeoutError(
                        f"emit job {self._job.id!r} timed out after {timeout:.3f} seconds"
                    ) from exc
        except OrchidPipelineEventsDisabledError:
            raise
        except OrchidPipelineTimeoutError:
            raise
        except Exception as exc:
            raise OrchidPipelineStepError(
                f"emit job {self._job.id!r} failed to emit signal {self._config.signal!r} ({type(exc).__name__})",
                job_id=self._job.id,
                kind=self.kind,
            ) from exc

        return OrchidPipelineStepResult(
            output={
                "signal": self._config.signal,
                "signal_id": str(result.signal_id),
                "deduplicated": bool(getattr(result, "deduplicated", False)),
                "queue_msg_id": getattr(result, "queue_msg_id", None),
                "source": envelope.source,
                "dedupe_key": dedupe_key,
            },
            usage={"llm_calls": 0},
        )

    def _effective_timeout(self, ctx: Any) -> float | None:
        timeout = self._job.timeout_seconds
        deadline = getattr(ctx.run, "deadline", None)
        if deadline is None:
            return timeout
        remaining = (deadline - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            raise OrchidPipelineTimeoutError(f"emit job {self._job.id!r} cannot start: the run deadline has passed")
        return remaining if timeout is None else min(timeout, remaining)


def _require_json_payload(payload: dict[str, Any]) -> None:
    try:
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        detail = str(exc)[:_MAX_ERROR_CHARS]
        raise OrchidPipelineSerializationError(f"emit payload is not JSON-serializable: {detail}") from exc
