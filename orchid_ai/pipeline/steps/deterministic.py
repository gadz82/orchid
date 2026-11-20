"""Deterministic pipeline step: operator code resolved from a dotted path.

A deterministic job is trusted operator code. The compiler resolves ``uses`` at
build time; this step executes it with the step context and the declared
``params``, off the event loop for synchronous callables, and normalizes the
return value into an :class:`OrchidPipelineStepResult`.

Return contract (spec §7.8.1):

* ``None`` → no output, no mutations;
* ``dict`` → the job output;
* ``OrchidPipelineStepResult`` → used as-is;
* a list of :class:`OrchidGraphMutation` → mutations with empty output;
* anything else → :class:`OrchidPipelineStepError`.

A synchronous callable runs in a worker thread (``asyncio.to_thread``) so a
blocking step cannot stall a parallel job. The per-attempt timeout is applied
here, because this is the layer that knows how long the callable may run.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ...core.pipeline import (
    OrchidGraphMutation,
    OrchidPipelineStep,
    OrchidPipelineStepError,
    OrchidPipelineStepResult,
    OrchidPipelineTimeoutError,
)
from ..definition import CompiledJob

__all__ = ["DeterministicStep", "normalize_result"]


# ORCHID-MEM: 01a1ed8b-246a-73cb-a0d8-6723e41eb334 — Deterministic step; resolves operator code at build time, offloads sync calls, enforces timeout, normalizes returns.
class DeterministicStep(OrchidPipelineStep):
    """Execute a ``kind: deterministic`` job's resolved callable."""

    kind = "deterministic"

    def __init__(
        self,
        job: CompiledJob,
        *,
        handler: Callable[..., Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if not isinstance(job, CompiledJob):
            raise TypeError("DeterministicStep requires a CompiledJob")
        self._job = job
        self._handler = handler if handler is not None else job.handler
        self._logger = logger or logging.getLogger(__name__)
        if not callable(self._handler):
            raise OrchidPipelineStepError(
                f"deterministic job {job.id!r} has no callable 'uses' handler",
                job_id=job.id,
                kind=job.kind,
            )

    async def execute(self, ctx: Any) -> OrchidPipelineStepResult:
        """Invoke the callable with ``ctx`` and the job params, then normalize."""
        params = dict(self._job.params)
        timeout = self._job.timeout_seconds
        if timeout is None:
            return await self._invoke(ctx, params)
        try:
            return await asyncio.wait_for(self._invoke(ctx, params), timeout=timeout)
        except TimeoutError as exc:
            raise OrchidPipelineTimeoutError(
                f"deterministic job {self._job.id!r} timed out after {timeout} seconds"
            ) from exc

    async def _invoke(self, ctx: Any, params: Mapping[str, Any]) -> OrchidPipelineStepResult:
        handler = self._handler
        if inspect.iscoroutinefunction(handler):
            result = await handler(ctx, **params)
        else:
            result = await asyncio.to_thread(handler, ctx, **params)
            if inspect.isawaitable(result):
                # A callable that returns an awaitable without being declared
                # ``async`` still has to be awaited on the event loop.
                result = await result
        return normalize_result(result, job_id=self._job.id, kind=self._job.kind)


def normalize_result(result: Any, *, job_id: str, kind: str) -> OrchidPipelineStepResult:
    """Turn a deterministic return value into a step result or fail clearly."""
    if result is None:
        return OrchidPipelineStepResult()
    if isinstance(result, OrchidPipelineStepResult):
        return result
    if isinstance(result, Mapping):
        return OrchidPipelineStepResult(output=dict(result))
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
        mutations = tuple(result)
        if mutations and all(isinstance(mutation, OrchidGraphMutation) for mutation in mutations):
            return OrchidPipelineStepResult(output={}, mutations=mutations)
        if not mutations:
            return OrchidPipelineStepResult()
    raise OrchidPipelineStepError(
        "deterministic step must return None, a dict, an OrchidPipelineStepResult "
        f"or a list of OrchidGraphMutation values; got {type(result).__name__}",
        job_id=job_id,
        kind=kind,
    )
