"""Graph-shaped job runner — final form (Phase 3).

The runner is what turns a :class:`JobRun` into a real LangGraph
invocation.  Phase 1 shipped a stubbed shape that delegated to an
injected callable; Phase 3 keeps the same injection point (so unit
tests don't have to spin up the supervisor) AND adds:

- §25 chat-binding resolution through :class:`ChatBindingDeliverer`,
  rejecting cross-user attempts at runtime regardless of what the
  signal carried.
- Final-``AIMessage`` persistence to :class:`OrchidChatStorage` on
  success when a binding resolved.
- Optional failure-message persistence when ``binding.on_failure ==
  "post_error"``.  Per §25.4 we never post when the binding lookup
  itself failed (target missing OR forbidden) — the runner cannot
  trust a chat target it just rejected.

The runner is composed by the framework, not constructed by
consumers directly.  The graph builder injects the compiled graph
behind ``invoker`` along with the chat-storage backend (when present)
when ``events.enabled: true``.
"""

from __future__ import annotations

import datetime as _dt
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from ...core.events.errors import (
    ChatBindingForbiddenError,
    ChatBindingTargetNotFoundError,
    JobRunnerError,
)
from ...core.events.job import JobRun, JobStatus
from ...core.events.runner import OrchidJobRunner
from .chat_binding import ChatBindingDeliverer

_logger = logging.getLogger(__name__)

# A graph-shaped callable: ``(run, auth) -> dict``.  Phase 1 + the
# unit tests inject a fake; the real wiring (a closure over the
# compiled LangGraph plus the session warmer) lands in the API
# lifespan in Phase 4.
GraphInvoker = Callable[[JobRun, Any], Awaitable[dict[str, Any]]]


def _now() -> _dt.datetime:
    return _dt.datetime.now(tz=_dt.UTC)


class GraphJobRunner(OrchidJobRunner):
    """Invokes a graph-shaped callable to satisfy a :class:`JobRun`.

    Constructor knobs:

    - ``invoker`` — required.  Async callable producing the result
      dict.  In production this is a closure over the compiled
      LangGraph; in tests it's a fake that returns a fixed payload.
    - ``chat_storage`` — optional.  When supplied AND the matched
      trigger has ``respect_chat_binding=true`` AND the signal
      carried a ``chat_binding`` AND the resolved auth can write to
      the target chat, the final ``AIMessage`` is persisted there.
    - ``clock`` — injected for tests that want to pin
      ``finished_at``.
    """

    def __init__(
        self,
        *,
        invoker: GraphInvoker,
        chat_storage: Any | None = None,
        clock: Callable[[], _dt.datetime] | None = None,
    ) -> None:
        self._invoker = invoker
        self._chat_storage = chat_storage
        self._clock = clock or _now
        self._deliverer = ChatBindingDeliverer(chat_storage=chat_storage, clock=self._clock)

    async def run(self, run: JobRun, *, auth: Any) -> None:
        # The processor has already moved the run to RUNNING and
        # persisted ``started_at`` — this method's job is to invoke
        # the graph, persist any chat-binding side effects, and
        # translate failures into the run's terminal state.

        # ── §25 chat-binding pre-flight ──────────────────────────
        # Resolved here BEFORE the graph runs so a failed-resolution
        # short-circuits without wasting the supervisor's time.
        try:
            binding = await self._deliverer.resolve_binding(run, auth)
        except (ChatBindingTargetNotFoundError, ChatBindingForbiddenError) as exc:
            _logger.warning(
                "chat-binding rejected for run %s (trigger=%s): %s",
                run.run_id,
                run.spec.trigger_id,
                exc,
            )
            # Non-retryable: the binding won't resolve next attempt
            # either.  Per §25.4 we do NOT post the on_failure error
            # message here — the runner cannot trust a chat target it
            # just rejected.
            run.status = JobStatus.FAILED
            run.error = repr(exc)
            run.finished_at = self._clock()
            raise JobRunnerError(repr(exc), retryable=False) from exc

        try:
            result = await self._invoker(run, auth)
        except JobRunnerError as exc:
            # Bubble up so the processor can read ``.retryable``.
            await self._deliverer.deliver_failure(binding, run, exc)
            raise
        except Exception as exc:
            _logger.exception(
                "graph invocation failed for run %s (trigger=%s)",
                run.run_id,
                run.spec.trigger_id,
            )
            await self._deliverer.deliver_failure(binding, run, exc)
            raise JobRunnerError(repr(exc), retryable=False) from exc

        run.result = result
        run.status = JobStatus.SUCCEEDED
        run.finished_at = self._clock()

        # On success: persist the final AIMessage to the bound chat.
        if binding is not None:
            await self._deliverer.deliver_success(binding, run, result)
