"""Graph-shaped job runner — final form (Phase 3).

The runner is what turns a :class:`JobRun` into a real LangGraph
invocation.  Phase 1 shipped a stubbed shape that delegated to an
injected callable; Phase 3 keeps the same injection point (so unit
tests don't have to spin up the supervisor) AND adds:

- §25 chat-binding resolution via :meth:`_resolve_chat_binding`,
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
from typing import Any, Awaitable, Callable

from ...core.events.errors import (
    ChatBindingForbiddenError,
    ChatBindingTargetNotFoundError,
    JobRunnerError,
)
from ...core.events.job import JobRun, JobStatus
from ...core.events.runner import OrchidJobRunner

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

    async def run(self, run: JobRun, *, auth: Any) -> None:
        # The processor has already moved the run to RUNNING and
        # persisted ``started_at`` — this method's job is to invoke
        # the graph, persist any chat-binding side effects, and
        # translate failures into the run's terminal state.

        # ── §25 chat-binding pre-flight ──────────────────────────
        # Resolved here BEFORE the graph runs so a failed-resolution
        # short-circuits without wasting the supervisor's time.
        try:
            binding = await self._resolve_chat_binding(run, auth)
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
            await self._maybe_post_failure(binding, run, exc)
            raise
        except Exception as exc:
            _logger.exception(
                "graph invocation failed for run %s (trigger=%s)",
                run.run_id,
                run.spec.trigger_id,
            )
            await self._maybe_post_failure(binding, run, exc)
            raise JobRunnerError(repr(exc), retryable=False) from exc

        run.result = result
        run.status = JobStatus.SUCCEEDED
        run.finished_at = self._clock()

        # On success: persist the final AIMessage to the bound chat.
        if binding is not None:
            await self._persist_final_message(binding, run, result)

    # ── Chat-binding helpers ─────────────────────────────────

    async def _resolve_chat_binding(self, run: JobRun, auth: Any) -> dict[str, Any] | None:
        """Per §25.4 — return the binding dict iff:

        (a) the trigger opted in via ``respect_chat_binding=true`` AND
        (b) the signal carried a ``chat_binding`` AND
        (c) the chat target exists AND
        (d) the resolved auth has write permission on it.

        When ``proactive_chat=true`` and no explicit binding is present,
        creates a new chat for the resolved user and returns a synthetic
        binding pointing at it.

        Raises :class:`ChatBindingTargetNotFoundError` / forbidden
        for (c)/(d) failures.  Returns ``None`` for the no-binding
        path (which is the default — chat invisibility is the spec's
        safe default).
        """
        binding = run.spec.chat_binding
        if binding is None:
            if run.spec.proactive_chat:
                return await self._create_proactive_chat(run, auth)
            return None
        if self._chat_storage is None:
            # Trigger asked for binding but framework wasn't given a
            # chat storage to write into — log + drop.  Visible only
            # in tests / mis-wired demos; Phase 4's lifespan ensures
            # the storage is always present in production.
            _logger.warning(
                "trigger %s opted into chat binding but no chat_storage is wired into the runner — dropping binding",
                run.spec.trigger_id,
            )
            return None

        chat_id = binding.get("chat_id")
        if not isinstance(chat_id, str) or not chat_id:
            raise ChatBindingTargetNotFoundError(str(chat_id))

        chat = await self._chat_storage.get_chat_metadata(chat_id)
        if chat is None:
            raise ChatBindingTargetNotFoundError(chat_id)

        # ``can_write`` is concrete on OrchidChatStorage with a
        # tenant-and-owner default; consumers can override.
        allowed = await self._chat_storage.can_write(chat, auth)
        if not allowed:
            raise ChatBindingForbiddenError(chat_id, getattr(auth, "user_id", ""))
        return binding

    async def _create_proactive_chat(self, run: JobRun, auth: Any) -> dict[str, Any] | None:
        """Create a new chat owned by the resolved user and return a
        synthetic binding pointing at it.

        Uses the first non-empty line of the rendered prompt as the chat
        title (truncated to 120 chars) — readable in the chat list without
        any extra config.  Returns ``None`` and logs a warning when
        ``chat_storage`` is not wired (mis-configured demo / test).
        """
        if self._chat_storage is None:
            _logger.warning(
                "trigger %s has proactive_chat=true but no chat_storage is wired into the runner — skipping",
                run.spec.trigger_id,
            )
            return None

        tenant_id = getattr(auth, "tenant_key", "default")
        user_id = getattr(auth, "user_id", "")
        title = next(
            (line.strip() for line in run.spec.prompt.splitlines() if line.strip()),
            run.spec.trigger_id,
        )[:120]

        session = await self._chat_storage.create_chat(
            tenant_id=tenant_id,
            user_id=user_id,
            title=title,
        )
        _logger.info(
            "[GraphJobRunner] proactive_chat: created chat %s for user %s (trigger=%s)",
            session.id,
            user_id,
            run.spec.trigger_id,
        )
        return {"chat_id": session.id}

    async def _persist_final_message(
        self,
        binding: dict[str, Any],
        run: JobRun,
        result: dict[str, Any],
    ) -> None:
        if self._chat_storage is None:  # pragma: no cover — guarded earlier
            return
        chat_id = binding["chat_id"]
        content = self._extract_final_content(result)
        metadata: dict[str, Any] = {
            "origin": "bloom",
            "bloom_run_id": str(run.run_id),
            "trigger_id": run.spec.trigger_id,
            "signal_id": str(run.spec.signal_id),
            "delivered_at": self._clock().isoformat(),
        }
        if binding.get("mode") == "append_with_metadata":
            metadata["bloom_metadata"] = {
                "trigger_id": run.spec.trigger_id,
                "signal_id": str(run.spec.signal_id),
                "agent": run.spec.agent_name,
            }
        try:
            await self._chat_storage.add_message(
                chat_id=chat_id,
                role="assistant",
                content=content,
                agents_used=[run.spec.agent_name],
                metadata=metadata,
            )
        except Exception:
            # Bloom result is already persisted to ``job_runs.result``
            # — losing the chat write is bad but not catastrophic.
            # Logged for operator visibility; the run itself remains
            # SUCCEEDED.
            _logger.exception(
                "chat-binding append failed for run %s — result is still available via /runs/%s",
                run.run_id,
                run.run_id,
            )

    async def _maybe_post_failure(
        self,
        binding: dict[str, Any] | None,
        run: JobRun,
        exc: BaseException,
    ) -> None:
        """Per §25 ``on_failure``: post a brief error message when
        ``post_error``, do nothing when ``silent``."""
        if binding is None or self._chat_storage is None:
            return
        if binding.get("on_failure", "post_error") != "post_error":
            return
        try:
            await self._chat_storage.add_message(
                chat_id=binding["chat_id"],
                role="assistant",
                content=self._render_failure_message(run, exc),
                agents_used=[run.spec.agent_name],
                metadata={
                    "origin": "bloom",
                    "bloom_run_id": str(run.run_id),
                    "trigger_id": run.spec.trigger_id,
                    "signal_id": str(run.spec.signal_id),
                    "status": "failed",
                    "delivered_at": self._clock().isoformat(),
                },
            )
        except Exception:
            _logger.exception(
                "failure-path chat-binding append failed for run %s",
                run.run_id,
            )

    # ── Result extraction ────────────────────────────────────

    @staticmethod
    def _extract_final_content(result: dict[str, Any]) -> str:
        """Pull a sensible string out of the runner result dict.

        Order of preference:

        1. ``result["final_response"]`` (the supervisor convention).
        2. ``result["content"]``.
        3. ``str(result)`` as a last-resort.

        Phase 1's invoker stub returns whatever the test fixture
        emits, so this helper has to be permissive.
        """
        if not result:
            return ""
        for key in ("final_response", "content", "message"):
            value = result.get(key)
            if isinstance(value, str) and value:
                return value
        return str(result)

    @staticmethod
    def _render_failure_message(run: JobRun, exc: BaseException) -> str:
        # Short, non-technical — frontend will render the bloom badge
        # alongside; users get a link to ``/runs/{run_id}`` for
        # detail.
        return (
            "I wasn't able to finish that background task. "
            f"You can inspect the run for details: run id "
            f"{run.run_id} (trigger {run.spec.trigger_id})."
        )
