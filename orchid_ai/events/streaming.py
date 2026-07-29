"""``BloomEventStream`` — fan-out of ``bloom.*`` events to SSE listeners.

Tiny in-process pub-sub keyed by **channel string**.  Each
:class:`BloomEventStream.publish` puts an event onto every active
subscriber's queue for the same channel; :meth:`subscribe` returns
an async iterator that yields events until the consumer
disconnects, the queue's ``idle_timeout_seconds`` elapses with no
events, or a *terminal* event arrives.

Two well-known channel families are used in v1:

- ``"run:{run_id}"`` — per-run events, consumed by
  ``GET /runs/{run_id}/stream``.  Backwards-compat wrappers
  :meth:`publish_run` and :meth:`subscribe_run` target this family.
- ``"chat:{chat_id}"`` — per-chat events, consumed by the new
  ``GET /chats/{chat_id}/events/stream`` endpoint (Phase 4.5 §LS6).
  The processor dual-publishes :class:`BloomEvent`s here as
  :class:`ChatBloomEvent`s when the underlying run carries a chat
  binding.

The streaming surface is opt-in.  When events are disabled (the
default), nothing constructs a :class:`BloomEventStream`, the
processor's ``publish_event`` calls fall through, and zero overhead
is paid.

Concrete out-of-process implementations (Redis, Postgres pub/sub)
can implement the same channel API without touching callers.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import uuid as _uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BloomEvent:
    """A single ``bloom.*`` SSE event.

    Spec §18 names five canonical event types:

    - ``bloom.signal.ingested`` — dispatcher returned.
    - ``bloom.run.queued`` — processor inserted JobRun.
    - ``bloom.run.started`` — processor invoked the runner.
    - ``bloom.run.tick`` — agent step (forwarded from graph events).
    - ``bloom.run.finished`` — runner returned (success OR failure).

    Concrete payloads are domain-specific dicts; the helper
    constructors below cover the typical shapes.
    """

    type: str
    run_id: _uuid.UUID
    payload: dict[str, Any] = field(default_factory=dict)
    occurred_at: _dt.datetime = field(default_factory=lambda: _dt.datetime.now(tz=_dt.UTC))

    def is_terminal(self) -> bool:
        return self.type == "bloom.run.finished"


# Convenience constructors (kept as functions, not classmethods, so
# tests can spread them with ``BloomEvent(**...)`` if they prefer).


def queued_event(*, run_id: _uuid.UUID, trigger_id: str, queued_at: _dt.datetime) -> BloomEvent:
    return BloomEvent(
        type="bloom.run.queued",
        run_id=run_id,
        payload={
            "run_id": str(run_id),
            "trigger_id": trigger_id,
            "queued_at": queued_at.isoformat(),
        },
    )


def started_event(*, run_id: _uuid.UUID, attempt_number: int, started_at: _dt.datetime) -> BloomEvent:
    return BloomEvent(
        type="bloom.run.started",
        run_id=run_id,
        payload={
            "run_id": str(run_id),
            "attempt_number": attempt_number,
            "started_at": started_at.isoformat(),
        },
    )


def finished_event(
    *,
    run_id: _uuid.UUID,
    status: str,
    finished_at: _dt.datetime,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> BloomEvent:
    payload: dict[str, Any] = {
        "run_id": str(run_id),
        "status": status,
        "finished_at": finished_at.isoformat(),
    }
    if result is not None:
        payload["result"] = result
    if error is not None:
        payload["error"] = error
    return BloomEvent(
        type="bloom.run.finished",
        run_id=run_id,
        payload=payload,
    )


def signal_ingested_event(*, run_id: _uuid.UUID, signal_id: _uuid.UUID, signal_type: str, source: str) -> BloomEvent:
    return BloomEvent(
        type="bloom.signal.ingested",
        run_id=run_id,
        payload={
            "signal_id": str(signal_id),
            "type": signal_type,
            "source": source,
        },
    )


# ── Chat-channel events (§LS6 + LS8) ────────────────────────


# Allowed type literals for the chat channel.  Three only, by
# design — ``attached`` collapses ``queued`` + ``started`` from the
# run channel because the chat consumer doesn't care about the
# queue-vs-started distinction.
ChatBloomEventType = Literal[
    "chat.bloom.attached",
    "chat.bloom.tick",
    "chat.bloom.finished",
]


@dataclass(frozen=True, slots=True)
class ChatBloomEvent:
    """Bloom event narrowed to the shape a chat-stream consumer needs.

    Channel: ``f"chat:{chat_id}"``.  Constructed by
    :func:`_wrap_for_chat` from the per-run :class:`BloomEvent`
    plus the resolved chat binding.  The ``result`` of a finished
    run is **not** carried on ``chat.bloom.finished`` — by §25.5 the
    final ``AIMessage`` is already persisted to chat storage; the
    frontend learns of it through the chat reload, not through the
    progress card stream (LS2 — zero persistence of intermediate
    progress).
    """

    type: ChatBloomEventType
    chat_id: str
    run_id: _uuid.UUID
    occurred_at: _dt.datetime
    payload: dict[str, Any] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        return self.type == "chat.bloom.finished"


# ── Stream ──────────────────────────────────────────────────


class BloomEventStream:
    """In-process channel → ``[asyncio.Queue]`` fan-out.

    Channels are arbitrary strings.  Two well-known families:

    - ``"run:{run_id}"`` — the run channel; backed by
      :meth:`subscribe_run` / :meth:`publish_run` for callers
      that don't want to spell the channel string themselves.
    - ``"chat:{chat_id}"`` — the chat channel for in-chat live
      progress (§LS6).

    Subscribers call :meth:`subscribe` to get an async iterator;
    each iteration yields one event.  The iterator closes
    automatically after a *terminal* event (anything whose
    ``is_terminal()`` returns ``True``) OR after the queue's
    ``idle_timeout_seconds`` elapses with no traffic, so a
    long-lived endpoint doesn't hang forever on a dead run.

    :meth:`publish` is fire-and-forget: a slow subscriber backs up
    its own queue, never the publisher.  When a queue is full
    (``per_run_buffer`` exceeded), the oldest event is dropped —
    better to drop one ``tick`` than to block the processor.
    """

    def __init__(
        self,
        *,
        per_run_buffer: int = 64,
        idle_timeout_seconds: float = 300.0,
    ) -> None:
        self._buf = per_run_buffer
        self._idle = idle_timeout_seconds
        # Channel → list of subscriber queues.  Channel strings are
        # the public addressing scheme; the run-id and chat-id
        # families compose those strings inside the backwards-compat
        # wrappers and the dual-publish branch in the processor.
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    # ── Generic channel API (primary) ────────────────────────

    async def publish(self, channel: str, event: Any) -> None:
        """Push ``event`` to every subscriber of ``channel``."""
        async with self._lock:
            queues = list(self._subscribers.get(channel, ()))
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop the oldest event to make room — better than
                # blocking the publisher on a slow client.
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    _logger.warning(
                        "BloomEventStream: dropped event on channel %s (subscriber queue still full after eviction)",
                        channel,
                    )

    async def subscribe(self, channel: str) -> AsyncIterator[Any]:
        """Yield events for ``channel`` until terminal OR idle timeout."""
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=self._buf)
        async with self._lock:
            self._subscribers.setdefault(channel, []).append(queue)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=self._idle)
                except TimeoutError:
                    # No traffic for ``idle_timeout_seconds`` — close
                    # the stream rather than hang indefinitely.
                    return
                yield event
                if _is_terminal(event):
                    return
        finally:
            async with self._lock:
                subs = self._subscribers.get(channel)
                if subs is not None and queue in subs:
                    subs.remove(queue)
                if subs is not None and not subs:
                    self._subscribers.pop(channel, None)

    # ── Backwards-compat per-run wrappers ────────────────────
    # The pre-channel API, preserved verbatim in shape so existing
    # callers (``processors/asyncio_pool.py``, ``routers/runs.py``)
    # don't need to know about channels.  These thin wrappers
    # delegate to the channel API at ``run:{run_id}`` and MUST NOT
    # be removed in v1 — see Phase 4.5 §LS6.

    async def publish_run(self, event: BloomEvent) -> None:
        """Publish a per-run event to ``f"run:{event.run_id}"``."""
        await self.publish(f"run:{event.run_id}", event)

    async def subscribe_run(self, run_id: _uuid.UUID) -> AsyncIterator[BloomEvent]:
        """Yield per-run events for ``run_id`` until terminal/idle."""
        async for ev in self.subscribe(f"run:{run_id}"):
            yield ev

    @property
    def subscriber_count(self) -> int:
        """Total active subscriptions across every channel.  Test/observability hook."""
        return sum(len(qs) for qs in self._subscribers.values())


def _wrap_for_chat(event: BloomEvent, *, run: Any, binding: dict[str, Any]) -> ChatBloomEvent | None:
    """Map a per-run :class:`BloomEvent` to its chat-channel equivalent.

    Phase 4.5 §LS8 — three event types only:

    - ``bloom.run.queued`` AND ``bloom.run.started`` BOTH collapse
      into ``chat.bloom.attached`` (the chat consumer doesn't care
      about the queue-vs-started distinction; the frontend dedupes
      by ``run_id``).
    - ``bloom.run.tick`` forwards as ``chat.bloom.tick`` with a
      **redacted** payload by default (LSQ6 default — awaiting
      decision).  Only ``kind``, ``agent``, ``tool``, ``status``,
      and ``message`` are carried; raw tool result bodies and
      internal reasoning are dropped.
    - ``bloom.run.finished`` forwards as ``chat.bloom.finished``
      WITHOUT the run ``result`` (LS2 — the final ``AIMessage`` is
      already persisted to chat storage by §25.5; the frontend
      learns of it through the chat reload, not through the
      progress-card stream).

    Returns ``None`` for event types the chat consumer doesn't need
    (notably ``bloom.signal.ingested`` — the user already saw their
    own holding message).
    """
    chat_id = binding["chat_id"]
    source_message_id = binding.get("source_message_id")
    base: dict[str, Any] = {
        "run_id": str(event.run_id),
        "trigger_id": run.spec.trigger_id,
        "agent_name": run.spec.agent_name,
        "source_message_id": source_message_id,
    }
    # Identity-mode hint so the frontend can gate the in-chat
    # cancel button per §LS3 (act_as_user → owner can cancel;
    # addressed_to_user → button hidden).
    identity_claim = run.spec.identity_claim or {}
    base["identity_mode"] = identity_claim.get("mode")

    if event.type in ("bloom.run.queued", "bloom.run.started"):
        return ChatBloomEvent(
            type="chat.bloom.attached",
            chat_id=chat_id,
            run_id=event.run_id,
            occurred_at=event.occurred_at,
            payload={**base, "attached_at": event.occurred_at.isoformat()},
        )

    if event.type == "bloom.run.tick":
        # Default redaction (LSQ6 default — awaiting decision).
        # Only carry the few fields that are safe to surface to the
        # chat owner.  ``chat_progress_verbosity: verbose`` per-trigger
        # opt-in is deferred until LSQ6 is decided.
        p = event.payload or {}
        redacted: dict[str, Any] = {**base}
        for key in ("kind", "agent", "tool", "status", "message"):
            if key in p:
                redacted[key] = p[key]
        return ChatBloomEvent(
            type="chat.bloom.tick",
            chat_id=chat_id,
            run_id=event.run_id,
            occurred_at=event.occurred_at,
            payload=redacted,
        )

    if event.type == "bloom.run.finished":
        p = event.payload or {}
        finished_payload: dict[str, Any] = {**base}
        if "status" in p:
            finished_payload["status"] = p["status"]
        if "finished_at" in p:
            finished_payload["finished_at"] = p["finished_at"]
        # ``result`` deliberately NOT forwarded (LS2).  ``error`` is
        # kept so the chat-side can render a "this run failed" hint
        # while the persisted §25 failure message catches up.
        if "error" in p:
            finished_payload["error"] = p["error"]
        return ChatBloomEvent(
            type="chat.bloom.finished",
            chat_id=chat_id,
            run_id=event.run_id,
            occurred_at=event.occurred_at,
            payload=finished_payload,
        )

    # bloom.signal.ingested and any future unknown types: not
    # forwarded to the chat channel.
    return None


def _is_terminal(event: Any) -> bool:
    """Duck-typed terminal check.

    Both :class:`BloomEvent` and :class:`ChatBloomEvent` expose
    ``is_terminal()`` so the channel-API ``subscribe`` loop can
    close cleanly regardless of which event family flowed through.
    Unknown event objects without the method default to non-terminal
    so the stream relies on ``idle_timeout_seconds`` for shutdown.
    """
    is_terminal = getattr(event, "is_terminal", None)
    if callable(is_terminal):
        try:
            return bool(is_terminal())
        except Exception:  # pragma: no cover — defensive
            return False
    return False
