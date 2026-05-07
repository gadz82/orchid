"""Processor dual-publish branch (Phase 4.5 §LS6/LS7).

Covers ``AsyncioWorkerPoolProcessor._publish_event(event, *, run=run)``:

- Chat-bound run: events go to BOTH the run channel and the chat
  channel (mapped through ``_wrap_for_chat``).
- Non-bound run: events go ONLY to the run channel; the chat
  channel is never touched.
- ``bloom.signal.ingested`` is NOT forwarded to the chat channel
  even when the run is bound (the chat consumer doesn't need it).
- A failure on the chat-channel publish does NOT affect the
  run-channel publish (the two ``try/except`` blocks are
  independent — the run channel is the authoritative path).
- A failure on the run-channel publish does NOT affect the
  chat-channel publish (mirror invariant).
- ``bloom.run.finished`` propagates ``status`` + ``finished_at`` +
  ``error`` but NOT ``result`` (LS2 — final AIMessage flows through
  chat reload, not the progress card stream).
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import uuid as _uuid

from orchid_ai.core.events.job import JobRun, JobSpec, JobStatus
from orchid_ai.events.processors.asyncio_pool import AsyncioWorkerPoolProcessor
from orchid_ai.events.streaming import (
    BloomEventStream,
    ChatBloomEvent,
    finished_event,
    queued_event,
    signal_ingested_event,
    started_event,
)


def _now() -> _dt.datetime:
    return _dt.datetime.now(tz=_dt.UTC)


def _make_run(*, chat_binding: dict | None) -> JobRun:
    spec = JobSpec(
        trigger_id="t-1",
        signal_id=_uuid.uuid4(),
        agent_name="reviews",
        prompt="p",
        identity_claim={"mode": "act_as_user", "user_id": "u-7"},
        correlation_id="corr",
        parallelism_key="user:t-1:u-7",
        visibility="actor",
        visibility_user_id="u-7",
        chat_binding=chat_binding,
    )
    return JobRun(
        run_id=_uuid.uuid4(),
        spec=spec,
        attempt_number=1,
        status=JobStatus.RUNNING,
        queued_at=_now(),
        started_at=_now(),
    )


# ── Dual publish on chat-bound runs ─────────────────────────


async def test_chat_bound_run_publishes_to_both_channels() -> None:
    stream = BloomEventStream()
    proc = AsyncioWorkerPoolProcessor()
    proc._event_stream = stream
    run = _make_run(
        chat_binding={
            "chat_id": "C-1",
            "mode": "append_final_message",
            "on_failure": "post_error",
            "source_message_id": "m-42",
        }
    )

    run_received: list = []
    chat_received: list[ChatBloomEvent] = []

    async def consume_run() -> None:
        async for evt in stream.subscribe_run(run.run_id):
            run_received.append(evt)

    async def consume_chat() -> None:
        async for evt in stream.subscribe(f"chat:{run.spec.chat_binding['chat_id']}"):
            chat_received.append(evt)

    t_run = asyncio.create_task(consume_run())
    t_chat = asyncio.create_task(consume_chat())
    await asyncio.sleep(0)

    await proc._publish_event(queued_event(run_id=run.run_id, trigger_id="t-1", queued_at=_now()), run=run)
    await proc._publish_event(started_event(run_id=run.run_id, attempt_number=1, started_at=_now()), run=run)
    await proc._publish_event(
        finished_event(run_id=run.run_id, status="succeeded", finished_at=_now(), result={"ok": True}),
        run=run,
    )
    await asyncio.wait_for(t_run, timeout=1.0)
    await asyncio.wait_for(t_chat, timeout=1.0)

    # Run channel: 3 events as before.
    assert [e.type for e in run_received] == [
        "bloom.run.queued",
        "bloom.run.started",
        "bloom.run.finished",
    ]
    # Chat channel: queued+started collapsed into TWO ``attached``
    # events (per LS8 — the frontend dedupes by run_id), then a
    # single ``finished``.  ``signal.ingested`` would be filtered
    # out but we didn't publish one in this run.
    assert [e.type for e in chat_received] == [
        "chat.bloom.attached",
        "chat.bloom.attached",
        "chat.bloom.finished",
    ]
    # Anchor information is forwarded.
    assert chat_received[0].payload["source_message_id"] == "m-42"
    assert chat_received[0].payload["trigger_id"] == "t-1"
    assert chat_received[0].payload["agent_name"] == "reviews"
    assert chat_received[0].payload["identity_mode"] == "act_as_user"
    # Finished payload: ``result`` is NOT forwarded; ``status`` +
    # ``finished_at`` are.
    assert "result" not in chat_received[2].payload
    assert chat_received[2].payload["status"] == "succeeded"
    assert "finished_at" in chat_received[2].payload


async def test_finished_with_error_propagates_error_field() -> None:
    """Failure-mode finished events forward ``error`` to the chat side."""
    stream = BloomEventStream()
    proc = AsyncioWorkerPoolProcessor()
    proc._event_stream = stream
    run = _make_run(chat_binding={"chat_id": "C-1"})
    received: list[ChatBloomEvent] = []

    async def consume() -> None:
        async for evt in stream.subscribe("chat:C-1"):
            received.append(evt)
            if evt.is_terminal():
                return

    t = asyncio.create_task(consume())
    await asyncio.sleep(0)

    await proc._publish_event(
        finished_event(run_id=run.run_id, status="failed", finished_at=_now(), error="boom"),
        run=run,
    )
    await asyncio.wait_for(t, timeout=1.0)
    assert received[0].payload["status"] == "failed"
    assert received[0].payload["error"] == "boom"


# ── Non-bound runs: chat channel must NOT be touched ────────


async def test_unbound_run_only_publishes_to_run_channel() -> None:
    stream = BloomEventStream()
    proc = AsyncioWorkerPoolProcessor()
    proc._event_stream = stream
    run = _make_run(chat_binding=None)

    chat_received: list = []

    async def consume_chat() -> None:
        async for evt in stream.subscribe("chat:NONEXISTENT"):
            chat_received.append(evt)

    t = asyncio.create_task(consume_chat())
    await asyncio.sleep(0)

    await proc._publish_event(queued_event(run_id=run.run_id, trigger_id="t", queued_at=_now()), run=run)
    await proc._publish_event(
        finished_event(run_id=run.run_id, status="succeeded", finished_at=_now()),
        run=run,
    )
    # Cancel the chat consumer (idle timeout would also work but
    # is slower).
    t.cancel()
    try:
        await t
    except asyncio.CancelledError:
        pass
    assert chat_received == []


# ── Filtering on the chat channel ───────────────────────────


async def test_signal_ingested_event_is_not_forwarded_to_chat() -> None:
    """``bloom.signal.ingested`` doesn't reach the chat side."""
    stream = BloomEventStream()
    proc = AsyncioWorkerPoolProcessor()
    proc._event_stream = stream
    run = _make_run(chat_binding={"chat_id": "C-1"})
    received: list[ChatBloomEvent] = []

    async def consume() -> None:
        async for evt in stream.subscribe("chat:C-1"):
            received.append(evt)
            if evt.is_terminal():
                return

    t = asyncio.create_task(consume())
    await asyncio.sleep(0)

    # Publish a signal-ingested event (which DOES go to the run
    # channel) — the chat-channel must not see it.
    await proc._publish_event(
        signal_ingested_event(
            run_id=run.run_id,
            signal_id=_uuid.uuid4(),
            signal_type="x",
            source="src",
        ),
        run=run,
    )
    # Then close the chat channel.
    await proc._publish_event(
        finished_event(run_id=run.run_id, status="succeeded", finished_at=_now()),
        run=run,
    )
    await asyncio.wait_for(t, timeout=1.0)

    assert [e.type for e in received] == ["chat.bloom.finished"]


# ── Independent error handling ──────────────────────────────


async def test_chat_publish_failure_does_not_affect_run_publish() -> None:
    """Chat-channel publish raising must NOT swallow run-channel publish.

    The run channel is the authoritative path for ``/runs/{id}/stream``;
    a chat-channel misconfiguration must never silently degrade the
    operator-facing surface.
    """
    stream = BloomEventStream()
    real_publish = stream.publish

    async def fake_publish(channel: str, event):
        if channel.startswith("chat:"):
            raise RuntimeError("simulated chat-channel outage")
        await real_publish(channel, event)

    stream.publish = fake_publish  # type: ignore[assignment]
    proc = AsyncioWorkerPoolProcessor()
    proc._event_stream = stream
    run = _make_run(chat_binding={"chat_id": "C-1"})
    run_received: list = []

    async def consume_run() -> None:
        async for evt in stream.subscribe_run(run.run_id):
            run_received.append(evt)
            if evt.is_terminal():
                return

    t = asyncio.create_task(consume_run())
    await asyncio.sleep(0)

    # The chat-channel publish raises; the run-channel publish
    # must still complete.
    await proc._publish_event(
        finished_event(run_id=run.run_id, status="succeeded", finished_at=_now()),
        run=run,
    )
    await asyncio.wait_for(t, timeout=1.0)
    assert [e.type for e in run_received] == ["bloom.run.finished"]


async def test_run_publish_failure_does_not_affect_chat_publish() -> None:
    """Run-channel publish raising must NOT swallow chat-channel publish.

    Mirror invariant — chat consumers are still allowed to learn that
    a run finished even when the operator-facing path failed.
    """
    stream = BloomEventStream()
    real_publish_run = stream.publish_run

    async def fake_publish_run(event):
        raise RuntimeError("simulated run-channel outage")

    stream.publish_run = fake_publish_run  # type: ignore[assignment]
    proc = AsyncioWorkerPoolProcessor()
    proc._event_stream = stream
    run = _make_run(chat_binding={"chat_id": "C-1"})
    chat_received: list[ChatBloomEvent] = []

    async def consume_chat() -> None:
        async for evt in stream.subscribe("chat:C-1"):
            chat_received.append(evt)
            if evt.is_terminal():
                return

    t = asyncio.create_task(consume_chat())
    await asyncio.sleep(0)

    await proc._publish_event(
        finished_event(run_id=run.run_id, status="succeeded", finished_at=_now()),
        run=run,
    )
    await asyncio.wait_for(t, timeout=1.0)
    assert [e.type for e in chat_received] == ["chat.bloom.finished"]
    # Reset the monkeypatch so other tests aren't affected.
    stream.publish_run = real_publish_run  # type: ignore[assignment]


async def test_publish_with_no_run_kwarg_is_run_channel_only() -> None:
    """Backwards compat: callers that don't pass ``run=`` see only the run channel."""
    stream = BloomEventStream()
    proc = AsyncioWorkerPoolProcessor()
    proc._event_stream = stream
    run_id = _uuid.uuid4()

    chat_received: list = []

    async def consume_chat() -> None:
        async for evt in stream.subscribe("chat:ANY"):
            chat_received.append(evt)

    t = asyncio.create_task(consume_chat())
    await asyncio.sleep(0)

    # No ``run=`` kwarg — chat channel is never touched.
    await proc._publish_event(queued_event(run_id=run_id, trigger_id="t", queued_at=_now()))
    t.cancel()
    try:
        await t
    except asyncio.CancelledError:
        pass
    assert chat_received == []
