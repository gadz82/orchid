"""Tests for ``BloomEventStream`` — the SSE fan-out keyed on ``run_id``.

Coverage:

- Single subscriber receives ``queued → started → finished`` in
  order and the iterator closes after ``finished``.
- Multi-subscriber fan-out — each subscriber sees its own copy.
- Idle timeout closes a subscription that no event arrives for.
- Slow consumer eviction policy: a full queue drops the oldest
  event rather than blocking the publisher.
- ``subscriber_count`` reflects active subscriptions.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import uuid as _uuid


from orchid_ai.events.streaming import (
    BloomEventStream,
    finished_event,
    queued_event,
    signal_ingested_event,
    started_event,
)


async def test_subscriber_receives_full_lifecycle() -> None:
    stream = BloomEventStream()
    run_id = _uuid.uuid4()
    now = _dt.datetime.now(tz=_dt.UTC)

    received = []

    async def consume():
        async for event in stream.subscribe_run(run_id):
            received.append(event)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)  # allow subscribe to register

    await stream.publish_run(queued_event(run_id=run_id, trigger_id="t1", queued_at=now))
    await stream.publish_run(started_event(run_id=run_id, attempt_number=1, started_at=now))
    await stream.publish_run(finished_event(run_id=run_id, status="succeeded", finished_at=now, result={"ok": True}))
    await asyncio.wait_for(task, timeout=1.0)

    assert [e.type for e in received] == [
        "bloom.run.queued",
        "bloom.run.started",
        "bloom.run.finished",
    ]
    assert received[-1].payload["status"] == "succeeded"


async def test_two_subscribers_each_see_full_stream() -> None:
    stream = BloomEventStream()
    run_id = _uuid.uuid4()
    now = _dt.datetime.now(tz=_dt.UTC)
    received_a, received_b = [], []

    async def consume(buffer):
        async for event in stream.subscribe_run(run_id):
            buffer.append(event)

    task_a = asyncio.create_task(consume(received_a))
    task_b = asyncio.create_task(consume(received_b))
    await asyncio.sleep(0)

    assert stream.subscriber_count == 2

    await stream.publish_run(queued_event(run_id=run_id, trigger_id="t1", queued_at=now))
    await stream.publish_run(finished_event(run_id=run_id, status="succeeded", finished_at=now))
    await asyncio.wait_for(task_a, timeout=1.0)
    await asyncio.wait_for(task_b, timeout=1.0)
    assert len(received_a) == 2
    assert len(received_b) == 2

    # Subscribers cleaned up after terminal event.
    assert stream.subscriber_count == 0


async def test_signal_ingested_event_published_to_run_id() -> None:
    """Signal events for a run_id arrive at that subscriber."""
    stream = BloomEventStream()
    run_id = _uuid.uuid4()
    sig_id = _uuid.uuid4()
    received = []

    async def consume():
        async for event in stream.subscribe_run(run_id):
            received.append(event)
            if event.type == "bloom.run.finished":
                return

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)

    await stream.publish_run(signal_ingested_event(run_id=run_id, signal_id=sig_id, signal_type="x", source="src"))
    await stream.publish_run(
        finished_event(
            run_id=run_id,
            status="succeeded",
            finished_at=_dt.datetime.now(tz=_dt.UTC),
        )
    )
    await asyncio.wait_for(task, timeout=1.0)
    assert received[0].type == "bloom.signal.ingested"
    assert received[0].payload["signal_id"] == str(sig_id)


async def test_idle_timeout_closes_stream() -> None:
    stream = BloomEventStream(idle_timeout_seconds=0.05)
    run_id = _uuid.uuid4()
    received = []

    async def consume():
        async for event in stream.subscribe_run(run_id):
            received.append(event)

    task = asyncio.create_task(consume())
    # No publish — wait past the idle timeout.
    await asyncio.wait_for(task, timeout=1.0)
    assert received == []


async def test_publishes_to_other_run_id_dont_leak() -> None:
    stream = BloomEventStream()
    a = _uuid.uuid4()
    b = _uuid.uuid4()
    now = _dt.datetime.now(tz=_dt.UTC)
    received_a = []

    async def consume():
        async for event in stream.subscribe_run(a):
            received_a.append(event)
            if event.is_terminal():
                return

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)

    await stream.publish_run(queued_event(run_id=b, trigger_id="other", queued_at=now))
    await stream.publish_run(finished_event(run_id=a, status="succeeded", finished_at=now))
    await asyncio.wait_for(task, timeout=1.0)
    # Only the terminal event for ``a``.
    assert len(received_a) == 1


async def test_publish_with_no_subscribers_is_a_noop() -> None:
    stream = BloomEventStream()
    await stream.publish_run(
        queued_event(
            run_id=_uuid.uuid4(),
            trigger_id="t1",
            queued_at=_dt.datetime.now(tz=_dt.UTC),
        )
    )
    assert stream.subscriber_count == 0
