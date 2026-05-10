"""Channel-API tests for ``BloomEventStream`` (Phase 4.5 §LS6).

Covers:

- The new generic ``subscribe(channel)`` / ``publish(channel, event)``
  API on both the run-channel and chat-channel families.
- The backwards-compat ``subscribe_run`` / ``publish_run`` wrappers
  behave identically to the underlying ``run:{run_id}`` channel —
  no observable drift.
- Cross-channel isolation: chat-channel events never reach run-channel
  subscribers and vice-versa.
- ``ChatBloomEvent.is_terminal()`` closes a chat-channel subscription
  on ``chat.bloom.finished``.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import uuid as _uuid

from orchid_ai.events.streaming import (
    BloomEventStream,
    ChatBloomEvent,
    finished_event,
    queued_event,
)


def _now() -> _dt.datetime:
    return _dt.datetime.now(tz=_dt.UTC)


# ── Generic channel API ─────────────────────────────────────


async def test_generic_subscribe_publish_round_trip_on_run_channel() -> None:
    """``subscribe('run:X')`` + ``publish('run:X', evt)`` deliver."""
    stream = BloomEventStream()
    run_id = _uuid.uuid4()
    channel = f"run:{run_id}"
    received: list = []

    async def consume() -> None:
        async for evt in stream.subscribe(channel):
            received.append(evt)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)

    await stream.publish(channel, queued_event(run_id=run_id, trigger_id="t", queued_at=_now()))
    await stream.publish(channel, finished_event(run_id=run_id, status="succeeded", finished_at=_now()))
    await asyncio.wait_for(task, timeout=1.0)

    assert [e.type for e in received] == ["bloom.run.queued", "bloom.run.finished"]


async def test_generic_subscribe_publish_round_trip_on_chat_channel() -> None:
    """Chat channel: ``ChatBloomEvent`` round-trip + terminal closes."""
    stream = BloomEventStream()
    chat_id = "C-742"
    run_id = _uuid.uuid4()
    channel = f"chat:{chat_id}"
    received: list[ChatBloomEvent] = []

    async def consume() -> None:
        async for evt in stream.subscribe(channel):
            received.append(evt)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)

    await stream.publish(
        channel,
        ChatBloomEvent(
            type="chat.bloom.attached",
            chat_id=chat_id,
            run_id=run_id,
            occurred_at=_now(),
            payload={"trigger_id": "deep-research"},
        ),
    )
    await stream.publish(
        channel,
        ChatBloomEvent(
            type="chat.bloom.tick",
            chat_id=chat_id,
            run_id=run_id,
            occurred_at=_now(),
            payload={"kind": "tool.called", "tool": "search"},
        ),
    )
    await stream.publish(
        channel,
        ChatBloomEvent(
            type="chat.bloom.finished",
            chat_id=chat_id,
            run_id=run_id,
            occurred_at=_now(),
            payload={"status": "succeeded"},
        ),
    )
    await asyncio.wait_for(task, timeout=1.0)

    assert [e.type for e in received] == [
        "chat.bloom.attached",
        "chat.bloom.tick",
        "chat.bloom.finished",
    ]
    # Terminal close: subscriber list cleaned up.
    assert stream.subscriber_count == 0


# ── Backwards-compat wrapper parity ─────────────────────────


async def test_publish_run_is_observable_on_explicit_run_channel() -> None:
    """``publish_run(evt)`` reaches ``subscribe('run:{id}')`` consumers.

    The wrappers MUST delegate to the same channel string so a
    consumer using either API sees the same events.  This is the
    invariant that makes the dual-publish branch in the processor
    safe — production callers can keep using ``publish_run`` while
    the new chat-channel callers use ``publish('chat:...', evt)``.
    """
    stream = BloomEventStream()
    run_id = _uuid.uuid4()
    received: list = []

    async def consume_explicit() -> None:
        async for evt in stream.subscribe(f"run:{run_id}"):
            received.append(evt)

    task = asyncio.create_task(consume_explicit())
    await asyncio.sleep(0)

    await stream.publish_run(queued_event(run_id=run_id, trigger_id="t", queued_at=_now()))
    await stream.publish_run(finished_event(run_id=run_id, status="succeeded", finished_at=_now()))
    await asyncio.wait_for(task, timeout=1.0)

    assert [e.type for e in received] == ["bloom.run.queued", "bloom.run.finished"]


async def test_subscribe_run_observes_explicit_publish_to_run_channel() -> None:
    """The reverse direction: ``subscribe_run`` sees ``publish('run:...', evt)``."""
    stream = BloomEventStream()
    run_id = _uuid.uuid4()
    received: list = []

    async def consume_wrapped() -> None:
        async for evt in stream.subscribe_run(run_id):
            received.append(evt)

    task = asyncio.create_task(consume_wrapped())
    await asyncio.sleep(0)

    await stream.publish(f"run:{run_id}", finished_event(run_id=run_id, status="succeeded", finished_at=_now()))
    await asyncio.wait_for(task, timeout=1.0)

    assert [e.type for e in received] == ["bloom.run.finished"]


# ── Cross-channel isolation ─────────────────────────────────


async def test_run_channel_subscribers_do_not_receive_chat_events() -> None:
    """A run-channel consumer is invisible to ``publish('chat:...', evt)``."""
    stream = BloomEventStream()
    run_id = _uuid.uuid4()
    chat_id = "C-iso"
    received: list = []

    async def consume() -> None:
        async for evt in stream.subscribe_run(run_id):
            received.append(evt)
            if getattr(evt, "is_terminal", lambda: False)():
                return

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)

    # Publish to the chat channel — must NOT leak into the run subscriber.
    await stream.publish(
        f"chat:{chat_id}",
        ChatBloomEvent(
            type="chat.bloom.tick",
            chat_id=chat_id,
            run_id=run_id,
            occurred_at=_now(),
            payload={"kind": "x"},
        ),
    )
    # Then publish a real run-channel terminal event so the
    # subscriber closes.
    await stream.publish_run(finished_event(run_id=run_id, status="succeeded", finished_at=_now()))
    await asyncio.wait_for(task, timeout=1.0)

    assert [e.type for e in received] == ["bloom.run.finished"]


async def test_chat_channel_subscribers_do_not_receive_run_events() -> None:
    """The mirror: chat-channel subscriber is invisible to ``publish_run``."""
    stream = BloomEventStream()
    run_id = _uuid.uuid4()
    chat_id = "C-iso2"
    received: list[ChatBloomEvent] = []

    async def consume() -> None:
        async for evt in stream.subscribe(f"chat:{chat_id}"):
            received.append(evt)
            if evt.is_terminal():
                return

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)

    # Publish on the run channel — chat consumer must not see it.
    await stream.publish_run(queued_event(run_id=run_id, trigger_id="t", queued_at=_now()))
    # Then close the chat subscription cleanly.
    await stream.publish(
        f"chat:{chat_id}",
        ChatBloomEvent(
            type="chat.bloom.finished",
            chat_id=chat_id,
            run_id=run_id,
            occurred_at=_now(),
            payload={"status": "succeeded"},
        ),
    )
    await asyncio.wait_for(task, timeout=1.0)

    assert [e.type for e in received] == ["chat.bloom.finished"]


async def test_chat_channel_idle_timeout_closes_subscription() -> None:
    """Chat channel honours the same ``idle_timeout_seconds`` contract."""
    stream = BloomEventStream(idle_timeout_seconds=0.05)
    received: list[ChatBloomEvent] = []

    async def consume() -> None:
        async for evt in stream.subscribe("chat:abandoned"):
            received.append(evt)

    task = asyncio.create_task(consume())
    await asyncio.wait_for(task, timeout=1.0)

    assert received == []


async def test_chat_bloom_event_is_terminal_only_on_finished() -> None:
    """``ChatBloomEvent.is_terminal()`` discriminates the three types."""
    run_id = _uuid.uuid4()
    occurred = _now()
    attached = ChatBloomEvent(
        type="chat.bloom.attached",
        chat_id="C-1",
        run_id=run_id,
        occurred_at=occurred,
    )
    tick = ChatBloomEvent(
        type="chat.bloom.tick",
        chat_id="C-1",
        run_id=run_id,
        occurred_at=occurred,
    )
    finished = ChatBloomEvent(
        type="chat.bloom.finished",
        chat_id="C-1",
        run_id=run_id,
        occurred_at=occurred,
    )
    assert attached.is_terminal() is False
    assert tick.is_terminal() is False
    assert finished.is_terminal() is True
