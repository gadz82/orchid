"""Phase-3 exit demo: end-to-end Bloom on SQLite for both
``service_account`` and ``act_as_user`` flavours, plus the
chat-bound ``act_as_user`` flow.

The Postgres-gated counterpart lives in
``test_postgres_queue.py``; this file is the always-on SQLite
equivalent.  Both demonstrate the same mechanical contract:

1. Dispatcher persists + enqueues a signal.
2. Processor dequeues, matches a trigger, resolves identity,
   creates a ``JobRun`` with the right ``visibility`` /
   ``visibility_user_id`` for §26, and invokes the runner.
3. Runner executes (optional chat-binding side-effects via §25)
   and persists the terminal status.

A fake ``GraphInvoker`` stands in for the real LangGraph; the
runner doesn't care what produced the result dict.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import aiosqlite
import pytest

from orchid_ai.config.schema_events import (
    ActAsUserIdentity,
    OrchidTriggerConfig,
    OrchidTriggerEmitConfig,
    OrchidTriggerMatchConfig,
    ServiceAccountIdentity,
)
from orchid_ai.core.events.dispatcher import OrchidSignalDispatcher
from orchid_ai.core.events.errors import OrchidServiceAccountUnknownError
from orchid_ai.core.events.job import JobStatus
from orchid_ai.core.events.signal import SignalEnvelope
from orchid_ai.core.identity import OrchidIdentityResolver
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.events.backends.sqlite import SQLiteEventStorage
from orchid_ai.events.processors.asyncio_pool import AsyncioWorkerPoolProcessor
from orchid_ai.events.queues.sqlite import SQLiteSignalQueue
from orchid_ai.events.registry import build_registry_from_config
from orchid_ai.events.runners.graph_runner import GraphJobRunner
from orchid_ai.persistence.models import OrchidChatSession


# ── Fixture resolver — supports both flavours ───────────────


class _FixtureResolver(OrchidIdentityResolver):
    """Knows two service accounts and mints user identities for any
    user (matches §26's act_as_user → actor visibility expectation)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    async def resolve(self, domain, bearer_token):  # pragma: no cover
        return OrchidAuthContext(access_token=bearer_token)

    async def resolve_service_account(self, name: str) -> OrchidAuthContext:
        self.calls.append(("resolve_service_account", (name,)))
        if name not in {"digest-bot", "research-bot"}:
            raise OrchidServiceAccountUnknownError(name)
        ctx = OrchidAuthContext(
            access_token=f"sa:{name}",
            tenant_key="t-1",
            user_id="",
        )
        ctx.extra["service_account"] = name
        return ctx

    async def mint_for_user(self, tenant_key: str, user_id: str) -> OrchidAuthContext:
        self.calls.append(("mint_for_user", (tenant_key, user_id)))
        return OrchidAuthContext(
            access_token=f"user:{tenant_key}:{user_id}",
            tenant_key=tenant_key,
            user_id=user_id,
        )


# ── Common fixtures ─────────────────────────────────────────


@pytest.fixture
async def shared_db(tmp_path: Path):
    dsn = str(tmp_path / "bloom.db")
    conn = await aiosqlite.connect(dsn)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")
    storage = SQLiteEventStorage(conn=conn)
    await storage.init_db()
    queue = SQLiteSignalQueue(conn=conn)
    yield {"storage": storage, "queue": queue, "conn": conn, "dsn": dsn}
    await conn.close()


# A no-op runner-injectable so the processor's full lifecycle is
# exercised — including warm-call + per-key lock + status writes.
async def _ok_invoker(run, auth) -> dict:
    return {
        "final_response": (
            f"Bloom completed for run {run.run_id} (agent={run.spec.agent_name}, auth.access_token={auth.access_token})"
        )
    }


# ── 1. Service-account Bloom ────────────────────────────────


async def test_end_to_end_service_account_bloom(shared_db) -> None:
    storage: SQLiteEventStorage = shared_db["storage"]
    queue: SQLiteSignalQueue = shared_db["queue"]

    # Wire dispatcher + registry + processor.
    dispatcher = OrchidSignalDispatcher(store=storage.signals, queue=queue)
    trigger = OrchidTriggerConfig(
        id="morning-digest",
        on=OrchidTriggerMatchConfig(signal="cron", cron="0 7 * * 1-5"),
        emits=OrchidTriggerEmitConfig(
            agent="notifications",
            prompt_template="Build digest for {{tenant_key}}",
            identity=ServiceAccountIdentity(name="digest-bot"),
            # Default visibility for service_account = "admin".
        ),
    )
    registry = build_registry_from_config([trigger], known_agents={"notifications"})
    resolver = _FixtureResolver()
    runner = GraphJobRunner(invoker=_ok_invoker)
    processor = AsyncioWorkerPoolProcessor()

    # Ingest a synthetic cron signal.
    envelope = SignalEnvelope(
        type="cron",
        payload={"schedule_id": "morning-digest", "fire_time": "2026-05-07T07:00:00Z"},
        source="scheduler:morning-digest",
        occurred_at=_dt.datetime.now(tz=_dt.UTC),
        tenant_key="t-1",
        identity_claim={"mode": "service_account", "name": "digest-bot"},
    )
    result = await dispatcher.ingest(envelope)
    assert result.deduplicated is False

    processed = await processor.process_until_idle(
        queue=queue,
        signal_store=storage.signals,
        triggers=registry,
        identity_resolver=resolver,
        job_store=storage.jobs,
        job_runner=runner,
    )
    assert processed >= 1

    runs = await storage.jobs.list()
    assert len(runs) == 1
    [run] = runs
    assert run.status == JobStatus.SUCCEEDED
    assert run.spec.trigger_id == "morning-digest"
    assert run.spec.visibility == "admin"  # §26 default for service_account
    assert run.spec.visibility_user_id is None
    assert run.spec.parallelism_key.startswith("sa:t-1:digest-bot")
    # Service-account auth was passed to the invoker — the fake echoes
    # the access token back so we can assert on identity wiring.
    assert "sa:digest-bot" in run.result.get("final_response", "")

    # The fixture resolver got a service-account call, NOT mint.
    assert any(call == ("resolve_service_account", ("digest-bot",)) for call in resolver.calls)


# ── 2. act_as_user Bloom ────────────────────────────────────


async def test_end_to_end_act_as_user_bloom(shared_db) -> None:
    storage: SQLiteEventStorage = shared_db["storage"]
    queue: SQLiteSignalQueue = shared_db["queue"]

    dispatcher = OrchidSignalDispatcher(store=storage.signals, queue=queue)
    trigger = OrchidTriggerConfig(
        id="ticket-triage",
        on=OrchidTriggerMatchConfig(signal="support.ticket.created"),
        emits=OrchidTriggerEmitConfig(
            agent="helpdesk",
            prompt_template="Triage ticket {{payload.ticket_id}}",
            identity=ActAsUserIdentity(user_id_from="signal.user_id"),
            # Default visibility for act_as_user = "actor".
        ),
    )
    registry = build_registry_from_config([trigger], known_agents={"helpdesk"}, identity_resolver=_FixtureResolver())
    resolver = _FixtureResolver()
    runner = GraphJobRunner(invoker=_ok_invoker)
    processor = AsyncioWorkerPoolProcessor()

    envelope = SignalEnvelope(
        type="support.ticket.created",
        payload={"ticket_id": "T-42"},
        source="webhook:support",
        occurred_at=_dt.datetime.now(tz=_dt.UTC),
        tenant_key="t-1",
        user_id="u-7",
        identity_claim={
            "mode": "act_as_user",
            "user_id_from": "signal.user_id",
        },
    )
    await dispatcher.ingest(envelope)

    await processor.process_until_idle(
        queue=queue,
        signal_store=storage.signals,
        triggers=registry,
        identity_resolver=resolver,
        job_store=storage.jobs,
        job_runner=runner,
    )

    runs = await storage.jobs.list()
    assert len(runs) == 1
    [run] = runs
    assert run.status == JobStatus.SUCCEEDED
    assert run.spec.trigger_id == "ticket-triage"
    assert run.spec.visibility == "actor"  # §26 default for act_as_user
    assert run.spec.visibility_user_id == "u-7"
    # Minted user-token threaded through to the runner's invoker.
    assert "user:t-1:u-7" in run.result.get("final_response", "")

    # The resolver was used to MINT for the user, NOT a service account.
    assert any(call == ("mint_for_user", ("t-1", "u-7")) for call in resolver.calls)


# ── 3. Chat-bound act_as_user Bloom (the §25 flagship) ──────


class _TinyChatStorage:
    """Just enough to satisfy the runner's chat-binding contract."""

    def __init__(self) -> None:
        self.chats: dict[str, OrchidChatSession] = {}
        self.messages: list[dict] = []

    async def get_chat_metadata(self, chat_id: str):
        return self.chats.get(chat_id)

    async def can_write(self, chat: OrchidChatSession, auth) -> bool:
        if chat.tenant_id != auth.tenant_key:
            return False
        return chat.user_id == auth.user_id or "admin" in auth.roles

    async def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        agents_used: list[str] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        record = {
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "agents_used": list(agents_used or []),
            "metadata": dict(metadata or {}),
        }
        self.messages.append(record)
        return record


async def test_end_to_end_chat_bound_act_as_user_bloom(shared_db) -> None:
    storage: SQLiteEventStorage = shared_db["storage"]
    queue: SQLiteSignalQueue = shared_db["queue"]
    chat_storage = _TinyChatStorage()
    chat_storage.chats["C-7"] = OrchidChatSession(
        id="C-7",
        tenant_id="t-1",
        user_id="u-7",
        title="My research chat",
        created_at=_dt.datetime.now(tz=_dt.UTC),
        updated_at=_dt.datetime.now(tz=_dt.UTC),
    )

    dispatcher = OrchidSignalDispatcher(store=storage.signals, queue=queue)
    trigger = OrchidTriggerConfig(
        id="deep-research",
        on=OrchidTriggerMatchConfig(signal="research.requested"),
        emits=OrchidTriggerEmitConfig(
            agent="research",
            prompt_template="Carry out: {{payload.question}}",
            identity=ActAsUserIdentity(user_id_from="signal.user_id"),
            respect_chat_binding=True,
        ),
    )
    registry = build_registry_from_config(
        [trigger],
        known_agents={"research"},
        identity_resolver=_FixtureResolver(),
    )
    resolver = _FixtureResolver()
    runner = GraphJobRunner(invoker=_ok_invoker, chat_storage=chat_storage)
    processor = AsyncioWorkerPoolProcessor()

    envelope = SignalEnvelope(
        type="research.requested",
        payload={"question": "What's the carbon footprint of olive oil?"},
        source="internal:agent:concierge",
        occurred_at=_dt.datetime.now(tz=_dt.UTC),
        tenant_key="t-1",
        user_id="u-7",
        identity_claim={
            "mode": "act_as_user",
            "user_id_from": "signal.user_id",
        },
        chat_binding={
            "chat_id": "C-7",
            "mode": "append_final_message",
            "on_failure": "post_error",
        },
    )
    await dispatcher.ingest(envelope)

    await processor.process_until_idle(
        queue=queue,
        signal_store=storage.signals,
        triggers=registry,
        identity_resolver=resolver,
        job_store=storage.jobs,
        job_runner=runner,
    )

    [run] = await storage.jobs.list()
    assert run.status == JobStatus.SUCCEEDED
    assert run.spec.visibility == "actor"
    assert run.spec.visibility_user_id == "u-7"

    # The final AIMessage landed in the chat with bloom origin.
    assert len(chat_storage.messages) == 1
    msg = chat_storage.messages[0]
    assert msg["chat_id"] == "C-7"
    assert msg["metadata"]["origin"] == "bloom"
    assert msg["metadata"]["bloom_run_id"] == str(run.run_id)
    assert msg["metadata"]["trigger_id"] == "deep-research"


async def test_cross_user_chat_binding_is_rejected_at_runtime(
    shared_db,
) -> None:
    """Same trigger, but the signal's binding points at a chat owned
    by a DIFFERENT user.  The runner's authorization gate must
    refuse — the run finishes FAILED and no message lands in the
    other user's chat."""
    storage: SQLiteEventStorage = shared_db["storage"]
    queue: SQLiteSignalQueue = shared_db["queue"]
    chat_storage = _TinyChatStorage()
    chat_storage.chats["C-OTHER"] = OrchidChatSession(
        id="C-OTHER",
        tenant_id="t-1",
        user_id="u-OTHER",  # ← different owner
        title="Someone else's chat",
        created_at=_dt.datetime.now(tz=_dt.UTC),
        updated_at=_dt.datetime.now(tz=_dt.UTC),
    )

    dispatcher = OrchidSignalDispatcher(store=storage.signals, queue=queue)
    trigger = OrchidTriggerConfig(
        id="deep-research",
        on=OrchidTriggerMatchConfig(signal="research.requested"),
        emits=OrchidTriggerEmitConfig(
            agent="research",
            prompt_template="x",
            identity=ActAsUserIdentity(user_id_from="signal.user_id"),
            respect_chat_binding=True,
        ),
    )
    registry = build_registry_from_config(
        [trigger],
        known_agents={"research"},
        identity_resolver=_FixtureResolver(),
    )
    runner = GraphJobRunner(invoker=_ok_invoker, chat_storage=chat_storage)
    processor = AsyncioWorkerPoolProcessor()

    envelope = SignalEnvelope(
        type="research.requested",
        payload={},
        source="internal:agent:concierge",
        occurred_at=_dt.datetime.now(tz=_dt.UTC),
        tenant_key="t-1",
        user_id="u-7",  # auth ends up resolving as u-7
        identity_claim={
            "mode": "act_as_user",
            "user_id_from": "signal.user_id",
        },
        chat_binding={"chat_id": "C-OTHER"},
    )
    await dispatcher.ingest(envelope)
    await processor.process_until_idle(
        queue=queue,
        signal_store=storage.signals,
        triggers=registry,
        identity_resolver=_FixtureResolver(),
        job_store=storage.jobs,
        job_runner=runner,
    )

    [run] = await storage.jobs.list()
    assert run.status == JobStatus.FAILED
    # NO message landed in the other user's chat — §25.4 contract
    # explicitly forbids posting on_failure when binding lookup
    # itself failed.
    assert chat_storage.messages == []


async def test_session_warmer_called_before_runner(shared_db) -> None:
    """The processor calls ``session_warmer.warm_for_user(auth)``
    before invoking the runner — captured here via a tiny spy."""
    storage: SQLiteEventStorage = shared_db["storage"]
    queue: SQLiteSignalQueue = shared_db["queue"]

    warmed: list[str] = []

    class _Warmer:
        async def warm_for_user(self, auth):
            warmed.append(auth.user_id or "<sa>")

    dispatcher = OrchidSignalDispatcher(store=storage.signals, queue=queue)
    trigger = OrchidTriggerConfig(
        id="t1",
        on=OrchidTriggerMatchConfig(signal="x"),
        emits=OrchidTriggerEmitConfig(
            agent="agent",
            prompt_template="hi",
            identity=ActAsUserIdentity(user_id_from="signal.user_id"),
        ),
    )
    registry = build_registry_from_config([trigger], known_agents={"agent"}, identity_resolver=_FixtureResolver())
    runner = GraphJobRunner(invoker=_ok_invoker)
    processor = AsyncioWorkerPoolProcessor()

    await dispatcher.ingest(
        SignalEnvelope(
            type="x",
            payload={},
            source="src",
            occurred_at=_dt.datetime.now(tz=_dt.UTC),
            tenant_key="t-1",
            user_id="u-7",
            identity_claim={
                "mode": "act_as_user",
                "user_id_from": "signal.user_id",
            },
        )
    )
    await processor.process_until_idle(
        queue=queue,
        signal_store=storage.signals,
        triggers=registry,
        identity_resolver=_FixtureResolver(),
        job_store=storage.jobs,
        job_runner=runner,
        session_warmer=_Warmer(),
    )
    assert warmed == ["u-7"]
