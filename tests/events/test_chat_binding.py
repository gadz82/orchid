"""Phase-3 chat-binding tests (§25) + proactive_chat.

Coverage:

- ``ChatBinding`` Pydantic round-trip including ``extra="forbid"``.
- ``OrchidTriggerEmit`` Pydantic-level rejection of
  ``respect_chat_binding=true`` + ``service_account``.
- Pydantic-level rejection of ``proactive_chat=true`` + ``service_account``.
- Registry-level rejection of both combinations at boot.
- ``_resolve_chat_binding`` returns ``None`` when no binding present.
- ``_resolve_chat_binding`` raises
  :class:`ChatBindingTargetNotFoundError` when chat does not exist.
- ``_resolve_chat_binding`` raises
  :class:`ChatBindingForbiddenError` when auth cannot write to chat.
- Happy-path: bound run persists ``AIMessage`` to
  :class:`OrchidChatStorage` with ``metadata.origin="bloom"``.
- Failure with ``on_failure="post_error"`` appends an error message;
  ``silent`` does not.
- proactive_chat: runner creates a new chat and posts the result there.
- proactive_chat: explicit chat_binding takes precedence when both present.
- proactive_chat: graceful no-op when chat_storage is absent.
"""

from __future__ import annotations

import datetime as _dt
import uuid as _uuid

import pytest
from pydantic import ValidationError

from orchid_ai.config.schema_events import (
    ActAsUserIdentity,
    ChatBinding,
    OrchidTriggerConfig,
    OrchidTriggerEmitConfig,
    OrchidTriggerMatchConfig,
    ServiceAccountIdentity,
)
from orchid_ai.core.events.errors import (
    ChatBindingForbiddenError,
    ChatBindingTargetNotFoundError,
    JobRunnerError,
    TriggerRegistrationError,
)
from orchid_ai.core.events.job import JobRun, JobSpec, JobStatus
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.events.registry import build_registry_from_config
from orchid_ai.events.runners.graph_runner import GraphJobRunner
from orchid_ai.persistence.models import OrchidChatSession

# ── Pydantic-level ──────────────────────────────────────────


def test_chat_binding_round_trip_strict() -> None:
    cb = ChatBinding(chat_id="C-1")
    assert cb.mode == "append_final_message"
    assert cb.on_failure == "post_error"

    explicit = ChatBinding(chat_id="C-2", mode="append_with_metadata", on_failure="silent")
    assert explicit.on_failure == "silent"


def test_chat_binding_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        ChatBinding(chat_id="C-1", weird="extra")  # type: ignore[arg-type]


def test_emit_config_rejects_chat_binding_with_service_account() -> None:
    with pytest.raises(ValidationError) as exc_info:
        OrchidTriggerEmitConfig(
            agent="notifications",
            prompt_template="hi",
            identity=ServiceAccountIdentity(name="bot"),
            respect_chat_binding=True,
        )
    assert "respect_chat_binding" in str(exc_info.value)


def test_emit_config_allows_chat_binding_with_act_as_user() -> None:
    cfg = OrchidTriggerEmitConfig(
        agent="research",
        prompt_template="hi",
        identity=ActAsUserIdentity(user_id_from="signal.user_id"),
        respect_chat_binding=True,
    )
    assert cfg.respect_chat_binding is True


def test_emit_config_rejects_proactive_chat_with_service_account() -> None:
    with pytest.raises(ValidationError) as exc_info:
        OrchidTriggerEmitConfig(
            agent="notifications",
            prompt_template="hi",
            identity=ServiceAccountIdentity(name="bot"),
            proactive_chat=True,
        )
    assert "proactive_chat" in str(exc_info.value)


def test_emit_config_allows_proactive_chat_with_act_as_user() -> None:
    cfg = OrchidTriggerEmitConfig(
        agent="support",
        prompt_template="hi",
        identity=ActAsUserIdentity(user_id_from="signal.user_id"),
        proactive_chat=True,
    )
    assert cfg.proactive_chat is True


# ── Registry-level (defence in depth) ───────────────────────


def test_registry_rejects_chat_binding_with_service_account() -> None:
    """Build both Pydantic layers via ``model_construct`` so the
    registry is the one that catches the bad combo (defence-in-depth)."""
    bad_emit = OrchidTriggerEmitConfig.model_construct(
        agent="notifications",
        prompt_template="hi",
        identity=ServiceAccountIdentity(name="bot"),
        respect_chat_binding=True,
        proactive_chat=False,
        visibility=None,
    )
    bad_trigger = OrchidTriggerConfig.model_construct(
        id="bad-trigger",
        on=OrchidTriggerMatchConfig(signal="demo.event"),
        emits=bad_emit,
    )
    with pytest.raises(TriggerRegistrationError) as exc_info:
        build_registry_from_config([bad_trigger], known_agents={"notifications"})
    assert "chat" in str(exc_info.value).lower()
    assert "bad-trigger" in str(exc_info.value)


def test_registry_rejects_proactive_chat_with_service_account() -> None:
    bad_emit = OrchidTriggerEmitConfig.model_construct(
        agent="notifications",
        prompt_template="hi",
        identity=ServiceAccountIdentity(name="bot"),
        respect_chat_binding=False,
        proactive_chat=True,
        visibility=None,
    )
    bad_trigger = OrchidTriggerConfig.model_construct(
        id="bad-trigger-proactive",
        on=OrchidTriggerMatchConfig(signal="demo.event"),
        emits=bad_emit,
    )
    with pytest.raises(TriggerRegistrationError) as exc_info:
        build_registry_from_config([bad_trigger], known_agents={"notifications"})
    assert "proactive_chat" in str(exc_info.value)
    assert "bad-trigger-proactive" in str(exc_info.value)


# ── Runtime authorisation gate (_resolve_chat_binding) ──────


class _FakeChatStorage:
    """Minimal stand-in implementing the methods the runner calls."""

    def __init__(self) -> None:
        self.chats: dict[str, OrchidChatSession] = {}
        self.messages: list[dict] = []
        self.fail_append: bool = False
        self._next_id: int = 1

    async def create_chat(
        self,
        tenant_id: str,
        user_id: str,
        title: str = "",
    ) -> OrchidChatSession:
        chat_id = f"proactive-{self._next_id}"
        self._next_id += 1
        session = OrchidChatSession(
            id=chat_id,
            tenant_id=tenant_id,
            user_id=user_id,
            title=title,
            created_at=_dt.datetime.now(tz=_dt.UTC),
            updated_at=_dt.datetime.now(tz=_dt.UTC),
        )
        self.chats[chat_id] = session
        return session

    async def get_chat_metadata(self, chat_id: str) -> OrchidChatSession | None:
        return self.chats.get(chat_id)

    async def can_write(self, chat: OrchidChatSession, auth: OrchidAuthContext) -> bool:
        if chat.tenant_id != auth.tenant_key:
            return False
        if chat.user_id == auth.user_id:
            return True
        return "admin" in auth.roles

    async def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        agents_used: list[str] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        if self.fail_append:
            raise RuntimeError("simulated chat append failure")
        record = {
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "agents_used": list(agents_used or []),
            "metadata": dict(metadata or {}),
        }
        self.messages.append(record)
        return record


def _make_run(*, chat_binding: dict | None, proactive_chat: bool = False) -> JobRun:
    spec = JobSpec(
        trigger_id="t1",
        signal_id=_uuid.uuid4(),
        agent_name="research",
        prompt="Run the deep-research task",
        identity_claim={"mode": "act_as_user", "user_id_from": "signal.user_id"},
        correlation_id="corr",
        parallelism_key="user:t-1:u-7",
        visibility="actor",
        visibility_user_id="u-7",
        chat_binding=chat_binding,
        proactive_chat=proactive_chat,
    )
    return JobRun(
        run_id=_uuid.uuid4(),
        spec=spec,
        attempt_number=1,
        status=JobStatus.RUNNING,
        queued_at=_dt.datetime.now(tz=_dt.UTC),
        started_at=_dt.datetime.now(tz=_dt.UTC),
    )


async def _ok_invoker(run: JobRun, auth) -> dict:
    return {"final_response": "All done — here's the summary.", "items": []}


async def _failing_invoker(run: JobRun, auth) -> dict:
    raise RuntimeError("graph blew up")


async def test_no_binding_returns_none_and_no_chat_writes() -> None:
    storage = _FakeChatStorage()
    runner = GraphJobRunner(invoker=_ok_invoker, chat_storage=storage)

    run = _make_run(chat_binding=None)
    auth = OrchidAuthContext(access_token="t", tenant_key="t-1", user_id="u-7")
    await runner.run(run, auth=auth)

    assert run.status == JobStatus.SUCCEEDED
    assert storage.messages == []  # nothing landed in any chat


async def test_target_chat_missing_raises_target_not_found() -> None:
    storage = _FakeChatStorage()  # no chats added
    runner = GraphJobRunner(invoker=_ok_invoker, chat_storage=storage)

    run = _make_run(chat_binding={"chat_id": "C-missing"})
    auth = OrchidAuthContext(access_token="t", tenant_key="t-1", user_id="u-7")

    with pytest.raises(JobRunnerError) as exc_info:
        await runner.run(run, auth=auth)
    assert isinstance(exc_info.value.__cause__, ChatBindingTargetNotFoundError)
    assert exc_info.value.retryable is False
    # Per §25.4: no on_failure post when binding lookup itself failed.
    assert storage.messages == []
    assert run.status == JobStatus.FAILED


async def test_cross_user_chat_binding_raises_forbidden() -> None:
    storage = _FakeChatStorage()
    storage.chats["C-OWNED-BY-OTHER"] = OrchidChatSession(
        id="C-OWNED-BY-OTHER",
        tenant_id="t-1",
        user_id="u-OTHER",
        title="Other user's chat",
        created_at=_dt.datetime.now(tz=_dt.UTC),
        updated_at=_dt.datetime.now(tz=_dt.UTC),
    )
    runner = GraphJobRunner(invoker=_ok_invoker, chat_storage=storage)

    run = _make_run(chat_binding={"chat_id": "C-OWNED-BY-OTHER"})
    auth = OrchidAuthContext(access_token="t", tenant_key="t-1", user_id="u-7")

    with pytest.raises(JobRunnerError) as exc_info:
        await runner.run(run, auth=auth)
    assert isinstance(exc_info.value.__cause__, ChatBindingForbiddenError)
    # Per §25.4: no on_failure post when binding lookup failed.
    assert storage.messages == []


async def test_happy_path_persists_final_aimessage_with_bloom_metadata() -> None:
    storage = _FakeChatStorage()
    storage.chats["C-7"] = OrchidChatSession(
        id="C-7",
        tenant_id="t-1",
        user_id="u-7",
        title="My chat",
        created_at=_dt.datetime.now(tz=_dt.UTC),
        updated_at=_dt.datetime.now(tz=_dt.UTC),
    )
    runner = GraphJobRunner(invoker=_ok_invoker, chat_storage=storage)

    run = _make_run(chat_binding={"chat_id": "C-7"})
    auth = OrchidAuthContext(access_token="t", tenant_key="t-1", user_id="u-7")
    await runner.run(run, auth=auth)

    assert run.status == JobStatus.SUCCEEDED
    [appended] = storage.messages
    assert appended["chat_id"] == "C-7"
    assert appended["role"] == "assistant"
    assert "All done" in appended["content"]
    md = appended["metadata"]
    assert md["origin"] == "bloom"
    assert md["bloom_run_id"] == str(run.run_id)
    assert md["trigger_id"] == "t1"


async def test_happy_path_with_metadata_mode() -> None:
    storage = _FakeChatStorage()
    storage.chats["C-7"] = OrchidChatSession(
        id="C-7",
        tenant_id="t-1",
        user_id="u-7",
        title="My chat",
        created_at=_dt.datetime.now(tz=_dt.UTC),
        updated_at=_dt.datetime.now(tz=_dt.UTC),
    )
    runner = GraphJobRunner(invoker=_ok_invoker, chat_storage=storage)
    run = _make_run(chat_binding={"chat_id": "C-7", "mode": "append_with_metadata"})
    auth = OrchidAuthContext(access_token="t", tenant_key="t-1", user_id="u-7")
    await runner.run(run, auth=auth)
    [appended] = storage.messages
    assert "bloom_metadata" in appended["metadata"]


async def test_failure_with_post_error_appends_error_message() -> None:
    storage = _FakeChatStorage()
    storage.chats["C-7"] = OrchidChatSession(
        id="C-7",
        tenant_id="t-1",
        user_id="u-7",
        title="My chat",
        created_at=_dt.datetime.now(tz=_dt.UTC),
        updated_at=_dt.datetime.now(tz=_dt.UTC),
    )
    runner = GraphJobRunner(invoker=_failing_invoker, chat_storage=storage)
    run = _make_run(chat_binding={"chat_id": "C-7", "on_failure": "post_error"})
    auth = OrchidAuthContext(access_token="t", tenant_key="t-1", user_id="u-7")
    with pytest.raises(JobRunnerError):
        await runner.run(run, auth=auth)
    [appended] = storage.messages
    assert appended["metadata"]["origin"] == "bloom"
    assert appended["metadata"]["status"] == "failed"


async def test_failure_with_silent_does_not_append() -> None:
    storage = _FakeChatStorage()
    storage.chats["C-7"] = OrchidChatSession(
        id="C-7",
        tenant_id="t-1",
        user_id="u-7",
        title="My chat",
        created_at=_dt.datetime.now(tz=_dt.UTC),
        updated_at=_dt.datetime.now(tz=_dt.UTC),
    )
    runner = GraphJobRunner(invoker=_failing_invoker, chat_storage=storage)
    run = _make_run(chat_binding={"chat_id": "C-7", "on_failure": "silent"})
    auth = OrchidAuthContext(access_token="t", tenant_key="t-1", user_id="u-7")
    with pytest.raises(JobRunnerError):
        await runner.run(run, auth=auth)
    assert storage.messages == []  # silent path is silent


async def test_admin_role_allows_cross_user_chat_write() -> None:
    """An admin-flagged auth can write to any chat in the tenant —
    matches the default ``can_write`` rule.  Operationally relevant
    when ops triggers run as the admin role to drop messages into
    user chats deliberately."""
    storage = _FakeChatStorage()
    storage.chats["C-OWNED-BY-OTHER"] = OrchidChatSession(
        id="C-OWNED-BY-OTHER",
        tenant_id="t-1",
        user_id="u-OTHER",
        title="x",
        created_at=_dt.datetime.now(tz=_dt.UTC),
        updated_at=_dt.datetime.now(tz=_dt.UTC),
    )
    runner = GraphJobRunner(invoker=_ok_invoker, chat_storage=storage)
    run = _make_run(chat_binding={"chat_id": "C-OWNED-BY-OTHER"})
    auth = OrchidAuthContext(
        access_token="t",
        tenant_key="t-1",
        user_id="u-admin",
        roles={"admin"},
    )
    await runner.run(run, auth=auth)
    assert run.status == JobStatus.SUCCEEDED
    assert len(storage.messages) == 1


async def test_chat_storage_failure_does_not_fail_the_run() -> None:
    """The Bloom result is already persisted to ``job_runs.result``;
    a chat-write failure logs but doesn't flip the run to FAILED."""
    storage = _FakeChatStorage()
    storage.chats["C-7"] = OrchidChatSession(
        id="C-7",
        tenant_id="t-1",
        user_id="u-7",
        title="x",
        created_at=_dt.datetime.now(tz=_dt.UTC),
        updated_at=_dt.datetime.now(tz=_dt.UTC),
    )
    storage.fail_append = True
    runner = GraphJobRunner(invoker=_ok_invoker, chat_storage=storage)
    run = _make_run(chat_binding={"chat_id": "C-7"})
    auth = OrchidAuthContext(access_token="t", tenant_key="t-1", user_id="u-7")
    await runner.run(run, auth=auth)
    assert run.status == JobStatus.SUCCEEDED
    assert run.result == {
        "final_response": "All done — here's the summary.",
        "items": [],
    }


# ── proactive_chat ──────────────────────────────────────────


async def test_proactive_chat_creates_chat_and_posts_result() -> None:
    """When ``proactive_chat=true`` and no binding on the signal, the
    runner creates a new chat and persists the result into it."""
    storage = _FakeChatStorage()
    runner = GraphJobRunner(invoker=_ok_invoker, chat_storage=storage)

    run = _make_run(chat_binding=None, proactive_chat=True)
    auth = OrchidAuthContext(access_token="t", tenant_key="t-1", user_id="u-7")
    await runner.run(run, auth=auth)

    assert run.status == JobStatus.SUCCEEDED
    # A new chat was created for the user.
    assert len(storage.chats) == 1
    [session] = storage.chats.values()
    assert session.tenant_id == "t-1"
    assert session.user_id == "u-7"
    # The first non-empty line of the prompt becomes the title.
    assert "deep-research" in session.title.lower()
    # The result was posted to that chat.
    [msg] = storage.messages
    assert msg["chat_id"] == session.id
    assert msg["role"] == "assistant"
    assert "All done" in msg["content"]
    assert msg["metadata"]["origin"] == "bloom"


async def test_proactive_chat_explicit_binding_takes_precedence() -> None:
    """When the signal carries an explicit ``chat_binding`` AND the
    trigger has ``proactive_chat=true``, the existing chat wins."""
    storage = _FakeChatStorage()
    storage.chats["C-existing"] = OrchidChatSession(
        id="C-existing",
        tenant_id="t-1",
        user_id="u-7",
        title="Existing chat",
        created_at=_dt.datetime.now(tz=_dt.UTC),
        updated_at=_dt.datetime.now(tz=_dt.UTC),
    )
    runner = GraphJobRunner(invoker=_ok_invoker, chat_storage=storage)

    # proactive_chat=True BUT an explicit binding is also present.
    run = _make_run(chat_binding={"chat_id": "C-existing"}, proactive_chat=True)
    auth = OrchidAuthContext(access_token="t", tenant_key="t-1", user_id="u-7")
    await runner.run(run, auth=auth)

    assert run.status == JobStatus.SUCCEEDED
    # No new chat was created — only the pre-existing one.
    assert list(storage.chats.keys()) == ["C-existing"]
    [msg] = storage.messages
    assert msg["chat_id"] == "C-existing"


async def test_proactive_chat_no_storage_is_noop() -> None:
    """Missing chat_storage logs a warning but the run still succeeds."""
    runner = GraphJobRunner(invoker=_ok_invoker, chat_storage=None)

    run = _make_run(chat_binding=None, proactive_chat=True)
    auth = OrchidAuthContext(access_token="t", tenant_key="t-1", user_id="u-7")
    await runner.run(run, auth=auth)

    assert run.status == JobStatus.SUCCEEDED  # run succeeds despite no storage


async def test_emit_signal_self_outside_chat_run_raises() -> None:
    """``chat_id='self'`` requires a current chat — when emitted
    outside one (e.g. from a background producer), it must raise."""
    from orchid_ai.core.agent import OrchidAgent
    from orchid_ai.core.events.dispatcher import OrchidSignalDispatcher
    from orchid_ai.events.producers.internal import DispatcherSignalEmitter
    from orchid_ai.events.queues.inmemory import (
        InMemorySignalQueue,
        InMemorySignalStore,
    )

    class _Agent(OrchidAgent):
        @property
        def name(self) -> str:
            return "test"

        @property
        def description(self) -> str:
            return "x"

        async def run(self, state):
            return state

    queue = InMemorySignalQueue()
    store = InMemorySignalStore()
    dispatcher = OrchidSignalDispatcher(store=store, queue=queue)

    agent = _Agent(reader=None)  # type: ignore[arg-type]
    agent._signal_emitter = DispatcherSignalEmitter(dispatcher)
    agent._current_auth = OrchidAuthContext(access_token="t", tenant_key="t-1", user_id="u-7")
    agent._current_chat_id = None  # outside a chat run

    with pytest.raises(RuntimeError, match="chat_id='self'"):
        await agent.emit_signal("demo.event", {"x": 1}, chat_id="self")
