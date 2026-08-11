"""Tests for ``orchid_ai.orchid.Orchid``."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphInterrupt

from orchid_ai import Orchid, OrchidInvokeResult, OrchidPendingApproval
from orchid_ai.config.schema import OrchidAgentConfig, OrchidAgentsConfig, OrchidRAGConfig
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.persistence.sqlite import OrchidSQLiteChatStorage
from orchid_ai.runtime import OrchidRuntime

# ── Test doubles ────────────────────────────────────────────────


class _FakeInterrupt:
    """Stand-in for langgraph's Interrupt instances."""

    def __init__(self, value, id_="int-1"):
        self.value = value
        self.id = id_


def _fake_graph(*, response="hi from the agent", agents=None, interrupts=None):
    """Build a MagicMock graph compatible with client.invoke / resume."""
    graph = MagicMock()

    async def ainvoke(initial_state, config=None):
        if interrupts is not None:
            raise GraphInterrupt(interrupts)
        prior = initial_state.get("messages", []) if isinstance(initial_state, dict) else []
        return {
            "final_response": response,
            "active_agents": list(agents or []),
            "messages": prior + [AIMessage(content=response)],
            "mcp_context": {"agent_x": {"tool": "data"}},
            "rag_context": {"agent_x": [{"text": "chunk"}]},
        }

    graph.ainvoke.side_effect = ainvoke
    return graph


@pytest.fixture
def minimal_config():
    return OrchidAgentsConfig(
        agents={
            "helper": OrchidAgentConfig(
                description="Answers questions.",
                prompt="You are a helper.",
                rag=OrchidRAGConfig(enabled=False),
            )
        }
    )


@pytest.fixture
def noop_runtime():
    return OrchidRuntime(default_model="ollama/llama3.2")


# ── OrchidInvokeResult / OrchidPendingApproval dataclasses ─────────────────


class TestDataclasses:
    def test_invoke_result_defaults(self):
        r = OrchidInvokeResult(response="ok", chat_id="c-1")
        assert r.agents_used == []
        assert r.messages == []
        assert r.interrupted is False
        assert r.approvals_needed == []
        assert r.mcp_context == {}
        assert r.rag_context == {}

    def test_pending_approval_fields(self):
        p = OrchidPendingApproval(tool="t", args={"a": 1}, agent="x", interrupt_id="id-1")
        assert p.tool == "t"
        assert p.args == {"a": 1}
        assert p.agent == "x"
        assert p.interrupt_id == "id-1"


# ── Basic invoke flow ──────────────────────────────────────────


class TestInvoke:
    @pytest.mark.asyncio
    async def test_generates_chat_id_when_missing(self, minimal_config, noop_runtime):
        with patch("orchid_ai.orchid.lifecycle.build_graph") as build:
            build.return_value = _fake_graph(response="hello", agents=["helper"])
            client = Orchid(config=minimal_config, runtime=noop_runtime)

            result = await client.invoke("hi", user_id="alice", persist=False)

        # chat_id must be a valid UUID4
        uuid.UUID(result.chat_id)
        assert result.response == "hello"
        assert result.agents_used == ["helper"]
        assert result.interrupted is False

    @pytest.mark.asyncio
    async def test_passes_thread_id_via_config(self, minimal_config, noop_runtime):
        chat_id = "chat-xyz"
        with patch("orchid_ai.orchid.lifecycle.build_graph") as build:
            graph = _fake_graph()
            build.return_value = graph
            client = Orchid(config=minimal_config, runtime=noop_runtime)

            await client.invoke("hi", chat_id=chat_id, persist=False)

        call_kwargs = graph.ainvoke.call_args.kwargs
        configurable = call_kwargs["config"]["configurable"]
        assert configurable["thread_id"] == chat_id
        # Auth now travels in the config, not the state.
        assert isinstance(configurable["auth_context"], OrchidAuthContext)

    @pytest.mark.asyncio
    async def test_auth_fallback_is_auth_context(self, minimal_config, noop_runtime):
        with patch("orchid_ai.orchid.lifecycle.build_graph") as build:
            graph = _fake_graph()
            build.return_value = graph
            client = Orchid(config=minimal_config, runtime=noop_runtime)

            await client.invoke(
                "hi",
                chat_id="c-1",
                user_id="u-42",
                tenant_id="acme",
                access_token="T",
                persist=False,
            )

        auth_ctx = graph.ainvoke.call_args.kwargs["config"]["configurable"]["auth_context"]
        assert isinstance(auth_ctx, OrchidAuthContext)
        assert auth_ctx.user_id == "u-42"
        assert auth_ctx.tenant_key == "acme"
        assert auth_ctx.access_token == "T"

    @pytest.mark.asyncio
    async def test_explicit_auth_wins(self, minimal_config, noop_runtime):
        with patch("orchid_ai.orchid.lifecycle.build_graph") as build:
            graph = _fake_graph()
            build.return_value = graph
            client = Orchid(config=minimal_config, runtime=noop_runtime)

            custom = OrchidAuthContext(access_token="X", user_id="u-orig", tenant_key="org")
            await client.invoke(
                "hi",
                chat_id="c-1",
                user_id="ignored",
                tenant_id="ignored",
                auth=custom,
                persist=False,
            )

        assert graph.ainvoke.call_args.kwargs["config"]["configurable"]["auth_context"] is custom

    @pytest.mark.asyncio
    async def test_explicit_history_passed_through(self, minimal_config, noop_runtime):
        with patch("orchid_ai.orchid.lifecycle.build_graph") as build:
            graph = _fake_graph()
            build.return_value = graph
            client = Orchid(config=minimal_config, runtime=noop_runtime)

            history = [HumanMessage(content="prev"), AIMessage(content="ok")]
            await client.invoke(
                "next",
                chat_id="c-1",
                history=history,
                persist=False,
            )

        messages = graph.ainvoke.call_args.args[0]["messages"]
        assert len(messages) == 3
        assert messages[0].content == "prev"
        assert messages[1].content == "ok"
        assert messages[2].content == "next"


# ── Interrupt handling ─────────────────────────────────────────


class TestInterrupts:
    @pytest.mark.asyncio
    async def test_graph_interrupt_returns_result(self, minimal_config, noop_runtime):
        pending = _FakeInterrupt({"tool": "book", "args": {"time": "8pm"}, "agent": "bookings"})
        with patch("orchid_ai.orchid.lifecycle.build_graph") as build:
            build.return_value = _fake_graph(interrupts=[pending])
            client = Orchid(config=minimal_config, runtime=noop_runtime)

            result = await client.invoke("book a table", chat_id="c-1", persist=False)

        assert result.interrupted is True
        assert result.response == ""
        assert len(result.approvals_needed) == 1
        approval = result.approvals_needed[0]
        assert approval.tool == "book"
        assert approval.args == {"time": "8pm"}
        assert approval.agent == "bookings"
        assert approval.interrupt_id == "int-1"

    @pytest.mark.asyncio
    async def test_non_dict_interrupt_value(self, minimal_config, noop_runtime):
        pending = _FakeInterrupt("raw-text", id_="abc")
        with patch("orchid_ai.orchid.lifecycle.build_graph") as build:
            build.return_value = _fake_graph(interrupts=[pending])
            client = Orchid(config=minimal_config, runtime=noop_runtime)

            result = await client.invoke("hi", chat_id="c-1", persist=False)

        assert result.interrupted
        assert result.approvals_needed[0].tool == "raw-text"


# ── Resume ─────────────────────────────────────────────────────


class TestResume:
    @pytest.mark.asyncio
    async def test_resume_requires_checkpointer(self, minimal_config, noop_runtime):
        with patch("orchid_ai.orchid.lifecycle.build_graph") as build:
            build.return_value = _fake_graph()
            client = Orchid(config=minimal_config, runtime=noop_runtime)

            with pytest.raises(RuntimeError, match="checkpointer"):
                await client.resume("c-1")

    @pytest.mark.asyncio
    async def test_resume_sends_command(self, minimal_config):
        from langgraph.checkpoint.memory import MemorySaver

        runtime = OrchidRuntime(
            default_model="ollama/llama3.2",
            checkpointer=MemorySaver(),
        )

        with patch("orchid_ai.orchid.lifecycle.build_graph") as build:
            graph = _fake_graph(response="booked!", agents=["bookings"])
            build.return_value = graph
            client = Orchid(config=minimal_config, runtime=runtime)

            result = await client.resume("c-1", approved=True, persist=False)

        assert result.response == "booked!"
        # The first positional arg must be a langgraph Command(resume={...})
        cmd = graph.ainvoke.call_args.args[0]
        assert hasattr(cmd, "resume")
        assert cmd.resume == {"approved": True}

    @pytest.mark.asyncio
    async def test_resume_injects_auth_via_config(self, minimal_config):
        """Resume re-supplies a fresh auth via config — never read from the checkpoint."""
        from langgraph.checkpoint.memory import MemorySaver

        runtime = OrchidRuntime(default_model="ollama/llama3.2", checkpointer=MemorySaver())
        auth = OrchidAuthContext(access_token="fresh", tenant_key="acme", user_id="u-1")

        with patch("orchid_ai.orchid.lifecycle.build_graph") as build:
            graph = _fake_graph(response="done")
            build.return_value = graph
            client = Orchid(config=minimal_config, runtime=runtime)

            await client.resume("c-1", auth=auth, approved=True, persist=False)

        config = graph.ainvoke.call_args.kwargs["config"]
        assert config["configurable"]["auth_context"] is auth
        assert config["configurable"]["thread_id"] == "c-1"


# ── Persistence ────────────────────────────────────────────────


class TestPersistence:
    @pytest.mark.asyncio
    async def test_persists_messages_when_enabled(self, minimal_config, noop_runtime):
        chat_repo = OrchidSQLiteChatStorage(dsn=":memory:")
        await chat_repo.init_db()

        with patch("orchid_ai.orchid.lifecycle.build_graph") as build:
            build.return_value = _fake_graph(response="the answer", agents=["helper"])
            client = Orchid(
                config=minimal_config,
                runtime=noop_runtime,
                chat_repo=chat_repo,
            )

            # No chat_id supplied → client creates a row for us
            result = await client.invoke("What's up?", user_id="u-1", tenant_id="t-1")

        assert result.chat_id  # backend-generated UUID
        rows = await chat_repo.get_messages(result.chat_id)
        assert len(rows) == 2
        assert rows[0].role == "user"
        assert rows[0].content == "What's up?"
        assert rows[1].role == "assistant"
        assert rows[1].content == "the answer"

        await chat_repo.close()

    @pytest.mark.asyncio
    async def test_persist_false_skips_storage(self, minimal_config, noop_runtime):
        chat_repo = OrchidSQLiteChatStorage(dsn=":memory:")
        await chat_repo.init_db()

        with patch("orchid_ai.orchid.lifecycle.build_graph") as build:
            build.return_value = _fake_graph(response="hi")
            client = Orchid(
                config=minimal_config,
                runtime=noop_runtime,
                chat_repo=chat_repo,
            )

            result = await client.invoke(
                "msg",
                chat_id="c-ephemeral",
                user_id="u",
                persist=False,
            )

        assert result.response == "hi"
        rows = await chat_repo.get_messages("c-ephemeral")
        assert rows == []

        await chat_repo.close()

    @pytest.mark.asyncio
    async def test_history_loaded_from_repo_when_chat_exists(self, minimal_config, noop_runtime):
        chat_repo = OrchidSQLiteChatStorage(dsn=":memory:")
        await chat_repo.init_db()

        chat = await chat_repo.create_chat(tenant_id="t", user_id="u", title="seed")
        await chat_repo.add_message(chat.id, "user", "earlier question")
        await chat_repo.add_message(chat.id, "assistant", "earlier answer")

        with patch("orchid_ai.orchid.lifecycle.build_graph") as build:
            graph = _fake_graph(response="fresh response")
            build.return_value = graph
            client = Orchid(
                config=minimal_config,
                runtime=noop_runtime,
                chat_repo=chat_repo,
            )

            await client.invoke("follow-up", chat_id=chat.id, user_id="u")

        messages = graph.ainvoke.call_args.args[0]["messages"]
        # 2 history messages + 1 new user message
        assert len(messages) == 3
        assert messages[0].content == "earlier question"
        assert messages[1].content == "earlier answer"
        assert messages[2].content == "follow-up"

        await chat_repo.close()

    @pytest.mark.asyncio
    async def test_checkpointer_skips_history_loading(self, minimal_config):
        from langgraph.checkpoint.memory import MemorySaver

        runtime = OrchidRuntime(
            default_model="ollama/llama3.2",
            checkpointer=MemorySaver(),
        )

        chat_repo = OrchidSQLiteChatStorage(dsn=":memory:")
        await chat_repo.init_db()
        chat = await chat_repo.create_chat(tenant_id="t", user_id="u")
        await chat_repo.add_message(chat.id, "user", "old")
        await chat_repo.add_message(chat.id, "assistant", "older")

        with patch("orchid_ai.orchid.lifecycle.build_graph") as build:
            graph = _fake_graph(response="new")
            build.return_value = graph
            client = Orchid(
                config=minimal_config,
                runtime=runtime,
                chat_repo=chat_repo,
            )

            await client.invoke("q", chat_id=chat.id, user_id="u")

        # Only the new user message should be in messages;
        # the graph owns prior state via the checkpointer.
        messages = graph.ainvoke.call_args.args[0]["messages"]
        assert len(messages) == 1
        assert messages[0].content == "q"

        await chat_repo.close()


# ── Lifecycle / close ──────────────────────────────────────────


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, minimal_config, noop_runtime):
        with patch("orchid_ai.orchid.lifecycle.build_graph") as build:
            build.return_value = _fake_graph()
            client = Orchid(config=minimal_config, runtime=noop_runtime)

        await client.close()
        await client.close()  # must not raise

    @pytest.mark.asyncio
    async def test_invoke_after_close_raises(self, minimal_config, noop_runtime):
        with patch("orchid_ai.orchid.lifecycle.build_graph") as build:
            build.return_value = _fake_graph()
            client = Orchid(config=minimal_config, runtime=noop_runtime)

        await client.close()
        with pytest.raises(RuntimeError, match="closed"):
            await client.invoke("hi", persist=False)

    @pytest.mark.asyncio
    async def test_async_context_manager(self, minimal_config, noop_runtime):
        with patch("orchid_ai.orchid.lifecycle.build_graph") as build:
            build.return_value = _fake_graph(response="ctx")
            async with Orchid(config=minimal_config, runtime=noop_runtime) as client:
                result = await client.invoke("hi", persist=False)
                assert result.response == "ctx"

    @pytest.mark.asyncio
    async def test_owned_resources_are_closed(self, minimal_config, noop_runtime):
        chat_repo = MagicMock()

        async def _aclose():
            pass

        chat_repo.close.return_value = _aclose()

        with patch("orchid_ai.orchid.lifecycle.build_graph") as build:
            build.return_value = _fake_graph()
            client = Orchid(
                config=minimal_config,
                runtime=noop_runtime,
                chat_repo=chat_repo,
                _owns_resources=True,
            )

        await client.close()
        chat_repo.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_unowned_resources_not_closed(self, minimal_config, noop_runtime):
        chat_repo = MagicMock()
        chat_repo.close = MagicMock()

        with patch("orchid_ai.orchid.lifecycle.build_graph") as build:
            build.return_value = _fake_graph()
            client = Orchid(
                config=minimal_config,
                runtime=noop_runtime,
                chat_repo=chat_repo,
                _owns_resources=False,
            )

        await client.close()
        chat_repo.close.assert_not_called()


# ── Public accessors ───────────────────────────────────────────


class TestAccessors:
    @pytest.mark.asyncio
    async def test_accessors_return_wired_values(self, minimal_config, noop_runtime):
        with patch("orchid_ai.orchid.lifecycle.build_graph") as build:
            build.return_value = _fake_graph()
            client = Orchid(config=minimal_config, runtime=noop_runtime)

        assert client.config is minimal_config
        assert client.runtime is noop_runtime
        assert client.graph is not None
        assert client.chat_repo is None
        # ``mcp_token_store`` must also be exposed on the public facade —
        # both ``orchid-cli`` (pre-flight MCP auth check) and ``orchid-api``
        # (``get_mcp_token_store_optional`` FastAPI dep) dereference it.
        assert client.mcp_token_store is None

    @pytest.mark.asyncio
    async def test_mcp_token_store_accessor_returns_injected_store(self, minimal_config, noop_runtime):
        """Regression for the 1.3.2 release that shipped without this accessor."""
        sentinel = object()
        with patch("orchid_ai.orchid.lifecycle.build_graph") as build:
            build.return_value = _fake_graph()
            client = Orchid(
                config=minimal_config,
                runtime=noop_runtime,
                mcp_token_store=sentinel,  # type: ignore[arg-type]
            )
        assert client.mcp_token_store is sentinel
