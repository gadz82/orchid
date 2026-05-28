"""Tests for the shared helpers on ``Orchid``.

Exercises ``_prepare_invocation``, ``_result_from_graph_output``, and
the static ``_interrupt_to_result`` — the common plumbing that
``invoke`` / ``stream`` / ``resume`` all depend on.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphInterrupt

from orchid_ai import Orchid, OrchidPendingApproval
from orchid_ai.config.schema import OrchidAgentConfig, OrchidAgentsConfig, OrchidRAGConfig
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.runtime import OrchidRuntime


class _FakeInterrupt:
    def __init__(self, value, id_="id-1"):
        self.value = value
        self.id = id_


@pytest.fixture
def client():
    config = OrchidAgentsConfig(
        agents={
            "helper": OrchidAgentConfig(
                description="t",
                prompt="t",
                rag=OrchidRAGConfig(enabled=False),
            )
        }
    )
    runtime = OrchidRuntime(default_model="ollama/llama3.2")
    with patch("orchid_ai.orchid.lifecycle.build_graph") as build:
        build.return_value = MagicMock()
        yield Orchid(config=config, runtime=runtime)


class TestPrepareInvocation:
    @pytest.mark.asyncio
    async def test_generates_chat_id_when_missing_and_no_auto_create(self, client):
        prepared = await client._prepare_invocation(
            message="hi",
            chat_id=None,
            user_id="u",
            tenant_id="t",
            access_token="",
            auth=None,
            history=None,
            persist=False,
        )
        assert prepared.chat_id  # uuid
        assert prepared.graph_config == {"configurable": {"thread_id": prepared.chat_id}}
        assert isinstance(prepared.auth_ctx, OrchidAuthContext)
        assert prepared.auth_ctx.user_id == "u"

    @pytest.mark.asyncio
    async def test_respects_explicit_chat_id(self, client):
        prepared = await client._prepare_invocation(
            message="hi",
            chat_id="stable-id",
            user_id="u",
            tenant_id="t",
            access_token="",
            auth=None,
            history=None,
            persist=False,
        )
        assert prepared.chat_id == "stable-id"

    @pytest.mark.asyncio
    async def test_uses_explicit_auth_when_provided(self, client):
        my_auth = OrchidAuthContext(access_token="T", user_id="real", tenant_key="org")
        prepared = await client._prepare_invocation(
            message="hi",
            chat_id="c",
            user_id="ignored",
            tenant_id="ignored",
            access_token="",
            auth=my_auth,
            history=None,
            persist=False,
        )
        assert prepared.auth_ctx is my_auth

    @pytest.mark.asyncio
    async def test_history_is_prepended_in_state(self, client):
        history = [HumanMessage(content="earlier")]
        prepared = await client._prepare_invocation(
            message="now",
            chat_id="c",
            user_id="u",
            tenant_id="t",
            access_token="",
            auth=None,
            history=history,
            persist=False,
        )
        msgs = prepared.state["messages"]
        assert len(msgs) == 2
        assert msgs[0].content == "earlier"
        assert msgs[1].content == "now"


class TestResultFromGraphOutput:
    def test_maps_all_fields(self, client):
        result = client._result_from_graph_output(
            {
                "final_response": "done",
                "active_agents": ["a"],
                "messages": [AIMessage(content="done")],
                "mcp_context": {"a": 1},
                "rag_context": {"a": [{"k": "v"}]},
            },
            chat_id="abc",
        )
        assert result.response == "done"
        assert result.chat_id == "abc"
        assert result.agents_used == ["a"]
        assert len(result.messages) == 1
        assert result.mcp_context == {"a": 1}
        assert result.rag_context == {"a": [{"k": "v"}]}
        assert result.interrupted is False

    def test_missing_fields_become_empty(self, client):
        result = client._result_from_graph_output({}, chat_id="c")
        assert result.response == ""
        assert result.agents_used == []
        assert result.messages == []
        assert result.mcp_context == {}
        assert result.rag_context == {}


class TestInterruptToResult:
    def test_dict_interrupt_extracts_fields(self):
        exc = GraphInterrupt(
            [
                _FakeInterrupt(
                    {"tool": "t1", "args": {"a": 1}, "agent": "x"},
                    id_="int-1",
                ),
            ]
        )
        result = Orchid._interrupt_to_result(exc, chat_id="c")
        assert result.interrupted is True
        assert result.chat_id == "c"
        assert len(result.approvals_needed) == 1
        approval = result.approvals_needed[0]
        assert isinstance(approval, OrchidPendingApproval)
        assert approval.tool == "t1"
        assert approval.args == {"a": 1}
        assert approval.agent == "x"
        assert approval.interrupt_id == "int-1"

    def test_non_dict_interrupt_stringifies(self):
        exc = GraphInterrupt([_FakeInterrupt("raw-text", id_="id-x")])
        result = Orchid._interrupt_to_result(exc, chat_id="c")
        assert result.approvals_needed[0].tool == "raw-text"
        assert result.approvals_needed[0].args == {}

    def test_empty_interrupt_list(self):
        exc = GraphInterrupt([])
        result = Orchid._interrupt_to_result(exc, chat_id="c")
        assert result.approvals_needed == []
