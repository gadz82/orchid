"""Tests for the ``persist_memory`` opt-out on the agent run context.

The default stays ``True`` so every existing chat turn keeps writing
conversation memory. Pipeline agent jobs set it to ``False`` and expect no
memory write at all.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from orchid_ai.agents.generic_agent import GenericAgent
from orchid_ai.agents.memory import OrchidInMemoryConversationMemory
from orchid_ai.agents.memory_rag import OrchidRAGConversationMemory
from orchid_ai.agents.skill_executor import SkillExecutor
from orchid_ai.config.schema import OrchidAgentConfig, OrchidRAGConfig
from orchid_ai.core.agent import OrchidAgent, OrchidAgentRunContext
from orchid_ai.core.agent import current_run_context as module_current_run_context
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.graph.agent_executor import OrchidAgentExecutor


class _RecordingRAGMemory(OrchidRAGConversationMemory):
    """RAG memory double that records turns instead of writing vectors."""

    def __init__(self) -> None:
        self.turns: list[dict[str, Any]] = []

    async def store_conversation_turn(
        self,
        chat_id: str,
        tenant_id: str,
        user_id: str,
        turn: dict[str, str],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.turns.append(
            {
                "chat_id": chat_id,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "turn": turn,
                "metadata": metadata,
            }
        )


def _agent(memory: _RecordingRAGMemory | None) -> GenericAgent:
    agent = GenericAgent(
        config=OrchidAgentConfig(name="a", description="agent", prompt="prompt"),
        reader=None,  # type: ignore[arg-type]
    )
    agent._summary_config = {"memory": memory} if memory is not None else None
    return agent


def _state() -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content="question", id="m-1")],
        "chat_id": "c-1",
    }


def _auth() -> OrchidAuthContext:
    return OrchidAuthContext(access_token="tok", tenant_key="t-1", user_id="u-1")


def test_persist_memory_defaults_to_true() -> None:
    assert OrchidAgentRunContext().persist_memory is True
    assert OrchidAgentRunContext(persist_memory=False).persist_memory is False


def test_current_run_context_is_none_without_a_binding() -> None:
    agent = _agent(None)
    assert agent.current_run_context() is None
    assert module_current_run_context() is None


def test_current_run_context_returns_the_whole_context() -> None:
    agent = _agent(None)
    context = OrchidAgentRunContext(auth=_auth(), chat_id="c-1", message_id="m-1", persist_memory=False)

    token = agent.set_run_context(context)
    try:
        assert agent.current_run_context() is context
        assert module_current_run_context() is context
        assert agent.current_run_context().persist_memory is False
    finally:
        agent.reset_run_context(token)

    assert agent.current_run_context() is None
    assert module_current_run_context() is None


@pytest.mark.asyncio
async def test_default_context_writes_conversation_memory() -> None:
    memory = _RecordingRAGMemory()
    agent = _agent(memory)
    context = OrchidAgentRunContext(auth=_auth(), chat_id="c-1", message_id="m-1")

    token = agent.set_run_context(context)
    try:
        await agent._store_agent_turn(_state(), "answer")
    finally:
        agent.reset_run_context(token)

    assert [entry["turn"] for entry in memory.turns] == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    assert all(entry["chat_id"] == "c-1" for entry in memory.turns)
    assert all(entry["tenant_id"] == "t-1" for entry in memory.turns)
    assert all(entry["user_id"] == "u-1" for entry in memory.turns)


@pytest.mark.asyncio
async def test_persist_memory_false_writes_nothing() -> None:
    memory = _RecordingRAGMemory()
    agent = _agent(memory)
    context = OrchidAgentRunContext(auth=_auth(), chat_id="c-1", message_id="m-1", persist_memory=False)

    token = agent.set_run_context(context)
    try:
        await agent._store_agent_turn(_state(), "answer")
    finally:
        agent.reset_run_context(token)

    assert memory.turns == []


@pytest.mark.asyncio
async def test_missing_run_context_keeps_the_legacy_write() -> None:
    memory = _RecordingRAGMemory()
    agent = _agent(memory)

    await agent._store_agent_turn(_state(), "answer")

    assert len(memory.turns) == 2


# ── Summary-memory path (running summary writes) ─────────────


class _StubChatModel:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> AIMessage:
        self.calls.append((messages, kwargs))
        return AIMessage(content="stub answer")


class _FakeChatStorage:
    def __init__(self, summary: str = "existing summary") -> None:
        self._summary = summary
        self.saved: list[dict[str, Any]] = []

    async def get_conversation_summary(self, chat_id: str) -> str | None:
        return self._summary

    async def save_conversation_summary(self, chat_id: str, summary: str, turn_number: int) -> None:
        self.saved.append({"chat_id": chat_id, "summary": summary, "turn_number": turn_number})


class _CountingMemory(OrchidInMemoryConversationMemory):
    def __init__(self, chat_storage: Any, chat_model: Any) -> None:
        super().__init__(chat_storage=chat_storage, chat_model=chat_model, structured_output=False)
        self.updates: list[dict[str, Any]] = []

    async def update_running_summary(
        self,
        chat_id: str,
        new_messages: list[dict[str, str]],
        existing_summary: str | None,
    ) -> str:
        self.updates.append({"chat_id": chat_id, "new_messages": new_messages, "existing_summary": existing_summary})
        return await super().update_running_summary(chat_id, new_messages, existing_summary)


def _summary_stack() -> tuple[GenericAgent, _CountingMemory, _FakeChatStorage]:
    chat_model = _StubChatModel()
    storage = _FakeChatStorage()
    memory = _CountingMemory(chat_storage=storage, chat_model=chat_model)
    agent = GenericAgent(
        config=OrchidAgentConfig(
            name="a",
            description="agent",
            prompt="prompt",
            rag=OrchidRAGConfig(enabled=False),
        ),
        reader=None,  # type: ignore[arg-type]
        chat_model=chat_model,
        summary_config={"memory": memory, "recent_turns": 3, "structured_output": False},
    )
    return agent, memory, storage


def _history_state() -> dict[str, Any]:
    return {
        "messages": [
            HumanMessage(content="earlier question", id="m-1"),
            AIMessage(content="[A Agent]\nolder answer"),
            HumanMessage(content="current question", id="m-2"),
        ],
        "chat_id": "c-1",
        "mcp_context": {},
    }


@pytest.mark.asyncio
async def test_summary_memory_is_not_written_when_persist_memory_is_false() -> None:
    agent, memory, storage = _summary_stack()
    context = OrchidAgentRunContext(auth=_auth(), chat_id="c-1", message_id="m-2", persist_memory=False)

    token = agent.set_run_context(context)
    try:
        result = await agent.run(_history_state())
    finally:
        agent.reset_run_context(token)

    assert "stub answer" in result["messages"][-1].content
    assert memory.updates == []
    assert storage.saved == []


@pytest.mark.asyncio
async def test_summary_memory_is_written_when_persist_memory_is_true() -> None:
    agent, memory, storage = _summary_stack()
    context = OrchidAgentRunContext(auth=_auth(), chat_id="c-1", message_id="m-2")

    token = agent.set_run_context(context)
    try:
        await agent.run(_history_state())
    finally:
        agent.reset_run_context(token)

    assert len(memory.updates) == 1
    assert memory.updates[0]["chat_id"] == "c-1"
    assert len(storage.saved) == 1
    assert storage.saved[0]["chat_id"] == "c-1"


@pytest.mark.asyncio
async def test_executor_run_with_persist_memory_false_writes_no_memory() -> None:
    agent, memory, storage = _summary_stack()
    executor = OrchidAgentExecutor(agent)

    result = await executor.run(
        _history_state(),
        auth=_auth(),
        chat_id="c-1",
        message_id="m-2",
        persist_memory=False,
    )

    assert "stub answer" in result["messages"][-1].content
    assert memory.updates == []
    assert storage.saved == []


# ── Skill-step peer context inheritance ──────────────────────


class _PeerAgent(OrchidAgent):
    def __init__(self, name: str = "peer") -> None:
        super().__init__(reader=None)  # type: ignore[arg-type]
        self._name = name
        self.seen_contexts: list[OrchidAgentRunContext | None] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "peer agent"

    async def run(self, state):
        self.seen_contexts.append(self.current_run_context())
        return {"messages": [AIMessage(content="peer answer")], "mcp_context": {}}


def _skill_executor(peer: _PeerAgent) -> SkillExecutor:
    return SkillExecutor(
        agent_name="parent",
        mcp_dispatcher=None,
        builtin_tool_caller=lambda **kwargs: None,
        agent_peers={"peer": peer},
    )


@pytest.mark.asyncio
async def test_skill_step_peer_inherits_persist_memory_false() -> None:
    peer = _PeerAgent()
    parent = _PeerAgent("parent")
    parent_context = OrchidAgentRunContext(
        auth=_auth(),
        correlation_id="corr-1",
        chat_id="c-1",
        message_id="m-1",
        persist_memory=False,
    )

    token = parent.set_run_context(parent_context)
    try:
        result = await _skill_executor(peer)._run_agent_step("peer", "do it", "query", _auth(), {})
        # The peer's binding is restored to the parent context after the step.
        assert peer.current_run_context() is parent_context
    finally:
        parent.reset_run_context(token)

    assert result["response"] == "peer answer"
    assert peer.seen_contexts[0] is not None
    assert peer.seen_contexts[0].persist_memory is False
    assert peer.seen_contexts[0].correlation_id == "corr-1"
    assert peer.seen_contexts[0].chat_id == "c-1"
    assert peer.seen_contexts[0].message_id == "m-1"
    assert peer.seen_contexts[0].auth is not None
    assert peer.current_run_context() is None


@pytest.mark.asyncio
async def test_skill_step_peer_defaults_to_persisting_memory() -> None:
    peer = _PeerAgent()

    await _skill_executor(peer)._run_agent_step("peer", "do it", "query", _auth(), {})

    assert peer.seen_contexts[0] is not None
    assert peer.seen_contexts[0].persist_memory is True
    assert peer.current_run_context() is None
