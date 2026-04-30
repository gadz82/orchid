"""Tests for ``MiniAgentDecomposer`` and the structured-output models.

Covers spec §16 cases:
  - T9  — strict allowlist rejects unknown tool names.
  - T15 — ``MiniAgentDecomposition`` validator caps ``len(sub_tasks) <= max_count``.
  - T19 — decomposer model fallback to the parent's ``llm.model``.

Plus internal correctness tests on the structured-output validator
(``should_fork`` / ``sub_tasks`` consistency, duplicate id rejection)
and the three ``tool_allowlist_mode`` variants.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from orchid_ai.agents.mini_agent_decomposer import (
    DEFAULT_DECOMPOSER_PROMPT,
    MiniAgentDecomposer,
    MiniAgentDecomposition,
    MiniAgentDecompositionError,
    MiniAgentSubTask,
)
from orchid_ai.config.schema import (
    OrchidAgentConfig,
    OrchidLLMConfig,
    OrchidMiniAgentConfig,
    OrchidRAGConfig,
)


# ── Helpers ────────────────────────────────────────────────────


def _make_config(
    *,
    name: str = "support",
    max_count: int = 3,
    tool_allowlist_mode: str = "strict",
    decomposer_model: str | None = None,
    parent_model: str = "gemini/gemini-2.5-flash",
) -> OrchidAgentConfig:
    """Build a minimal opt-in agent config for the decomposer."""
    cfg = OrchidAgentConfig(
        name=name,
        description=f"{name} agent",
        prompt="be helpful",
        rag=OrchidRAGConfig(enabled=False),
        llm=OrchidLLMConfig(model=parent_model),
        mini_agent=OrchidMiniAgentConfig(
            enabled=True,
            max_count=max_count,
            tool_allowlist_mode=tool_allowlist_mode,
            decomposer_model=decomposer_model,
        ),
    )
    return cfg


def _make_chat_model(decomposition: MiniAgentDecomposition) -> MagicMock:
    """Construct a chat-model double whose ``with_structured_output``
    returns an ``ainvoke`` that yields the given decomposition.

    Mirrors how LangChain's ``BaseChatModel.with_structured_output``
    behaves — chains a follow-up ``ainvoke`` returning a Pydantic model.
    """
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=decomposition)
    chat = MagicMock()
    chat.with_structured_output = MagicMock(return_value=structured)
    return chat


# ── Pydantic validator tests (independent of the LLM) ─────────


class TestMiniAgentDecompositionValidator:
    def test_should_fork_false_with_subtasks_rejected(self):
        with pytest.raises(ValidationError):
            MiniAgentDecomposition(
                should_fork=False,
                sub_tasks=[
                    MiniAgentSubTask(
                        id="mini_0",
                        description="x",
                        instruction="x",
                        rationale="x",
                    ),
                ],
            )

    def test_should_fork_true_requires_min_two_subtasks(self):
        with pytest.raises(ValidationError):
            MiniAgentDecomposition(
                should_fork=True,
                sub_tasks=[
                    MiniAgentSubTask(
                        id="mini_0",
                        description="x",
                        instruction="x",
                        rationale="x",
                    ),
                ],
            )

    def test_duplicate_subtask_id_rejected(self):
        with pytest.raises(ValidationError):
            MiniAgentDecomposition(
                should_fork=True,
                sub_tasks=[
                    MiniAgentSubTask(id="mini_0", description="a", instruction="a", rationale="r"),
                    MiniAgentSubTask(id="mini_0", description="b", instruction="b", rationale="r"),
                ],
            )

    def test_should_fork_false_no_subtasks_accepted(self):
        d = MiniAgentDecomposition(should_fork=False)
        assert d.should_fork is False
        assert d.sub_tasks == []


# ── Decomposer behaviour ───────────────────────────────────────


class TestMiniAgentDecomposerHappyPath:
    @pytest.mark.asyncio
    async def test_should_fork_true_returns_decomposition(self):
        decomposition = MiniAgentDecomposition(
            should_fork=True,
            sub_tasks=[
                MiniAgentSubTask(
                    id="mini_0",
                    description="lookup user",
                    instruction="find the user record",
                    allowed_tools=["lookup_user"],
                    rationale="independent of orders",
                ),
                MiniAgentSubTask(
                    id="mini_1",
                    description="lookup orders",
                    instruction="list recent orders",
                    allowed_tools=["lookup_order"],
                    rationale="independent of users",
                ),
            ],
            reasoning="two independent reads",
        )
        chat = _make_chat_model(decomposition)
        decomposer = MiniAgentDecomposer(agent_config=_make_config(), chat_model=chat)

        result = await decomposer.decompose(
            user_query="show me user X and their orders",
            conversation_history=None,
            tool_inventory=["lookup_user", "lookup_order"],
        )
        assert result.should_fork is True
        assert [s.id for s in result.sub_tasks] == ["mini_0", "mini_1"]
        chat.with_structured_output.assert_called_once_with(MiniAgentDecomposition)

    @pytest.mark.asyncio
    async def test_should_fork_false_returns_no_subtasks(self):
        decomposition = MiniAgentDecomposition(should_fork=False, reasoning="single coherent task")
        chat = _make_chat_model(decomposition)
        decomposer = MiniAgentDecomposer(agent_config=_make_config(), chat_model=chat)

        result = await decomposer.decompose(
            user_query="hi",
            conversation_history=None,
            tool_inventory=["search_kb"],
        )
        assert result.should_fork is False
        assert result.sub_tasks == []


class TestMaxCountCap:
    """T15 — decomposer's structured output capped to ``max_count``."""

    @pytest.mark.asyncio
    async def test_overage_raises(self):
        # Build 9 sub-tasks; max_count=8 (the hard cap).
        sub_tasks = [
            MiniAgentSubTask(
                id=f"mini_{i}",
                description=f"task {i}",
                instruction="x",
                allowed_tools=["t"],
                rationale="r",
            )
            for i in range(9)
        ]
        decomposition = MiniAgentDecomposition(should_fork=True, sub_tasks=sub_tasks)
        chat = _make_chat_model(decomposition)
        decomposer = MiniAgentDecomposer(agent_config=_make_config(max_count=8), chat_model=chat)

        with pytest.raises(MiniAgentDecompositionError) as excinfo:
            await decomposer.decompose(
                user_query="big request",
                conversation_history=None,
                tool_inventory=["t"],
            )
        assert "max_count" in str(excinfo.value)


class TestStrictAllowlist:
    """T9 — strict mode rejects unknown tool names."""

    @pytest.mark.asyncio
    async def test_unknown_tool_rejected(self):
        decomposition = MiniAgentDecomposition(
            should_fork=True,
            sub_tasks=[
                MiniAgentSubTask(
                    id="mini_0",
                    description="a",
                    instruction="a",
                    allowed_tools=["lookup_user"],
                    rationale="r",
                ),
                MiniAgentSubTask(
                    id="mini_1",
                    description="b",
                    instruction="b",
                    allowed_tools=["nonexistent_tool"],  # not in inventory
                    rationale="r",
                ),
            ],
        )
        chat = _make_chat_model(decomposition)
        decomposer = MiniAgentDecomposer(
            agent_config=_make_config(tool_allowlist_mode="strict"),
            chat_model=chat,
        )

        with pytest.raises(MiniAgentDecompositionError) as excinfo:
            await decomposer.decompose(
                user_query="x",
                conversation_history=None,
                tool_inventory=["lookup_user", "lookup_order"],  # no nonexistent_tool
            )
        assert "nonexistent_tool" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_empty_allowed_tools_in_strict_mode_rejected(self):
        decomposition = MiniAgentDecomposition(
            should_fork=True,
            sub_tasks=[
                MiniAgentSubTask(
                    id="mini_0",
                    description="a",
                    instruction="a",
                    allowed_tools=["lookup_user"],
                    rationale="r",
                ),
                MiniAgentSubTask(
                    id="mini_1",
                    description="b",
                    instruction="b",
                    allowed_tools=[],  # empty in strict mode → reject
                    rationale="r",
                ),
            ],
        )
        chat = _make_chat_model(decomposition)
        decomposer = MiniAgentDecomposer(
            agent_config=_make_config(tool_allowlist_mode="strict"),
            chat_model=chat,
        )

        with pytest.raises(MiniAgentDecompositionError):
            await decomposer.decompose(
                user_query="x",
                conversation_history=None,
                tool_inventory=["lookup_user"],
            )


class TestParentFullMode:
    """``parent_full`` mode ignores ``allowed_tools`` entirely."""

    @pytest.mark.asyncio
    async def test_unknown_tool_accepted(self):
        decomposition = MiniAgentDecomposition(
            should_fork=True,
            sub_tasks=[
                MiniAgentSubTask(
                    id="mini_0",
                    description="a",
                    instruction="a",
                    allowed_tools=["nonexistent_tool"],
                    rationale="r",
                ),
                MiniAgentSubTask(
                    id="mini_1",
                    description="b",
                    instruction="b",
                    allowed_tools=[],  # also accepted in parent_full
                    rationale="r",
                ),
            ],
        )
        chat = _make_chat_model(decomposition)
        decomposer = MiniAgentDecomposer(
            agent_config=_make_config(tool_allowlist_mode="parent_full"),
            chat_model=chat,
        )

        result = await decomposer.decompose(
            user_query="x",
            conversation_history=None,
            tool_inventory=["lookup_user"],
        )
        assert result.should_fork is True

    def test_resolve_tool_subset_returns_full_inventory(self):
        decomposer = MiniAgentDecomposer(
            agent_config=_make_config(tool_allowlist_mode="parent_full"),
            chat_model=MagicMock(),
        )
        sub_task = MiniAgentSubTask(
            id="mini_0",
            description="x",
            instruction="x",
            allowed_tools=["only_one"],
            rationale="r",
        )
        subset = decomposer.resolve_tool_subset(
            sub_task=sub_task,
            tool_inventory=["a", "b", "c"],
        )
        assert subset == ["a", "b", "c"]


class TestInferredMode:
    """``inferred`` mode falls back to full inventory when allowed_tools is empty."""

    @pytest.mark.asyncio
    async def test_empty_allowed_tools_falls_back(self):
        decomposition = MiniAgentDecomposition(
            should_fork=True,
            sub_tasks=[
                MiniAgentSubTask(
                    id="mini_0",
                    description="a",
                    instruction="a",
                    allowed_tools=[],  # empty triggers fallback
                    rationale="r",
                ),
                MiniAgentSubTask(
                    id="mini_1",
                    description="b",
                    instruction="b",
                    allowed_tools=["t1"],
                    rationale="r",
                ),
            ],
        )
        chat = _make_chat_model(decomposition)
        decomposer = MiniAgentDecomposer(
            agent_config=_make_config(tool_allowlist_mode="inferred"),
            chat_model=chat,
        )

        result = await decomposer.decompose(
            user_query="x",
            conversation_history=None,
            tool_inventory=["t1", "t2"],
        )
        assert result.should_fork is True

    def test_resolve_subset_inferred_empty_falls_back(self):
        decomposer = MiniAgentDecomposer(
            agent_config=_make_config(tool_allowlist_mode="inferred"),
            chat_model=MagicMock(),
        )
        empty_sub_task = MiniAgentSubTask(
            id="mini_0",
            description="x",
            instruction="x",
            allowed_tools=[],
            rationale="r",
        )
        assert decomposer.resolve_tool_subset(
            sub_task=empty_sub_task,
            tool_inventory=["a", "b"],
        ) == ["a", "b"]

    def test_resolve_subset_inferred_nonempty_uses_subset(self):
        decomposer = MiniAgentDecomposer(
            agent_config=_make_config(tool_allowlist_mode="inferred"),
            chat_model=MagicMock(),
        )
        sub_task = MiniAgentSubTask(
            id="mini_0",
            description="x",
            instruction="x",
            allowed_tools=["a"],
            rationale="r",
        )
        assert decomposer.resolve_tool_subset(
            sub_task=sub_task,
            tool_inventory=["a", "b"],
        ) == ["a"]


# ── T19: decomposer model fallback ────────────────────────────


class TestDecomposerModelFallback:
    """T19 — when ``decomposer_model`` is unset, the parent's chat model
    is used.  Asserts the precedence at the ``maybe_decompose`` free
    function (Phase B / B3 graph-level wrapper hook)."""

    @pytest.mark.asyncio
    async def test_unset_decomposer_model_uses_parent_chat_model(self, monkeypatch):
        from orchid_ai.agents.mini_agent_decomposer import maybe_decompose
        from orchid_ai.core.state import OrchidAuthContext
        from langchain_core.messages import HumanMessage

        decomposition = MiniAgentDecomposition(should_fork=False, reasoning="single")
        parent_chat = _make_chat_model(decomposition)

        # No build_chat_model call should happen — fallback path.
        called = {"build": 0}

        def _fake_build(*args, **kwargs):
            called["build"] += 1
            return MagicMock()

        monkeypatch.setattr("orchid_ai.llm_factory.build_chat_model", _fake_build)

        auth = OrchidAuthContext(access_token="t", tenant_key="x", user_id="u")
        update = await maybe_decompose(
            agent_config=_make_config(decomposer_model=None, parent_model="gemini/foo"),
            chat_model=parent_chat,
            mcp_clients=[],
            auth=auth,
            state={"messages": [HumanMessage(content="x")]},
        )
        assert update is None  # should_fork=False → no state update
        assert called["build"] == 0
        # Parent chat model was used directly for structured output.
        parent_chat.with_structured_output.assert_called_once_with(MiniAgentDecomposition)

    @pytest.mark.asyncio
    async def test_set_decomposer_model_triggers_factory(self, monkeypatch):
        from orchid_ai.agents.mini_agent_decomposer import maybe_decompose
        from orchid_ai.core.state import OrchidAuthContext
        from langchain_core.messages import HumanMessage

        decomposition = MiniAgentDecomposition(should_fork=False, reasoning="single")
        custom_chat = _make_chat_model(decomposition)
        parent_chat = MagicMock()

        called = {"build_args": []}

        def _fake_build(model, **kwargs):
            called["build_args"].append((model, kwargs))
            return custom_chat

        monkeypatch.setattr("orchid_ai.llm_factory.build_chat_model", _fake_build)

        auth = OrchidAuthContext(access_token="t", tenant_key="x", user_id="u")
        await maybe_decompose(
            agent_config=_make_config(decomposer_model="gemini/lite", parent_model="gemini/big"),
            chat_model=parent_chat,
            mcp_clients=[],
            auth=auth,
            state={"messages": [HumanMessage(content="x")]},
        )
        # Custom factory invoked with the configured decomposer model.
        assert len(called["build_args"]) == 1
        assert called["build_args"][0][0] == "gemini/lite"
        # Parent chat model NOT used; the custom one was.
        custom_chat.with_structured_output.assert_called_once_with(MiniAgentDecomposition)
        parent_chat.with_structured_output.assert_not_called()


# ── Prompt rendering ──────────────────────────────────────────


class TestPromptRendering:
    @pytest.mark.asyncio
    async def test_default_template_substitutes_placeholders(self):
        captured: dict[str, list] = {"messages": []}

        async def _capture(messages):
            captured["messages"].append(messages)
            return MiniAgentDecomposition(should_fork=False)

        chat = MagicMock()
        structured = MagicMock()
        structured.ainvoke = _capture
        chat.with_structured_output = MagicMock(return_value=structured)

        decomposer = MiniAgentDecomposer(agent_config=_make_config(name="support"), chat_model=chat)
        await decomposer.decompose(
            user_query="why is my account locked?",
            conversation_history=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            tool_inventory=["lookup_user", "lookup_account"],
        )
        prompt = captured["messages"][0][0]["content"]
        assert "support" in prompt
        assert "lookup_user" in prompt
        assert "lookup_account" in prompt
        assert "why is my account locked?" in prompt
        # History rendered.
        assert "user: hi" in prompt
        assert "assistant: hello" in prompt
        # max_count made it in.
        assert "2..3" in prompt

    @pytest.mark.asyncio
    async def test_overridden_template_used(self):
        captured: dict[str, list] = {"messages": []}

        async def _capture(messages):
            captured["messages"].append(messages)
            return MiniAgentDecomposition(should_fork=False)

        chat = MagicMock()
        structured = MagicMock()
        structured.ainvoke = _capture
        chat.with_structured_output = MagicMock(return_value=structured)

        cfg = _make_config()
        cfg.mini_agent.decomposer_prompt = (
            "CUSTOM template for {agent_name}: {user_query} (tools={tool_inventory}, "
            "history={history}, history_max_turns={history_max_turns}, "
            "max_count={max_count}, agent_description={agent_description}, "
            "agent_prompt={agent_prompt})"
        )
        decomposer = MiniAgentDecomposer(agent_config=cfg, chat_model=chat)
        await decomposer.decompose(
            user_query="hello",
            conversation_history=None,
            tool_inventory=["t1"],
        )
        prompt = captured["messages"][0][0]["content"]
        assert prompt.startswith("CUSTOM template for support: hello")
        # Default template's specific phrasing is absent.
        assert "Decide whether this request decomposes" not in prompt


def test_default_prompt_has_all_placeholders():
    """Defensive: format() against a complete kwargs dict must not raise."""
    rendered = DEFAULT_DECOMPOSER_PROMPT.format(
        agent_name="x",
        agent_description="y",
        agent_prompt="z",
        tool_inventory="t1, t2",
        user_query="q",
        history="h",
        history_max_turns=10,
        max_count=3,
    )
    assert "x" in rendered
    assert "q" in rendered
