"""``GenericAgent._step_dynamic_injection`` honours per-tool overrides."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from orchid_ai.agents.generic_agent import GenericAgent
from orchid_ai.config.schema import OrchidAgentsConfig
from orchid_ai.core.agent import OrchidAgentRunContext
from orchid_ai.core.state import OrchidAuthContext


async def _run_with_auth(agent, state):
    """Run an agent with auth bound on the run-context ContextVar (auth is
    execution context now, not graph state)."""
    token = agent.set_run_context(OrchidAgentRunContext(auth=state.get("auth_context"), chat_id=state.get("chat_id")))
    try:
        return await agent.run(state)
    finally:
        agent.reset_run_context(token)


def _state(query: str = "go") -> dict[str, Any]:
    return {
        "messages": [MagicMock(content=query)],
        "auth_context": OrchidAuthContext(access_token="tok", tenant_key="t1", user_id="u1"),
        "chat_id": "chat-1",
    }


def _make_agent(cfg_yaml: str, agent_name: str) -> GenericAgent:
    cfg = OrchidAgentsConfig.model_validate(yaml.safe_load(cfg_yaml))
    reader = MagicMock()
    reader.retrieve = AsyncMock(return_value=[])
    chat_model = MagicMock()
    chat_model.ainvoke = AsyncMock(return_value=MagicMock(content="summary"))
    return GenericAgent(
        config=cfg.agents[agent_name],
        model_id="test",
        reader=reader,
        mcp_clients=[],
        chat_model=chat_model,
    )


@pytest.mark.asyncio
async def test_mcp_tool_override_namespace_reaches_inject_to_rag():
    yaml_cfg = """
    version: '1'
    agents:
      search:
        description: KB search
        prompt: 'p'
        rag:
          namespace: kb-default
          ingestion: { strategy: recursive, chunk_size: 1000 }
        mcp_servers:
          - name: srv
            url: http://srv
            tools:
              - name: lookup
                inject_to_rag: true
                rag:
                  namespace: lookup-cache
                  ingestion: { chunk_size: 300 }
    """
    agent = _make_agent(yaml_cfg, "search")

    with (
        patch.object(
            agent,
            "_agentic_tool_loop",
            new_callable=AsyncMock,
            return_value=(None, {"lookup": "result text"}, []),
        ),
        patch("orchid_ai.agents.rag_pipeline.inject_to_rag", new_callable=AsyncMock) as mock_inject,
        patch(
            "orchid_ai.agents.rag_pipeline.build_ingestion_strategy",
        ) as mock_build,
    ):
        sentinel = MagicMock(name="ingestion-strategy")
        mock_build.return_value = sentinel

        await _run_with_auth(agent, _state())

        mock_inject.assert_awaited_once()
        kwargs = mock_inject.call_args.kwargs
        assert kwargs["tool_name"] == "lookup"
        # Override namespace flowed through.
        assert kwargs["namespace"] == "lookup-cache"
        assert kwargs["ingestion"] is sentinel

        # ``build_ingestion_strategy`` saw the merged ingestion config —
        # chunk_size from the override, strategy inherited from the agent.
        ingestion_arg = mock_build.call_args.args[0]
        assert ingestion_arg.chunk_size == 300
        assert ingestion_arg.strategy == "recursive"


@pytest.mark.asyncio
async def test_no_override_uses_agent_namespace():
    yaml_cfg = """
    version: '1'
    agents:
      search:
        description: KB search
        prompt: 'p'
        rag:
          namespace: kb-default
        mcp_servers:
          - name: srv
            url: http://srv
            tools:
              - name: lookup
                inject_to_rag: true
    """
    agent = _make_agent(yaml_cfg, "search")

    with (
        patch.object(
            agent,
            "_agentic_tool_loop",
            new_callable=AsyncMock,
            return_value=(None, {"lookup": "result"}, []),
        ),
        patch("orchid_ai.agents.rag_pipeline.inject_to_rag", new_callable=AsyncMock) as mock_inject,
        patch("orchid_ai.agents.rag_pipeline.build_ingestion_strategy", return_value=MagicMock()),
    ):
        await _run_with_auth(agent, _state())
        kwargs = mock_inject.call_args.kwargs
        assert kwargs["namespace"] == "kb-default"


@pytest.mark.asyncio
async def test_builtin_tool_override_namespace_reaches_inject_to_rag():
    yaml_cfg = """
    version: '1'
    tools:
      format_date:
        handler: tests.tools.format_date
        inject_to_rag: true
        rag:
          namespace: dates-cache
    agents:
      utility:
        description: utility
        prompt: 'p'
        tools: [format_date]
        rag:
          namespace: utility-default
    """
    agent = _make_agent(yaml_cfg, "utility")

    with (
        patch.object(
            agent,
            "_agentic_tool_loop",
            new_callable=AsyncMock,
            return_value=(None, {"format_date": "2024-01-01"}, []),
        ),
        patch("orchid_ai.agents.rag_pipeline.inject_to_rag", new_callable=AsyncMock) as mock_inject,
        patch("orchid_ai.agents.rag_pipeline.build_ingestion_strategy", return_value=MagicMock()),
    ):
        await _run_with_auth(agent, _state())
        mock_inject.assert_awaited_once()
        kwargs = mock_inject.call_args.kwargs
        assert kwargs["tool_name"] == "format_date"
        assert kwargs["namespace"] == "dates-cache"


@pytest.mark.asyncio
async def test_mixed_injectable_and_non_injectable_tools_iterate_only_injectables():
    yaml_cfg = """
    version: '1'
    agents:
      search:
        description: KB search
        prompt: 'p'
        rag: { namespace: kb-default }
        mcp_servers:
          - name: srv
            url: http://srv
            tools:
              - { name: t_keep, inject_to_rag: true, rag: { namespace: keep-ns } }
              - { name: t_skip }
              - { name: t_keep_2, inject_to_rag: true }
    """
    agent = _make_agent(yaml_cfg, "search")

    with (
        patch.object(
            agent,
            "_agentic_tool_loop",
            new_callable=AsyncMock,
            return_value=(None, {"t_keep": "a", "t_skip": "b", "t_keep_2": "c"}, []),
        ),
        patch("orchid_ai.agents.rag_pipeline.inject_to_rag", new_callable=AsyncMock) as mock_inject,
        patch("orchid_ai.agents.rag_pipeline.build_ingestion_strategy", return_value=MagicMock()),
    ):
        await _run_with_auth(agent, _state())
        # Two injectable tools → two calls; t_skip was filtered out.
        assert mock_inject.await_count == 2
        names = [c.kwargs["tool_name"] for c in mock_inject.call_args_list]
        assert names == ["t_keep", "t_keep_2"]


@pytest.mark.asyncio
async def test_exclude_dynamic_injects_negative_filter_into_retrieval():
    """``retrieval.exclude_dynamic`` must surface as a ``dynamic: {"not": True}``
    clause so dynamically-injected output stays out of retrieval results."""
    yaml_cfg = """
    version: '1'
    agents:
      search:
        description: KB search
        prompt: 'p'
        rag:
          namespace: kb-default
          retrieval:
            strategy: simple
            exclude_dynamic: true
    """
    agent = _make_agent(yaml_cfg, "search")

    captured: dict[str, Any] = {}

    fake_strategy = MagicMock()

    async def _retrieve(**kwargs):
        captured.update(kwargs)
        return []

    fake_strategy.retrieve = _retrieve

    with patch(
        "orchid_ai.agents.rag_pipeline.get_retrieval_strategy",
        return_value=fake_strategy,
    ):
        from orchid_ai.rag.scopes import OrchidRAGScope

        await agent._step_rag_retrieval(
            "q",
            OrchidRAGScope(tenant_id="t1", user_id="u1", chat_id="c1", agent_id="search"),
        )

    assert captured["metadata_filters"] == {"dynamic": {"not": True}}


@pytest.mark.asyncio
async def test_exclude_dynamic_off_leaves_filters_alone():
    yaml_cfg = """
    version: '1'
    agents:
      search:
        description: KB search
        prompt: 'p'
        rag:
          namespace: kb-default
          retrieval:
            strategy: simple
            metadata_filters: { language: en }
    """
    agent = _make_agent(yaml_cfg, "search")

    captured: dict[str, Any] = {}

    fake_strategy = MagicMock()

    async def _retrieve(**kwargs):
        captured.update(kwargs)
        return []

    fake_strategy.retrieve = _retrieve

    with patch(
        "orchid_ai.agents.rag_pipeline.get_retrieval_strategy",
        return_value=fake_strategy,
    ):
        from orchid_ai.rag.scopes import OrchidRAGScope

        await agent._step_rag_retrieval(
            "q",
            OrchidRAGScope(tenant_id="t1", user_id="u1", chat_id="c1", agent_id="search"),
        )

    assert captured["metadata_filters"] == {"language": "en"}
