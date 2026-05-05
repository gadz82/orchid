"""ADR-026 — ``graph_store`` flows from runtime → agent → retrieval strategy."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from orchid_ai.agents.generic_agent import GenericAgent
from orchid_ai.config.schema import OrchidAgentsConfig
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.rag.backends.in_memory_graph import InMemoryGraphStore
from orchid_ai.rag.scopes import OrchidRAGScope


def _state(query: str = "go") -> dict[str, Any]:
    return {
        "messages": [MagicMock(content=query)],
        "auth_context": OrchidAuthContext(access_token="tok", tenant_key="t1", user_id="u1"),
        "chat_id": "chat-1",
    }


def _build_agent(yaml_cfg: str, agent_name: str, *, graph_store=None) -> GenericAgent:
    cfg = OrchidAgentsConfig.model_validate(yaml.safe_load(yaml_cfg))
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
        graph_store=graph_store,
    )


@pytest.mark.asyncio
async def test_graph_store_propagates_to_retrieval_kwargs():
    yaml_cfg = """
    version: '1'
    agents:
      kg_agent:
        description: KG agent
        prompt: 'p'
        rag:
          namespace: graph_kb
          retrieval: { strategy: graph_rag }
    """
    store = InMemoryGraphStore()
    agent = _build_agent(yaml_cfg, "kg_agent", graph_store=store)

    captured: dict[str, Any] = {}

    fake_strategy = MagicMock()

    async def _retrieve(**kwargs):
        captured.update(kwargs)
        return []

    fake_strategy.retrieve = _retrieve

    with patch(
        "orchid_ai.agents.generic_agent.get_retrieval_strategy",
        return_value=fake_strategy,
    ):
        await agent._step_rag_retrieval(
            "q",
            OrchidRAGScope(tenant_id="t1", user_id="u1", chat_id="c1", agent_id="kg_agent"),
        )

    assert captured["graph_store"] is store


@pytest.mark.asyncio
async def test_no_graph_store_yields_none():
    yaml_cfg = """
    version: '1'
    agents:
      simple_agent:
        description: simple agent
        prompt: 'p'
        rag: { namespace: kb, retrieval: { strategy: simple } }
    """
    agent = _build_agent(yaml_cfg, "simple_agent", graph_store=None)

    captured: dict[str, Any] = {}

    fake_strategy = MagicMock()

    async def _retrieve(**kwargs):
        captured.update(kwargs)
        return []

    fake_strategy.retrieve = _retrieve

    with patch(
        "orchid_ai.agents.generic_agent.get_retrieval_strategy",
        return_value=fake_strategy,
    ):
        await agent._step_rag_retrieval(
            "q",
            OrchidRAGScope(tenant_id="t1", user_id="u1", chat_id="c1", agent_id="simple_agent"),
        )

    assert captured["graph_store"] is None
