"""Unit tests for ``bootstrap._build_runner``.

Verifies the two code-paths:
  - ``graph_invoker=None``  → stub invoker that returns a formatted string
  - ``graph_invoker=<fn>``  → real invoker passed through to GraphJobRunner
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchid_ai.events.bootstrap import _build_runner
from orchid_ai.events.runners.graph_runner import GraphJobRunner


def _make_run(run_id: str = "r-1", agent: str = "support", prompt: str = "test") -> MagicMock:
    run = MagicMock()
    run.run_id = run_id
    run.spec.agent_name = agent
    run.spec.prompt = prompt
    return run


@pytest.mark.asyncio
async def test_build_runner_stub_when_no_invoker():
    """With no graph_invoker, the runner uses a stub that marks runs SUCCEEDED."""
    runner = _build_runner(chat_storage=None)
    assert isinstance(runner, GraphJobRunner)

    run = _make_run()
    auth = MagicMock()

    # Call the stub invoker directly through the runner's internal invoker.
    result = await runner._invoker(run, auth)

    assert "bloom-default-invoker" in result["final_response"]
    assert "r-1" in result["final_response"]
    assert "support" in result["final_response"]


@pytest.mark.asyncio
async def test_build_runner_uses_provided_invoker():
    """With a graph_invoker, the runner delegates to it without modification."""
    real_invoker = AsyncMock(return_value={"final_response": "done"})

    runner = _build_runner(chat_storage=None, graph_invoker=real_invoker)
    assert isinstance(runner, GraphJobRunner)

    run = _make_run()
    auth = MagicMock()

    result = await runner._invoker(run, auth)

    real_invoker.assert_awaited_once_with(run, auth)
    assert result == {"final_response": "done"}


def test_build_runner_returns_graph_job_runner_in_both_cases():
    """Both paths return a GraphJobRunner instance (not a subclass or stub object)."""
    without_invoker = _build_runner(chat_storage=None)
    with_invoker = _build_runner(chat_storage=None, graph_invoker=AsyncMock())
    assert isinstance(without_invoker, GraphJobRunner)
    assert isinstance(with_invoker, GraphJobRunner)
