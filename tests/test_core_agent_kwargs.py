"""Regression tests for ``OrchidAgent.__init__`` keyword tolerance.

The graph builder's ``_instantiate_agent`` always passes
``config=agent_config`` (and conditionally ``summary_config=...``)
to every agent class it loads, regardless of whether the target
class needs them.  ``GenericAgent`` captures both explicitly; custom
subclasses that don't override ``__init__`` (e.g. the helpdesk
``SupportAgent``) must still construct cleanly.

These tests pin down the contract: ``OrchidAgent.__init__`` absorbs
any extra kwargs the framework injects without raising ``TypeError``
— matching the documented "subclasses pick what they need and
ignore the rest" rule from ``orchid/CLAUDE.md``.
"""

from __future__ import annotations

from typing import Any

import pytest

from orchid_ai.core.agent import OrchidAgent
from orchid_ai.rag.backends.null import NullVectorReader


class _MinimalAgent(OrchidAgent):
    """Subclass that does not override ``__init__`` — same shape as
    ``examples.helpdesk.agents.support.SupportAgent``."""

    @property
    def name(self) -> str:
        return "minimal"

    @property
    def description(self) -> str:
        return "minimal agent"

    async def run(self, state: Any) -> Any:  # pragma: no cover — not exercised
        return {}


class TestOrchidAgentKwargTolerance:
    def test_accepts_named_kwargs(self):
        """Sanity — the documented kwargs still work."""
        agent = _MinimalAgent(
            model_id="gemini/test",
            reader=NullVectorReader(),
            mcp_clients=[],
            chat_model=None,
        )
        assert agent.model_id == "gemini/test"
        assert agent.mcp_clients == []

    def test_absorbs_config_kwarg(self):
        """Framework-injected ``config`` does not raise."""
        agent = _MinimalAgent(
            reader=NullVectorReader(),
            config={"any": "shape"},  # absorbed by **_kwargs
        )
        # Doesn't expose it (subclasses that care override __init__).
        assert not hasattr(agent, "_config") or agent._config is not None

    def test_absorbs_summary_config_kwarg(self):
        """Framework-injected ``summary_config`` does not raise."""
        _MinimalAgent(
            reader=NullVectorReader(),
            summary_config={"model": "gemini/lite", "recent_turns": 10},
        )

    def test_absorbs_arbitrary_extras(self):
        """Defence-in-depth — unknown kwargs the graph builder may add
        in the future do not break existing subclasses."""
        _MinimalAgent(
            reader=NullVectorReader(),
            config="x",
            summary_config={},
            something_brand_new=42,
        )

    def test_required_kwargs_still_required(self):
        """``reader`` is required — the absorber MUST NOT swallow it."""
        with pytest.raises(TypeError):
            _MinimalAgent()  # type: ignore[call-arg]

    def test_extras_do_not_become_attributes(self):
        """The base class doesn't materialise extras onto ``self`` —
        keeps the public surface predictable.
        """
        agent = _MinimalAgent(
            reader=NullVectorReader(),
            config={"x": 1},
            summary_config={"y": 2},
        )
        # Only the four documented attributes exist.
        assert agent.model_id == ""
        assert isinstance(agent.reader, NullVectorReader)
        assert agent.mcp_clients == []
        assert agent._chat_model is None
        # Extras were not promoted.
        assert not hasattr(agent, "config")
        assert not hasattr(agent, "summary_config")
