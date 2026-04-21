"""Tests for the OrchidOAuthStateStore factory's subclass-contract error path."""

from __future__ import annotations

import pytest

from orchid_ai.mcp.oauth_state import OrchidOAuthStateStore, build_oauth_state_store


class _BadSignatureStore(OrchidOAuthStateStore):
    """Declares a constructor that rejects the factory's kwargs."""

    def __init__(self, *, wrong_name: str = "") -> None:  # noqa: D401
        self.wrong_name = wrong_name

    async def put(self, state, payload):  # pragma: no cover — never called
        return None

    async def pop(self, state):  # pragma: no cover — never called
        return None


class TestCustomSubclassContract:
    @pytest.mark.asyncio
    async def test_helpful_error_on_constructor_mismatch(self):
        """A subclass whose ``__init__`` rejects ``ttl_seconds`` must raise
        a clear ``TypeError`` pointing at the contract (not the raw
        ``unexpected keyword argument`` message)."""
        path = f"{__name__}._BadSignatureStore"
        with pytest.raises(TypeError, match="must accept `ttl_seconds="):
            await build_oauth_state_store(path, ttl_seconds=60.0)
