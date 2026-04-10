"""Tests for src.rag.null — NullVectorReader."""

from __future__ import annotations

import pytest

from orchid_ai.rag.null import NullVectorReader
from orchid_ai.rag.scopes import RAGScope


class TestNullVectorReader:
    @pytest.mark.asyncio
    async def test_retrieve_returns_empty_list(self):
        reader = NullVectorReader()
        result = await reader.retrieve("any query", "any_namespace")
        assert result == []

    @pytest.mark.asyncio
    async def test_retrieve_with_scope(self):
        reader = NullVectorReader()
        scope = RAGScope(tenant_id="t-1", user_id="u-1", chat_id="c-1")
        result = await reader.retrieve("query", "ns", k=10, scope=scope)
        assert result == []

    @pytest.mark.asyncio
    async def test_retrieve_with_custom_k(self):
        reader = NullVectorReader()
        result = await reader.retrieve("q", "ns", k=100)
        assert result == []

    @pytest.mark.asyncio
    async def test_retrieve_with_no_optional_args(self):
        reader = NullVectorReader()
        result = await reader.retrieve("hello world", "namespace")
        assert isinstance(result, list)
        assert len(result) == 0
