from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchid_ai.core.truncation import (
    OrchidTruncationStrategy,
    truncate_content,
    truncate_content_async,
)


class TestTruncationStrategy:
    def test_enum_values(self):
        assert OrchidTruncationStrategy.HARD.value == "hard"
        assert OrchidTruncationStrategy.MIDDLE.value == "middle"
        assert OrchidTruncationStrategy.LLM.value == "llm"
        assert OrchidTruncationStrategy.SEMANTIC.value == "semantic"


class TestTruncateContent:
    def test_under_limit_returns_unchanged(self):
        result = truncate_content("short", 100)
        assert result == "short"

    def test_exact_limit_returns_unchanged(self):
        result = truncate_content("exact", 5)
        assert result == "exact"

    def test_hard_truncation(self):
        content = "this is a longer message that needs truncation"
        result = truncate_content(content, 20, OrchidTruncationStrategy.HARD)
        assert len(result) == 20
        assert result.endswith("…")

    def test_middle_truncation(self):
        content = "A" * 100 + "B" * 100 + "C" * 100
        result = truncate_content(content, 100, OrchidTruncationStrategy.MIDDLE)
        assert len(result) <= 100
        assert "…[truncated]…" in result
        assert result.startswith("A")
        assert result.endswith("C")

    def test_middle_falls_back_to_hard_when_remaining_negative(self):
        content = "A" * 10
        result = truncate_content(content, 5, OrchidTruncationStrategy.MIDDLE)
        assert result == "AAAA…"

    def test_llm_falls_back_to_middle_sync(self):
        content = "A" * 50 + "B" * 50 + "C" * 50
        result = truncate_content(content, 100, OrchidTruncationStrategy.LLM)
        assert "…[truncated]…" in result

    def test_semantic_falls_back_to_middle_sync(self):
        content = "A" * 50 + "B" * 50 + "C" * 50
        result = truncate_content(content, 100, OrchidTruncationStrategy.SEMANTIC)
        assert "…[truncated]…" in result

    def test_default_strategy_is_hard(self):
        content = "long content here" * 10
        result = truncate_content(content, 20)
        assert len(result) == 20
        assert result.endswith("…")

    def test_empty_content(self):
        result = truncate_content("", 100)
        assert result == ""

    def test_unicode_content(self):
        content = "héllo wörld 🙏 this is longer than max"
        result = truncate_content(content, 15, OrchidTruncationStrategy.HARD)
        assert len(result) == 15
        assert result.endswith("…")


class TestTruncateContentAsync:
    @pytest.mark.asyncio
    async def test_under_limit_returns_unchanged(self):
        result = await truncate_content_async("short", 100)
        assert result == "short"

    @pytest.mark.asyncio
    async def test_hard_truncation(self):
        content = "this is a longer message that needs truncation"
        result = await truncate_content_async(content, 20, OrchidTruncationStrategy.HARD)
        assert len(result) == 20
        assert result.endswith("…")

    @pytest.mark.asyncio
    async def test_middle_truncation(self):
        content = "A" * 100 + "B" * 100 + "C" * 100
        result = await truncate_content_async(content, 100, OrchidTruncationStrategy.MIDDLE)
        assert len(result) <= 100
        assert "…[truncated]…" in result

    @pytest.mark.asyncio
    async def test_llm_with_model(self):
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock()
        mock_model.ainvoke.return_value = MagicMock(content="concise summary")

        content = "A" * 200
        result = await truncate_content_async(content, 100, OrchidTruncationStrategy.LLM, chat_model=mock_model)
        assert result == "concise summary"
        mock_model.ainvoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_llm_falls_back_when_no_model(self):
        content = "A" * 50 + "B" * 50 + "C" * 50
        result = await truncate_content_async(content, 100, OrchidTruncationStrategy.LLM)
        assert "…[truncated]…" in result

    @pytest.mark.asyncio
    async def test_llm_falls_back_on_exception(self):
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(side_effect=RuntimeError("LLM error"))

        content = "A" * 50 + "B" * 50 + "C" * 50
        result = await truncate_content_async(content, 100, OrchidTruncationStrategy.LLM, chat_model=mock_model)
        assert "…[truncated]…" in result

    @pytest.mark.asyncio
    async def test_llm_truncates_overlong_summary(self):
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock()
        mock_model.ainvoke.return_value = MagicMock(content="X" * 200)

        content = "A" * 200
        result = await truncate_content_async(content, 50, OrchidTruncationStrategy.LLM, chat_model=mock_model)
        assert len(result) == 50

    @pytest.mark.asyncio
    async def test_semantic_falls_back_to_middle(self):
        content = "A" * 50 + "B" * 50 + "C" * 50
        result = await truncate_content_async(content, 100, OrchidTruncationStrategy.SEMANTIC)
        assert "…[truncated]…" in result

    @pytest.mark.asyncio
    async def test_empty_content(self):
        result = await truncate_content_async("", 100)
        assert result == ""
