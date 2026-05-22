"""Tests for built-in content tool handlers (list/search/read)."""

from __future__ import annotations

import pytest

from orchid_ai.agents.content_tools import (
    list_content_files,
    read_content_file,
    search_content_files,
)
from orchid_ai.content.local import LocalFileContentSource


@pytest.fixture
def source_dir(tmp_path):
    (tmp_path / "subdir").mkdir()
    (tmp_path / "a.txt").write_text("hello world")
    (tmp_path / "b.md").write_text("# markdown")
    (tmp_path / "subdir" / "c.txt").write_text("nested")
    return tmp_path


@pytest.fixture
def sources(source_dir):
    return [LocalFileContentSource(path=str(source_dir))]


class TestListContentFiles:
    @pytest.mark.asyncio
    async def test_returns_items(self, sources):
        results = await list_content_files(content_sources=sources)
        assert len(results) >= 1
        names = {r["name"] for r in results}
        assert "a.txt" in names
        assert "b.md" in names

    @pytest.mark.asyncio
    async def test_recursive(self, sources):
        results = await list_content_files(content_sources=sources, recursive=True)
        names = {r["name"] for r in results}
        assert "c.txt" in names

    @pytest.mark.asyncio
    async def test_non_recursive(self, sources):
        results = await list_content_files(content_sources=sources, recursive=False)
        names = {r["name"] for r in results}
        assert "c.txt" not in names

    @pytest.mark.asyncio
    async def test_limit(self, sources):
        results = await list_content_files(content_sources=sources, limit=1)
        assert len(results) <= 1

    @pytest.mark.asyncio
    async def test_empty_sources(self):
        results = await list_content_files(content_sources=None)
        assert results == []

    @pytest.mark.asyncio
    async def test_empty_sources_list(self):
        results = await list_content_files(content_sources=[])
        assert results == []

    @pytest.mark.asyncio
    async def test_content_is_none(self, sources):
        results = await list_content_files(content_sources=sources)
        for r in results:
            assert r["content"] is None


class TestSearchContentFiles:
    @pytest.mark.asyncio
    async def test_search_finds_by_name(self, sources):
        results = await search_content_files(query="a", content_sources=sources)
        names = {r["name"] for r in results}
        assert "a.txt" in names

    @pytest.mark.asyncio
    async def test_search_no_match(self, sources):
        results = await search_content_files(query="zzz", content_sources=sources)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_limit(self, sources):
        results = await search_content_files(query=".", content_sources=sources, limit=1)
        assert len(results) <= 1

    @pytest.mark.asyncio
    async def test_empty_sources(self):
        results = await search_content_files(query="x", content_sources=None)
        assert results == []


class TestReadContentFile:
    @pytest.mark.asyncio
    async def test_reads_file(self, sources):
        result = await read_content_file(path="a.txt", content_sources=sources)
        assert result["name"] == "a.txt"
        assert "hello world" in result["content"]

    @pytest.mark.asyncio
    async def test_read_nonexistent(self, sources):
        result = await read_content_file(path="nonexistent.txt", content_sources=sources)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_empty_sources(self):
        result = await read_content_file(path="x", content_sources=None)
        assert "error" in result
