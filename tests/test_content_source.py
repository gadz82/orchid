"""Tests for OrchidContentItem, OrchidContentSource ABC, registry, and LocalFileContentSource."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from orchid_ai.content import (
    CONTENT_SOURCE_REGISTRY,
    build_content_source,
    register_content_source,
)
from orchid_ai.content.local import LocalFileContentSource
from orchid_ai.core.content import OrchidContentItem, OrchidContentSource


class TestOrchidContentItem:
    def test_creation_defaults(self):
        item = OrchidContentItem(path="a/b.txt", name="b.txt", content_type=".txt")
        assert item.path == "a/b.txt"
        assert item.name == "b.txt"
        assert item.content_type == ".txt"
        assert item.metadata == {}
        assert item.content is None

    def test_creation_full(self):
        item = OrchidContentItem(
            path="a/b.pdf",
            name="b.pdf",
            content_type=".pdf",
            metadata={"size": 1024},
            content="document text",
        )
        assert item.metadata == {"size": 1024}
        assert item.content == "document text"

    def test_frozen(self):
        item = OrchidContentItem(path="x", name="x", content_type=".txt")
        with pytest.raises(FrozenInstanceError):
            item.path = "y"  # type: ignore[misc]


class TestOrchidContentSourceABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            OrchidContentSource()  # type: ignore[abstract]

    def test_subclass_must_implement_all(self):
        class Partial(OrchidContentSource):
            async def list(self, path="", recursive=False, limit=100):
                return []

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]

    def test_concrete_subclass(self):
        class Full(OrchidContentSource):
            async def list(self, path="", recursive=False, limit=100):
                return []

            async def get(self, path):
                return OrchidContentItem(path=path, name="x", content_type=".txt")

            async def search(self, query, recursive=True, limit=10):
                return []

        instance = Full()
        assert isinstance(instance, OrchidContentSource)


class TestContentSourceRegistry:
    def test_local_registered_by_default(self):
        assert "local" in CONTENT_SOURCE_REGISTRY
        assert CONTENT_SOURCE_REGISTRY["local"] is LocalFileContentSource

    def test_register_and_build(self, tmp_path):
        class DummySource(OrchidContentSource):
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            async def list(self, path="", recursive=False, limit=100):
                return []

            async def get(self, path):
                return OrchidContentItem(path=path, name="x", content_type=".txt")

            async def search(self, query, recursive=True, limit=10):
                return []

        register_content_source("dummy", DummySource)
        try:
            instance = build_content_source("dummy", foo="bar", baz=42)
            assert isinstance(instance, DummySource)
            assert instance.kwargs == {"foo": "bar", "baz": 42}
        finally:
            CONTENT_SOURCE_REGISTRY.pop("dummy", None)

    def test_build_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown content source 'nonexistent'"):
            build_content_source("nonexistent")

    def test_reregister_overwrites(self):
        original = CONTENT_SOURCE_REGISTRY.get("local")
        try:
            register_content_source("local", LocalFileContentSource)
            assert CONTENT_SOURCE_REGISTRY["local"] is LocalFileContentSource
        finally:
            if original is not None:
                CONTENT_SOURCE_REGISTRY["local"] = original


class TestLocalFileContentSource:
    @pytest.fixture
    def source_dir(self, tmp_path):
        (tmp_path / "subdir").mkdir()
        (tmp_path / "a.txt").write_text("hello world")
        (tmp_path / "b.md").write_text("# markdown")
        (tmp_path / "c.csv").write_text("col1,col2\n1,2")
        (tmp_path / "notes").write_text("no extension")
        (tmp_path / "subdir" / "d.txt").write_text("nested")
        return tmp_path

    @pytest.fixture
    def source(self, source_dir):
        return LocalFileContentSource(path=str(source_dir))

    @pytest.mark.asyncio
    async def test_list_returns_items_with_content_none(self, source):
        items = await source.list()
        assert len(items) >= 1
        names = {item.name for item in items}
        assert "a.txt" in names
        assert "b.md" in names
        for item in items:
            assert item.content is None

    @pytest.mark.asyncio
    async def test_list_non_recursive_finds_top_level_only(self, source):
        items = await source.list(recursive=False)
        names = {item.name for item in items}
        assert "d.txt" not in names
        assert "a.txt" in names

    @pytest.mark.asyncio
    async def test_list_recursive_finds_nested(self, source):
        items = await source.list(recursive=True)
        names = {item.name for item in items}
        assert "d.txt" in names

    @pytest.mark.asyncio
    async def test_list_limit_truncation(self, source):
        items = await source.list(limit=2)
        assert len(items) <= 2

    @pytest.mark.asyncio
    async def test_get_returns_item_with_content(self, source):
        item = await source.get("a.txt")
        assert item.name == "a.txt"
        assert item.content is not None
        assert "hello world" in item.content
        assert item.metadata.get("size_bytes", 0) > 0

    @pytest.mark.asyncio
    async def test_get_nonexistent_raises(self, source):
        with pytest.raises(FileNotFoundError):
            await source.get("nonexistent.txt")

    @pytest.mark.asyncio
    async def test_search_finds_by_filename(self, source):
        results = await source.search("a")
        names = {item.name for item in results}
        assert len(results) >= 1
        assert "a.txt" in names

    @pytest.mark.asyncio
    async def test_search_returns_content_none(self, source):
        results = await source.search("b")
        for item in results:
            assert item.content is None

    @pytest.mark.asyncio
    async def test_search_no_match_returns_empty(self, source):
        results = await source.search("zzz_nonexistent")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_limit_respected(self, source):
        results = await source.search(".", limit=1)
        assert len(results) <= 1

    @pytest.mark.asyncio
    async def test_extension_filtering(self, source_dir):
        source = LocalFileContentSource(
            path=str(source_dir),
            file_extensions=[".txt"],
        )
        items = await source.list()
        names = {item.name for item in items}
        assert "a.txt" in names
        assert "b.md" not in names
        assert "c.csv" not in names

    @pytest.mark.asyncio
    async def test_custom_metadata(self, source_dir):
        source = LocalFileContentSource(
            path=str(source_dir),
            category="test-cat",
            language="en",
        )
        items = await source.list()
        for item in items:
            assert item.metadata["category"] == "test-cat"
            assert item.metadata["language"] == "en"

    @pytest.mark.asyncio
    async def test_list_in_subdirectory(self, source):
        items = await source.list(path="subdir", recursive=False)
        names = {item.name for item in items}
        assert "d.txt" in names
        assert "a.txt" not in names

    @pytest.mark.asyncio
    async def test_list_nonexistent_directory(self, source):
        items = await source.list(path="no_such_dir")
        assert items == []
