"""Tests for Document, SearchResult, and vector store ABCs from src/core/repository.py."""

from __future__ import annotations

import pytest

from orchid_ai.core.repository import (
    Document,
    SearchResult,
    VectorReader,
    VectorStoreAdmin,
    VectorStoreRepository,
    VectorWriter,
)


# ── Document ──


def test_document_defaults():
    doc = Document(id="d-1", page_content="hello")
    assert doc.id == "d-1"
    assert doc.page_content == "hello"
    assert doc.metadata == {}


def test_document_all_fields():
    doc = Document(id="d-2", page_content="world", metadata={"k": "v"})
    assert doc.metadata == {"k": "v"}
    assert doc.id == "d-2"


# ── SearchResult ──


def test_search_result():
    doc = Document(id="d-1", page_content="hello")
    sr = SearchResult(document=doc, score=0.95)
    assert sr.document is doc
    assert sr.score == 0.95


# ── ABCs cannot be instantiated ──


def test_vector_reader_is_abstract():
    with pytest.raises(TypeError):
        VectorReader()


def test_vector_writer_is_abstract():
    with pytest.raises(TypeError):
        VectorWriter()


def test_vector_store_admin_is_abstract():
    with pytest.raises(TypeError):
        VectorStoreAdmin()


def test_vector_store_repository_is_abstract():
    with pytest.raises(TypeError):
        VectorStoreRepository()


# ── Concrete implementations work (via conftest mocks) ──


@pytest.mark.asyncio
async def test_mock_reader_records_calls(mock_reader):
    results = await mock_reader.retrieve("q", "ns", k=3)
    assert results == []
    assert len(mock_reader.calls) == 1
    assert mock_reader.calls[0]["query"] == "q"
    assert mock_reader.calls[0]["namespace"] == "ns"
    assert mock_reader.calls[0]["k"] == 3


@pytest.mark.asyncio
async def test_mock_writer_records_index(mock_writer):
    docs = [Document(id="d-1", page_content="hi")]
    await mock_writer.index(docs, "ns")
    assert len(mock_writer.indexed) == 1
    assert mock_writer.indexed[0] == (docs, "ns")


@pytest.mark.asyncio
async def test_mock_writer_records_upsert(mock_writer):
    docs = [Document(id="d-1", page_content="hi")]
    await mock_writer.upsert(docs, "ns")
    assert len(mock_writer.upserted) == 1


@pytest.mark.asyncio
async def test_mock_writer_records_delete(mock_writer):
    await mock_writer.delete(["d-1"], "ns")
    assert len(mock_writer.deleted) == 1
    assert mock_writer.deleted[0] == (["d-1"], "ns")


@pytest.mark.asyncio
async def test_mock_repository_has_reader_and_writer(mock_repository):
    results = await mock_repository.retrieve("q", "ns")
    assert results == []
    docs = [Document(id="d-1", page_content="hi")]
    await mock_repository.upsert(docs, "ns")
    assert len(mock_repository.upserted) == 1
    await mock_repository.ensure_collections(["ns"])


@pytest.mark.asyncio
async def test_null_reader_returns_empty(null_reader):
    results = await null_reader.retrieve("q", "ns", k=10)
    assert results == []
