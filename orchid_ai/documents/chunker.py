"""
Text chunking for document ingestion.

Wraps LangChain's ``RecursiveCharacterTextSplitter`` with a simple
``ChunkConfig`` dataclass for YAML-driven configuration.

When ``parent_chunk_size`` > 0, the **Parent Document** pattern is used:
child chunks (small, high-precision embeddings) carry their parent chunk
content in metadata.  The retriever returns the richer parent context
when a child chunk matches, improving answer quality.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ChunkConfig:
    """Configuration for text chunking.

    When ``parent_chunk_size`` is 0 (default), standard chunking is used:
    the text is split into ``chunk_size``-char pieces with overlap.

    When ``parent_chunk_size`` > 0, the Parent Document pattern is enabled:
    text is first split into large parent chunks, then each parent is
    sub-split into small child chunks.  Child chunks are embedded for
    precise retrieval, but their parent's content is stored in metadata
    so the LLM receives richer context.
    """

    chunk_size: int = 1000  # characters per child chunk
    chunk_overlap: int = 200  # overlap between consecutive chunks
    separator: str = "\n\n"  # primary split boundary
    parent_chunk_size: int = 0  # 0 = disabled; > 0 = parent chunk size (e.g. 2000)
    parent_chunk_overlap: int = 200  # overlap between parent chunks


@dataclass
class ParentChildChunk:
    """A child chunk with a reference to its parent content."""

    child_text: str
    parent_text: str
    parent_index: int
    child_index: int


def chunk_text(text: str, config: ChunkConfig | None = None) -> list[str]:
    """
    Split text into overlapping chunks using LangChain's
    ``RecursiveCharacterTextSplitter``.

    The splitter recursively tries ``["\\n\\n", "\\n", " ", ""]``
    as separators, producing chunks that respect natural boundaries.
    """
    cfg = config or ChunkConfig()

    if not text.strip():
        return []

    if len(text) <= cfg.chunk_size:
        return [text.strip()]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=[cfg.separator, "\n", " ", ""],
    )

    return splitter.split_text(text)


def parent_child_chunk_text(
    text: str,
    config: ChunkConfig | None = None,
) -> list[ParentChildChunk]:
    """Split text into parent chunks, then sub-split each into child chunks.

    Returns a flat list of ``ParentChildChunk`` objects where each child
    carries its parent's full text.  The child text is used for embedding
    (precise retrieval), while the parent text is stored in metadata and
    returned to the LLM (richer context).

    Requires ``config.parent_chunk_size`` > 0.
    """
    cfg = config or ChunkConfig()

    if not text.strip():
        return []

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.parent_chunk_size,
        chunk_overlap=cfg.parent_chunk_overlap,
        separators=[cfg.separator, "\n", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=[cfg.separator, "\n", " ", ""],
    )

    parent_chunks = parent_splitter.split_text(text)
    result: list[ParentChildChunk] = []

    for pi, parent_text in enumerate(parent_chunks):
        child_chunks = child_splitter.split_text(parent_text)
        if not child_chunks:
            # Parent is small enough to be a single child
            child_chunks = [parent_text.strip()]
        for ci, child_text in enumerate(child_chunks):
            result.append(
                ParentChildChunk(
                    child_text=child_text,
                    parent_text=parent_text,
                    parent_index=pi,
                    child_index=ci,
                )
            )

    return result
