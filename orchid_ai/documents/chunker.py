"""
Text chunking for document ingestion.

Wraps LangChain's ``RecursiveCharacterTextSplitter`` with a simple
``ChunkConfig`` dataclass for YAML-driven configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ChunkConfig:
    """Configuration for text chunking."""

    chunk_size: int = 1000  # characters per chunk
    chunk_overlap: int = 200  # overlap between consecutive chunks
    separator: str = "\n\n"  # primary split boundary


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
