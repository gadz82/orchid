"""
Text chunking for document ingestion.

Splits extracted text into overlapping chunks suitable for embedding.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChunkConfig:
    """Configuration for text chunking."""

    chunk_size: int = 1000       # characters per chunk
    chunk_overlap: int = 200     # overlap between consecutive chunks
    separator: str = "\n\n"      # primary split boundary


def chunk_text(text: str, config: ChunkConfig | None = None) -> list[str]:
    """
    Split text into overlapping chunks.

    Uses a simple recursive approach: first split on the separator,
    then merge small pieces into chunks of the target size with overlap.
    """
    cfg = config or ChunkConfig()

    if not text.strip():
        return []

    if len(text) <= cfg.chunk_size:
        return [text.strip()]

    # Split on separator first
    paragraphs = text.split(cfg.separator)

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph exceeds chunk size, finalize current chunk
        if current and len(current) + len(para) + 2 > cfg.chunk_size:
            chunks.append(current.strip())
            # Start new chunk with overlap from the end of current
            overlap_text = current[-cfg.chunk_overlap:] if cfg.chunk_overlap > 0 else ""
            current = overlap_text + "\n\n" + para if overlap_text else para
        else:
            current = current + "\n\n" + para if current else para

    # Don't forget the last chunk
    if current.strip():
        chunks.append(current.strip())

    # Handle paragraphs that are individually larger than chunk_size
    final_chunks: list[str] = []
    for chunk in chunks:
        if len(chunk) <= cfg.chunk_size:
            final_chunks.append(chunk)
        else:
            # Force-split long chunks by character
            for i in range(0, len(chunk), cfg.chunk_size - cfg.chunk_overlap):
                piece = chunk[i:i + cfg.chunk_size]
                if piece.strip():
                    final_chunks.append(piece.strip())

    return final_chunks
