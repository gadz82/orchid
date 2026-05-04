"""
``ContextualHeaderPostProcessor`` — prepend ``# {title}\\n## {section}\\n``.

ADR-022 §"OrchidChunkPostProcessor" describes the goal: the embedder
and the LLM should both see the document's title and the chunk's
section heading, so semantically-similar chunks from different
sections rank correctly during retrieval.

The Stage 2 implementation derives:

* ``title`` from the filename (stem, ``_``/``-`` → space, title-case).
* ``section`` by walking the original document for markdown headings
  (``#``, ``##``, ``###``…) and tagging each chunk with the **nearest
  preceding heading**.  Falls back to ``"Document"`` when none.

LLM-generated summaries (``# {title}\\n## {section}\\n{summary}\\n\\n``
in the original ADR sketch) land in a future stage — the post-processor
already accepts ``chat_model`` so wiring summaries is purely additive.

The ``section`` value also lands in chunk metadata so retrieval
strategies and metadata filters (ADR-027) can use it directly.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ...core.ingestion import OrchidChunk, OrchidChunkPostProcessor


_HEADING_LINE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _filename_to_title(filename: str) -> str:
    """Derive a human-readable title from a filename.

    Empty filename → ``"Document"``.  Stems with underscores or hyphens
    are split on those characters before title-casing.
    """
    if not filename:
        return "Document"
    stem = Path(filename).stem
    cleaned = stem.replace("_", " ").replace("-", " ").strip()
    return cleaned.title() if cleaned else "Document"


def _build_heading_index(text: str) -> list[tuple[int, str]]:
    """Return ``[(start_position, heading_text)]`` for every markdown heading.

    Sorted by position so a single linear scan can find the nearest
    preceding heading for any character offset.
    """
    return [(m.start(), m.group(2).strip()) for m in _HEADING_LINE.finditer(text)]


def _section_for_position(headings: list[tuple[int, str]], pos: int) -> str:
    """Find the nearest heading whose position is ``<= pos``."""
    section = "Document"
    for h_pos, h_text in headings:
        if h_pos <= pos:
            section = h_text
        else:
            break
    return section


class ContextualHeaderPostProcessor(OrchidChunkPostProcessor):
    """Prepend a two-line header (title + section) to every chunk.

    Idempotent: chunks already carrying a ``contextual_header`` flag in
    metadata are returned unchanged so chaining the post-processor twice
    (e.g. through accidental double-registration) doesn't double-prefix.
    """

    HEADER_TEMPLATE = "# {title}\n## {section}\n\n"

    async def process(
        self,
        chunks: list[OrchidChunk],
        *,
        text: str,
        filename: str,
        chat_model: Any | None = None,
    ) -> list[OrchidChunk]:
        if not chunks:
            return []

        title = _filename_to_title(filename)
        headings = _build_heading_index(text)

        out: list[OrchidChunk] = []
        cursor = 0
        for chunk in chunks:
            if chunk.metadata.get("contextual_header"):
                out.append(chunk)
                continue

            # Locate the chunk in the original text (linear scan with a
            # forward-only cursor to keep cost O(N) for the common case
            # where chunks land in document order).  ``find`` returns -1
            # for unanchored chunks (e.g. semantic strategy that strips
            # whitespace) — fall back to the cursor's previous position.
            pos = text.find(chunk.text, cursor)
            if pos == -1:
                pos = cursor
            else:
                cursor = pos + len(chunk.text)

            section = _section_for_position(headings, pos)
            prefix = self.HEADER_TEMPLATE.format(title=title, section=section)
            out.append(
                OrchidChunk(
                    text=prefix + chunk.text,
                    metadata={
                        **chunk.metadata,
                        "section": section,
                        "title": title,
                        "contextual_header": True,
                    },
                )
            )
        return out
