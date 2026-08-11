from __future__ import annotations

from orchid_ai.documents.chunker import ChunkConfig, chunk_text

# ── ChunkConfig defaults ───────────────────────────────────


def test_chunk_config_defaults():
    cfg = ChunkConfig()
    assert cfg.chunk_size == 1000
    assert cfg.chunk_overlap == 200
    assert cfg.separator == "\n\n"


# ── chunk_text ──────────────────────────────────────────────


def test_empty_text():
    assert chunk_text("") == []


def test_whitespace_only():
    assert chunk_text("   \n\n  \t  ") == []


def test_short_text_single_chunk():
    text = "Hello, this is a short text."
    result = chunk_text(text)
    assert result == [text]


def test_paragraphs_fitting_one_chunk():
    text = "Paragraph one.\n\nParagraph two."
    cfg = ChunkConfig(chunk_size=2000)
    result = chunk_text(text, cfg)
    assert len(result) == 1
    assert "Paragraph one." in result[0]
    assert "Paragraph two." in result[0]


def test_paragraphs_exceeding_chunk_size():
    para = "A" * 100
    # 5 paragraphs of 100 chars each, chunk_size=250
    text = ("\n\n").join([para] * 5)
    cfg = ChunkConfig(chunk_size=250, chunk_overlap=50)
    result = chunk_text(text, cfg)
    assert len(result) > 1


def test_overlap_between_consecutive_chunks():
    para = "word " * 60  # ~300 chars per paragraph
    text = ("\n\n").join([para.strip()] * 5)
    cfg = ChunkConfig(chunk_size=400, chunk_overlap=100)
    result = chunk_text(text, cfg)
    assert len(result) >= 2
    # The end of the first chunk should appear at the start of the second
    tail_of_first = result[0][-100:]
    assert tail_of_first in result[1]


def test_force_split_long_paragraph():
    # A single paragraph longer than chunk_size with no separator
    long_text = "X" * 3000
    cfg = ChunkConfig(chunk_size=500, chunk_overlap=100)
    result = chunk_text(long_text, cfg)
    assert len(result) > 1
    for chunk in result:
        assert len(chunk) <= cfg.chunk_size


def test_custom_separator():
    text = "Part one---Part two---Part three"
    cfg = ChunkConfig(chunk_size=20, chunk_overlap=0, separator="---")
    result = chunk_text(text, cfg)
    assert len(result) >= 2


def test_none_config_uses_defaults():
    text = "Short text."
    result = chunk_text(text, None)
    assert result == [text]
