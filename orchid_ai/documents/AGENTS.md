# documents/ — Document Parsing, Chunking & Ingestion

## Overview

Handles file uploads: extracting text from various formats, chunking for RAG, and ingesting into Qdrant. Used by the `POST /chats/{id}/messages` endpoint when files are attached.

## The Parse-Once Pattern (Critical)

Files are parsed ONCE. The extracted text is reused for two purposes:

```
file_bytes
    │
    ├──→ extract_text()  ──→ text  ──→ prepend to user message (LLM sees it NOW)
    │                          │
    │                          └──→ ingest_document(pre_extracted_text=text)
    │                                  → chunk → embed → Qdrant (RAG for LATER)
    └── NEVER parse again
```

This is especially important for images — vision models are slow and non-deterministic. Parsing twice would give different results and waste time.

## Files

### `parsers.py` — Pluggable Parser Registry

```python
PARSER_REGISTRY = {
    ".pdf":  PDFParser,       # PyMuPDF (fitz)
    ".docx": DOCXParser,      # python-docx
    ".xlsx": XLSXParser,      # openpyxl → pipe-delimited text
    ".csv":  CSVParser,       # stdlib csv
    ".md":   TextParser,      # passthrough
    ".txt":  TextParser,      # passthrough
    ".png":  ImageParser,     # LiteLLM vision model
    ".jpg":  ImageParser,
    ".jpeg": ImageParser,
}

get_parser(filename, *, vision_model="") → DocumentParser
```

**Adding a new format:** Create a class extending `DocumentParser`, implement `async parse(file_bytes, filename) → str`, add to `PARSER_REGISTRY`.

### `chunker.py` — Text Chunking

```python
ChunkConfig(chunk_size=1000, chunk_overlap=200, separator="\n\n")
chunk_text(text, config) → list[str]
```

Splits on paragraph boundaries first, then force-splits long paragraphs. Overlap ensures context continuity across chunks.

### `pipeline.py` — Orchestrator

```python
# Step 1: Extract text (standalone — used by API for prompt augmentation)
text = await extract_text(file_bytes=..., filename=..., vision_model=...)

# Step 2: Chunk + embed + store (with pre-extracted text to avoid re-parsing)
count = await ingest_document(
    file_bytes=..., filename=..., scope=RAGScope(...),
    writer=..., pre_extracted_text=text,  # ← reuse!
)
```

## ImageParser Details

- Uses LiteLLM `acompletion()` with vision model (e.g., `ollama/minicpm-v`)
- Encodes image as base64, sends as `image_url` content block
- Prompt: "Extract all text and describe the content of this image in detail."
- `temperature=0` for deterministic output
- Returns placeholder text if no vision model is configured
- Debug logging: logs image size, model used, and first 300 chars of result

## Common Mistakes

- **Parsing a file twice.** Always use `extract_text()` → pass `pre_extracted_text` to `ingest_document()`.
- **Not passing `vision_model` to `get_parser()`.** Without it, `ImageParser` returns a placeholder, not actual content.
- **Assuming synchronous parsing.** All parsers are `async` — they may call external APIs (vision model).
- **Forgetting scope metadata.** Every indexed chunk must have `tenant_id`, `user_id`, `chat_id`, and `scope` in its metadata.
