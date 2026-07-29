"""YAML frontmatter parser for Markdown configuration files.

Self-contained — uses only ``yaml`` (already a framework dependency)
and stdlib.  Strips BOM, normalises line endings, and computes
deterministic SHA-256 hashes for hot-reload change detection.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class MarkdownFile:
    """Parsed Markdown file with YAML frontmatter and body.

    Attributes
    ----------
    frontmatter : dict[str, Any]
        Parsed YAML frontmatter, or empty dict when absent.
    body : str
        Markdown body text (stripped of leading/trailing whitespace).
    path : Path
        Absolute path to the source file.
    sha256 : str
        Hex-encoded SHA-256 digest of the raw file bytes (for change detection).
    """

    frontmatter: dict[str, Any]
    body: str
    path: Path
    sha256: str


def compute_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split YAML frontmatter from Markdown body.

    Parameters
    ----------
    text : str
        Raw file content (UTF-8 decoded).

    Returns
    -------
    tuple[dict[str, Any], str]
        (frontmatter_dict, body_text).
        The returned dict is empty when no valid frontmatter block
        is present.  The body is stripped of leading/trailing
        whitespace and leading/trailing blank lines from the
        frontmatter delimiters.
    """
    text = text.removeprefix("\ufeff")

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if not text.startswith("---\n"):
        return {}, text.strip()

    # Skip the opening delimiter line (--- followed by newline)
    opening_end = text.index("\n") + 1
    rest = text[opening_end:]

    # Handle empty frontmatter: after opening, rest starts with ---
    if rest.startswith("---\n"):
        body = rest[4:].strip()
        return {}, body
    if rest.rstrip() == "---":
        body = ""
        return {}, body

    parts = rest.split("\n---", 1)
    if len(parts) == 1:
        return {}, text.strip()

    fm_text = parts[0]
    body = parts[1]

    # Strip the leading newline on body (everything after \n--- starts with \n)
    body = body.removeprefix("\n")
    body = body.strip()

    if not fm_text.strip():
        return {}, body

    try:
        parsed = yaml.safe_load(fm_text)
    except yaml.YAMLError:
        return {}, body

    if not isinstance(parsed, dict):
        return {}, body

    return parsed, body


def load_markdown_file(path: Path | str) -> MarkdownFile:
    """Load and parse a Markdown file with optional YAML frontmatter.

    Parameters
    ----------
    path : Path | str
        Path to the ``.md`` file.

    Returns
    -------
    MarkdownFile
        Parsed result with frontmatter dict, body, and SHA-256 hash.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.is_absolute():
        path = path.resolve()

    raw_bytes = path.read_bytes()
    raw_text = raw_bytes.decode("utf-8")

    frontmatter, body = parse_frontmatter(raw_text)
    sha256 = compute_sha256(raw_bytes)

    return MarkdownFile(
        frontmatter=frontmatter,
        body=body,
        path=path,
        sha256=sha256,
    )
