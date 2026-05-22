from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..core.content import OrchidContentItem, OrchidContentSource

logger = logging.getLogger(__name__)

_DEFAULT_EXTENSIONS = [".pdf", ".txt", ".md", ".docx", ".xlsx", ".csv"]


class LocalFileContentSource(OrchidContentSource):
    def __init__(
        self,
        *,
        path: str,
        file_extensions: list[str] | None = None,
        **metadata: Any,
    ) -> None:
        self._root = Path(path).resolve()
        self._extensions = file_extensions or _DEFAULT_EXTENSIONS
        self._metadata = metadata

    def _matches_extension(self, name: str) -> bool:
        ext = Path(name).suffix.lower()
        return ext in {e.lower() for e in self._extensions}

    async def list(self, path: str = "", recursive: bool = False, limit: int = 100) -> list[OrchidContentItem]:
        search_dir = self._root / path
        if not search_dir.exists():
            return []
        pattern = "**/*" if recursive else "*"
        items: list[OrchidContentItem] = []
        for file_path in search_dir.glob(pattern):
            if not file_path.is_file():
                continue
            if not self._matches_extension(file_path.name):
                continue
            if len(items) >= limit:
                break
            stat = file_path.stat()
            rel = str(file_path.relative_to(self._root))
            items.append(
                OrchidContentItem(
                    path=rel,
                    name=file_path.name,
                    content_type=file_path.suffix.lower(),
                    metadata={
                        "size_bytes": stat.st_size,
                        "last_modified": stat.st_mtime,
                        **self._metadata,
                    },
                    content=None,
                )
            )
        return items

    async def get(self, path: str) -> OrchidContentItem:
        file_path = (self._root / path).resolve()
        if not file_path.is_relative_to(self._root):
            raise ValueError(f"Path {path!r} escapes root directory")
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not file_path.is_file():
            raise IsADirectoryError(f"Path is not a file: {path}")

        file_bytes = file_path.read_bytes()
        from ..documents.pipeline import extract_text

        text = await extract_text(file_bytes=file_bytes, filename=file_path.name)
        stat = file_path.stat()
        return OrchidContentItem(
            path=str(file_path.relative_to(self._root)),
            name=file_path.name,
            content_type=file_path.suffix.lower(),
            metadata={
                "size_bytes": stat.st_size,
                "last_modified": stat.st_mtime,
                **self._metadata,
            },
            content=text,
        )

    async def search(self, query: str, recursive: bool = True, limit: int = 10) -> list[OrchidContentItem]:
        query_lower = query.lower()
        pattern = "**/*" if recursive else "*"
        items: list[OrchidContentItem] = []
        for file_path in self._root.glob(pattern):
            if not file_path.is_file():
                continue
            if not self._matches_extension(file_path.name):
                continue
            if query_lower not in file_path.name.lower():
                continue
            if len(items) >= limit:
                break
            stat = file_path.stat()
            rel = str(file_path.relative_to(self._root))
            items.append(
                OrchidContentItem(
                    path=rel,
                    name=file_path.name,
                    content_type=file_path.suffix.lower(),
                    metadata={
                        "size_bytes": stat.st_size,
                        "last_modified": stat.st_mtime,
                        **self._metadata,
                    },
                    content=None,
                )
            )
        return items
