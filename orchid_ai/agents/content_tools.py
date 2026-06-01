from __future__ import annotations

import logging
from typing import Any

from ..core.content import OrchidContentSource

logger = logging.getLogger(__name__)


def _serialise_item(item: Any) -> dict[str, Any]:
    if hasattr(item, "__dict__"):
        return {
            "path": getattr(item, "path", ""),
            "name": getattr(item, "name", ""),
            "content_type": getattr(item, "content_type", ""),
            "metadata": dict(getattr(item, "metadata", {})),
            "content": getattr(item, "content", None),
        }
    return {"error": "unknown item format"}


async def list_content_files(
    *,
    path: str = "",
    recursive: bool = False,
    limit: int | str = 100,
    content_sources: list[OrchidContentSource] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    if not content_sources:
        return []
    _limit = int(limit)
    results: list[dict[str, Any]] = []
    for source in content_sources:
        try:
            items = await source.list(path=path, recursive=recursive, limit=_limit)
            for item in items:
                results.append(_serialise_item(item))
            if len(results) >= _limit:
                results = results[:_limit]
                break
        except Exception as exc:
            logger.warning("[ContentTools] list_content_files failed on source: %s", exc)
            continue
    return results


async def search_content_files(
    *,
    query: str,
    recursive: bool = True,
    limit: int | str = 10,
    content_sources: list[OrchidContentSource] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    if not content_sources:
        return []
    _limit = int(limit)
    results: list[dict[str, Any]] = []
    for source in content_sources:
        try:
            items = await source.search(query=query, recursive=recursive, limit=_limit)
            for item in items:
                results.append(_serialise_item(item))
            if len(results) >= _limit:
                results = results[:_limit]
                break
        except Exception as exc:
            logger.warning("[ContentTools] search_content_files failed on source: %s", exc)
            continue
    return results


async def read_content_file(
    *,
    path: str,
    source: str = "",
    content_sources: list[OrchidContentSource] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    if not content_sources:
        return {"error": "no content sources configured"}
    for cs in content_sources:
        try:
            item = await cs.get(path)
            return _serialise_item(item)
        except (FileNotFoundError, IsADirectoryError):
            continue
        except Exception as exc:
            logger.warning("[ContentTools] read_content_file failed: %s", exc)
            continue
    return {"error": f"file not found in any content source: {path}"}
