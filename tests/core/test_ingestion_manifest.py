from __future__ import annotations

import pytest

from orchid_ai.core.ingestion_manifest import OrchidIngestionManifest


class MinimalManifest(OrchidIngestionManifest):
    async def should_skip(self, source_id: str, content_hash: str, namespace: str, scope: str = "") -> bool:
        return False

    async def record(
        self,
        source_id: str,
        content_hash: str,
        namespace: str,
        document_ids: list[str],
        scope: str = "",
    ) -> None:
        return None

    async def remove(self, source_id: str, namespace: str, scope: str = "") -> None:
        return None

    async def list_known(self, namespace: str, scope: str = "") -> set[str]:
        return set()

    async def get_document_ids(self, source_id: str, namespace: str, scope: str = "") -> list[str]:
        return []

    async def close(self) -> None:
        return None


def test_orchid_ingestion_manifest_abc_cannot_instantiate():
    with pytest.raises(TypeError):
        OrchidIngestionManifest()


@pytest.mark.asyncio
async def test_minimal_manifest_conforms():
    manifest = MinimalManifest()

    assert await manifest.should_skip("a", "b", "c") is False
    assert await manifest.record("a", "b", "c", ["d"]) is None
    assert await manifest.remove("a", "c") is None
    assert await manifest.list_known("c") == set()
    assert await manifest.get_document_ids("a", "c") == []
    assert await manifest.close() is None
