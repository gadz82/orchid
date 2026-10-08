"""Built-in in-memory config storage — the framework default.

Process-local, non-durable implementation of :class:`OrchidConfigStorage`.
Used when config storage is enabled but no class path is configured (or the
``"memory"`` sentinel is passed to
:func:`orchid_ai.config.storage_factory.build_config_storage`).

Timestamps use ISO-8601 UTC strings — the same shape every backend returns,
so consumers see an identical row contract regardless of the backend.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime

from .schema_agent import OrchidAgentConfig, _deep_merge
from .storage import OrchidConfigStorage


class OrchidInMemoryConfigStorage(OrchidConfigStorage):
    """Process-local agent-configuration store (dict-backed)."""

    def __init__(self, *, dsn: str = "") -> None:
        # ``dsn`` is accepted for factory-contract compatibility and ignored.
        self._rows: dict[str, dict] = {}

    # ── Lifecycle ────────────────────────────────────────────

    async def init_db(self) -> None:
        """No-op — nothing to open or migrate."""

    async def close(self) -> None:
        """No-op — nothing to release."""

    # ── CRUD ─────────────────────────────────────────────────

    async def list_configs(self) -> list[dict]:
        rows = sorted(self._rows.values(), key=lambda row: row["updated_at"], reverse=True)
        return [copy.deepcopy(row) for row in rows]

    async def get_config(self, name: str) -> dict | None:
        row = self._rows.get(name)
        return copy.deepcopy(row) if row is not None else None

    async def upsert_config(self, name: str, config: dict) -> dict:
        now = _now_iso()
        existing = self._rows.get(name)
        row = {
            "name": name,
            "config": copy.deepcopy(config),
            "created_at": existing["created_at"] if existing is not None else now,
            "updated_at": now,
        }
        self._rows[name] = row
        return copy.deepcopy(row)

    async def patch_config(self, name: str, patch: dict) -> dict | None:
        existing = self._rows.get(name)
        if existing is None:
            return None
        merged = _deep_merge(existing["config"], patch)
        OrchidAgentConfig.model_validate(merged)
        row = {
            "name": name,
            "config": merged,
            "created_at": existing["created_at"],
            "updated_at": _now_iso(),
        }
        self._rows[name] = row
        return copy.deepcopy(row)

    async def delete_config(self, name: str) -> None:
        self._rows.pop(name, None)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["OrchidInMemoryConfigStorage"]
