"""
Abstract config storage interface for database-backed agent configurations.

Any concrete backend (PostgreSQL, SQLite, etc.) must implement the
`OrchidConfigStorage` ABC. The Orchid facade uses this to merge
DB-sourced configs into the `OrchidAgentsConfig` at startup.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class OrchidConfigStorage(ABC):
    """Abstract base class for agent configuration storage backends."""

    @abstractmethod
    async def init_db(self) -> None:
        """Initialise the database connection and run pending migrations."""

    @abstractmethod
    async def close(self) -> None:
        """Release database connections / pools."""

    @abstractmethod
    async def list_configs(self) -> list[dict]:
        """Return all rows as ``[{"name": ..., "config": ..., "created_at": ..., "updated_at": ...}, ...]``.

        The ``config`` value is a deserialized Python dict (JSON object).
        Timestamps are ISO-format strings.
        """

    @abstractmethod
    async def get_config(self, name: str) -> dict | None:
        """Return one row as a dict or ``None`` when not found.

        The ``config`` value is a deserialized Python dict.
        """

    @abstractmethod
    async def upsert_config(self, name: str, config: dict) -> dict:
        """Create or replace a config. Returns the full row dict.

        Parameters
        ----------
        name : str
            Unique agent name (primary key).
        config : dict
            Validated ``OrchidAgentConfig`` data as a Python dict
            (typically from ``.model_dump(mode="json")``).

        Returns
        -------
        dict
            Full row including server-assigned ``created_at`` and ``updated_at``.
        """

    @abstractmethod
    async def patch_config(self, name: str, patch: dict) -> dict | None:
        """Partial update — merge ``patch`` over the existing config.

        Reads the existing row, deep-merges ``patch`` into it, validates
        the result, writes back, and returns the full updated row.
        Returns ``None`` if the row does not exist.

        Parameters
        ----------
        name : str
            Agent name (row must exist).
        patch : dict
            Partial config fields to overlay.

        Returns
        -------
        dict | None
            Full updated row, or ``None`` if not found.
        """

    @abstractmethod
    async def delete_config(self, name: str) -> None:
        """Delete a config by name. Idempotent — no error if already gone."""
