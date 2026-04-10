"""
Migration runner — discovers and applies up/down migrations.

Each migration module must expose:
    VERSION: str          — unique ordered identifier (e.g. "001")
    DESCRIPTION: str      — human-readable summary
    async def up(conn, *, dialect)    — apply the migration
    async def down(conn, *, dialect)  — revert the migration

The runner tracks applied versions in a `_migrations` table.
The `conn` passed to up/down is a backend-specific connection object
(e.g. asyncpg.Connection for PostgreSQL, aiosqlite.Connection for SQLite).

The `dialect` keyword ("postgres", "sqlite") tells migrations which
SQL flavour to emit.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


@dataclass
class Migration:
    """A discovered migration module."""

    version: str
    description: str
    up: Callable[..., Coroutine]
    down: Callable[..., Coroutine]


def discover_migrations(package: str | None = None) -> list[Migration]:
    """
    Scan a Python package for migration modules, sorted by version.

    Parameters
    ----------
    package : str | None
        Dotted import path of the package to scan (e.g.
        ``"examples.basketball.storage.migrations"``).  If *None*, falls back
        to ``src.persistence.migrations`` (the library default, used by the
        built-in ``PostgresChatStorage``).

    Each module in the package whose name starts with ``v`` and
    exposes ``VERSION``, ``up()``, and ``down()`` is collected.
    """
    if package is None:
        from . import __path__ as pkg_path, __name__ as pkg_name
    else:
        mod = importlib.import_module(package)
        pkg_path = mod.__path__
        pkg_name = mod.__name__

    migrations: list[Migration] = []
    for _importer, modname, ispkg in pkgutil.iter_modules(pkg_path):
        if modname.startswith("v") and not ispkg:
            full_name = f"{pkg_name}.{modname}"
            loaded = importlib.import_module(full_name)
            if hasattr(loaded, "VERSION") and hasattr(loaded, "up") and hasattr(loaded, "down"):
                migrations.append(
                    Migration(
                        version=loaded.VERSION,
                        description=getattr(loaded, "DESCRIPTION", ""),
                        up=loaded.up,
                        down=loaded.down,
                    )
                )
    return sorted(migrations, key=lambda m: m.version)


class MigrationRunner:
    """
    Backend-agnostic migration runner.

    Subclasses must implement the tracking-table hooks and set
    ``dialect`` and ``migrations_package`` so that migration SQL is
    generated for the correct database engine and discovered from
    the correct location.
    """

    dialect: str = "postgres"
    migrations_package: str | None = None  # dotted path — set by subclass

    async def ensure_migrations_table(self, conn: Any) -> None:
        """Create the _migrations tracking table if it doesn't exist."""
        raise NotImplementedError

    async def get_applied_versions(self, conn: Any) -> set[str]:
        """Return the set of already-applied migration versions."""
        raise NotImplementedError

    async def record_version(self, conn: Any, version: str, description: str) -> None:
        """Record a migration as applied."""
        raise NotImplementedError

    async def remove_version(self, conn: Any, version: str) -> None:
        """Remove a migration record (on rollback)."""
        raise NotImplementedError

    async def run_up(self, conn: Any) -> list[str]:
        """Apply all pending migrations. Returns list of applied versions."""
        await self.ensure_migrations_table(conn)
        applied = await self.get_applied_versions(conn)
        all_migrations = discover_migrations(self.migrations_package)

        newly_applied: list[str] = []
        for m in all_migrations:
            if m.version in applied:
                continue
            logger.info("[Migration] Applying v%s — %s", m.version, m.description)
            await m.up(conn, dialect=self.dialect)
            await self.record_version(conn, m.version, m.description)
            newly_applied.append(m.version)

        if newly_applied:
            logger.info("[Migration] Applied %d migration(s): %s", len(newly_applied), newly_applied)
        else:
            logger.info("[Migration] Database is up to date")
        return newly_applied

    async def run_down(self, conn: Any, target_version: str = "") -> list[str]:
        """
        Roll back migrations down to (but not including) target_version.
        If target_version is empty, rolls back ALL migrations.
        """
        await self.ensure_migrations_table(conn)
        applied = await self.get_applied_versions(conn)
        all_migrations = discover_migrations(self.migrations_package)

        to_rollback = [m for m in reversed(all_migrations) if m.version in applied and m.version > target_version]

        rolled_back: list[str] = []
        for m in to_rollback:
            logger.info("[Migration] Rolling back v%s — %s", m.version, m.description)
            await m.down(conn, dialect=self.dialect)
            await self.remove_version(conn, m.version)
            rolled_back.append(m.version)

        if rolled_back:
            logger.info("[Migration] Rolled back %d migration(s): %s", len(rolled_back), rolled_back)
        return rolled_back
