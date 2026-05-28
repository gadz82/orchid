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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

# Version prefix for integrator-supplied migrations. Records land in the
# same ``_migrations`` table as framework migrations; the prefix keeps the
# primary key unique when a consumer starts their own numbering at "001".
EXTRA_NAMESPACE_PREFIX = "ext:"


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
        to ``orchid_ai.persistence.migrations`` (the library default, used by
        the built-in ``OrchidSQLiteChatStorage``).

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


class OrchidMigrationRunner(ABC):
    """
    Backend-agnostic migration runner.

    Subclasses must implement the tracking-table hooks and set
    ``dialect`` and ``migrations_package`` so that migration SQL is
    generated for the correct database engine and discovered from
    the correct location.

    Integrators may also supply ``extra_migrations_package`` (dotted
    import path) via the constructor.  Migrations in that package run
    AFTER framework migrations and are recorded in the same tracking
    table with an ``"ext:"`` prefix (see :data:`EXTRA_NAMESPACE_PREFIX`)
    so their version numbers can overlap the framework's without
    colliding on the primary key.
    """

    dialect: str = "postgres"
    migrations_package: str | None = None  # dotted path — set by subclass

    def __init__(self, *, extra_migrations_package: str | None = None) -> None:
        self.extra_migrations_package = extra_migrations_package

    @abstractmethod
    async def ensure_migrations_table(self, conn: Any) -> None:
        """Create the _migrations tracking table if it doesn't exist."""
        ...

    @abstractmethod
    async def get_applied_versions(self, conn: Any) -> set[str]:
        """Return the set of already-applied migration versions."""
        ...

    @abstractmethod
    async def record_version(self, conn: Any, version: str, description: str) -> None:
        """Record a migration as applied."""
        ...

    @abstractmethod
    async def remove_version(self, conn: Any, version: str) -> None:
        """Remove a migration record (on rollback)."""
        ...

    async def _apply_pass(
        self,
        conn: Any,
        applied: set[str],
        package: str | None,
        *,
        namespace: str = "",
    ) -> list[str]:
        """Apply all pending migrations from ``package``, prefixing their
        recorded version with ``namespace``.  Returns the list of
        recorded keys (namespace-prefixed)."""
        if package is None:
            return []

        newly_applied: list[str] = []
        for m in discover_migrations(package):
            key = f"{namespace}{m.version}"
            if key in applied:
                continue
            logger.info("[Migration] Applying %s — %s", key, m.description)
            await m.up(conn, dialect=self.dialect)
            await self.record_version(conn, key, m.description)
            newly_applied.append(key)
        return newly_applied

    async def run_up(self, conn: Any) -> list[str]:
        """Apply all pending migrations. Returns list of applied versions.

        Framework migrations run first (bare version keys); integrator
        migrations run second (``ext:`` prefixed keys) when an
        ``extra_migrations_package`` is configured.
        """
        await self.ensure_migrations_table(conn)
        applied = await self.get_applied_versions(conn)

        newly_applied: list[str] = []
        newly_applied.extend(await self._apply_pass(conn, applied, self.migrations_package))
        newly_applied.extend(
            await self._apply_pass(
                conn,
                applied,
                self.extra_migrations_package,
                namespace=EXTRA_NAMESPACE_PREFIX,
            )
        )

        if newly_applied:
            logger.info("[Migration] Applied %d migration(s): %s", len(newly_applied), newly_applied)
        else:
            logger.info("[Migration] Database is up to date")
        return newly_applied

    async def _rollback_pass(
        self,
        conn: Any,
        applied: set[str],
        package: str | None,
        target_version: str,
        *,
        namespace: str = "",
    ) -> list[str]:
        """Roll back migrations from ``package`` down to (but not
        including) ``target_version``.  Versions are compared in the
        namespaced form so ``ext:`` keys sort independently of framework
        keys."""
        if package is None:
            return []

        rolled_back: list[str] = []
        migrations = discover_migrations(package)
        for m in reversed(migrations):
            key = f"{namespace}{m.version}"
            if key not in applied or key <= target_version:
                continue
            logger.info("[Migration] Rolling back %s — %s", key, m.description)
            await m.down(conn, dialect=self.dialect)
            await self.remove_version(conn, key)
            rolled_back.append(key)
        return rolled_back

    async def run_down(self, conn: Any, target_version: str = "") -> list[str]:
        """Roll back migrations down to (but not including)
        ``target_version``.  If ``target_version`` is empty, rolls back
        ALL migrations.

        Integrator migrations are rolled back first (reverse dependency
        order), then framework migrations.
        """
        await self.ensure_migrations_table(conn)
        applied = await self.get_applied_versions(conn)

        rolled_back: list[str] = []
        rolled_back.extend(
            await self._rollback_pass(
                conn,
                applied,
                self.extra_migrations_package,
                target_version,
                namespace=EXTRA_NAMESPACE_PREFIX,
            )
        )
        rolled_back.extend(await self._rollback_pass(conn, applied, self.migrations_package, target_version))

        if rolled_back:
            logger.info("[Migration] Rolled back %d migration(s): %s", len(rolled_back), rolled_back)
        return rolled_back
