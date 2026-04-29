"""End-to-end SQLite test for integrator migrations.

Exercises ``OrchidSQLiteChatStorage(extra_migrations_package=...)`` against a
real in-memory aiosqlite database: the framework's v001 runs first, then
the integrator migration creates an extra table.  Verifies the table
exists and the recorded version keys use the ``ext:`` prefix.
"""

from __future__ import annotations

import sys
import types

import pytest

from orchid_ai.persistence.sqlite import OrchidSQLiteChatStorage

_EXT_PACKAGE = "tests._fake_integrator_migrations_extras"


def _install_fake_integrator_package() -> None:
    """Register a one-file integrator migration package in ``sys.modules``."""
    if _EXT_PACKAGE in sys.modules:
        return

    pkg = types.ModuleType(_EXT_PACKAGE)
    pkg.__path__ = []

    mod_name = f"{_EXT_PACKAGE}.v001_integrator_table"
    mod = types.ModuleType(mod_name)
    mod.VERSION = "001"
    mod.DESCRIPTION = "Integrator-specific table"

    async def _up(conn, *, dialect: str = "sqlite") -> None:
        await conn.execute("CREATE TABLE IF NOT EXISTS integrator_widgets (id TEXT PRIMARY KEY)")
        await conn.commit()

    async def _down(conn, *, dialect: str = "sqlite") -> None:
        await conn.execute("DROP TABLE IF EXISTS integrator_widgets")
        await conn.commit()

    mod.up = _up
    mod.down = _down
    sys.modules[mod_name] = mod
    setattr(pkg, "v001_integrator_table", mod)
    pkg._fake_submodules = [mod_name]  # type: ignore[attr-defined]
    sys.modules[_EXT_PACKAGE] = pkg


@pytest.fixture(autouse=True)
def _patch_pkgutil_iter_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the synthetic integrator package discoverable by ``pkgutil``.

    Mirrors the helper in ``test_migration_runner.py``: returns the
    ``_fake_submodules`` list for any fake package whose ``__path__`` is
    the one pkgutil was asked to scan.
    """
    import pkgutil

    real_iter = pkgutil.iter_modules

    def _iter(path=None, prefix=""):
        for mod in list(sys.modules.values()):
            if getattr(mod, "__path__", None) is path and hasattr(mod, "_fake_submodules"):
                for sub in mod._fake_submodules:
                    yield (None, sub.rsplit(".", 1)[-1], False)
                return
        yield from real_iter(path=path, prefix=prefix)

    monkeypatch.setattr(pkgutil, "iter_modules", _iter)


@pytest.mark.asyncio
async def test_integrator_migration_runs_after_framework() -> None:
    _install_fake_integrator_package()

    storage = OrchidSQLiteChatStorage(
        dsn=":memory:",
        extra_migrations_package=_EXT_PACKAGE,
    )
    await storage.init_db()
    try:
        # The integrator table must exist.
        cursor = await storage._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('chat_sessions', 'mcp_oauth_tokens', "
            "'mcp_client_registrations', 'integrator_widgets')"
        )
        tables = {row[0] async for row in cursor}
        assert tables == {
            "chat_sessions",
            "mcp_oauth_tokens",
            "mcp_client_registrations",
            "integrator_widgets",
        }

        # Framework migration runs first (bare key), integrator
        # migration runs last with the ``ext:`` prefix.
        cursor = await storage._conn.execute("SELECT version FROM _migrations ORDER BY version")
        versions = [row[0] async for row in cursor]
        assert versions == ["001", "ext:001"]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_without_extras_only_framework_tables() -> None:
    storage = OrchidSQLiteChatStorage(dsn=":memory:")
    await storage.init_db()
    try:
        cursor = await storage._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('chat_sessions', 'mcp_oauth_tokens', "
            "'mcp_client_registrations', 'integrator_widgets')"
        )
        tables = {row[0] async for row in cursor}
        assert "chat_sessions" in tables
        assert "mcp_oauth_tokens" in tables
        assert "mcp_client_registrations" in tables  # unified v001
        assert "integrator_widgets" not in tables

        cursor = await storage._conn.execute("SELECT version FROM _migrations ORDER BY version")
        versions = [row[0] async for row in cursor]
        # Unified framework migration — no integrator extras.
        assert versions == ["001"]
    finally:
        await storage.close()
