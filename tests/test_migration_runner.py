"""Tests for the two-pass MigrationRunner (framework + integrator extras).

The runner is exercised against a minimal in-memory fake of the
backend-specific connection contract (``execute`` / ``get_applied_versions``)
so the tests stay backend-agnostic and do not need aiosqlite / asyncpg.

Fixture migration packages are injected into ``sys.modules`` at import
time — no temp dirs or real imports required.
"""

from __future__ import annotations

import sys
import types

import pytest

from orchid_ai.persistence.migrations.runner import (
    EXTRA_NAMESPACE_PREFIX,
    MigrationRunner,
    discover_migrations,
)


# ── Fake connection + runner used by the tests ──────────────────────


class _FakeConn:
    """Records every statement executed against it — nothing more."""

    def __init__(self) -> None:
        self.executed: list[str] = []

    async def execute(self, sql: str, *_args: object) -> None:
        self.executed.append(sql)


class _FakeRunner(MigrationRunner):
    """Concrete runner that tracks applied versions in a Python set."""

    dialect = "sqlite"
    # ``migrations_package`` is set per-test below.

    def __init__(self, *, extra_migrations_package: str | None = None) -> None:
        super().__init__(extra_migrations_package=extra_migrations_package)
        self.applied: set[str] = set()
        self.ensured = False

    async def ensure_migrations_table(self, conn: object) -> None:
        self.ensured = True

    async def get_applied_versions(self, conn: object) -> set[str]:
        return set(self.applied)

    async def record_version(self, conn: object, version: str, description: str) -> None:
        self.applied.add(version)

    async def remove_version(self, conn: object, version: str) -> None:
        self.applied.discard(version)


# ── Fixture migration-package builders ──────────────────────────────


def _register_fake_package(name: str, migrations: dict[str, str]) -> None:
    """Create ``sys.modules[name]`` with submodules ``v{VERSION}_...`` that
    each expose ``up``/``down`` appending a marker to ``conn.executed``.

    ``migrations`` maps ``version -> description``.
    """
    pkg = types.ModuleType(name)
    pkg.__path__ = []  # treat as a namespace package so pkgutil walks it
    pkg_submodules: list[str] = []

    for version, description in migrations.items():
        mod_name = f"{name}.v{version}_test"
        mod = types.ModuleType(mod_name)
        mod.VERSION = version
        mod.DESCRIPTION = description

        async def _up(conn: _FakeConn, *, dialect: str = "sqlite", _v: str = version) -> None:
            conn.executed.append(f"UP v{_v}")

        async def _down(conn: _FakeConn, *, dialect: str = "sqlite", _v: str = version) -> None:
            conn.executed.append(f"DOWN v{_v}")

        mod.up = _up
        mod.down = _down
        sys.modules[mod_name] = mod
        pkg_submodules.append(mod_name)
        # Attach as attribute so pkgutil.iter_modules can find it.
        setattr(pkg, f"v{version}_test", mod)

    sys.modules[name] = pkg
    # pkgutil.iter_modules scans ``__path__``; since it's empty, we override
    # ``iter_modules`` via the package's ``__path__`` trick below — instead,
    # we monkey-patch the package to use a custom ``__path__`` that points
    # to a generator-backed finder.  Simpler: emulate the finder by
    # attaching attributes and overriding ``discover_migrations`` lookup.
    # Cleanest solution — point ``__path__`` at an iterator-like object that
    # pkgutil recognises.  We instead lean on a dedicated finder:

    class _Finder:
        def __init__(self, submodules: list[str]) -> None:
            self.submodules = submodules

        def iter_modules(self, _path: object = None, _prefix: str = "") -> list:
            # pkgutil.iter_modules expects (importer, modname, ispkg) triples.
            return [(None, sub.rsplit(".", 1)[-1], False) for sub in self.submodules]

    # Replace ``__path__`` with a fake list and stash our fake finder on the
    # package; ``discover_migrations`` uses ``pkgutil.iter_modules(pkg_path)``
    # which iterates known finders — not reliable for synthetic pkgs.
    # For simplicity we just override ``pkgutil.iter_modules`` behaviour by
    # exposing the submodule names directly:
    pkg._fake_submodules = pkg_submodules  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def _patch_pkgutil_iter_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``pkgutil.iter_modules`` recognise our synthetic packages.

    The real implementation scans filesystem importers; our fixture packages
    live entirely in-memory.  We patch it so that when ``discover_migrations``
    asks for a package's children, we look at the ``_fake_submodules`` list
    we stored on the package.
    """
    import pkgutil

    real_iter = pkgutil.iter_modules

    def _iter(path=None, prefix=""):
        # Find any of our fake packages whose path-list is ``path``.
        for mod in list(sys.modules.values()):
            if getattr(mod, "__path__", None) is path and hasattr(mod, "_fake_submodules"):
                for sub in mod._fake_submodules:
                    yield (None, sub.rsplit(".", 1)[-1], False)
                return
        yield from real_iter(path=path, prefix=prefix)

    monkeypatch.setattr(pkgutil, "iter_modules", _iter)


# ── Tests ────────────────────────────────────────────────────────────


class TestDiscoverMigrations:
    def test_discovers_modules_sorted(self) -> None:
        _register_fake_package("_fake_fw_a", {"002": "two", "001": "one"})
        got = discover_migrations("_fake_fw_a")
        assert [m.version for m in got] == ["001", "002"]
        assert [m.description for m in got] == ["one", "two"]


class TestRunUp:
    @pytest.mark.asyncio
    async def test_framework_only(self) -> None:
        _register_fake_package("_fake_fw_b", {"001": "initial"})
        runner = _FakeRunner()
        runner.migrations_package = "_fake_fw_b"

        conn = _FakeConn()
        applied = await runner.run_up(conn)

        assert runner.ensured is True
        assert applied == ["001"]
        assert runner.applied == {"001"}
        assert "UP v001" in conn.executed

    @pytest.mark.asyncio
    async def test_framework_and_extras_namespaced(self) -> None:
        """Extras run after framework and are recorded with the ``ext:`` prefix."""
        _register_fake_package("_fake_fw_c", {"001": "core"})
        _register_fake_package("_fake_ext_c", {"001": "integrator"})

        runner = _FakeRunner(extra_migrations_package="_fake_ext_c")
        runner.migrations_package = "_fake_fw_c"

        conn = _FakeConn()
        applied = await runner.run_up(conn)

        assert applied == ["001", f"{EXTRA_NAMESPACE_PREFIX}001"]
        assert runner.applied == {"001", "ext:001"}
        # Order: framework first, then extras.
        assert conn.executed == ["UP v001", "UP v001"]

    @pytest.mark.asyncio
    async def test_version_collision_does_not_conflict(self) -> None:
        """Framework ``001`` and integrator ``001`` coexist thanks to the prefix."""
        _register_fake_package("_fake_fw_d", {"001": "fw"})
        _register_fake_package("_fake_ext_d", {"001": "ext"})

        runner = _FakeRunner(extra_migrations_package="_fake_ext_d")
        runner.migrations_package = "_fake_fw_d"

        await runner.run_up(_FakeConn())
        # Both were recorded under distinct keys.
        assert "001" in runner.applied
        assert "ext:001" in runner.applied

    @pytest.mark.asyncio
    async def test_idempotent_second_run(self) -> None:
        """A second ``run_up`` is a no-op when everything is already applied."""
        _register_fake_package("_fake_fw_e", {"001": "once"})
        _register_fake_package("_fake_ext_e", {"001": "once-ext"})

        runner = _FakeRunner(extra_migrations_package="_fake_ext_e")
        runner.migrations_package = "_fake_fw_e"

        conn = _FakeConn()
        await runner.run_up(conn)
        second = await runner.run_up(conn)

        assert second == []
        assert conn.executed.count("UP v001") == 2  # 1 framework + 1 extra on first pass only

    @pytest.mark.asyncio
    async def test_no_extras_package_only_framework(self) -> None:
        _register_fake_package("_fake_fw_f", {"001": "core"})
        runner = _FakeRunner()  # extras default None
        runner.migrations_package = "_fake_fw_f"

        applied = await runner.run_up(_FakeConn())

        assert applied == ["001"]
        assert runner.applied == {"001"}


class TestRunDown:
    @pytest.mark.asyncio
    async def test_rolls_back_extras_first_then_framework(self) -> None:
        _register_fake_package("_fake_fw_g", {"001": "core"})
        _register_fake_package("_fake_ext_g", {"001": "ext"})

        runner = _FakeRunner(extra_migrations_package="_fake_ext_g")
        runner.migrations_package = "_fake_fw_g"

        conn = _FakeConn()
        await runner.run_up(conn)
        rolled = await runner.run_down(conn)

        # Extras come off first (LIFO), then framework.
        assert rolled == ["ext:001", "001"]
        assert runner.applied == set()
        assert conn.executed[-2:] == ["DOWN v001", "DOWN v001"]

    @pytest.mark.asyncio
    async def test_target_version_stops_rollback(self) -> None:
        _register_fake_package("_fake_fw_h", {"001": "one", "002": "two"})
        runner = _FakeRunner()
        runner.migrations_package = "_fake_fw_h"

        conn = _FakeConn()
        await runner.run_up(conn)
        rolled = await runner.run_down(conn, target_version="001")

        assert rolled == ["002"]
        assert runner.applied == {"001"}
