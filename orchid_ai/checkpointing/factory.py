"""
Factory for LangGraph checkpointers — pluggable state persistence.

Resolves well-known type strings (``"memory"``, ``"sqlite"``) or any dotted
class path to a ``BaseCheckpointSaver`` instance.  SQLite is built-in;
PostgreSQL ships in ``orchid-storage-postgres``.

Example — built-in types::

    from orchid_ai.checkpointing import build_checkpointer

    # In-memory (testing / dev)
    saver = await build_checkpointer("memory")

    # SQLite (built-in)
    saver = await build_checkpointer("sqlite", dsn="~/.orchid/checkpoints.db")

    # PostgreSQL (requires: pip install orchid-storage-postgres)
    saver = await build_checkpointer("postgres", dsn="postgresql://...")

Example — custom class::

    saver = await build_checkpointer(
        "myproject.checkpointing.RedisCheckpointer",
        dsn="redis://localhost:6379/0",
    )

Lifecycle::

    saver = await build_checkpointer("sqlite", dsn="cp.db")
    runtime.checkpointer = saver
    graph = build_graph(config=config, runtime=runtime)
    # ... use graph ...
    await shutdown_checkpointer(saver)
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Coroutine
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

from ..plugins import iter_entry_point_plugins
from ..utils import import_class

logger = logging.getLogger(__name__)

# Registry for custom built-in checkpointer types.
# Maps type string → async factory callable(dsn) -> BaseCheckpointSaver
_CHECKPOINTER_REGISTRY: dict[str, Callable[..., Coroutine[Any, Any, BaseCheckpointSaver]]] = {}

_CHECKPOINTER_PACKAGE_HINTS: dict[str, str] = {
    "orchid_storage_postgres.": "orchid-storage-postgres",
}


def _augment_checkpointer_import_error(class_path: str, exc: Exception) -> ImportError:
    msg = (
        f"Cannot resolve checkpointer class '{class_path}'. "
        f"Ensure it is a valid dotted import path to a BaseCheckpointSaver subclass. "
        f"Error: {exc}"
    )
    for prefix, package in _CHECKPOINTER_PACKAGE_HINTS.items():
        if class_path.startswith(prefix):
            msg += f" Install the missing plugin: pip install {package}"
            break
    return ImportError(msg)


def register_checkpointer(
    type_name: str,
    factory: Callable[..., Coroutine[Any, Any, BaseCheckpointSaver]],
) -> None:
    """Register a custom checkpointer type for ``build_checkpointer()``.

    The factory is an async callable ``(dsn: str) -> BaseCheckpointSaver``.

    Example::

        async def build_redis(dsn: str) -> BaseCheckpointSaver:
            from myproject.checkpointing import RedisCheckpointer
            return RedisCheckpointer(dsn)

        register_checkpointer("redis", build_redis)
        saver = await build_checkpointer("redis", dsn="redis://localhost")
    """
    _CHECKPOINTER_REGISTRY[type_name] = factory
    logger.info("[Checkpointer] Registered custom type: %s", type_name)


async def build_checkpointer(
    checkpointer_type: str,
    dsn: str = "",
) -> BaseCheckpointSaver:
    """Build a LangGraph checkpointer from a type string or dotted class path.

    Parameters
    ----------
    checkpointer_type : str
        One of ``"memory"``, ``"sqlite"``, ``"postgres"``, or a fully-qualified
        dotted class path to a ``BaseCheckpointSaver`` subclass.
    dsn : str
        Connection string or file path.  Required for ``"sqlite"`` and
        ``"postgres"``.  Ignored for ``"memory"``.  Supports ``~`` expansion.

    Returns
    -------
    BaseCheckpointSaver
        A ready-to-use checkpointer.  For ``"postgres"``, the schema tables
        are created automatically via ``setup()``.

    Raises
    ------
    ImportError
        When the required checkpoint package is not installed.
    TypeError
        When a custom class path does not resolve to a ``BaseCheckpointSaver``
        subclass.
    ValueError
        When ``dsn`` is missing for types that require it.
    """
    resolved_dsn = os.path.expanduser(dsn) if dsn else dsn

    # Check custom registry first (integrators can register types)
    if checkpointer_type in _CHECKPOINTER_REGISTRY:
        factory_fn = _CHECKPOINTER_REGISTRY[checkpointer_type]
        logger.info("[Checkpointer] Using registered type: %s", checkpointer_type)
        return await factory_fn(resolved_dsn)

    if checkpointer_type == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        logger.info("[Checkpointer] Using MemorySaver (in-memory, non-persistent)")
        return MemorySaver()

    if checkpointer_type == "sqlite":
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        if not resolved_dsn:
            raise ValueError("DSN is required for sqlite checkpointer (e.g. ~/.orchid/checkpoints.db)")
        # ``AsyncSqliteSaver.from_conn_string`` is an async *context manager*,
        # not a saver — awaiting ``.setup()`` on it raises AttributeError.
        # Construct the saver from a long-lived connection so it outlives
        # this call; ``shutdown_checkpointer`` closes the connection.
        conn = await aiosqlite.connect(resolved_dsn)
        checkpointer = AsyncSqliteSaver(conn)
        await checkpointer.setup()
        logger.info("[Checkpointer] SQLite checkpointer ready — %s", _mask_dsn(resolved_dsn))
        return checkpointer

    # ── Custom dotted class path ──────────────────────────────
    try:
        cls = import_class(checkpointer_type)
    except ImportError as exc:
        raise _augment_checkpointer_import_error(checkpointer_type, exc) from exc

    if not (isinstance(cls, type) and issubclass(cls, BaseCheckpointSaver)):
        raise TypeError(f"'{checkpointer_type}' resolves to {cls!r}, which is not a BaseCheckpointSaver subclass.")

    logger.info("[Checkpointer] Using custom class: %s", checkpointer_type)
    return cls(dsn=resolved_dsn) if resolved_dsn else cls()


def _load_entry_point_checkpointers() -> None:
    for name, register_fn in iter_entry_point_plugins("orchid.checkpointers"):
        try:
            register_fn()
        except Exception as exc:
            logger.warning("[Checkpointers] Failed to load plugin '%s': %s", name, exc)


async def shutdown_checkpointer(saver: BaseCheckpointSaver | None) -> None:
    """Gracefully close a checkpointer's underlying connections.

    Safe to call with ``None`` (no-op).  Uses duck-typing to handle
    backends that implement different cleanup interfaces.
    """
    if saver is None:
        return

    name = type(saver).__name__
    try:
        if hasattr(saver, "aclose"):
            await saver.aclose()
        elif hasattr(saver, "close"):
            saver.close()
        elif hasattr(saver, "__aexit__"):
            await saver.__aexit__(None, None, None)
        elif (conn := getattr(saver, "conn", None)) is not None and hasattr(conn, "close"):
            # e.g. AsyncSqliteSaver wraps an aiosqlite.Connection
            await conn.close()
        logger.info("[Checkpointer] %s shut down", name)
    except Exception as exc:
        logger.warning("[Checkpointer] Error shutting down %s: %s", name, exc)


def _mask_dsn(dsn: str) -> str:
    """Mask password in DSN for safe logging."""
    if "@" in dsn and "://" in dsn:
        # postgresql://user:pass@host/db → postgresql://user:***@host/db
        prefix, rest = dsn.split("://", 1)
        if "@" in rest:
            creds, host = rest.rsplit("@", 1)
            if ":" in creds:
                user, _ = creds.split(":", 1)
                return f"{prefix}://{user}:***@{host}"
    return dsn
