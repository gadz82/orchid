"""
Factory for LangGraph checkpointers — pluggable state persistence.

Resolves well-known type strings (``"memory"``, ``"sqlite"``, ``"postgres"``)
or any dotted class path to a ``BaseCheckpointSaver`` instance.  Integrators
can use the built-in types or provide a fully-qualified class path to a
custom ``BaseCheckpointSaver`` subclass.

Example — built-in types::

    from orchid_ai.checkpointing import build_checkpointer

    # In-memory (testing / dev)
    saver = await build_checkpointer("memory")

    # SQLite (requires: pip install langgraph-checkpoint-sqlite)
    saver = await build_checkpointer("sqlite", dsn="~/.orchid/checkpoints.db")

    # PostgreSQL (requires: pip install langgraph-checkpoint-postgres)
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

from typing import Any, Callable, Coroutine

from langgraph.checkpoint.base import BaseCheckpointSaver

from ..utils import import_class

logger = logging.getLogger(__name__)

# Registry for custom built-in checkpointer types.
# Maps type string → async factory callable(dsn) -> BaseCheckpointSaver
_CHECKPOINTER_REGISTRY: dict[str, Callable[..., Coroutine[Any, Any, BaseCheckpointSaver]]] = {}


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
        if not resolved_dsn:
            raise ValueError("SQLite checkpointer requires a DSN (file path), e.g. '~/.orchid/checkpoints.db'")
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError as exc:
            raise ImportError(
                "SQLite checkpointer requires the 'langgraph-checkpoint-sqlite' package. "
                "Install it with: pip install langgraph-checkpoint-sqlite"
            ) from exc

        saver = AsyncSqliteSaver.from_conn_string(resolved_dsn)
        logger.info("[Checkpointer] Using AsyncSqliteSaver (dsn=%s)", resolved_dsn)
        return saver

    if checkpointer_type == "postgres":
        if not resolved_dsn:
            raise ValueError("PostgreSQL checkpointer requires a DSN, e.g. 'postgresql://user:pass@host/db'")
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ImportError as exc:
            raise ImportError(
                "PostgreSQL checkpointer requires the 'langgraph-checkpoint-postgres' package. "
                "Install it with: pip install langgraph-checkpoint-postgres"
            ) from exc

        saver = AsyncPostgresSaver.from_conn_string(resolved_dsn)
        await saver.setup()
        logger.info("[Checkpointer] Using AsyncPostgresSaver (dsn=%s)", _mask_dsn(resolved_dsn))
        return saver

    # ── Custom dotted class path ──────────────────────────────
    try:
        cls = import_class(checkpointer_type)
    except ImportError as exc:
        raise ImportError(
            f"Cannot resolve checkpointer class '{checkpointer_type}'. "
            f"Ensure it is a valid dotted import path to a BaseCheckpointSaver subclass. "
            f"Error: {exc}"
        ) from exc

    if not (isinstance(cls, type) and issubclass(cls, BaseCheckpointSaver)):
        raise TypeError(f"'{checkpointer_type}' resolves to {cls!r}, which is not a BaseCheckpointSaver subclass.")

    logger.info("[Checkpointer] Using custom class: %s", checkpointer_type)
    return cls(dsn=resolved_dsn) if resolved_dsn else cls()


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
