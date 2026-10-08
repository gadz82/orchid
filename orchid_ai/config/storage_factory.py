"""
Factory for building :class:`OrchidConfigStorage` instances from dotted
import paths.

The ``"memory"`` sentinel (also the empty string) selects the built-in
in-memory backend — the framework default when config storage is enabled
without a class path.  Configured dotted paths resolve strictly.
"""

from __future__ import annotations

from orchid_ai.config.storage import OrchidConfigStorage
from orchid_ai.config.storage_memory import OrchidInMemoryConfigStorage
from orchid_ai.persistence.plugin_hints import with_plugin_hint
from orchid_ai.utils import import_class, is_memory_storage_sentinel

__all__ = ["build_config_storage"]


def build_config_storage(class_path: str, dsn: str) -> OrchidConfigStorage:
    """Build a :class:`OrchidConfigStorage` from a dotted class path.

    Parameters
    ----------
    class_path : str
        Dotted import path, e.g.
        ``"orchid_storage_postgres.OrchidPostgresConfigStorage"``, or the
        ``"memory"`` sentinel (also the empty string) for the built-in
        in-memory backend.
    dsn : str
        Data-source name / connection string for the backend
        (e.g. ``"postgresql://user:pass@host:5432/db"``).  Ignored by the
        in-memory backend.

    Returns
    -------
    OrchidConfigStorage
        Initialised backend (caller must call ``init_db()`` before use).

    Raises
    ------
    ValueError
        If the class cannot be imported or does not subclass
        :class:`OrchidConfigStorage`.
    """
    if is_memory_storage_sentinel(class_path):
        return OrchidInMemoryConfigStorage(dsn=dsn)

    try:
        cls = import_class(class_path)
    except ImportError as exc:
        raise ImportError(with_plugin_hint(f"Cannot resolve class '{class_path}': {exc}", class_path)) from exc

    if not issubclass(cls, OrchidConfigStorage):
        raise TypeError(f"Class '{class_path}' is not a subclass of OrchidConfigStorage. Found: {cls.__mro__}")
    return cls(dsn=dsn)
