"""
Factory for building :class:`OrchidConfigStorage` instances from dotted
import paths.
"""

from __future__ import annotations

from orchid_ai.config.storage import OrchidConfigStorage
from orchid_ai.utils import import_class

__all__ = ["build_config_storage"]


def build_config_storage(class_path: str, dsn: str) -> OrchidConfigStorage:
    """Build a :class:`OrchidConfigStorage` from a dotted class path.

    Parameters
    ----------
    class_path : str
        Dotted import path, e.g.
        ``"orchid_storage_postgres.OrchidPostgresConfigStorage"``.
    dsn : str
        Data-source name / connection string for the backend
        (e.g. ``"postgresql://user:pass@host:5432/db"``).

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
    cls = import_class(class_path)
    if not issubclass(cls, OrchidConfigStorage):
        raise TypeError(f"Class '{class_path}' is not a subclass of OrchidConfigStorage. Found: {cls.__mro__}")
    return cls(dsn=dsn)
