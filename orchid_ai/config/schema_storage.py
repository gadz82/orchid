"""Database-backed agent configuration storage settings."""

from __future__ import annotations

from pydantic import BaseModel, Field


class OrchidConfigStorageConfig(BaseModel):
    """Configuration for the database-backed agent config store.

    When ``enabled`` is ``False`` (the default), Orchid skips config storage
    entirely and loads only from YAML / Markdown. When ``True``, the
    ``class`` and ``dsn`` fields are required and the store is initialised
    at bootstrap, with its configs merged into ``agents`` before the graph
    is built.

    Example YAML::

        config_storage:
          enabled: true
          class: orchid_storage_postgres.OrchidPostgresConfigStorage  # install orchid-storage-postgres plugin
          dsn: postgresql://user:pass@host:5432/db

    Attributes
    ----------
    enabled : bool
        ``False`` (default) means no DB config backend. ``True`` activates it.
    class : str
        Dotted Python import path of the storage backend.
        Must subclass ``OrchidConfigStorage``.
        Examples:
          - ``orchid_storage_postgres.OrchidPostgresConfigStorage``
    dsn : str
        Data-source name / connection string for the backend.
        For PostgreSQL: ``postgresql://user:pass@host:5432/dbname``
    """

    enabled: bool = False
    class_path: str = Field(default="", alias="class")
    dsn: str = ""

    model_config = {"populate_by_name": True}
