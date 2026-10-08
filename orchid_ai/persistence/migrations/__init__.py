"""
Database migrations for the persistence layer.

This package ships the migration **contract** only:

- :class:`~orchid_ai.persistence.migrations.runner.OrchidMigrationRunner`
  — the backend-agnostic runner ABC (tracking-table hooks + up/down
  orchestration, including the integrator ``extra_migrations_package``
  pass).
- :func:`~orchid_ai.persistence.migrations.runner.discover_migrations`
  — package scanner for ``v*`` migration modules.

Concrete migrations live in the storage plugin packages
(``orchid_storage_sqlite.migrations``, ``orchid_storage_postgres.migrations``)
or in consumer projects.
"""

from __future__ import annotations
