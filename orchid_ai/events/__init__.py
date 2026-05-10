"""Concrete implementations of the events ABCs.

This package depends on ``orchid_ai/core/events/`` (the pure ABCs) and
on the rest of the framework (config schema, identity resolver, …) but
**must not** be imported from ``orchid_ai/core/`` — the boundary is
verified by ``tests/test_dependency_boundaries.py``.

Subpackages:

- ``queues/`` — ``InMemorySignalQueue`` and the future Postgres /
  SQLite / Relay queues.
- ``producers/`` — built-in producers (HTTP ingest, scheduler tick,
  internal emission, MCP gateway forward).
- ``processors/`` — built-in processors (asyncio worker pool today;
  Celery / Lambda / consumer-group adapters live in integrators).
- ``runners/`` — ``GraphJobRunner`` plus integrator-supplied alternatives.
- ``registry`` (module) — the in-memory trigger registry with JMESPath
  match logic.

Phase 1 ships only the in-memory queue + store + job-store, the
in-memory trigger registry, the ``DispatcherSignalEmitter``, the
asyncio worker-pool processor, and a stubbed ``GraphJobRunner``.
Postgres / SQLite backends and APScheduler land in the next phase.
"""

from __future__ import annotations
