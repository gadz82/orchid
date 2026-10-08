"""Concrete implementations of the events ABCs.

This package depends on ``orchid_ai/core/events/`` (the pure ABCs) and
on the rest of the framework (config schema, identity resolver, …) but
**must not** be imported from ``orchid_ai/core/`` — the boundary is
verified by ``tests/test_dependency_boundaries.py``.

Subpackages:

- ``queues/`` — in-memory queue + stores, relay queue skeleton.  Durable
  SQLite / PostgreSQL queues live in the storage plugin packages.
- ``backends/`` — the ``InMemoryEventStorage`` facade (framework
  default).  Durable store backends live in the storage plugins.
- ``producers/`` — built-in producers (HTTP ingest, scheduler tick,
  internal emission, MCP gateway forward).
- ``processors/`` — built-in processors (asyncio worker pool today;
  Celery / Lambda / consumer-group adapters live in integrators).
- ``runners/`` — ``GraphJobRunner`` plus integrator-supplied alternatives.
- ``registry`` (module) — the in-memory trigger registry with JMESPath
  match logic.
"""

from __future__ import annotations
