"""Chat session persistence — pluggable storage backends for multi-chat support.

The framework ships dependency-free in-memory backends (see
:mod:`orchid_ai.persistence.in_memory`); durable backends live in plugin
packages (``orchid-storage-sqlite``, ``orchid-storage-postgres``) or in
consumer projects, referenced by dotted class path.
"""

from __future__ import annotations
