"""
Hierarchical RAG scoping — re-export of the pure-stdlib :class:`OrchidRAGScope`.

The dataclass itself lives in :mod:`orchid_ai.core.scopes` (zero
external deps).  This module re-exports it so callers can pull the
scope from the same package as the rest of the RAG surface.

Backend-specific filter builders (e.g. ``build_qdrant_filter``) live
next to their backend in ``orchid_ai/rag/backends/<name>.py`` — keeping
this module's import surface independent of any concrete vector DB
client.  Strategies and consumer code depend only on
:class:`OrchidRAGScope`; the translation to backend-native filters
happens inside the backend.
"""

from __future__ import annotations

from ..core.scopes import SHARED_TENANT, OrchidRAGScope, resolve_scope_level, scope_key

__all__ = ["SHARED_TENANT", "OrchidRAGScope", "resolve_scope_level", "scope_key"]
