"""
Chunk post-processors (ADR-022).

Concrete :class:`OrchidChunkPostProcessor` implementations that wrap the
output of any :class:`OrchidIngestionStrategy`.  Composable: the YAML
``ingestion.post_processors: [contextual_headers, pii_redact]`` chain
runs each in declared order, the next consuming the prior's output.

Stage 2 ships ``contextual_headers``.  Future stages add
``entity_extraction`` (Stage 5 — GraphRAG ingestion side) and
integrator-supplied processors for PII redaction, dedup, etc. via
:func:`orchid_ai.documents.strategies.register_post_processor`.
"""

from __future__ import annotations

from .contextual_headers import ContextualHeaderPostProcessor

__all__ = ["ContextualHeaderPostProcessor"]
