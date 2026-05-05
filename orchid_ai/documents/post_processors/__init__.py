"""
Chunk post-processors.

Concrete :class:`OrchidChunkPostProcessor` implementations that wrap the
output of any :class:`OrchidIngestionStrategy`.  Composable: the YAML
``ingestion.post_processors: [contextual_headers, pii_redact]`` chain
runs each in declared order, the next consuming the prior's output.

Built-ins include ``contextual_headers`` and ``entity_extraction``
(GraphRAG ingestion side).  Integrator-supplied processors for PII
redaction, dedup, etc. register via
:func:`orchid_ai.documents.strategies.register_post_processor`.
"""

from __future__ import annotations

from .contextual_headers import ContextualHeaderPostProcessor
from .entity_extraction import (
    EntityExtractionPostProcessor,
    LLMEntityExtractor,
    OrchidExtractedGraph,
)

__all__ = [
    "ContextualHeaderPostProcessor",
    "EntityExtractionPostProcessor",
    "LLMEntityExtractor",
    "OrchidExtractedGraph",
]
