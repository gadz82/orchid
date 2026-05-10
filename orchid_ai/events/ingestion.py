"""HTTP signal-ingestion domain types.

Pure Python — no FastAPI dependency.  These data structures represent
the in-memory view of the ``signal_sources`` configuration: which
sources are registered, which validator guards each one, and which
signal types each source may produce.

The concrete :class:`HTTPIngestionProducer` (FastAPI adapter) lives in
``orchid-api`` and imports these types from here.  The lifecycle helper
:func:`orchid_ai.events.bootstrap.build_signal_source_registry` compiles
:class:`OrchidIngestionSourceConfig` rows into a live
:class:`SignalSourceRegistry` that the producer can use immediately.
"""

from __future__ import annotations

from dataclasses import dataclass

from .auth.base import SignalAuthValidator


@dataclass(frozen=True, slots=True)
class SignalSource:
    """One ``signal_sources`` row, materialised in memory.

    The validator is the *resolved* validator (not the dotted-path
    config) — the lifecycle layer compiles
    :class:`OrchidIngestionSourceConfig` into this shape before
    handing it to the producer.
    """

    source_id: str
    validator: SignalAuthValidator
    allowed_types: frozenset[str]


class SignalSourceRegistry:
    """Lookup of registered HTTP-ingestion sources.

    Plain dict wrapper; the lifecycle layer constructs one from
    YAML at boot.  v1 doesn't support hot-reload — operators
    restart the process to pick up source changes.
    """

    def __init__(self, sources: list[SignalSource]) -> None:
        self._by_id: dict[str, SignalSource] = {s.source_id: s for s in sources}

    def get(self, source_id: str) -> SignalSource | None:
        return self._by_id.get(source_id)

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self):
        return iter(self._by_id.values())
