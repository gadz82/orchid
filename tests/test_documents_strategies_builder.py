"""``build_ingestion_strategy(config)`` honours per-strategy knobs."""

from __future__ import annotations

from orchid_ai.config.schema_rag import OrchidIngestionConfig
from orchid_ai.documents.strategies import (
    HeaderedIngestion,
    HierarchicalIngestion,
    RecursiveIngestion,
    SemanticIngestion,
    build_ingestion_strategy,
)


class TestBuildIngestionStrategy:
    def test_recursive_uses_chunk_config(self):
        cfg = OrchidIngestionConfig(strategy="recursive", chunk_size=400, chunk_overlap=50)
        strat = build_ingestion_strategy(cfg)
        assert isinstance(strat, RecursiveIngestion)
        assert strat._config.chunk_size == 400
        assert strat._config.chunk_overlap == 50

    def test_hierarchical_uses_chunk_config(self):
        cfg = OrchidIngestionConfig(strategy="hierarchical", chunk_size=600, parent_chunk_size=2400)
        strat = build_ingestion_strategy(cfg)
        assert isinstance(strat, HierarchicalIngestion)
        assert strat._config.chunk_size == 600
        assert strat._config.parent_chunk_size == 2400

    def test_headered_uses_chunk_config(self):
        cfg = OrchidIngestionConfig(strategy="headered", chunk_size=750)
        strat = build_ingestion_strategy(cfg)
        assert isinstance(strat, HeaderedIngestion)
        # ``HeaderedIngestion`` wraps a ``RecursiveIngestion`` internally —
        # the ChunkConfig flows through to the inner strategy.
        assert strat._inner._config.chunk_size == 750

    def test_semantic_built_with_defaults(self):
        """``semantic`` does not consume ChunkConfig knobs — built zero-arg."""
        cfg = OrchidIngestionConfig(strategy="semantic", chunk_size=999)
        strat = build_ingestion_strategy(cfg)
        assert isinstance(strat, SemanticIngestion)

    def test_unknown_strategy_falls_back_to_recursive(self):
        cfg = OrchidIngestionConfig(strategy="not-registered", chunk_size=300)
        strat = build_ingestion_strategy(cfg)
        assert isinstance(strat, RecursiveIngestion)
        # Falls back to a recursive strategy — chunk knobs still flow through.
        assert strat._config.chunk_size == 300

    def test_default_strategy_is_recursive(self):
        cfg = OrchidIngestionConfig()  # strategy=None → "recursive"
        strat = build_ingestion_strategy(cfg)
        assert isinstance(strat, RecursiveIngestion)
