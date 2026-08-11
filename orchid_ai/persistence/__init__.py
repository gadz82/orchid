"""Chat session persistence — pluggable storage backends for multi-chat support."""

from __future__ import annotations

from .sqlite_ingestion_manifest import OrchidSQLiteIngestionManifest

__all__ = ["OrchidSQLiteIngestionManifest"]
