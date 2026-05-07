"""Built-in producers — in-process emitter + scheduler.

Phase-1 shipped :class:`DispatcherSignalEmitter` only.  Phase-2 adds
:class:`SchedulerProducer` driven by APScheduler against an
:class:`OrchidScheduleStore`.  HTTP / MCP ingestion producers land in
the API surface phase.
"""

from __future__ import annotations

from .internal import DispatcherSignalEmitter
from .scheduler import SchedulerProducer

__all__ = ["DispatcherSignalEmitter", "SchedulerProducer"]
