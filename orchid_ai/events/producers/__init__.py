"""Built-in producers — in-process emitter + scheduler."""

from __future__ import annotations

from .internal import DispatcherSignalEmitter, InternalEmissionProducer
from .scheduler import SchedulerProducer

__all__ = ["DispatcherSignalEmitter", "InternalEmissionProducer", "SchedulerProducer"]
