"""Orchid sub-package — focused collaborators for the public ``Orchid`` facade.

M1 refactoring: splits the 1281-LOC god module into three SRP collaborators
plus grouped config dataclasses to replace the 16-kwarg factory explosion.

- ``config_loader``  — YAML / MD / hybrid auto-detection + loading.
- ``lifecycle``      — init / close, signal-emitter wiring, hot-reload.
- ``invoker``        — invoke / stream / resume, including persistence.
- ``orchid``         — thin public facade that composes the three.
"""

from .orchid import (
    CheckpointerOverrides,
    MCPStorageOverrides,
    Orchid,
    OrchidFactoryOverrides,
    OrchidInvokeResult,
    OrchidPendingApproval,
    StartupOverrides,
    StorageOverrides,
)

__all__ = [
    "CheckpointerOverrides",
    "MCPStorageOverrides",
    "Orchid",
    "OrchidFactoryOverrides",
    "OrchidInvokeResult",
    "OrchidPendingApproval",
    "StartupOverrides",
    "StorageOverrides",
]
