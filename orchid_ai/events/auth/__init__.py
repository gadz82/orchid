"""Signal-source authentication validators.

Each :class:`SignalAuthValidator` decides whether the headers + body
of an inbound HTTP signal genuinely came from a configured source.
The two built-ins are :class:`HMACValidator` (shared-secret signing
of the raw body) and :class:`BearerValidator` (constant-time bearer
comparison).  Integrators add their own (mTLS, JWT-from-IdP, …) by
subclassing the ABC and pointing
:class:`OrchidValidatorConfig.class_path` at the dotted import path.

The validator is invoked from
:class:`orchid_ai.events.producers.http.HTTPIngestionProducer` after
it has resolved the ``X-Orchid-Source`` header to a registered
``signal_sources`` row.  Validation runs BEFORE
:meth:`OrchidSignalDispatcher.ingest`, so a failed signature short-
circuits with a 401 before any persistence happens.
"""

from __future__ import annotations

from .base import SignalAuthValidator
from .bearer import BearerValidator
from .hmac import HMACValidator

__all__ = [
    "BearerValidator",
    "HMACValidator",
    "SignalAuthValidator",
]
