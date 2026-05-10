"""Validator ABC + the request shape it sees."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SignalAuthRequest:
    """Inputs handed to a :class:`SignalAuthValidator`.

    Captures only what the validator can legitimately use:

    - ``raw_body`` is the bytes the dispatcher will hash for HMAC.
    - ``headers`` are case-folded to lowercase keys, matching what
      Starlette / FastAPI hands us.
    - ``source_id`` echoes ``X-Orchid-Source`` so the validator can
      double-check (and so logs cite it).
    """

    source_id: str
    raw_body: bytes
    headers: dict[str, str]


class SignalAuthValidator(ABC):
    """Decides whether a signal request really came from this source.

    Concrete validators may need configuration (an HMAC secret, a
    static bearer, an IdP JWKS URL).  They take that via constructor
    kwargs which match the ``extra_args`` block in
    :class:`OrchidValidatorConfig` plus an optional ``secret_ref``
    string the loader has already resolved against env vars.
    """

    @abstractmethod
    async def validate(self, request: SignalAuthRequest) -> None:
        """Raise :class:`SignalAuthValidationError` (from
        :mod:`orchid_ai.core.events.errors`) on failure; return
        cleanly on success.

        Validators are async because some (mTLS via cert-store
        lookup, JWKS-with-rotation) need I/O; the synchronous ones
        ignore the async-ness and ``return`` immediately.
        """
