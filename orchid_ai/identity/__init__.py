"""Identity helpers — composable mixins consumers wire alongside
:class:`~orchid_ai.core.identity.OrchidIdentityResolver`.

Lives outside ``core/`` because the implementations may pull in
HTTP-capable collaborators (asyncpg / httpx via the chat-storage
pool) that ``core/`` cannot import.  ``core/identity.py`` carries the
ABC + raising defaults; this package carries the concrete-but-still-
optional helpers.
"""

from __future__ import annotations

from .oauth_minting import (
    OAuthMintingMixin,
    OrchidTokenRefresher,
)

__all__ = [
    "OAuthMintingMixin",
    "OrchidTokenRefresher",
]
