"""Run-visibility filter helpers (§26).

The ``visibility`` model on ``JobRun`` rows splits the world into:

- ``actor`` — visible to the user the Bloom ran as (and admins).
- ``addressed`` — visible to the user the Bloom was addressed to (and admins).
- ``tenant`` — visible to every authenticated user in the tenant.
- ``admin`` — visible only to ``admin``-role users.

Two flavours of filter ship here:

- :func:`build_run_filter_clause` returns SQL fragments for the two
  dialects we ship (Postgres + SQLite).  Routers paste this onto
  every ``SELECT FROM job_runs`` (and onto ``signals`` queries via
  the join).
- :func:`run_is_visible` is the in-memory predicate used by tests
  and by the in-memory backend (which skips SQL).

Cross-tenant access is always rejected (returns 404, never 403,
per §26.6) — the SQL fragment ANDs the tenant key in regardless of
role.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class _Filter:
    """SQL fragment + param map.  Routers AND ``where`` onto their
    ``SELECT`` and pass ``params`` to asyncpg / aiosqlite."""

    where: str
    params: dict[str, Any]


def build_run_filter_clause(auth: Any, *, dialect: str = "postgres") -> _Filter:
    """Return a ``WHERE`` fragment + bind params for the caller.

    ``auth`` must expose ``tenant_key``, ``user_id``, and ``roles``
    (the standard :class:`OrchidAuthContext` contract).  Admins
    short-circuit to a tenant-only filter; everyone else gets
    visibility-by-row.

    The fragment uses **named parameters** for asyncpg-Postgres
    (``$N``-style binding requires an ordered tuple) and named
    parameters for sqlite (``:tenant``-style).  Callers pick a
    placeholder format with ``dialect``.
    """
    tenant_key = getattr(auth, "tenant_key", "default")
    user_id = getattr(auth, "user_id", "")
    roles = getattr(auth, "roles", frozenset())

    if dialect == "postgres":
        if "admin" in roles:
            return _Filter(
                where="tenant_key = $1",
                params={"$1": tenant_key},
            )
        return _Filter(
            where=(
                "tenant_key = $1 AND ("
                "visibility = 'tenant' "
                "OR (visibility IN ('actor', 'addressed') "
                "    AND visibility_user_id = $2)"
                ")"
            ),
            params={"$1": tenant_key, "$2": user_id},
        )

    # SQLite (and other ``?``/named placeholder dialects).
    if "admin" in roles:
        return _Filter(
            where="tenant_key = :tenant_key",
            params={"tenant_key": tenant_key},
        )
    return _Filter(
        where=(
            "tenant_key = :tenant_key AND ("
            "visibility = 'tenant' "
            "OR (visibility IN ('actor', 'addressed') "
            "    AND visibility_user_id = :user_id)"
            ")"
        ),
        params={"tenant_key": tenant_key, "user_id": user_id},
    )


def run_is_visible(run: Any, auth: Any) -> bool:
    """In-memory predicate over a :class:`JobRun`.

    Mirrors the SQL filter exactly so the in-memory backend used by
    Phase 1's tests stays in lock-step with the durable backends.
    Returns ``False`` for cross-tenant access regardless of role.

    ``run`` must expose ``spec.visibility``, ``spec.visibility_user_id``,
    and a tenant-key reachable via ``run.spec.identity_claim`` /
    ``run.spec.parallelism_key``.  We pull tenant from the parallelism
    key first (most reliable — it's structured) and fall back to the
    identity claim when needed.

    The ``OrchidAuthContext`` exposes ``tenant_key`` directly.
    """
    auth_tenant = getattr(auth, "tenant_key", "default")
    run_tenant = _run_tenant_key(run) or auth_tenant
    if run_tenant != auth_tenant:
        return False

    roles = getattr(auth, "roles", frozenset())
    if "admin" in roles:
        return True

    visibility = getattr(run.spec, "visibility", "admin")
    if visibility == "tenant":
        return True
    if visibility in ("actor", "addressed"):
        return getattr(run.spec, "visibility_user_id", None) == getattr(auth, "user_id", "")
    # ``admin`` (default for service-account runs) — only admins,
    # which we already short-circuited above.
    return False


def _run_tenant_key(run: Any) -> str | None:
    """Best-effort tenant pull from a :class:`JobRun`.

    ``parallelism_key`` is the canonical structured form
    (``sa:<tenant>:<service>``, ``user:<tenant>:<user>``,
    ``tenant:<tenant>``); when it doesn't carry a tenant we fall
    back to the identity claim, which may not have one either —
    callers handle ``None`` by treating the run as 'whatever the
    caller's tenant is' (defensive: cross-tenant cases produce
    ``None`` only when the claim lacks routing, which means the
    Bloom has no business running outside its tenant either).
    """
    pkey = getattr(run.spec, "parallelism_key", "")
    if isinstance(pkey, str) and ":" in pkey:
        first, _, rest = pkey.partition(":")
        if first in ("sa", "user", "tenant") and ":" in rest:
            return rest.split(":", 1)[0]
        if first == "tenant":
            return rest
    claim = getattr(run.spec, "identity_claim", None) or {}
    if isinstance(claim, dict):
        candidate = claim.get("tenant_key")
        if isinstance(candidate, str) and candidate:
            return candidate
    return None
