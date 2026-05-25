"""Tests for ``orchid_ai.events.visibility``.

Verifies that the SQL fragment shape and the in-memory predicate
agree across the §26.3 matrix and that cross-tenant access always
returns false regardless of role.
"""

from __future__ import annotations

import datetime as _dt
import uuid as _uuid

import pytest

from orchid_ai.core.events.job import JobRun, JobSpec, JobStatus
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.events.visibility import (
    build_run_filter_clause,
    run_is_visible,
)


def _run(*, visibility: str, visibility_user_id: str | None, tenant: str = "t-1") -> JobRun:
    spec = JobSpec(
        trigger_id="t",
        signal_id=_uuid.uuid4(),
        agent_name="a",
        prompt="x",
        identity_claim={"mode": "service_account", "name": "bot"},
        correlation_id=None,
        parallelism_key=f"sa:{tenant}:bot",
        visibility=visibility,
        visibility_user_id=visibility_user_id,
    )
    return JobRun(
        run_id=_uuid.uuid4(),
        spec=spec,
        attempt_number=1,
        status=JobStatus.SUCCEEDED,
        queued_at=_dt.datetime.now(tz=_dt.UTC),
    )


# ── SQL fragment (SQLite only) ─────────────────────────────


def test_sqlite_admin_fragment() -> None:
    auth = OrchidAuthContext(access_token="t", tenant_key="t-1", user_id="u-7", roles={"admin"})
    f = build_run_filter_clause(auth, dialect="sqlite")
    assert "tenant_key = :tenant_key" in f.where
    assert f.params == {"tenant_key": "t-1"}


def test_sqlite_non_admin_fragment() -> None:
    auth = OrchidAuthContext(access_token="t", tenant_key="t-1", user_id="u-7")
    f = build_run_filter_clause(auth, dialect="sqlite")
    assert ":tenant_key" in f.where
    assert ":user_id" in f.where
    assert f.params == {"tenant_key": "t-1", "user_id": "u-7"}


# ── In-memory predicate ─────────────────────────────────────


@pytest.mark.parametrize(
    "visibility,user_id,auth_user,roles,expected",
    [
        # admin sees everything in their tenant
        ("admin", None, "u-X", {"admin"}, True),
        ("actor", "u-7", "u-X", {"admin"}, True),
        ("addressed", "u-7", "u-X", {"admin"}, True),
        ("tenant", None, "u-X", {"admin"}, True),
        # tenant visibility for non-admins (any user in tenant)
        ("tenant", None, "u-7", set(), True),
        ("tenant", None, "anyone", set(), True),
        # actor only matches the right user
        ("actor", "u-7", "u-7", set(), True),
        ("actor", "u-7", "u-OTHER", set(), False),
        # addressed only matches the addressed user
        ("addressed", "u-7", "u-7", set(), True),
        ("addressed", "u-7", "u-OTHER", set(), False),
        # admin visibility hides from non-admins
        ("admin", None, "u-7", set(), False),
    ],
)
def test_run_is_visible_matrix(visibility, user_id, auth_user, roles, expected) -> None:
    run = _run(visibility=visibility, visibility_user_id=user_id)
    auth = OrchidAuthContext(access_token="t", tenant_key="t-1", user_id=auth_user, roles=roles)
    assert run_is_visible(run, auth) is expected


def test_cross_tenant_is_invisible_even_to_admin() -> None:
    """An admin from tenant A cannot see a run from tenant B —
    cross-tenant access is the absolute boundary (§26.6)."""
    run = _run(visibility="admin", visibility_user_id=None, tenant="t-A")
    auth_other = OrchidAuthContext(
        access_token="t",
        tenant_key="t-B",
        user_id="u-admin",
        roles={"admin"},
    )
    assert run_is_visible(run, auth_other) is False
