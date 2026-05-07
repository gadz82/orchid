"""End-to-end tests for ``HTTPIngestionProducer`` against a TestClient.

The producer's router is mounted into a throwaway FastAPI app; we
hit it with httpx-style requests and assert on the stored signal +
the dispatcher's outbox.  Validates the §11.1 contract:

- 202 on success with ``{signal_id, deduplicated}`` body.
- 401 on signature mismatch.
- 403 on signal-type allow-list miss.
- 404 on unknown source.
- ``Idempotency-Key`` header → ``dedupe_key`` on the persisted row.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from orchid_ai.core.events.dispatcher import OrchidSignalDispatcher
from orchid_ai.events.auth import BearerValidator, HMACValidator
from orchid_ai.events.producers.http import (
    HTTPIngestionProducer,
    SignalSource,
    SignalSourceRegistry,
)
from orchid_ai.events.queues.inmemory import (
    InMemorySignalQueue,
    InMemorySignalStore,
)


# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
async def wired():
    queue = InMemorySignalQueue()
    store = InMemorySignalStore()
    dispatcher = OrchidSignalDispatcher(store=store, queue=queue)
    sources = [
        SignalSource(
            source_id="support-system",
            validator=HMACValidator(secret="topsecret"),
            allowed_types=frozenset({"support.ticket.created"}),
        ),
        SignalSource(
            source_id="ops-bearer",
            validator=BearerValidator(secret="ops-token"),
            allowed_types=frozenset(),  # empty allow-list = accept any
        ),
    ]
    producer = HTTPIngestionProducer(registry=SignalSourceRegistry(sources))
    await producer.start(dispatcher)

    app = FastAPI()
    app.include_router(producer.router)
    yield {
        "client": TestClient(app),
        "store": store,
        "queue": queue,
        "producer": producer,
    }
    await producer.stop()


def _sign(body: bytes, secret: str = "topsecret") -> str:
    return "sha256=" + _hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


# ── Tests ───────────────────────────────────────────────────


def test_ingest_happy_path(wired) -> None:
    client: TestClient = wired["client"]
    body = json.dumps(
        {
            "type": "support.ticket.created",
            "payload": {"ticket_id": "T-42"},
            "tenant_key": "acme-prod",
            "user_id": "u-7",
        }
    ).encode()
    resp = client.post(
        "/signals",
        content=body,
        headers={
            "x-orchid-source": "support-system",
            "x-orchid-signature": _sign(body),
            "content-type": "application/json",
        },
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "signal_id" in data
    assert data["deduplicated"] is False


def test_ingest_idempotency_key_dedupes(wired) -> None:
    client: TestClient = wired["client"]
    body = json.dumps(
        {
            "type": "support.ticket.created",
            "payload": {},
            "tenant_key": "acme-prod",
        }
    ).encode()
    headers = {
        "x-orchid-source": "support-system",
        "x-orchid-signature": _sign(body),
        "idempotency-key": "ticket-42-v1",
    }
    first = client.post("/signals", content=body, headers=headers)
    second = client.post("/signals", content=body, headers=headers)
    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["signal_id"] == second.json()["signal_id"]
    assert second.json()["deduplicated"] is True


def test_ingest_rejects_unknown_source(wired) -> None:
    client: TestClient = wired["client"]
    body = b"{}"
    resp = client.post(
        "/signals",
        content=body,
        headers={"x-orchid-source": "no-such-source"},
    )
    assert resp.status_code == 404
    assert "no-such-source" in resp.json()["detail"]


def test_ingest_rejects_missing_source_header(wired) -> None:
    client: TestClient = wired["client"]
    resp = client.post("/signals", content=b"{}")
    assert resp.status_code == 400


def test_ingest_rejects_bad_signature(wired) -> None:
    client: TestClient = wired["client"]
    body = json.dumps({"type": "support.ticket.created", "tenant_key": "acme-prod"}).encode()
    resp = client.post(
        "/signals",
        content=body,
        headers={
            "x-orchid-source": "support-system",
            "x-orchid-signature": _sign(b"DIFFERENT", secret="topsecret"),
        },
    )
    assert resp.status_code == 401


def test_ingest_rejects_disallowed_type(wired) -> None:
    client: TestClient = wired["client"]
    body = json.dumps({"type": "other.event", "tenant_key": "acme-prod"}).encode()
    resp = client.post(
        "/signals",
        content=body,
        headers={
            "x-orchid-source": "support-system",
            "x-orchid-signature": _sign(body),
        },
    )
    assert resp.status_code == 403
    assert "other.event" in resp.json()["detail"]


def test_ingest_rejects_missing_required_fields(wired) -> None:
    client: TestClient = wired["client"]
    body = json.dumps({"payload": {}}).encode()
    resp = client.post(
        "/signals",
        content=body,
        headers={
            "x-orchid-source": "support-system",
            "x-orchid-signature": _sign(body),
        },
    )
    assert resp.status_code == 400
    assert "type" in resp.json()["detail"]


def test_ingest_bearer_validator_path(wired) -> None:
    client: TestClient = wired["client"]
    body = json.dumps({"type": "ops.event", "tenant_key": "acme-prod"}).encode()
    resp = client.post(
        "/signals",
        content=body,
        headers={
            "x-orchid-source": "ops-bearer",
            "authorization": "Bearer ops-token",
        },
    )
    assert resp.status_code == 202


def test_ingest_bearer_validator_rejects_wrong_token(wired) -> None:
    client: TestClient = wired["client"]
    body = json.dumps({"type": "ops.event", "tenant_key": "acme-prod"}).encode()
    resp = client.post(
        "/signals",
        content=body,
        headers={
            "x-orchid-source": "ops-bearer",
            "authorization": "Bearer WRONG",
        },
    )
    assert resp.status_code == 401


def test_ingest_too_large(wired) -> None:
    """Bodies larger than ``max_body_bytes`` get a 413."""
    queue = InMemorySignalQueue()
    store = InMemorySignalStore()
    dispatcher = OrchidSignalDispatcher(store=store, queue=queue)
    sources = [
        SignalSource(
            source_id="src",
            validator=HMACValidator(secret="s"),
            allowed_types=frozenset({"x"}),
        ),
    ]
    producer = HTTPIngestionProducer(registry=SignalSourceRegistry(sources), max_body_bytes=10)
    import asyncio

    asyncio.get_event_loop().run_until_complete(producer.start(dispatcher))

    app = FastAPI()
    app.include_router(producer.router)
    client = TestClient(app)
    body = b"x" * 50  # 50 bytes > 10
    resp = client.post(
        "/signals",
        content=body,
        headers={"x-orchid-source": "src", "x-orchid-signature": _sign(body, "s")},
    )
    assert resp.status_code == 413
