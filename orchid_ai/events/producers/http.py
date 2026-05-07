"""HTTP ingestion producer (§11.1).

Mounts a FastAPI sub-router that exposes::

    POST {mount}      ← default ``/signals``

The route validates ``X-Orchid-Source`` against an in-memory
:class:`SignalSourceRegistry`, runs the matched
:class:`SignalAuthValidator`, builds a :class:`SignalEnvelope`, calls
``dispatcher.ingest`` and returns ``202 + {signal_id, deduplicated}``
in well under 50 ms p95 against a local Postgres (per the spec exit
criterion).

FastAPI is imported lazily so the library stays platform-agnostic —
projects that don't enable events (or don't use HTTP ingestion) pay
no import cost and don't need fastapi as a runtime dependency.

The producer's :meth:`router` is read by ``orchid-api``'s lifespan
and ``app.include_router(...)``-ed once the dispatcher is built.
"""

from __future__ import annotations

import datetime as _dt
import json as _json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ...core.events.dispatcher import OrchidSignalDispatcher
from ...core.events.errors import (
    SignalAuthValidationError,
    SignalSourceTypeNotAllowedError,
    SignalSourceUnknownError,
)
from ...core.events.producer import OrchidSignalProducer
from ...core.events.signal import SignalEnvelope
from ..auth.base import SignalAuthRequest, SignalAuthValidator

if TYPE_CHECKING:
    from fastapi import APIRouter
    from fastapi.responses import JSONResponse


# FastAPI introspects route-handler annotations via ``get_type_hints``;
# a string-quoted ``"Request"`` would fail to resolve and the request
# body would get treated as a JSON-body model.  We import the real
# symbol at module level when fastapi is present so the annotation
# below is resolvable, and fall back to ``Any`` (which FastAPI accepts
# as a request-typed parameter) when fastapi is missing.
try:
    from fastapi import Request as _FastAPIRequest
except ImportError:  # pragma: no cover — degrades gracefully
    _FastAPIRequest = Any  # type: ignore[assignment,misc]


_logger = logging.getLogger(__name__)


# ── In-memory source registry ───────────────────────────────


@dataclass(frozen=True, slots=True)
class SignalSource:
    """One ``signal_sources`` row, materialised in memory.

    The validator is the *resolved* validator (not the dotted-path
    config) — the lifecycle layer compiles
    :class:`OrchidIngestionSourceConfig` into this shape before
    handing it to the producer.
    """

    source_id: str
    validator: SignalAuthValidator
    allowed_types: frozenset[str]


class SignalSourceRegistry:
    """Lookup of registered HTTP-ingestion sources.

    Plain dict wrapper; the lifecycle layer constructs one from
    YAML at boot.  v1 doesn't support hot-reload — operators
    restart the process to pick up source changes.
    """

    def __init__(self, sources: list[SignalSource]) -> None:
        self._by_id: dict[str, SignalSource] = {s.source_id: s for s in sources}

    def get(self, source_id: str) -> SignalSource | None:
        return self._by_id.get(source_id)

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self):
        return iter(self._by_id.values())


# ── Producer ────────────────────────────────────────────────


class HTTPIngestionProducer(OrchidSignalProducer):
    """FastAPI-backed signal source.

    The dispatcher is set in :meth:`start` per the
    :class:`OrchidSignalProducer` contract; the router is built
    eagerly so :class:`orchid-api` lifespan can mount it before
    ``start`` runs.

    Headers:

    - ``X-Orchid-Source`` — required.  Must match a registered source.
    - ``X-Orchid-Signature`` — consumed by the validator (HMAC).
    - ``Idempotency-Key`` — copied verbatim into ``dedupe_key`` so
      re-deliveries are deduplicated by the
      ``UNIQUE (source, dedupe_key)`` index on ``signals``.
    - ``Authorization`` — consumed by :class:`BearerValidator`.

    Body (JSON):

    .. code-block:: json

        {
          "type": "support.ticket.created",
          "payload": {...},
          "occurred_at": "2026-05-07T08:51:42Z",
          "tenant_key": "acme-prod",
          "user_id": "u-7abc12",
          "correlation_id": "...",
          "identity_claim": {...},
          "chat_binding": {...}
        }

    Only ``type`` and ``tenant_key`` are required; the rest default
    to None / now.
    """

    def __init__(
        self,
        *,
        registry: SignalSourceRegistry,
        mount: str = "/signals",
        max_body_bytes: int = 1_000_000,
    ) -> None:
        # Lazy fastapi import so the library stays platform-agnostic.
        # Construction-time failure surfaces as a clear RuntimeError
        # rather than an obscure NameError when the route fires.
        try:
            from fastapi import APIRouter, HTTPException
            from fastapi import status as _status
            from fastapi.responses import JSONResponse
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "HTTPIngestionProducer requires fastapi — install via "
                "`pip install fastapi` or include orchid-api which "
                "carries it as a core dependency."
            ) from exc

        self._registry = registry
        self._mount = mount.rstrip("/") or "/signals"
        self._max_body = max_body_bytes
        self._dispatcher: OrchidSignalDispatcher | None = None

        self._JSONResponse = JSONResponse
        self._HTTPException = HTTPException
        self._fastapi_status = _status
        self._router = APIRouter()
        self._router.add_api_route(
            self._mount,
            self._ingest,
            methods=["POST"],
            status_code=_status.HTTP_202_ACCEPTED,
            response_class=JSONResponse,
            tags=["events"],
            include_in_schema=True,
        )

    # ── Lifecycle ────────────────────────────────────────

    @property
    def name(self) -> str:
        return "HTTPIngestionProducer"

    @property
    def router(self) -> "APIRouter":
        return self._router

    @property
    def mount(self) -> str:
        return self._mount

    async def start(self, dispatcher: OrchidSignalDispatcher) -> None:
        self._dispatcher = dispatcher
        _logger.info(
            "[HTTPIngestionProducer] started — mount=%s sources=%d",
            self._mount,
            len(self._registry),
        )

    async def stop(self) -> None:
        # Stateless — the FastAPI lifespan tears the router down with
        # the app; nothing to release here.
        self._dispatcher = None

    # ── Handler ──────────────────────────────────────────

    async def _ingest(self, request: _FastAPIRequest) -> "JSONResponse":
        """The route body.  Wired into the FastAPI router at construction.

        Returns ``202 + {signal_id, deduplicated}`` on success, or
        a structured error otherwise:

        - 400 — malformed JSON / missing required field.
        - 401 — signature / bearer rejected by the validator.
        - 403 — declared signal ``type`` is not in the source's
          allow-list.
        - 404 — ``X-Orchid-Source`` not registered.
        - 413 — body exceeds ``max_body_bytes``.
        - 503 — dispatcher not yet started (lifespan ordering bug).
        """
        if self._dispatcher is None:
            raise self._HTTPException(503, "events dispatcher not started")

        source_id = request.headers.get("x-orchid-source", "")
        if not source_id:
            raise self._HTTPException(400, "missing X-Orchid-Source header")

        source = self._registry.get(source_id)
        if source is None:
            raise self._HTTPException(404, f"unknown source {source_id!r}")

        raw_body = await request.body()
        if len(raw_body) > self._max_body:
            raise self._HTTPException(413, "request body too large")

        # Validate before parsing — a bad signature should never
        # surface payload contents in the error response, and the
        # HMAC validator hashes the raw bytes.
        headers = {k.lower(): v for k, v in request.headers.items()}
        try:
            await source.validator.validate(
                SignalAuthRequest(
                    source_id=source_id,
                    raw_body=raw_body,
                    headers=headers,
                )
            )
        except SignalAuthValidationError as exc:
            raise self._HTTPException(401, str(exc)) from exc

        # Body must be JSON.
        try:
            body = _json.loads(raw_body or b"{}")
        except Exception as exc:
            raise self._HTTPException(400, f"body is not valid JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise self._HTTPException(400, "body must be a JSON object")

        signal_type = body.get("type")
        tenant_key = body.get("tenant_key")
        if not isinstance(signal_type, str) or not signal_type:
            raise self._HTTPException(400, "missing required field 'type'")
        if not isinstance(tenant_key, str) or not tenant_key:
            raise self._HTTPException(400, "missing required field 'tenant_key'")

        if source.allowed_types and signal_type not in source.allowed_types:
            raise self._HTTPException(
                403,
                f"signal type {signal_type!r} is not allowed for source "
                f"{source_id!r} (allow-list: {sorted(source.allowed_types)})",
            )

        dedupe_key = request.headers.get("idempotency-key") or body.get("dedupe_key")

        envelope = SignalEnvelope(
            type=signal_type,
            payload=body.get("payload") if isinstance(body.get("payload"), dict) else {},
            source=source_id,
            occurred_at=_parse_occurred_at(body.get("occurred_at")),
            tenant_key=tenant_key,
            user_id=_str_or_none(body.get("user_id")),
            correlation_id=_str_or_none(body.get("correlation_id")),
            dedupe_key=dedupe_key if isinstance(dedupe_key, str) and dedupe_key else None,
            identity_claim=body.get("identity_claim") if isinstance(body.get("identity_claim"), dict) else None,
            chat_binding=body.get("chat_binding") if isinstance(body.get("chat_binding"), dict) else None,
        )

        try:
            result = await self._dispatcher.ingest(envelope)
        except SignalSourceUnknownError as exc:
            raise self._HTTPException(404, str(exc)) from exc
        except SignalSourceTypeNotAllowedError as exc:
            raise self._HTTPException(403, str(exc)) from exc

        return self._JSONResponse(
            status_code=self._fastapi_status.HTTP_202_ACCEPTED,
            content={
                "signal_id": str(result.signal_id),
                "deduplicated": result.deduplicated,
            },
        )


# ── Helpers ─────────────────────────────────────────────────


def _str_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _parse_occurred_at(value: Any) -> _dt.datetime:
    """Parse the ``occurred_at`` field — RFC 3339 / ISO 8601.

    Falls back to ``now()`` when missing or unparseable rather than
    rejecting; replayed historical events sometimes lack the field
    and the framework prefers to ingest with a best-effort timestamp
    over dropping the signal.
    """
    if isinstance(value, str) and value:
        try:
            iso = value.replace("Z", "+00:00")
            parsed = _dt.datetime.fromisoformat(iso)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=_dt.UTC)
            return parsed
        except Exception:
            return _dt.datetime.now(tz=_dt.UTC)
    return _dt.datetime.now(tz=_dt.UTC)
