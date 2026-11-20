"""Pipeline contracts — value objects, ports, and errors.

A pipeline is a typed DAG of jobs over an in-memory information graph. This
module owns the dependency-free half of that contract: the immutable value
objects exchanged between a run session and its jobs, the read-only graph view,
the step/observer ports, and the error hierarchy.

The module is deliberately stdlib-only. The concrete run session, graph
mutation engine, shape compiler, and LangGraph topology live outside ``core/``;
everything here is a contract a consumer can import without pulling a backend,
a transport, or a schema library into the process.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from .events.signal import SignalEnvelope

__all__ = [
    "PIPELINE_INPUT_OBJECT_ID",
    "PIPELINE_INPUT_OBJECT_TYPE",
    "PIPELINE_RESERVED_OBJECT_IDS",
    "OrchidDataGraphView",
    "OrchidGraphMutation",
    "OrchidGraphObject",
    "OrchidGraphRelation",
    "OrchidPipelineApprovalRequiredError",
    "OrchidPipelineAuthError",
    "OrchidPipelineConfigError",
    "OrchidPipelineDisabledError",
    "OrchidPipelineError",
    "OrchidPipelineEvent",
    "OrchidPipelineEventsDisabledError",
    "OrchidPipelineGraphConflictError",
    "OrchidPipelineInputError",
    "OrchidPipelineJobContext",
    "OrchidPipelineLimitError",
    "OrchidPipelineMappingError",
    "OrchidPipelineNotFoundError",
    "OrchidPipelineObserver",
    "OrchidPipelineOutputError",
    "OrchidPipelineResult",
    "OrchidPipelineRetryableError",
    "OrchidPipelineRunContext",
    "OrchidPipelineSerializationError",
    "OrchidPipelineShapeError",
    "OrchidPipelineStep",
    "OrchidPipelineStepContext",
    "OrchidPipelineStepError",
    "OrchidPipelineStepResult",
    "OrchidPipelineTimeoutError",
    "OrchidPipelineWriteNotAllowedError",
]

PIPELINE_INPUT_OBJECT_ID = "__input__"
PIPELINE_INPUT_OBJECT_TYPE = "__pipeline_input__"

#: Object IDs the graph owns and a job may never write.
PIPELINE_RESERVED_OBJECT_IDS: frozenset[str] = frozenset({PIPELINE_INPUT_OBJECT_ID})

_MUTATION_OPS: frozenset[str] = frozenset({"create", "update", "upsert", "delete", "link", "unlink"})
_MUTATION_MODES: frozenset[str] = frozenset({"merge", "replace"})
_RUN_MODES: frozenset[str] = frozenset({"direct", "chat", "bloom"})
_RESULT_STATUSES: frozenset[str] = frozenset({"succeeded", "failed", "partial", "cancelled"})
_SUMMARY_SOURCES: frozenset[str] = frozenset({"none", "template", "llm", "caller"})

_MAX_OBJECT_ID_CHARS = 256
_MAX_RELATION_ID_CHARS = 1024
_MAX_ERROR_MESSAGE_CHARS = 2000
_MAX_JSON_DEPTH = 50


# ── Small helpers ────────────────────────────────────────────


def _jsonable(
    value: Any,
    *,
    path: str = "$",
    depth: int = 0,
    seen: frozenset[int] | None = None,
) -> Any:
    """Return ``value`` as a JSON-compatible structure or raise.

    Used by every ``to_dict`` in this module. Datetimes become ISO-8601
    strings, mappings become plain dicts with string keys, and sequences become
    lists. Anything a JSON encoder cannot represent — a live object handle,
    bytes, a non-finite float, a circular reference — raises
    :class:`OrchidPipelineSerializationError` instead of leaking through to a
    later encoder failure.
    """
    if depth > _MAX_JSON_DEPTH:
        raise OrchidPipelineSerializationError(f"value at {path} exceeds the maximum nesting depth")
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise OrchidPipelineSerializationError(f"value at {path} is a non-finite float")
        return value
    if isinstance(value, datetime):
        return value.isoformat()

    ancestors = seen or frozenset()
    identifier = id(value)
    if identifier in ancestors:
        raise OrchidPipelineSerializationError(f"value at {path} contains a circular reference")
    children = ancestors | {identifier}

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise OrchidPipelineSerializationError(
                    f"mapping key at {path} must be a string, got {type(key).__name__}"
                )
            result[key] = _jsonable(item, path=f"{path}.{key}", depth=depth + 1, seen=children)
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            _jsonable(item, path=f"{path}[{index}]", depth=depth + 1, seen=children) for index, item in enumerate(value)
        ]

    raise OrchidPipelineSerializationError(f"value at {path} is not JSON-serializable: {type(value).__name__}")


def _require_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _require_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return dict(value)


def _validate_object_id(value: Any, *, field_name: str = "object id") -> str:
    text = _require_text(value, field_name=field_name)
    if len(text) > _MAX_OBJECT_ID_CHARS:
        raise ValueError(f"{field_name} must be at most {_MAX_OBJECT_ID_CHARS} characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in text):
        raise ValueError(f"{field_name} must not contain control characters")
    return text


def _validate_relation_id(value: Any, *, field_name: str = "relation id") -> str:
    """Bound a deterministic ``type:source:target`` id without capping endpoints.

    Two 256-character object ids plus a relation type stay well under 1024
    characters, so a legitimate link is never rejected.
    """
    text = _require_text(value, field_name=field_name)
    if len(text) > _MAX_RELATION_ID_CHARS:
        raise ValueError(f"{field_name} must be at most {_MAX_RELATION_ID_CHARS} characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in text):
        raise ValueError(f"{field_name} must not contain control characters")
    return text


def _bounded_message(message: str) -> str:
    if len(message) <= _MAX_ERROR_MESSAGE_CHARS:
        return message
    return message[: _MAX_ERROR_MESSAGE_CHARS - 1] + "…"


# ── Value objects ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class OrchidGraphObject:
    """One typed object in a run-scoped information graph.

    ``revision`` starts at ``1`` on create and increments on every update, so a
    read-modify-write job can pass ``expected_revision`` on its mutation and let
    the session reject a lost update instead of silently overwriting one.
    """

    id: str
    type: str
    data: Mapping[str, Any]
    revision: int = 1
    shape_version: str = "1"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    created_by: str = ""
    updated_by: str = ""

    def __post_init__(self) -> None:
        _validate_object_id(self.id)
        _require_text(self.type, field_name="object type")
        _require_text(self.shape_version, field_name="object shape_version")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("object revision must be an integer")
        if self.revision < 1:
            raise ValueError("object revision must be at least 1")
        object.__setattr__(self, "data", _require_mapping(self.data, field_name="object data"))
        object.__setattr__(self, "metadata", _require_mapping(self.metadata, field_name="object metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": _jsonable(self.id, path="$.id"),
            "type": _jsonable(self.type, path="$.type"),
            "data": _jsonable(self.data, path="$.data"),
            "revision": _jsonable(self.revision, path="$.revision"),
            "shape_version": _jsonable(self.shape_version, path="$.shape_version"),
            "metadata": _jsonable(self.metadata, path="$.metadata"),
            "created_at": _jsonable(self.created_at, path="$.created_at"),
            "updated_at": _jsonable(self.updated_at, path="$.updated_at"),
            "created_by": _jsonable(self.created_by, path="$.created_by"),
            "updated_by": _jsonable(self.updated_by, path="$.updated_by"),
        }


@dataclass(frozen=True, slots=True)
class OrchidGraphRelation:
    """One typed edge between two graph objects.

    The ID is deterministic (``"{type}:{source_id}:{target_id}"``) so a link is
    an idempotent relation upsert rather than a duplicate edge.
    """

    id: str
    type: str
    source_id: str
    target_id: str
    data: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = ""
    created_by: str = ""

    def __post_init__(self) -> None:
        _validate_relation_id(self.id)
        _require_text(self.type, field_name="relation type")
        _validate_object_id(self.source_id, field_name="relation source_id")
        _validate_object_id(self.target_id, field_name="relation target_id")
        object.__setattr__(self, "data", _require_mapping(self.data, field_name="relation data"))
        object.__setattr__(self, "metadata", _require_mapping(self.metadata, field_name="relation metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": _jsonable(self.id, path="$.id"),
            "type": _jsonable(self.type, path="$.type"),
            "source_id": _jsonable(self.source_id, path="$.source_id"),
            "target_id": _jsonable(self.target_id, path="$.target_id"),
            "data": _jsonable(self.data, path="$.data"),
            "metadata": _jsonable(self.metadata, path="$.metadata"),
            "created_at": _jsonable(self.created_at, path="$.created_at"),
            "created_by": _jsonable(self.created_by, path="$.created_by"),
        }


# ORCHID-MEM: 01a1e991-b914-7705-883e-c7a94fad7f55 — Graph mutation value object; jobs return mutations and the session commits them atomically.
@dataclass(frozen=True, slots=True)
class OrchidGraphMutation:
    """A declarative write a step asks the run session to commit.

    Jobs never mutate the graph directly: they return mutations, and the
    session validates the whole batch against the current graph before applying
    any of it. ``merge`` deep-merges ``data``; ``replace`` swaps it wholesale.
    ``link`` / ``unlink`` carry the relation to upsert or delete.
    """

    op: Literal["create", "update", "upsert", "delete", "link", "unlink"]
    object_id: str = ""
    object_type: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)
    mode: Literal["merge", "replace"] = "merge"
    expected_revision: int | None = None
    relation: OrchidGraphRelation | None = None
    strict: bool = False

    def __post_init__(self) -> None:
        if self.op not in _MUTATION_OPS:
            raise ValueError(f"mutation op must be one of {sorted(_MUTATION_OPS)}, got {self.op!r}")
        if self.mode not in _MUTATION_MODES:
            raise ValueError(f"mutation mode must be one of {sorted(_MUTATION_MODES)}, got {self.mode!r}")
        object.__setattr__(self, "data", _require_mapping(self.data, field_name="mutation data"))

        if self.op in {"link", "unlink"}:
            if not isinstance(self.relation, OrchidGraphRelation):
                raise ValueError(f"mutation op {self.op!r} requires a relation")
        elif self.relation is not None:
            raise ValueError(f"mutation op {self.op!r} must not carry a relation")
        else:
            _validate_object_id(self.object_id, field_name="mutation object_id")
            # ``update`` / ``upsert`` may address an existing object whose type
            # the session already knows, so only ``create`` needs one.
            if self.op == "create":
                _require_text(self.object_type, field_name="mutation object_type")

        if self.expected_revision is not None:
            if isinstance(self.expected_revision, bool) or not isinstance(self.expected_revision, int):
                raise TypeError("mutation expected_revision must be an integer or None")
            if self.expected_revision < 1:
                raise ValueError("mutation expected_revision must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "object_id": self.object_id,
            "object_type": self.object_type,
            "data": _jsonable(self.data),
            "mode": self.mode,
            "expected_revision": self.expected_revision,
            "relation": self.relation.to_dict() if self.relation is not None else None,
            "strict": self.strict,
        }


@dataclass(frozen=True, slots=True)
class OrchidPipelineRunContext:
    """Identity and provenance of one pipeline run.

    Auth is execution context, not state: the session holds the
    ``OrchidAuthContext`` out-of-band. This object carries only identifiers
    that are safe to surface in results, events, and logs.
    """

    run_id: str
    pipeline: str
    pipeline_version: str
    mode: Literal["direct", "chat", "bloom"]
    tenant_key: str
    user_id: str
    started_at: datetime
    chat_id: str | None = None
    correlation_id: str | None = None
    parent_run_id: str | None = None
    trigger_id: str | None = None
    signal_id: str | None = None
    deadline: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # ORCHID-MEM: 01a1f2fe-cd1f-7ede-b321-a8b5ecaaffa1 — Nested run depth; bounds sub-pipeline recursion at runtime.
    subpipeline_depth: int = 0
    # ORCHID-MEM: 01a1f31c-f71d-77e1-a6de-0dd56f84c521 — Root sub-pipeline cap; inherited by nested runs so per-definition caps apply only at their own root.
    subpipeline_max_depth: int = 3

    def __post_init__(self) -> None:
        _require_text(self.run_id, field_name="run_id")
        _require_text(self.pipeline, field_name="pipeline")
        _require_text(self.pipeline_version, field_name="pipeline_version")
        if self.mode not in _RUN_MODES:
            raise ValueError(f"run mode must be one of {sorted(_RUN_MODES)}, got {self.mode!r}")
        if not isinstance(self.started_at, datetime):
            raise TypeError("run started_at must be a datetime")
        if self.deadline is not None:
            if not isinstance(self.deadline, datetime):
                raise TypeError("run deadline must be a datetime or None")
            if self.deadline < self.started_at:
                raise ValueError("run deadline must not precede started_at")
        if isinstance(self.subpipeline_depth, bool) or not isinstance(self.subpipeline_depth, int):
            raise TypeError("run subpipeline_depth must be an integer")
        if self.subpipeline_depth < 0:
            raise ValueError("run subpipeline_depth must be non-negative")
        if isinstance(self.subpipeline_max_depth, bool) or not isinstance(self.subpipeline_max_depth, int):
            raise TypeError("run subpipeline_max_depth must be an integer")
        if self.subpipeline_max_depth < 1:
            raise ValueError("run subpipeline_max_depth must be at least 1")
        object.__setattr__(self, "metadata", _require_mapping(self.metadata, field_name="run metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "pipeline": self.pipeline,
            "pipeline_version": self.pipeline_version,
            "mode": self.mode,
            "tenant_key": self.tenant_key,
            "user_id": self.user_id,
            "chat_id": self.chat_id,
            "correlation_id": self.correlation_id,
            "parent_run_id": self.parent_run_id,
            "trigger_id": self.trigger_id,
            "signal_id": self.signal_id,
            "started_at": self.started_at.isoformat(),
            "deadline": self.deadline.isoformat() if self.deadline is not None else None,
            "subpipeline_depth": self.subpipeline_depth,
            "subpipeline_max_depth": self.subpipeline_max_depth,
            "metadata": _jsonable(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class OrchidPipelineJobContext:
    """The slice of run context a single step receives about its own job.

    ``item`` / ``index`` / ``parent_job`` are populated only for jobs executed
    through a ``map`` job; a top-level job leaves them unset.
    """

    job_id: str
    kind: str
    attempt: int = 1
    params: Mapping[str, Any] = field(default_factory=dict)
    item: Mapping[str, Any] | None = None
    index: int | None = None
    parent_job: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.job_id, field_name="job_id")
        _require_text(self.kind, field_name="job kind")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int):
            raise TypeError("job attempt must be an integer")
        if self.attempt < 1:
            raise ValueError("job attempt must be at least 1")
        object.__setattr__(self, "params", _require_mapping(self.params, field_name="job params"))
        if self.item is not None:
            object.__setattr__(self, "item", _require_mapping(self.item, field_name="job item"))
        if self.index is not None:
            if isinstance(self.index, bool) or not isinstance(self.index, int):
                raise TypeError("job index must be an integer or None")
            if self.index < 0:
                raise ValueError("job index must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "attempt": self.attempt,
            "params": _jsonable(self.params),
            "item": _jsonable(self.item) if self.item is not None else None,
            "index": self.index,
            "parent_job": self.parent_job,
        }


@dataclass(frozen=True, slots=True)
class OrchidPipelineStepResult:
    """What a step returns to the job node.

    ``output`` is the step's structured result (a dict for deterministic steps,
    a validated model dump for LLM/agent jobs). ``mutations`` are requests, not
    commits: the session applies them atomically after the step returns.
    """

    output: Any = None
    mutations: tuple[OrchidGraphMutation, ...] = ()
    usage: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mutations = tuple(self.mutations)
        for mutation in mutations:
            if not isinstance(mutation, OrchidGraphMutation):
                raise TypeError("step result mutations must be OrchidGraphMutation values")
        object.__setattr__(self, "mutations", mutations)
        object.__setattr__(self, "usage", _require_mapping(self.usage, field_name="step result usage"))
        object.__setattr__(self, "metadata", _require_mapping(self.metadata, field_name="step result metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": _jsonable(self.output),
            "mutations": [mutation.to_dict() for mutation in self.mutations],
            "usage": _jsonable(self.usage),
            "metadata": _jsonable(self.metadata),
        }


# ORCHID-MEM: 01a1e991-b942-72a1-a8a1-bca56713c32d — Terminal pipeline result; typed output, job report, graph stats, JSON-safe dict form.
@dataclass(frozen=True, slots=True)
class OrchidPipelineResult:
    """The terminal result of a run, safe to persist or return to a caller.

    ``to_dict(include_graph=False)`` is the API/tool form: it keeps the graph
    statistics but omits the serialized graph snapshot. ``graph`` always remains
    available on the Python object.
    """

    run_id: str
    pipeline: str
    status: Literal["succeeded", "failed", "partial", "cancelled"]
    output: Mapping[str, Any] | None = None
    graph: Mapping[str, Any] = field(default_factory=dict)
    summary: str = ""
    summary_source: Literal["none", "template", "llm", "caller"] = "none"
    jobs: Mapping[str, Any] = field(default_factory=dict)
    graph_stats: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    started_at: datetime = field(kw_only=True)
    finished_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.run_id, field_name="run_id")
        _require_text(self.pipeline, field_name="pipeline")
        if not isinstance(self.started_at, datetime):
            raise TypeError("result started_at must be a datetime")
        if self.status not in _RESULT_STATUSES:
            raise ValueError(f"result status must be one of {sorted(_RESULT_STATUSES)}, got {self.status!r}")
        if self.summary_source not in _SUMMARY_SOURCES:
            raise ValueError(
                f"result summary_source must be one of {sorted(_SUMMARY_SOURCES)}, got {self.summary_source!r}"
            )
        if self.finished_at is not None:
            if not isinstance(self.finished_at, datetime):
                raise TypeError("result finished_at must be a datetime or None")
            if self.finished_at < self.started_at:
                raise ValueError("result finished_at must not precede started_at")
        if self.error is not None:
            if not isinstance(self.error, str):
                raise TypeError("result error must be a string or None")
            object.__setattr__(self, "error", _bounded_message(self.error))
        if self.output is not None:
            object.__setattr__(self, "output", _require_mapping(self.output, field_name="result output"))
        object.__setattr__(self, "graph", _require_mapping(self.graph, field_name="result graph"))
        object.__setattr__(self, "jobs", _require_mapping(self.jobs, field_name="result jobs"))
        object.__setattr__(self, "graph_stats", _require_mapping(self.graph_stats, field_name="result graph_stats"))
        object.__setattr__(self, "metadata", _require_mapping(self.metadata, field_name="result metadata"))

    @property
    def duration_ms(self) -> float | None:
        """Wall-clock duration in milliseconds, or ``None`` while unfinished."""
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds() * 1000.0

    def to_dict(self, *, include_graph: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "pipeline": self.pipeline,
            "status": self.status,
            "output": _jsonable(self.output, path="$.output"),
            "summary": _jsonable(self.summary, path="$.summary"),
            "summary_source": self.summary_source,
            "jobs": _jsonable(self.jobs, path="$.jobs"),
            "graph_stats": _jsonable(self.graph_stats, path="$.graph_stats"),
            "error": _jsonable(self.error, path="$.error"),
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at is not None else None,
            "duration_ms": _jsonable(self.duration_ms, path="$.duration_ms"),
            "metadata": _jsonable(self.metadata, path="$.metadata"),
        }
        if include_graph:
            payload["graph"] = _jsonable(self.graph, path="$.graph")
        return payload


@dataclass(frozen=True, slots=True)
class OrchidPipelineEvent:
    """One bounded observability event emitted during a run.

    Observers receive the typed event; stream adapters serialize it with
    :meth:`to_dict`. Observer failures are logged and never break a run.
    """

    type: str
    run_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.type, field_name="event type")
        _require_text(self.run_id, field_name="event run_id")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("event timestamp must be a datetime")
        object.__setattr__(self, "data", _require_mapping(self.data, field_name="event data"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "data": _jsonable(self.data),
        }


# ── Ports ────────────────────────────────────────────────────


class OrchidDataGraphView(ABC):
    """Read-only view of the run's information graph.

    Jobs receive this view and cannot mutate through it; writes are returned as
    :class:`OrchidGraphMutation` values and committed by the session.
    ``neighbours`` is traversal-bounded by the concrete implementation.
    """

    @abstractmethod
    def get(self, object_id: str) -> OrchidGraphObject | None:
        """Return one object, or ``None`` when it is not in the graph."""

    @abstractmethod
    def find(
        self,
        object_type: str | None = None,
        where: Mapping[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[OrchidGraphObject]:
        """Return objects matching ``object_type`` and top-level equality ``where``."""

    @abstractmethod
    def relations(
        self,
        *,
        type: str | None = None,
        source_id: str | None = None,
        target_id: str | None = None,
    ) -> list[OrchidGraphRelation]:
        """Return relations matching every supplied filter."""

    @abstractmethod
    def neighbours(
        self,
        object_id: str,
        *,
        relation: str | None = None,
        direction: Literal["out", "in", "both"] = "out",
        max_hops: int = 1,
    ) -> tuple[list[OrchidGraphObject], list[OrchidGraphRelation]]:
        """Return the objects and relations reachable within ``max_hops``."""


@runtime_checkable
class OrchidPipelineStepContext(Protocol):
    """Protocol for the runtime context handed to a pipeline step.

    A protocol, not a dataclass, so ``core/`` stays dependency-free and a test
    step can supply a tiny fake. The concrete implementation lives in the
    pipeline runtime and is the only object that touches auth, models, RAG, or
    signals.
    """

    run: OrchidPipelineRunContext
    job: OrchidPipelineJobContext
    input: Mapping[str, Any]
    graph: OrchidDataGraphView
    auth: Any
    chat_model: Any | None
    reader: Any | None
    logger: logging.Logger
    idempotency_key: str

    async def call_tool(self, name: str, **kwargs: Any) -> Any:
        """Invoke a built-in tool with the run's auth injected."""
        ...

    async def emit_signal(self, envelope: SignalEnvelope) -> Any:
        """Emit a signal envelope; raises when the events layer is disabled."""
        ...

    def stable_id(self, *parts: str) -> str:
        """Derive a deterministic ID from the run namespace and ``parts``."""
        ...


# ORCHID-MEM: 01a1e991-b972-7b02-b8a5-3c20ef800d05 — Custom pipeline step extension point; stateless across runs, bound per compiled pipeline.
class OrchidPipelineStep(ABC):
    """A custom pipeline step resolved by ``kind`` or dotted ``class``.

    Instances are created once per compiled pipeline and must be stateless
    across runs: all per-run data arrives through the context.
    """

    kind: ClassVar[str] = "custom"

    @abstractmethod
    async def execute(self, ctx: OrchidPipelineStepContext) -> OrchidPipelineStepResult:
        """Run the step and return its output plus requested mutations."""


class OrchidPipelineObserver(ABC):
    """Observer port for run events (SSE, logging, metrics, …).

    Implementations must not raise: the runtime logs and swallows observer
    failures so observability can never break a run. The default runtime
    observer is a no-op.
    """

    @abstractmethod
    async def emit(self, event: OrchidPipelineEvent) -> None:
        """Receive one event."""


# ── Errors ───────────────────────────────────────────────────


# ORCHID-MEM: 01a1e991-b9a2-7451-aeb5-ec683eb38c5f — Pipeline error hierarchy; stable codes and bounded messages for API/CLI mapping.
class OrchidPipelineError(Exception):
    """Base class for every pipeline failure.

    ``code`` is a stable identifier safe for API/CLI mapping and logs. Messages
    never carry bearer tokens or unbounded object dumps.
    """

    code: ClassVar[str] = "pipeline_error"

    def __init__(self, message: str, *, code: str | None = None, retryable: bool = False) -> None:
        if not isinstance(message, str):
            raise TypeError("error message must be a string")
        super().__init__(_bounded_message(message))
        if code is not None:
            self.code = code
        self.retryable = retryable


class OrchidPipelineConfigError(OrchidPipelineError):
    code = "pipeline_config_error"


class OrchidPipelineNotFoundError(OrchidPipelineError):
    code = "pipeline_not_found"


class OrchidPipelineDisabledError(OrchidPipelineError):
    code = "pipeline_disabled"


class OrchidPipelineAuthError(OrchidPipelineError):
    code = "pipeline_auth_error"


class OrchidPipelineInputError(OrchidPipelineError):
    code = "pipeline_input_error"


class OrchidPipelineOutputError(OrchidPipelineError):
    code = "pipeline_output_error"


class OrchidPipelineShapeError(OrchidPipelineError):
    code = "pipeline_shape_error"


class OrchidPipelineMappingError(OrchidPipelineError):
    code = "pipeline_mapping_error"


class OrchidPipelineGraphConflictError(OrchidPipelineError):
    code = "pipeline_graph_conflict"


class OrchidPipelineWriteNotAllowedError(OrchidPipelineError):
    code = "pipeline_write_not_allowed"


class OrchidPipelineSerializationError(OrchidPipelineError):
    code = "pipeline_serialization_error"


class OrchidPipelineLimitError(OrchidPipelineError):
    code = "pipeline_limit_error"


class OrchidPipelineTimeoutError(OrchidPipelineError):
    code = "pipeline_timeout"


class OrchidPipelineStepError(OrchidPipelineError):
    """A step failed in a way the runtime records against its job.

    ``job_id`` and ``kind`` let the job node attribute the failure without
    re-parsing the message; ``retryable`` tells the retry policy whether another
    attempt can succeed.
    """

    code = "pipeline_step_error"

    def __init__(
        self,
        message: str,
        *,
        job_id: str = "",
        kind: str = "",
        retryable: bool = False,
    ) -> None:
        if not isinstance(message, str):
            raise TypeError("error message must be a string")
        super().__init__(_bounded_message(message), retryable=retryable)
        self.job_id = job_id
        self.kind = kind

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "job_id": self.job_id,
            "kind": self.kind,
            "retryable": self.retryable,
            "message": _bounded_message(str(self)),
        }


class OrchidPipelineRetryableError(OrchidPipelineStepError):
    """A step failed transiently; the retry policy may attempt it again."""

    code = "pipeline_retryable_step_error"

    def __init__(
        self,
        message: str,
        *,
        job_id: str = "",
        kind: str = "",
        retryable: bool = True,
    ) -> None:
        super().__init__(message, job_id=job_id, kind=kind, retryable=retryable)


class OrchidPipelineApprovalRequiredError(OrchidPipelineStepError):
    """A step (or its tool) needs human approval the runtime cannot obtain.

    Retrying the same step with the same context cannot succeed, so
    ``retryable`` defaults to ``False``.
    """

    code = "pipeline_approval_required"


class OrchidPipelineEventsDisabledError(OrchidPipelineError):
    code = "pipeline_events_disabled"
