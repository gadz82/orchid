"""Run-scoped information graph with atomic, revisioned commits.

A pipeline run owns exactly one :class:`DataGraph`. Jobs never touch it: they
read an :class:`orchid_ai.core.pipeline.OrchidDataGraphView` and return
``OrchidGraphMutation`` values, which the run session commits as one envelope.

This module is the only writer of object and relation state. A commit is
all-or-nothing:

1. every mutation is planned against a working copy of the graph;
2. ids, revisions, JSON safety, shape validation and size caps are checked;
3. only when the whole batch succeeds are the working maps swapped in and the
   graph ``version`` incremented once.

A failure at any point discards the working copy, so the committed graph is
untouched and no provenance or counter is stamped.

Semantics:

* ``create`` fails when the object exists;
* ``update`` fails when it does not; ``merge`` deep-merges mappings, replaces
  lists and scalars, and deletes a key whose patch value is ``None``;
  ``replace`` swaps ``data`` wholesale;
* ``upsert`` creates when absent and updates when present;
* ``delete`` removes the object and every incident relation; ``strict=False``
  makes a missing object a no-op, ``strict=True`` fails;
* ``link`` / ``unlink`` upsert/delete the relation whose id is derived from
  ``"{type}:{source_id}:{target_id}"``. Re-linking replaces the relation's data
  and metadata while preserving its original ``created_at`` / ``created_by``.

Provenance is stamped from the committing ``job_id``: ``created_by`` /
``created_at`` on create, ``updated_by`` / ``updated_at`` on update. Timestamps
are strictly increasing within a graph even when the injected clock is coarse.

Relation identity is the documented ``"{type}:{source_id}:{target_id}"`` string.
Object ids may themselves contain ``:``, so two distinct triples can derive the
same string. The graph detects that collision at commit time and never merges or
overwrites one triple with another; this ambiguity is a known v1 limitation.

Concurrency: this class is **not** thread-safe. The run session serialises
commits (an ``asyncio.Lock`` around ``commit``); a graph instance is confined to
one run and is never persisted or serialised into graph state.

Read-only boundary: the step-facing guarantee belongs to
:class:`DataGraphView`, which returns defensive copies. The graph's own
``objects`` and ``relations`` maps are session-facing internals that expose the
live value objects, and a view holds a private reference to the graph; steps
must receive only the view.
"""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any, Literal, Protocol

from ..core.pipeline import (
    PIPELINE_INPUT_OBJECT_ID,
    PIPELINE_INPUT_OBJECT_TYPE,
    PIPELINE_RESERVED_OBJECT_IDS,
    OrchidDataGraphView,
    OrchidGraphMutation,
    OrchidGraphObject,
    OrchidGraphRelation,
    OrchidPipelineError,
    OrchidPipelineGraphConflictError,
    OrchidPipelineLimitError,
    OrchidPipelineSerializationError,
    OrchidPipelineShapeError,
    OrchidPipelineWriteNotAllowedError,
)

__all__ = [
    "DataGraph",
    "DataGraphCommit",
    "DataGraphMetrics",
    "DataGraphView",
    "ShapeValidator",
]

_MAX_OBJECT_ID_CHARS = 256
_MAX_TEXT_CHARS = 256
_MAX_ALIAS_CHARS = 128
_MAX_REVISION = (1 << 63) - 1
_MAX_JSON_DEPTH = 50
_MAX_NEIGHBOUR_HOPS = 10
_MAX_NEIGHBOUR_OBJECTS = 10_000
_LINK_OPS = frozenset({"link", "unlink"})
_SEED_ACTOR = "pipeline"


# ── Validator port ───────────────────────────────────────────


class ShapeValidator(Protocol):
    """Injected shape validation hook; the run session passes its registry.

    The protocol mirrors ``OrchidShapeRegistry.dump``: a validator receives the
    shape name, the candidate data, and the object id/version, and returns the
    normalised mapping to store (or ``None`` to keep the candidate unchanged).
    It raises ``OrchidPipelineShapeError`` on invalid data. ``core/`` and this
    module stay free of a schema library: the concrete validator belongs to the
    compiler.
    """

    def __call__(
        self,
        shape: str,
        data: Mapping[str, Any],
        *,
        object_id: str,
        shape_version: str,
    ) -> Mapping[str, Any] | None: ...


# ── Small checks ─────────────────────────────────────────────


def _require_utf8(value: str, *, field_name: str) -> str:
    """Reject strings a UTF-8 encoder cannot represent (lone surrogates)."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field_name} must be valid UTF-8 (no lone surrogates)") from exc
    return value


def _checked_text(value: Any, *, field_name: str, max_chars: int = _MAX_TEXT_CHARS) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    if len(value) > max_chars:
        raise ValueError(f"{field_name} must be at most {max_chars} characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{field_name} must not contain control characters")
    return _require_utf8(value, field_name=field_name)


def _checked_object_id(value: Any, *, field_name: str = "object id") -> str:
    return _checked_text(value, field_name=field_name, max_chars=_MAX_OBJECT_ID_CHARS)


def _checked_limit(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _checked_cap(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 1:
        raise ValueError(f"{field_name} must be at least 1")
    return value


def _checked_actor(value: Any, *, field_name: str) -> str:
    """Validate an optional provenance actor (an empty string means unknown)."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if value == "":
        return ""
    return _checked_text(value, field_name=field_name)


def _checked_timestamp(value: Any, *, field_name: str) -> str:
    """Validate an optional ISO-8601 provenance timestamp (empty means unknown)."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if value == "":
        return ""
    if len(value) > _MAX_TEXT_CHARS:
        raise ValueError(f"{field_name} must be at most {_MAX_TEXT_CHARS} characters")
    _require_utf8(value, field_name=field_name)
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    return value


def _checked_revision(value: Any, *, field_name: str) -> int:
    """Bound revisions so they cannot overflow JSON encoders or arithmetic."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 1:
        raise ValueError(f"{field_name} must be at least 1")
    if value > _MAX_REVISION:
        raise ValueError(f"{field_name} must be at most {_MAX_REVISION}")
    return value


def _json_safe_copy(
    value: Any,
    *,
    path: str = "$",
    depth: int = 0,
    ancestors: frozenset[int] | None = None,
) -> Any:
    """Return a deep, JSON-safe copy of ``value`` or raise.

    Accepted values are exactly what a JSON encoder can represent: ``None``,
    booleans, strings, integers, finite floats, mappings with string keys, and
    lists/tuples (normalised to lists). Datetimes, bytes, sets, live handles,
    non-string mapping keys, non-finite floats, over-deep structures and cycles
    are rejected before they can enter the graph.
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

    active = ancestors or frozenset()
    identifier = id(value)
    if identifier in active:
        raise OrchidPipelineSerializationError(f"value at {path} contains a circular reference")
    children = active | {identifier}

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise OrchidPipelineSerializationError(
                    f"mapping key at {path} must be a string, got {type(key).__name__}"
                )
            result[key] = _json_safe_copy(item, path=f"{path}.{key}", depth=depth + 1, ancestors=children)
        return result
    if isinstance(value, (list, tuple)):
        return [
            _json_safe_copy(item, path=f"{path}[{index}]", depth=depth + 1, ancestors=children)
            for index, item in enumerate(value)
        ]

    raise OrchidPipelineSerializationError(f"value at {path} is not JSON-serializable: {type(value).__name__}")


def _json_bytes(value: Mapping[str, Any]) -> int:
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:  # pragma: no cover - guarded by _json_safe_copy
        raise OrchidPipelineSerializationError(f"value is not JSON-serializable: {exc}") from exc
    return len(encoded)


def _payload_bytes(data: Mapping[str, Any], metadata: Mapping[str, Any]) -> int:
    """Serialized size of a value object's ``data`` plus its ``metadata``."""
    return _json_bytes(data) + _json_bytes(metadata)


def _object_payload_bytes(data: Mapping[str, Any], metadata: Mapping[str, Any], shape_version: str) -> int:
    """Object payload size, including the bounded shape version string."""
    try:
        version_bytes = len(shape_version.encode("utf-8"))
    except UnicodeEncodeError as exc:  # pragma: no cover - shape versions are validated first
        raise OrchidPipelineSerializationError("shape_version is not valid UTF-8") from exc
    return _payload_bytes(data, metadata) + version_bytes


def _alias_entry_bytes(alias: str, object_id: str) -> int:
    """Serialized size of one alias entry, used for the graph byte cap."""
    return _json_bytes({alias: object_id})


def _job_alias_entry_bytes(job_id: str, alias: str, object_id: str) -> int:
    """Serialized size of one job-scoped alias entry, including its owner."""
    return _alias_entry_bytes(alias, object_id) + len(job_id.encode("utf-8"))


def _copy_object(obj: OrchidGraphObject) -> OrchidGraphObject:
    """Return a defensive copy so a view reader can never mutate stored state."""
    return replace(
        obj,
        data=_json_safe_copy(obj.data),
        metadata=_json_safe_copy(obj.metadata),
    )


def _copy_relation(relation: OrchidGraphRelation) -> OrchidGraphRelation:
    """Return a defensive copy so a view reader can never mutate stored state."""
    return replace(
        relation,
        data=_json_safe_copy(relation.data),
        metadata=_json_safe_copy(relation.metadata),
    )


def _merge_data(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge ``patch`` into ``base``: dicts merge, lists/scalars replace, ``None`` deletes."""
    merged = {key: _json_safe_copy(item) for key, item in base.items()}
    for key, value in patch.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_data(merged[key], value)
        else:
            merged[key] = _json_safe_copy(value)
    return merged


def _relation_id(relation_type: str, source_id: str, target_id: str) -> str:
    return f"{relation_type}:{source_id}:{target_id}"


def _limit_error(name: str, actual: int, maximum: int, message: str) -> OrchidPipelineLimitError:
    error = OrchidPipelineLimitError(f"{message} ({name}={maximum})")
    error.limit = name
    error.actual = actual
    error.maximum = maximum
    return error


def _conflict(message: str) -> OrchidPipelineGraphConflictError:
    return OrchidPipelineGraphConflictError(message)


# ── Result and stats ─────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class DataGraphCommit:
    """What one successful commit changed.

    ``object_ids`` and ``relation_ids`` are the unique live objects/relations
    the batch created or updated (in first-seen order). Deleted ids are reported
    separately. ``aliases`` is the alias map registered by this commit, already
    resolved to object ids.
    """

    job_id: str
    version: int
    object_ids: tuple[str, ...] = ()
    relation_ids: tuple[str, ...] = ()
    deleted_object_ids: tuple[str, ...] = ()
    deleted_relation_ids: tuple[str, ...] = ()
    aliases: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "aliases", MappingProxyType(dict(self.aliases)))


@dataclass(frozen=True, slots=True)
class DataGraphMetrics:
    """Operational counters for one graph; informational, never authoritative.

    The graph returns a frozen snapshot, so mutating it cannot influence
    versioning or any other graph behaviour.
    """

    commits: int = 0
    mutations: int = 0
    objects_created: int = 0
    objects_updated: int = 0
    objects_deleted: int = 0
    relations_created: int = 0
    relations_updated: int = 0
    relations_deleted: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "commits": self.commits,
            "mutations": self.mutations,
            "objects_created": self.objects_created,
            "objects_updated": self.objects_updated,
            "objects_deleted": self.objects_deleted,
            "relations_created": self.relations_created,
            "relations_updated": self.relations_updated,
            "relations_deleted": self.relations_deleted,
        }


@dataclass(slots=True)
class _MetricCounters:
    """Private mutable counters; never exposed directly."""

    commits: int = 0
    mutations: int = 0
    objects_created: int = 0
    objects_updated: int = 0
    objects_deleted: int = 0
    relations_created: int = 0
    relations_updated: int = 0
    relations_deleted: int = 0


@dataclass(slots=True)
class _CommitPlan:
    """Mutable working state for one in-flight commit."""

    objects: dict[str, OrchidGraphObject]
    relations: dict[str, OrchidGraphRelation]
    object_bytes: dict[str, int]
    relation_bytes: dict[str, int]
    total_bytes: int
    planned_deletes: set[str]
    incidence: dict[str, set[str]] | None = None
    last_timestamp: datetime | None = None
    touched_objects: dict[str, None] = field(default_factory=dict)
    touched_relations: dict[str, None] = field(default_factory=dict)
    deleted_objects: dict[str, None] = field(default_factory=dict)
    deleted_relations: dict[str, None] = field(default_factory=dict)
    objects_created: int = 0
    objects_updated: int = 0
    objects_deleted: int = 0
    relations_created: int = 0
    relations_updated: int = 0
    relations_deleted: int = 0


# ── The graph ────────────────────────────────────────────────


# ORCHID-MEM: 01a1ea82-7066-7188-8209-ebc0fb486bfb — Run-scoped information graph; atomic mutation commits, revisions, provenance and hard caps.
class DataGraph:
    """The single writer of a run's typed objects and relations.

    ``validator`` is optional so the graph can be unit-tested without the shape
    compiler; when present it runs on every object write, on the final data
    (after a merge), before that write is placed in the working state.
    ``shape_versions`` maps a shape name to the version stamped on new objects;
    updates keep the version the object was created with.

    A graph starts empty with ``version == 0``. Seeding the reserved input (and
    optional seeds) initialises it to version 1; a seed batch applied after a
    job commit increments the version like a commit. Every later successful
    commit with at least one mutation increments the version once. A commit that
    only registers or changes aliases is also observable state and increments
    the version. Alias registrations (global and job-scoped) are bounded by
    ``max_aliases`` (default: ``max_objects``), and their serialized size counts
    toward ``max_graph_bytes``. Object payload sizes include ``data``,
    ``metadata`` and ``shape_version``; relation payload sizes include ``data``
    and ``metadata``.
    """

    def __init__(
        self,
        *,
        validator: ShapeValidator | None = None,
        max_objects: int = 10_000,
        max_relations: int = 50_000,
        max_object_bytes: int = 1_000_000,
        max_graph_bytes: int = 20_000_000,
        max_neighbour_objects: int = 1_000,
        max_neighbour_hops: int = 8,
        max_aliases: int | None = None,
        shape_versions: Mapping[str, str] | None = None,
        default_shape_version: str = "1",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if validator is not None and not callable(validator):
            raise TypeError("validator must be callable")
        self._validator = validator
        self._max_objects = _checked_cap(max_objects, field_name="max_objects")
        self._max_relations = _checked_cap(max_relations, field_name="max_relations")
        self._max_object_bytes = _checked_cap(max_object_bytes, field_name="max_object_bytes")
        self._max_graph_bytes = _checked_cap(max_graph_bytes, field_name="max_graph_bytes")
        self._max_aliases = (
            self._max_objects if max_aliases is None else _checked_cap(max_aliases, field_name="max_aliases")
        )
        neighbour_objects = _checked_cap(max_neighbour_objects, field_name="max_neighbour_objects")
        if neighbour_objects > _MAX_NEIGHBOUR_OBJECTS:
            raise ValueError(f"max_neighbour_objects must be at most {_MAX_NEIGHBOUR_OBJECTS}, got {neighbour_objects}")
        self._max_neighbour_objects = neighbour_objects
        neighbour_hops = _checked_cap(max_neighbour_hops, field_name="max_neighbour_hops")
        if neighbour_hops > _MAX_NEIGHBOUR_HOPS:
            raise ValueError(f"max_neighbour_hops must be at most {_MAX_NEIGHBOUR_HOPS}, got {neighbour_hops}")
        self._max_neighbour_hops = neighbour_hops
        self._default_shape_version = _checked_text(
            default_shape_version,
            field_name="default_shape_version",
        )
        self._shape_versions = {
            name: _checked_text(version, field_name=f"shape version for {name!r}")
            for name, version in (shape_versions or {}).items()
        }
        self._clock = clock or (lambda: datetime.now(UTC))

        self._objects: dict[str, OrchidGraphObject] = {}
        self._relations: dict[str, OrchidGraphRelation] = {}
        self._object_bytes: dict[str, int] = {}
        self._relation_bytes: dict[str, int] = {}
        self._bytes = 0
        self._version = 0
        self._commit_count = 0
        self._last_timestamp: datetime | None = None
        self._metrics = _MetricCounters()
        self._aliases: dict[str, str] = {}
        self._alias_sizes: dict[str, int] = {}
        self._job_aliases: dict[tuple[str, str], str] = {}
        self._job_alias_sizes: dict[tuple[str, str], int] = {}
        self._job_alias_bytes = 0
        self._alias_bytes = 0

    def __repr__(self) -> str:
        return f"DataGraph(version={self._version}, objects={len(self._objects)}, relations={len(self._relations)})"

    # ── Read-only state ──────────────────────────────────────

    @property
    def version(self) -> int:
        """Committed graph version; incremented once per successful commit."""
        return self._version

    @property
    def objects(self) -> Mapping[str, OrchidGraphObject]:
        """Session-facing map of live objects; steps must read through :meth:`view`."""
        return MappingProxyType(self._objects)

    @property
    def relations(self) -> Mapping[str, OrchidGraphRelation]:
        """Session-facing map of live relations; steps must read through :meth:`view`."""
        return MappingProxyType(self._relations)

    @property
    def metrics(self) -> DataGraphMetrics:
        """A frozen snapshot of this graph's informational counters."""
        counters = self._metrics
        return DataGraphMetrics(
            commits=counters.commits,
            mutations=counters.mutations,
            objects_created=counters.objects_created,
            objects_updated=counters.objects_updated,
            objects_deleted=counters.objects_deleted,
            relations_created=counters.relations_created,
            relations_updated=counters.relations_updated,
            relations_deleted=counters.relations_deleted,
        )

    @property
    def bytes(self) -> int:
        """Total serialized bytes of object and relation ``data``/``metadata`` payloads."""
        return self._bytes

    @property
    def alias_bytes(self) -> int:
        """Serialized bytes of the global and job-scoped alias maps."""
        return self._alias_bytes

    @property
    def job_alias_bytes(self) -> int:
        """Serialized bytes of the job-scoped alias history."""
        return self._job_alias_bytes

    @property
    def aliases(self) -> Mapping[str, str]:
        """Latest alias-to-object-id map across every commit."""
        return MappingProxyType(dict(self._aliases))

    def aliases_for(self, job_id: str) -> Mapping[str, str]:
        """Return the aliases registered by one job (later commits win)."""
        job = _checked_text(job_id, field_name="job_id")
        return MappingProxyType(
            {alias: object_id for (owner, alias), object_id in self._job_aliases.items() if owner == job}
        )

    def alias(self, name: str) -> str | None:
        """Resolve one alias to an object id, or ``None`` when it is unknown."""
        return self._aliases.get(name)

    def stats(self) -> dict[str, Any]:
        """Counts by type plus the current size, suitable for ``graph_stats``."""
        by_type: dict[str, int] = {}
        for obj in self._objects.values():
            by_type[obj.type] = by_type.get(obj.type, 0) + 1
        return {
            "version": self._version,
            "objects": len(self._objects),
            "relations": len(self._relations),
            "bytes": self._bytes,
            "aliases": len(self._aliases),
            "alias_bytes": self._alias_bytes,
            "job_alias_bytes": self._job_alias_bytes,
            "by_type": dict(sorted(by_type.items())),
        }

    def snapshot(self) -> dict[str, Any]:
        """Serialize the graph in the documented ``{version, objects, relations}`` form."""
        return {
            "version": self._version,
            "objects": {object_id: obj.to_dict() for object_id, obj in self._objects.items()},
            "relations": {relation_id: rel.to_dict() for relation_id, rel in self._relations.items()},
        }

    def to_dict(self) -> dict[str, Any]:
        """Alias for :meth:`snapshot`."""
        return self.snapshot()

    def view(self) -> DataGraphView:
        """Return a read-only view for a step.

        The view is live: it reflects the latest *committed* state. The session
        serialises commits, so a step never observes a half-applied batch.
        """
        return DataGraphView(self)

    # ── Run initialisation ───────────────────────────────────

    def set_input(self, data: Mapping[str, Any], *, shape_version: str | None = None) -> OrchidGraphObject:
        """Seed the reserved, read-only ``__input__`` object.

        Called once by the run session before any job executes. The input shape
        is validated by the runner (the reserved type is not a declared shape),
        so this method only enforces JSON safety and the size caps. Seeding
        initialises an empty graph to version 1.
        """
        if PIPELINE_INPUT_OBJECT_ID in self._objects:
            raise _conflict(f"reserved object {PIPELINE_INPUT_OBJECT_ID!r} is already seeded")
        if not isinstance(data, Mapping):
            raise TypeError("pipeline input must be a mapping")
        version = (
            self._default_shape_version
            if shape_version is None
            else _checked_text(shape_version, field_name="input shape_version")
        )
        payload = _json_safe_copy(data, path="$.input")
        size = _object_payload_bytes(payload, {}, version)
        if size > self._max_object_bytes:
            raise _limit_error(
                "max_object_bytes",
                size,
                self._max_object_bytes,
                f"input object data is {size} bytes, above max_object_bytes",
            )
        if len(self._objects) + 1 > self._max_objects:
            raise _limit_error(
                "max_objects",
                len(self._objects) + 1,
                self._max_objects,
                "graph object cap exceeded",
            )
        if self._bytes + self._alias_bytes + size > self._max_graph_bytes:
            raise _limit_error(
                "max_graph_bytes",
                self._bytes + self._alias_bytes + size,
                self._max_graph_bytes,
                "graph data is above max_graph_bytes",
            )
        stamp, moment = self._next_timestamp(self._last_timestamp)
        obj = OrchidGraphObject(
            id=PIPELINE_INPUT_OBJECT_ID,
            type=PIPELINE_INPUT_OBJECT_TYPE,
            data=payload,
            revision=1,
            shape_version=version,
            created_at=stamp,
            updated_at=stamp,
            created_by=_SEED_ACTOR,
            updated_by=_SEED_ACTOR,
        )
        self._objects[obj.id] = obj
        self._object_bytes[obj.id] = size
        self._bytes += size
        self._last_timestamp = moment
        self._mark_seeded()
        return obj

    def seed(self, objects: Iterable[OrchidGraphObject]) -> tuple[str, ...]:
        """Insert pre-built objects supplied at run start.

        Seeds are validated by the caller (they usually come from a shape
        dump); this method only enforces identity, JSON safety and the caps.
        The batch is all-or-nothing, like a commit. A seed batch during run
        initialisation keeps version 1; a batch applied after a job commit
        increments the version.
        """
        prepared: list[tuple[OrchidGraphObject, int]] = []
        seen: set[str] = set()
        total_bytes = self._bytes + self._alias_bytes
        for obj in objects:
            if not isinstance(obj, OrchidGraphObject):
                raise TypeError("seeds must be OrchidGraphObject values")
            if obj.id in PIPELINE_RESERVED_OBJECT_IDS:
                raise OrchidPipelineWriteNotAllowedError(f"object {obj.id!r} is reserved; use set_input() to seed it")
            object_type = _checked_text(obj.type, field_name="seed object type")
            if object_type == PIPELINE_INPUT_OBJECT_TYPE:
                raise OrchidPipelineWriteNotAllowedError(
                    f"object type {PIPELINE_INPUT_OBJECT_TYPE!r} is reserved and read-only"
                )
            if obj.id in self._objects or obj.id in seen:
                raise _conflict(f"seed object {obj.id!r} already exists")
            shape_version = _checked_text(obj.shape_version, field_name="seed shape_version")
            revision = _checked_revision(obj.revision, field_name="seed revision")
            payload = _json_safe_copy(obj.data, path=f"$.objects[{obj.id!r}].data")
            metadata = _json_safe_copy(obj.metadata, path=f"$.objects[{obj.id!r}].metadata")
            created_at = _checked_timestamp(obj.created_at, field_name="seed created_at")
            updated_at = _checked_timestamp(obj.updated_at, field_name="seed updated_at")
            created_by = _checked_actor(obj.created_by, field_name="seed created_by")
            updated_by = _checked_actor(obj.updated_by, field_name="seed updated_by")
            size = _object_payload_bytes(payload, metadata, shape_version)
            if size > self._max_object_bytes:
                raise _limit_error(
                    "max_object_bytes",
                    size,
                    self._max_object_bytes,
                    f"seed object {obj.id!r} data is {size} bytes, above max_object_bytes",
                )
            if len(self._objects) + len(prepared) + 1 > self._max_objects:
                raise _limit_error(
                    "max_objects",
                    len(self._objects) + len(prepared) + 1,
                    self._max_objects,
                    "graph object cap exceeded",
                )
            if total_bytes + size > self._max_graph_bytes:
                raise _limit_error(
                    "max_graph_bytes",
                    total_bytes + size,
                    self._max_graph_bytes,
                    "graph data is above max_graph_bytes",
                )
            prepared.append(
                (
                    OrchidGraphObject(
                        id=obj.id,
                        type=object_type,
                        data=payload,
                        revision=revision,
                        shape_version=shape_version,
                        metadata=metadata,
                        created_at=created_at,
                        updated_at=updated_at,
                        created_by=created_by,
                        updated_by=updated_by,
                    ),
                    size,
                )
            )
            seen.add(obj.id)
            total_bytes += size

        for seeded_object, size in prepared:
            self._objects[seeded_object.id] = seeded_object
            self._object_bytes[seeded_object.id] = size
            self._bytes += size
        if prepared:
            self._mark_seeded()
        return tuple(seeded_object.id for seeded_object, _ in prepared)

    # ── Commit ───────────────────────────────────────────────

    def commit(
        self,
        job_id: str,
        mutations: Sequence[OrchidGraphMutation] = (),
        *,
        aliases: Mapping[str, str] | None = None,
    ) -> DataGraphCommit:
        """Apply one job's mutations atomically and return what was committed.

        An empty batch with no aliases is a no-op: the version does not change
        and nothing is recorded. Alias registration is observable state, so an
        alias-only commit that adds or changes an alias increments the version
        too.
        """
        job = _checked_text(job_id, field_name="commit job_id")
        mutation_list = tuple(mutations)
        for index, mutation in enumerate(mutation_list):
            if not isinstance(mutation, OrchidGraphMutation):
                raise TypeError(f"mutation {index} must be an OrchidGraphMutation")
        alias_map = self._normalise_aliases(aliases)

        if not mutation_list and not alias_map:
            return DataGraphCommit(job_id=job, version=self._version)

        planned_deletes = {mutation.object_id for mutation in mutation_list if mutation.op == "delete"}
        plan = _CommitPlan(
            objects=dict(self._objects),
            relations=dict(self._relations),
            object_bytes=dict(self._object_bytes),
            relation_bytes=dict(self._relation_bytes),
            total_bytes=self._bytes,
            planned_deletes=planned_deletes,
        )
        for mutation in mutation_list:
            if mutation.op in _LINK_OPS:
                self._apply_relation_mutation(plan, mutation, job)
            else:
                self._apply_object_mutation(plan, mutation, job)

        for alias, object_id in alias_map.items():
            if object_id not in plan.objects:
                raise _conflict(f"alias {alias!r} targets missing object {object_id!r}")

        alias_count, job_alias_count, alias_bytes, alias_changed = self._project_alias_usage(plan, job, alias_map)
        if alias_count > self._max_aliases:
            raise _limit_error(
                "max_aliases",
                alias_count,
                self._max_aliases,
                "graph alias cap exceeded",
            )
        if job_alias_count > self._max_aliases:
            raise _limit_error(
                "max_aliases",
                job_alias_count,
                self._max_aliases,
                "job alias cap exceeded",
            )
        if plan.total_bytes + alias_bytes > self._max_graph_bytes:
            raise _limit_error(
                "max_graph_bytes",
                plan.total_bytes + alias_bytes,
                self._max_graph_bytes,
                "graph data including aliases is above max_graph_bytes",
            )

        state_changed = bool(mutation_list) or alias_changed
        self._objects = plan.objects
        self._relations = plan.relations
        self._object_bytes = plan.object_bytes
        self._relation_bytes = plan.relation_bytes
        self._bytes = plan.total_bytes
        if plan.last_timestamp is not None:
            self._last_timestamp = plan.last_timestamp
        if state_changed:
            self._version += 1
            self._commit_count += 1

        self._record_metrics(plan, len(mutation_list))
        self._drop_dangling_aliases()
        self._register_aliases(job, alias_map)

        return DataGraphCommit(
            job_id=job,
            version=self._version,
            object_ids=tuple(plan.touched_objects),
            relation_ids=tuple(plan.touched_relations),
            deleted_object_ids=tuple(plan.deleted_objects),
            deleted_relation_ids=tuple(plan.deleted_relations),
            aliases=alias_map,
        )

    # ── Mutation dispatch ────────────────────────────────────

    def _apply_object_mutation(self, plan: _CommitPlan, mutation: OrchidGraphMutation, job_id: str) -> None:
        object_id = _checked_object_id(mutation.object_id, field_name="mutation object_id")
        if object_id in PIPELINE_RESERVED_OBJECT_IDS:
            raise OrchidPipelineWriteNotAllowedError(f"object {object_id!r} is reserved and read-only")
        if mutation.object_type == PIPELINE_INPUT_OBJECT_TYPE:
            raise OrchidPipelineWriteNotAllowedError(
                f"object type {PIPELINE_INPUT_OBJECT_TYPE!r} is reserved and read-only"
            )

        existing = plan.objects.get(object_id)
        if mutation.op == "create":
            if existing is not None:
                raise _conflict(f"create failed: object {object_id!r} already exists at revision {existing.revision}")
            self._create_object(plan, mutation, object_id, job_id)
        elif mutation.op == "update":
            if existing is None:
                raise _conflict(f"update failed: object {object_id!r} does not exist")
            self._update_object(plan, mutation, existing, job_id)
        elif mutation.op == "upsert":
            if existing is None:
                self._create_object(plan, mutation, object_id, job_id)
            else:
                self._update_object(plan, mutation, existing, job_id)
        elif mutation.op == "delete":
            if existing is None:
                if mutation.expected_revision is not None:
                    raise _conflict(
                        f"delete failed: object {object_id!r} does not exist "
                        f"(expected revision {mutation.expected_revision})"
                    )
                if mutation.strict:
                    raise _conflict(f"delete failed: object {object_id!r} does not exist")
                return
            self._check_expected_revision(existing, mutation)
            self._delete_object(plan, existing.id)
        else:  # pragma: no cover - OrchidGraphMutation restricts the op set
            raise ValueError(f"unsupported object mutation op {mutation.op!r}")

    def _create_object(
        self,
        plan: _CommitPlan,
        mutation: OrchidGraphMutation,
        object_id: str,
        job_id: str,
    ) -> None:
        if not mutation.object_type:
            raise _conflict(f"create failed: object {object_id!r} requires an object_type")
        if mutation.expected_revision is not None:
            raise _conflict(f"create failed: object {object_id!r} must not set expected_revision")
        object_type = _checked_text(mutation.object_type, field_name="object type")
        shape_version = self._shape_versions.get(object_type, self._default_shape_version)
        data = _json_safe_copy(mutation.data, path=f"$.objects[{object_id!r}].data")
        data = self._validate_data(object_type, data, object_id, shape_version)
        stamp = self._timestamp(plan)
        obj = OrchidGraphObject(
            id=object_id,
            type=object_type,
            data=data,
            revision=1,
            shape_version=shape_version,
            created_at=stamp,
            updated_at=stamp,
            created_by=job_id,
            updated_by=job_id,
        )
        self._store_object(plan, obj)
        plan.objects_created += 1
        plan.touched_objects.setdefault(object_id, None)

    def _update_object(
        self,
        plan: _CommitPlan,
        mutation: OrchidGraphMutation,
        existing: OrchidGraphObject,
        job_id: str,
    ) -> None:
        self._check_expected_revision(existing, mutation)
        object_type = _checked_text(mutation.object_type, field_name="object type") if mutation.object_type else ""
        if object_type and object_type != existing.type:
            raise _conflict(
                f"update failed: object {existing.id!r} has type {existing.type!r}, cannot change it to {object_type!r}"
            )
        if mutation.mode == "replace":
            data = _json_safe_copy(mutation.data, path=f"$.objects[{existing.id!r}].data")
        else:
            patch = _json_safe_copy(mutation.data, path=f"$.objects[{existing.id!r}].data")
            data = _merge_data(existing.data, patch)
        data = self._validate_data(existing.type, data, existing.id, existing.shape_version)
        next_revision = existing.revision + 1
        if next_revision > _MAX_REVISION:
            raise _limit_error(
                "max_revision",
                next_revision,
                _MAX_REVISION,
                f"object {existing.id!r} revision cap exceeded",
            )
        stamp = self._timestamp(plan)
        obj = OrchidGraphObject(
            id=existing.id,
            type=existing.type,
            data=data,
            revision=next_revision,
            shape_version=existing.shape_version,
            metadata=existing.metadata,
            created_at=existing.created_at,
            updated_at=stamp,
            created_by=existing.created_by,
            updated_by=job_id,
        )
        self._store_object(plan, obj)
        plan.objects_updated += 1
        plan.touched_objects.setdefault(existing.id, None)

    def _delete_object(self, plan: _CommitPlan, object_id: str) -> None:
        self._ensure_incidence(plan)
        incident = list(plan.incidence.get(object_id, ())) if plan.incidence is not None else []
        for relation_id in incident:
            relation = plan.relations.get(relation_id)
            if relation is None:  # pragma: no cover - incidence and relations are kept in sync
                continue
            other_id = relation.target_id if relation.source_id == object_id else relation.source_id
            # When both endpoints are deleted in this batch, the second delete
            # removes the edge. Skip the first pass to avoid touching it twice;
            # self-loops and already-deleted peers are removed here.
            if (
                other_id != object_id
                and other_id in plan.objects
                and other_id in plan.planned_deletes
                and other_id not in plan.deleted_objects
            ):
                continue
            self._drop_relation(plan, relation_id)
            plan.relations_deleted += 1
            plan.deleted_relations.setdefault(relation_id, None)
            plan.touched_relations.pop(relation_id, None)
        self._drop_object(plan, object_id)
        plan.objects_deleted += 1
        plan.deleted_objects.setdefault(object_id, None)
        plan.touched_objects.pop(object_id, None)

    def _apply_relation_mutation(self, plan: _CommitPlan, mutation: OrchidGraphMutation, job_id: str) -> None:
        relation = mutation.relation
        if relation is None:  # pragma: no cover - OrchidGraphMutation guarantees it
            raise ValueError(f"mutation op {mutation.op!r} requires a relation")
        relation_type = _checked_text(relation.type, field_name="relation type")
        source_id = _checked_object_id(relation.source_id, field_name="relation source_id")
        target_id = _checked_object_id(relation.target_id, field_name="relation target_id")
        relation_id = _relation_id(relation_type, source_id, target_id)
        if relation.id != relation_id:
            raise _conflict(
                f"relation id {relation.id!r} is not deterministic; expected {relation_id!r} "
                "from type:source_id:target_id"
            )

        triple = (relation_type, source_id, target_id)
        existing = plan.relations.get(relation_id)
        if existing is not None and (existing.type, existing.source_id, existing.target_id) != triple:
            if mutation.op == "unlink" and not mutation.strict:
                return
            raise _conflict(
                f"relation id collision at {relation_id!r}: stored "
                f"{existing.type!r}:{existing.source_id!r}:{existing.target_id!r}, requested "
                f"{relation_type!r}:{source_id!r}:{target_id!r}"
            )

        if mutation.op == "link":
            for endpoint in (source_id, target_id):
                if endpoint not in plan.objects:
                    raise _conflict(
                        f"link failed: relation {relation_id!r} endpoint object {endpoint!r} does not exist"
                    )
            data = _json_safe_copy(relation.data, path=f"$.relations[{relation_id!r}].data")
            metadata = _json_safe_copy(relation.metadata, path=f"$.relations[{relation_id!r}].metadata")
            size = _payload_bytes(data, metadata)
            if size > self._max_object_bytes:
                raise _limit_error(
                    "max_object_bytes",
                    size,
                    self._max_object_bytes,
                    f"relation {relation_id!r} data is {size} bytes, above max_object_bytes",
                )
            if existing is None:
                stamp = self._timestamp(plan)
                stored = OrchidGraphRelation(
                    id=relation_id,
                    type=relation_type,
                    source_id=source_id,
                    target_id=target_id,
                    data=data,
                    metadata=metadata,
                    created_at=stamp,
                    created_by=job_id,
                )
                plan.relations_created += 1
            else:
                stored = OrchidGraphRelation(
                    id=relation_id,
                    type=relation_type,
                    source_id=source_id,
                    target_id=target_id,
                    data=data,
                    metadata=metadata,
                    created_at=existing.created_at,
                    created_by=existing.created_by,
                )
                plan.relations_updated += 1
            self._store_relation(plan, stored, size)
            plan.touched_relations.setdefault(relation_id, None)
        elif mutation.op == "unlink":
            if existing is not None:
                self._drop_relation(plan, relation_id)
                plan.relations_deleted += 1
                plan.deleted_relations.setdefault(relation_id, None)
                plan.touched_relations.pop(relation_id, None)
            elif mutation.strict:
                raise _conflict(f"unlink failed: relation {relation_id!r} does not exist")
        else:  # pragma: no cover - OrchidGraphMutation restricts the op set
            raise ValueError(f"unsupported relation mutation op {mutation.op!r}")

    # ── Working-state helpers ────────────────────────────────

    def _check_expected_revision(self, existing: OrchidGraphObject, mutation: OrchidGraphMutation) -> None:
        if mutation.expected_revision is None:
            return
        if mutation.expected_revision != existing.revision:
            raise _conflict(
                f"object {existing.id!r} revision conflict: expected {mutation.expected_revision}, "
                f"found {existing.revision}"
            )

    def _validate_data(
        self,
        shape: str,
        data: Mapping[str, Any],
        object_id: str,
        shape_version: str,
    ) -> dict[str, Any]:
        if self._validator is None:
            return dict(data)
        try:
            validated = self._validator(shape, data, object_id=object_id, shape_version=shape_version)
        except OrchidPipelineError:
            raise
        except Exception as exc:
            raise OrchidPipelineShapeError(f"shape {shape!r} object {object_id!r} validation failed: {exc}") from exc
        if validated is None:
            return dict(data)
        if not isinstance(validated, Mapping):
            raise OrchidPipelineShapeError(
                f"shape {shape!r} object {object_id!r}: validator returned "
                f"{type(validated).__name__}, expected a mapping or None"
            )
        return _json_safe_copy(validated, path=f"$.objects[{object_id!r}].data")

    def _store_object(self, plan: _CommitPlan, obj: OrchidGraphObject) -> None:
        size = _object_payload_bytes(obj.data, obj.metadata, obj.shape_version)
        if size > self._max_object_bytes:
            raise _limit_error(
                "max_object_bytes",
                size,
                self._max_object_bytes,
                f"object {obj.id!r} data is {size} bytes, above max_object_bytes",
            )
        previous = plan.object_bytes.get(obj.id)
        if previous is not None:
            plan.total_bytes -= previous
        plan.objects[obj.id] = obj
        plan.object_bytes[obj.id] = size
        plan.total_bytes += size
        if len(plan.objects) > self._max_objects:
            raise _limit_error(
                "max_objects",
                len(plan.objects),
                self._max_objects,
                "graph object cap exceeded",
            )
        if plan.total_bytes > self._max_graph_bytes:
            raise _limit_error(
                "max_graph_bytes",
                plan.total_bytes,
                self._max_graph_bytes,
                "graph data is above max_graph_bytes",
            )

    def _drop_object(self, plan: _CommitPlan, object_id: str) -> None:
        plan.objects.pop(object_id, None)
        size = plan.object_bytes.pop(object_id, 0)
        plan.total_bytes -= size

    def _store_relation(self, plan: _CommitPlan, relation: OrchidGraphRelation, size: int) -> None:
        previous = plan.relation_bytes.get(relation.id)
        if previous is not None:
            plan.total_bytes -= previous
        plan.relations[relation.id] = relation
        plan.relation_bytes[relation.id] = size
        plan.total_bytes += size
        if previous is None and plan.incidence is not None:
            plan.incidence.setdefault(relation.source_id, set()).add(relation.id)
            plan.incidence.setdefault(relation.target_id, set()).add(relation.id)
        if len(plan.relations) > self._max_relations:
            raise _limit_error(
                "max_relations",
                len(plan.relations),
                self._max_relations,
                "graph relation cap exceeded",
            )
        if plan.total_bytes > self._max_graph_bytes:
            raise _limit_error(
                "max_graph_bytes",
                plan.total_bytes,
                self._max_graph_bytes,
                "graph data is above max_graph_bytes",
            )

    def _drop_relation(self, plan: _CommitPlan, relation_id: str) -> None:
        relation = plan.relations.pop(relation_id, None)
        size = plan.relation_bytes.pop(relation_id, 0)
        plan.total_bytes -= size
        if relation is not None and plan.incidence is not None:
            for endpoint in {relation.source_id, relation.target_id}:
                relation_ids = plan.incidence.get(endpoint)
                if relation_ids is None:
                    continue
                relation_ids.discard(relation_id)
                if not relation_ids:
                    plan.incidence.pop(endpoint, None)

    def _ensure_incidence(self, plan: _CommitPlan) -> None:
        """Build the endpoint→relation index once per batch, on the first delete."""
        if plan.incidence is not None:
            return
        incidence: dict[str, set[str]] = {}
        for relation_id, relation in plan.relations.items():
            incidence.setdefault(relation.source_id, set()).add(relation_id)
            incidence.setdefault(relation.target_id, set()).add(relation_id)
        plan.incidence = incidence

    def _record_metrics(self, plan: _CommitPlan, mutation_count: int) -> None:
        self._metrics.commits += 1
        self._metrics.mutations += mutation_count
        self._metrics.objects_created += plan.objects_created
        self._metrics.objects_updated += plan.objects_updated
        self._metrics.objects_deleted += plan.objects_deleted
        self._metrics.relations_created += plan.relations_created
        self._metrics.relations_updated += plan.relations_updated
        self._metrics.relations_deleted += plan.relations_deleted

    def _normalise_aliases(self, aliases: Mapping[str, str] | None) -> dict[str, str]:
        if aliases is None:
            return {}
        if not isinstance(aliases, Mapping):
            raise TypeError("commit aliases must be a mapping")
        normalised: dict[str, str] = {}
        for alias, object_id in aliases.items():
            name = _checked_text(alias, field_name="alias name", max_chars=_MAX_ALIAS_CHARS)
            normalised[name] = _checked_object_id(object_id, field_name=f"alias {name!r} target")
        return normalised

    def _project_alias_usage(
        self,
        plan: _CommitPlan,
        job_id: str,
        alias_map: Mapping[str, str],
    ) -> tuple[int, int, int, bool]:
        """Bounded projection of alias count, job-alias count, bytes and change."""
        surviving = {alias: target for alias, target in self._aliases.items() if target in plan.objects}
        sizes = {
            alias: self._alias_sizes.get(alias, _alias_entry_bytes(alias, target))
            for alias, target in surviving.items()
        }
        projected = dict(surviving)
        global_changed = False
        for alias, target in alias_map.items():
            if projected.get(alias) != target:
                global_changed = True
            projected[alias] = target
            sizes[alias] = _alias_entry_bytes(alias, target)

        surviving_job = {key: target for key, target in self._job_aliases.items() if target in plan.objects}
        job_sizes = {
            key: self._job_alias_sizes.get(key, _job_alias_entry_bytes(key[0], key[1], target))
            for key, target in surviving_job.items()
        }
        projected_job = dict(surviving_job)
        job_changed = False
        for alias, target in alias_map.items():
            key = (job_id, alias)
            if projected_job.get(key) != target:
                job_changed = True
            projected_job[key] = target
            job_sizes[key] = _job_alias_entry_bytes(job_id, alias, target)

        alias_bytes = sum(sizes.values()) + sum(job_sizes.values())
        return len(projected), len(projected_job), alias_bytes, global_changed or job_changed

    def _drop_dangling_aliases(self) -> None:
        """Drop aliases whose target no longer exists; bounded by the alias cap."""
        for alias, object_id in list(self._aliases.items()):
            if object_id not in self._objects:
                del self._aliases[alias]
                size = self._alias_sizes.pop(alias, 0)
                self._alias_bytes -= size
        for key, object_id in list(self._job_aliases.items()):
            if object_id not in self._objects:
                del self._job_aliases[key]
                size = self._job_alias_sizes.pop(key, 0)
                self._job_alias_bytes -= size
                self._alias_bytes -= size

    def _register_aliases(self, job_id: str, alias_map: Mapping[str, str]) -> None:
        for alias, object_id in alias_map.items():
            size = _alias_entry_bytes(alias, object_id)
            previous = self._alias_sizes.get(alias)
            self._alias_bytes += size if previous is None else size - previous
            self._alias_sizes[alias] = size
            self._aliases[alias] = object_id

            key = (job_id, alias)
            job_size = _job_alias_entry_bytes(job_id, alias, object_id)
            previous_job = self._job_alias_sizes.get(key)
            self._job_alias_bytes += job_size if previous_job is None else job_size - previous_job
            self._alias_bytes += job_size if previous_job is None else job_size - previous_job
            self._job_alias_sizes[key] = job_size
            self._job_aliases[key] = object_id

    def _mark_seeded(self) -> None:
        """Version a seed batch: run initialisation is 1, a later seed is a change."""
        if self._commit_count == 0:
            self._version = max(self._version, 1)
        else:
            self._version += 1

    def _next_timestamp(self, baseline: datetime | None) -> tuple[str, datetime]:
        now = self._clock()
        if not isinstance(now, datetime):
            raise TypeError("graph clock must return a datetime")
        if baseline is not None and now <= baseline:
            now = baseline + timedelta(microseconds=1)
        return now.isoformat(), now

    def _timestamp(self, plan: _CommitPlan) -> str:
        """Stage a timestamp in the plan; it is published only if the commit succeeds."""
        baseline = plan.last_timestamp if plan.last_timestamp is not None else self._last_timestamp
        stamp, moment = self._next_timestamp(baseline)
        plan.last_timestamp = moment
        return stamp


# ── Read-only view ───────────────────────────────────────────


# ORCHID-MEM: 01a1ea82-7095-7e6d-9bc2-5a17102f1f91 — Read-only graph view for steps; returns defensive copies with bounded neighbour traversal.
class DataGraphView(OrchidDataGraphView):
    """Live, read-only projection of a :class:`DataGraph`.

    A job receives this view and cannot mutate through it: it exposes only
    ``get``, ``find``, ``relations`` and a bounded ``neighbours`` traversal,
    each returning defensive copies. The view reflects the latest committed
    state; the session's commit lock means a reader never sees a partially
    applied batch.

    The read-only guarantee is scoped to this view. ``_graph`` is a private
    reference used by the view implementation, and ``DataGraph.objects`` /
    ``DataGraph.relations`` are session-facing maps of live value objects;
    neither is part of the step contract.
    """

    def __init__(self, graph: DataGraph) -> None:
        if not isinstance(graph, DataGraph):
            raise TypeError("DataGraphView requires a DataGraph")
        self._graph = graph

    def __repr__(self) -> str:
        return f"DataGraphView(version={self._graph.version}, objects={len(self._graph.objects)})"

    @property
    def version(self) -> int:
        """Version of the graph this view reads from."""
        return self._graph.version

    def get(self, object_id: str) -> OrchidGraphObject | None:
        """Return a defensive copy of one committed object, or ``None``."""
        if not isinstance(object_id, str):
            raise TypeError("object_id must be a string")
        obj = self._graph._objects.get(object_id)
        return _copy_object(obj) if obj is not None else None

    def find(
        self,
        object_type: str | None = None,
        where: Mapping[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[OrchidGraphObject]:
        """Return objects matching ``object_type`` and top-level equality ``where``.

        Only ``data`` keys are matched; a key absent from an object does not
        match, not even against ``None``. ``limit`` bounds the result in
        insertion order.
        """
        if object_type is not None and not isinstance(object_type, str):
            raise TypeError("object_type must be a string or None")
        if where is not None and not isinstance(where, Mapping):
            raise TypeError("find 'where' must be a mapping or None")
        conditions = dict(where or {})
        for key in conditions:
            if not isinstance(key, str):
                raise TypeError("find 'where' keys must be strings")
        maximum = None if limit is None else _checked_limit(limit, field_name="find limit")
        if maximum == 0:
            return []

        missing = object()
        matches: list[OrchidGraphObject] = []
        for obj in self._graph._objects.values():
            if object_type is not None and obj.type != object_type:
                continue
            if any(
                obj.data.get(key, missing) is missing or obj.data[key] != value for key, value in conditions.items()
            ):
                continue
            matches.append(_copy_object(obj))
            if maximum is not None and len(matches) >= maximum:
                break
        return matches

    def relations(
        self,
        *,
        type: str | None = None,
        source_id: str | None = None,
        target_id: str | None = None,
    ) -> list[OrchidGraphRelation]:
        """Return defensive copies of relations matching every supplied filter."""
        return [
            _copy_relation(relation)
            for relation in self._graph._relations.values()
            if (type is None or relation.type == type)
            and (source_id is None or relation.source_id == source_id)
            and (target_id is None or relation.target_id == target_id)
        ]

    def neighbours(
        self,
        object_id: str,
        *,
        relation: str | None = None,
        direction: Literal["out", "in", "both"] = "out",
        max_hops: int = 1,
        max_objects: int | None = None,
    ) -> tuple[list[OrchidGraphObject], list[OrchidGraphRelation]]:
        """Return objects and relations reachable within ``max_hops``.

        ``out`` follows source→target, ``in`` follows target→source, ``both``
        ignores direction. The result excludes the starting object and is
        bounded by ``max_objects`` (default: the graph's configured neighbour
        cap; a caller may only tighten it) and by the graph's maximum hop count.
        The traversal stops as soon as either bound is reached.
        """
        _checked_object_id(object_id, field_name="neighbours object_id")
        if direction not in {"out", "in", "both"}:
            raise ValueError(f"neighbours direction must be one of 'out', 'in', 'both', got {direction!r}")
        hops = _checked_limit(max_hops, field_name="max_hops")
        if hops > self._graph._max_neighbour_hops:
            raise ValueError(f"max_hops {hops} exceeds the graph's bound of {self._graph._max_neighbour_hops}")
        cap = self._graph._max_neighbour_objects
        if max_objects is not None:
            cap = min(cap, _checked_limit(max_objects, field_name="max_objects"))
        if hops == 0 or cap == 0 or object_id not in self._graph._objects:
            return [], []

        adjacency: dict[str, list[tuple[str, OrchidGraphRelation]]] = {}
        for candidate in self._graph._relations.values():
            if relation is not None and candidate.type != relation:
                continue
            if direction in {"out", "both"}:
                adjacency.setdefault(candidate.source_id, []).append((candidate.target_id, candidate))
            if direction in {"in", "both"}:
                adjacency.setdefault(candidate.target_id, []).append((candidate.source_id, candidate))

        visited = {object_id}
        queue: deque[tuple[str, int]] = deque([(object_id, 0)])
        objects: list[OrchidGraphObject] = []
        relations: list[OrchidGraphRelation] = []
        seen_relations: set[str] = set()

        while queue:
            current, depth = queue.popleft()
            if depth >= hops:
                continue
            for neighbour_id, candidate in adjacency.get(current, ()):
                if candidate.id not in seen_relations:
                    if len(relations) >= cap:
                        return objects, relations
                    seen_relations.add(candidate.id)
                    relations.append(_copy_relation(candidate))
                if neighbour_id in visited:
                    continue
                neighbour = self._graph._objects.get(neighbour_id)
                if neighbour is None:  # pragma: no cover - dangling relations are prevented on commit
                    continue
                if len(objects) >= cap:
                    return objects, relations
                visited.add(neighbour_id)
                objects.append(_copy_object(neighbour))
                if depth + 1 < hops:
                    queue.append((neighbour_id, depth + 1))
        return objects, relations
