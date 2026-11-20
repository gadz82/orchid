"""Atomic mutation, revision, provenance and view semantics for ``DataGraph``."""

from __future__ import annotations

import json
import time
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from orchid_ai.core.pipeline import (
    PIPELINE_INPUT_OBJECT_ID,
    PIPELINE_INPUT_OBJECT_TYPE,
    OrchidDataGraphView,
    OrchidGraphMutation,
    OrchidGraphObject,
    OrchidGraphRelation,
    OrchidPipelineGraphConflictError,
    OrchidPipelineLimitError,
    OrchidPipelineSerializationError,
    OrchidPipelineShapeError,
    OrchidPipelineWriteNotAllowedError,
)
from orchid_ai.pipeline.graph import DataGraph, DataGraphCommit, DataGraphMetrics, DataGraphView

START = datetime(2026, 11, 17, 9, 0, tzinfo=UTC)


class _Clock:
    """Monotonic fake clock so provenance timestamps are deterministic."""

    def __init__(self, start: datetime = START) -> None:
        self._now = start

    def __call__(self) -> datetime:
        current = self._now
        self._now += timedelta(seconds=1)
        return current


def _graph(**kwargs: Any) -> DataGraph:
    kwargs.setdefault("clock", _Clock())
    return DataGraph(**kwargs)


def _create(object_id: str, object_type: str = "Account", data: dict[str, Any] | None = None, **kwargs: Any):
    return OrchidGraphMutation(
        op="create",
        object_id=object_id,
        object_type=object_type,
        data=data or {},
        **kwargs,
    )


def _update(
    object_id: str,
    data: dict[str, Any] | None = None,
    *,
    mode: str = "merge",
    object_type: str = "",
    expected_revision: int | None = None,
) -> OrchidGraphMutation:
    return OrchidGraphMutation(
        op="update",
        object_id=object_id,
        object_type=object_type,
        data=data or {},
        mode=mode,  # type: ignore[arg-type]
        expected_revision=expected_revision,
    )


def _upsert(
    object_id: str,
    data: dict[str, Any] | None = None,
    *,
    object_type: str = "",
    mode: str = "merge",
    expected_revision: int | None = None,
) -> OrchidGraphMutation:
    return OrchidGraphMutation(
        op="upsert",
        object_id=object_id,
        object_type=object_type,
        data=data or {},
        mode=mode,  # type: ignore[arg-type]
        expected_revision=expected_revision,
    )


def _delete(
    object_id: str,
    *,
    strict: bool = False,
    expected_revision: int | None = None,
) -> OrchidGraphMutation:
    return OrchidGraphMutation(
        op="delete",
        object_id=object_id,
        strict=strict,
        expected_revision=expected_revision,
    )


def _relation(
    relation_type: str,
    source_id: str,
    target_id: str,
    *,
    data: dict[str, Any] | None = None,
    relation_id: str | None = None,
    created_by: str = "",
) -> OrchidGraphRelation:
    return OrchidGraphRelation(
        id=relation_id or f"{relation_type}:{source_id}:{target_id}",
        type=relation_type,
        source_id=source_id,
        target_id=target_id,
        data=data or {},
        created_by=created_by,
    )


def _link(
    relation_type: str,
    source_id: str,
    target_id: str,
    *,
    data: dict[str, Any] | None = None,
    relation_id: str | None = None,
    strict: bool = False,
) -> OrchidGraphMutation:
    return OrchidGraphMutation(
        op="link",
        relation=_relation(relation_type, source_id, target_id, data=data, relation_id=relation_id),
        strict=strict,
    )


def _unlink(
    relation_type: str,
    source_id: str,
    target_id: str,
    *,
    strict: bool = False,
) -> OrchidGraphMutation:
    return OrchidGraphMutation(
        op="unlink",
        relation=_relation(relation_type, source_id, target_id),
        strict=strict,
    )


def _accounts(graph: DataGraph, *ids: str) -> None:
    graph.commit("load", [_create(object_id, data={"name": object_id}) for object_id in ids])


# ── create ───────────────────────────────────────────────────


def test_create_stamps_revision_provenance_and_version() -> None:
    graph = _graph()
    commit = graph.commit("load", [_create("account:1", data={"name": "Acme"})])

    assert isinstance(commit, DataGraphCommit)
    assert commit.job_id == "load"
    assert commit.version == 1
    assert commit.object_ids == ("account:1",)
    assert graph.version == 1

    obj = graph.objects["account:1"]
    assert obj.revision == 1
    assert obj.type == "Account"
    assert obj.data == {"name": "Acme"}
    assert obj.created_by == "load"
    assert obj.updated_by == "load"
    assert obj.created_at == obj.updated_at
    assert obj.created_at


def test_create_fails_when_the_object_exists_and_leaves_it_untouched() -> None:
    graph = _graph()
    graph.commit("load", [_create("account:1")])

    with pytest.raises(OrchidPipelineGraphConflictError, match="already exists"):
        graph.commit("other", [_create("account:1", data={"name": "Overwrite"})])

    assert graph.version == 1
    assert graph.objects["account:1"].revision == 1
    assert graph.objects["account:1"].data == {}


def test_create_requires_an_object_type_and_rejects_expected_revision() -> None:
    graph = _graph()

    with pytest.raises(OrchidPipelineGraphConflictError, match="requires an object_type"):
        graph.commit("job", [_upsert("account:1")])
    with pytest.raises(OrchidPipelineGraphConflictError, match="expected_revision"):
        graph.commit("job", [_create("account:1", expected_revision=1)])


def test_object_type_is_bounded_like_other_graph_text() -> None:
    graph = _graph()
    long_type = _create("account:1")
    object.__setattr__(long_type, "object_type", "x" * 257)
    with pytest.raises(ValueError, match="at most 256"):
        graph.commit("job", [long_type])

    control_type = _create("account:1")
    object.__setattr__(control_type, "object_type", "bad\ntype")
    with pytest.raises(ValueError, match="control characters"):
        graph.commit("job", [control_type])

    graph.commit("load", [_create("account:1")])
    bad_update = _update("account:1", {"v": 1})
    object.__setattr__(bad_update, "object_type", "x" * 257)
    with pytest.raises(ValueError, match="at most 256"):
        graph.commit("job", [bad_update])

    with pytest.raises(ValueError, match="at most 256"):
        graph.seed([OrchidGraphObject(id="seed:long", type="x" * 257, data={})])
    with pytest.raises(ValueError, match="control characters"):
        graph.seed([OrchidGraphObject(id="seed:bad", type="bad\ntype", data={})])

    graph.seed([OrchidGraphObject(id="seed:ok", type="t" * 256, data={})])
    assert graph.objects["seed:ok"].type == "t" * 256


# ── update ───────────────────────────────────────────────────


def test_update_merge_is_deep_replaces_lists_and_scalars_and_deletes_none_keys() -> None:
    graph = _graph()
    graph.commit(
        "load",
        [
            _create(
                "account:1",
                data={
                    "name": "Acme",
                    "plan": "pro",
                    "nested": {"keep": 1, "drop": 2, "deeper": {"a": 1}},
                    "tags": ["a", "b"],
                    "count": 1,
                },
            )
        ],
    )

    graph.commit(
        "analyze",
        [
            _update(
                "account:1",
                {
                    "name": "Beta",
                    "nested": {"drop": None, "deeper": {"b": 2}},
                    "tags": ["c"],
                    "count": 5,
                    "fresh": True,
                },
                expected_revision=1,
            )
        ],
    )

    obj = graph.objects["account:1"]
    assert obj.data == {
        "name": "Beta",
        "plan": "pro",
        "nested": {"keep": 1, "deeper": {"a": 1, "b": 2}},
        "tags": ["c"],
        "count": 5,
        "fresh": True,
    }
    assert obj.revision == 2
    assert obj.created_by == "load"
    assert obj.updated_by == "analyze"
    assert datetime.fromisoformat(obj.updated_at) > datetime.fromisoformat(obj.created_at)


def test_update_replace_swaps_data_wholesale() -> None:
    graph = _graph()
    graph.commit("load", [_create("account:1", data={"a": 1, "b": 2})])

    graph.commit("analyze", [_update("account:1", {"only": 1}, mode="replace")])

    assert graph.objects["account:1"].data == {"only": 1}
    assert graph.objects["account:1"].revision == 2


def test_update_fails_when_absent_or_when_it_changes_the_type() -> None:
    graph = _graph()
    with pytest.raises(OrchidPipelineGraphConflictError, match="does not exist"):
        graph.commit("job", [_update("missing", {"a": 1})])

    graph.commit("load", [_create("account:1")])
    with pytest.raises(OrchidPipelineGraphConflictError, match="cannot change it"):
        graph.commit("job", [_update("account:1", {"a": 1}, object_type="UsageSnapshot")])


# ── upsert ───────────────────────────────────────────────────


def test_upsert_creates_when_absent_and_merges_when_present() -> None:
    graph = _graph()

    graph.commit("load", [_upsert("account:1", {"name": "Acme", "nested": {"a": 1}}, object_type="Account")])
    assert graph.objects["account:1"].revision == 1
    assert graph.objects["account:1"].type == "Account"

    graph.commit(
        "analyze",
        [_upsert("account:1", {"plan": "pro", "nested": {"b": 2}}, expected_revision=1)],
    )
    assert graph.objects["account:1"].revision == 2
    assert graph.objects["account:1"].data == {"name": "Acme", "plan": "pro", "nested": {"a": 1, "b": 2}}


def test_upsert_absent_with_expected_revision_is_a_conflict() -> None:
    graph = _graph()
    with pytest.raises(OrchidPipelineGraphConflictError, match="expected_revision"):
        graph.commit("job", [_upsert("account:1", {"a": 1}, object_type="Account", expected_revision=2)])
    assert graph.objects == {}


def test_upsert_with_replace_swaps_existing_data_and_creates_absent_objects() -> None:
    graph = _graph()
    graph.commit("load", [_upsert("account:1", {"x": 1, "nested": {"a": 1}}, object_type="Account")])

    graph.commit("job", [_upsert("account:1", {"only": 1}, mode="replace", expected_revision=1)])

    obj = graph.objects["account:1"]
    assert obj.data == {"only": 1}
    assert obj.revision == 2

    graph.commit("job", [_upsert("account:2", {"x": 2}, object_type="Account", mode="replace")])
    assert graph.objects["account:2"].data == {"x": 2}
    assert graph.objects["account:2"].revision == 1


# ── delete and cascade ───────────────────────────────────────


def test_delete_cascades_to_incident_relations() -> None:
    graph = _graph()
    _accounts(graph, "a", "b", "c")
    graph.commit("link", [_link("has_usage", "a", "b"), _link("has_usage", "c", "a"), _link("has_usage", "b", "c")])
    assert len(graph.relations) == 3

    commit = graph.commit("cleanup", [_delete("a")])

    assert commit.deleted_object_ids == ("a",)
    assert set(commit.deleted_relation_ids) == {"has_usage:a:b", "has_usage:c:a"}
    assert "a" not in graph.objects
    assert set(graph.relations) == {"has_usage:b:c"}
    assert graph.metrics.objects_deleted == 1
    assert graph.metrics.relations_deleted == 2


def test_delete_missing_object_is_a_noop_unless_strict() -> None:
    graph = _graph()
    commit = graph.commit("cleanup", [_delete("missing")])
    assert commit.deleted_object_ids == ()
    assert graph.objects == {}

    with pytest.raises(OrchidPipelineGraphConflictError, match="does not exist"):
        graph.commit("cleanup", [_delete("missing", strict=True)])


def test_delete_missing_with_expected_revision_is_a_conflict() -> None:
    graph = _graph()
    with pytest.raises(OrchidPipelineGraphConflictError, match="does not exist.*expected revision 1"):
        graph.commit("cleanup", [_delete("missing", expected_revision=1)])
    assert graph.objects == {}
    assert graph.version == 0


def test_cascade_delete_handles_both_endpoints_and_self_loops() -> None:
    graph = _graph()
    graph.commit("load", [_create("a"), _create("b")])
    graph.commit("link", [_link("edge", "a", "b"), _link("edge", "b", "a"), _link("edge", "a", "a")])

    commit = graph.commit("cleanup", [_delete("a"), _delete("b")])

    assert graph.objects == {}
    assert graph.relations == {}
    assert set(commit.deleted_object_ids) == {"a", "b"}
    assert set(commit.deleted_relation_ids) == {"edge:a:b", "edge:b:a", "edge:a:a"}


def test_cascade_delete_is_linear_at_the_object_and_relation_caps() -> None:
    graph = _graph(max_objects=10_000, max_relations=50_000)
    graph.commit("load", [_create(f"o{index}", data={}) for index in range(10_000)])
    mutations = []
    for type_index in range(5):
        relation_type = f"edge{type_index}"
        mutations.extend(_link(relation_type, f"o{index}", f"o{(index + 1) % 10_000}") for index in range(10_000))
    graph.commit("link", mutations)

    started = time.perf_counter()
    commit = graph.commit("cleanup", [_delete(f"o{index}") for index in range(0, 10_000, 2)])
    elapsed = time.perf_counter() - started

    assert len(commit.deleted_object_ids) == 5_000
    assert len(graph.objects) == 5_000
    # Every cycle edge touches an even endpoint, so all 50k relations cascade.
    assert graph.relations == {}
    assert elapsed < 1.5, f"cascade delete took {elapsed:.3f}s"


def test_cascade_index_tracks_relations_added_after_the_first_delete() -> None:
    graph = _graph()
    graph.commit("load", [_create("a"), _create("b"), _create("c")])
    graph.commit("link", [_link("edge", "a", "b")])

    commit = graph.commit("mixed", [_delete("a"), _link("edge", "b", "c"), _delete("b")])

    assert graph.objects.keys() == {"c"}
    assert graph.relations == {}
    assert set(commit.deleted_relation_ids) == {"edge:a:b", "edge:b:c"}


# ── revisions and conflicts ─────────────────────────────────


def test_stale_expected_revision_is_a_conflict_and_the_object_survives() -> None:
    graph = _graph()
    graph.commit("load", [_create("account:1", data={"v": 1})])
    graph.commit("first", [_update("account:1", {"v": 2}, expected_revision=1)])
    assert graph.objects["account:1"].revision == 2

    with pytest.raises(OrchidPipelineGraphConflictError, match="expected 1, found 2"):
        graph.commit("second", [_update("account:1", {"v": 3}, expected_revision=1)])

    assert graph.objects["account:1"].revision == 2
    assert graph.objects["account:1"].data == {"v": 2}
    assert graph.version == 2


def test_delete_and_upsert_check_expected_revision() -> None:
    graph = _graph()
    graph.commit("load", [_create("account:1", data={"v": 1})])
    graph.commit("first", [_update("account:1", {"v": 2})])

    with pytest.raises(OrchidPipelineGraphConflictError, match="expected 1, found 2"):
        graph.commit("second", [_delete("account:1", expected_revision=1)])
    with pytest.raises(OrchidPipelineGraphConflictError, match="expected 1, found 2"):
        graph.commit("second", [_upsert("account:1", {"v": 3}, expected_revision=1)])

    assert graph.objects["account:1"].data == {"v": 2}
    assert graph.objects["account:1"].revision == 2


# ── link / unlink ────────────────────────────────────────────


def test_link_is_idempotent_and_uses_a_deterministic_id() -> None:
    graph = _graph()
    _accounts(graph, "account:1", "usage:1")

    first = graph.commit(
        "load",
        [_link("has_usage", "account:1", "usage:1", data={"period": "2026-10"})],
    )
    assert first.relation_ids == ("has_usage:account:1:usage:1",)
    relation = graph.relations["has_usage:account:1:usage:1"]
    assert relation.created_by == "load"
    assert relation.data == {"period": "2026-10"}

    graph.commit("refresh", [_link("has_usage", "account:1", "usage:1", data={"period": "2026-11"})])

    assert len(graph.relations) == 1
    updated = graph.relations["has_usage:account:1:usage:1"]
    assert updated.data == {"period": "2026-11"}
    assert updated.created_by == "load"
    assert graph.metrics.relations_created == 1
    assert graph.metrics.relations_updated == 1


def test_link_rejects_a_mismatched_relation_id_and_missing_endpoints() -> None:
    graph = _graph()
    _accounts(graph, "a")

    with pytest.raises(OrchidPipelineGraphConflictError, match="not deterministic"):
        graph.commit("job", [_link("has_usage", "a", "a", relation_id="custom-id")])
    with pytest.raises(OrchidPipelineGraphConflictError, match="does not exist"):
        graph.commit("job", [_link("has_usage", "a", "missing")])

    assert graph.relations == {}
    assert graph.version == 1


def test_link_and_unlink_accept_the_input_object_as_an_endpoint() -> None:
    graph = _graph()
    graph.set_input({"account_id": "acme"})
    _accounts(graph, "account:1")

    graph.commit("load", [_link("from_input", PIPELINE_INPUT_OBJECT_ID, "account:1")])
    assert len(graph.relations) == 1

    graph.commit("cleanup", [_unlink("from_input", PIPELINE_INPUT_OBJECT_ID, "account:1")])
    assert graph.relations == {}


def test_unlink_is_idempotent_unless_strict() -> None:
    graph = _graph()
    _accounts(graph, "a", "b")
    graph.commit("load", [_link("has_usage", "a", "b")])

    graph.commit("cleanup", [_unlink("has_usage", "a", "b")])
    assert graph.relations == {}

    graph.commit("cleanup", [_unlink("has_usage", "a", "b")])
    with pytest.raises(OrchidPipelineGraphConflictError, match="does not exist"):
        graph.commit("cleanup", [_unlink("has_usage", "a", "b", strict=True)])


def test_link_detects_a_relation_id_collision_without_overwriting() -> None:
    graph = _graph()
    graph.commit("load", [_create("a:b"), _create("c"), _create("a"), _create("b:c")])
    graph.commit("link", [_link("edge", "a:b", "c")])

    with pytest.raises(OrchidPipelineGraphConflictError, match="collision"):
        graph.commit("link", [_link("edge", "a", "b:c")])

    stored = graph.relations["edge:a:b:c"]
    assert (stored.source_id, stored.target_id) == ("a:b", "c")
    assert stored.data == {}


def test_unlink_does_not_delete_a_colliding_triple() -> None:
    graph = _graph()
    graph.commit("load", [_create("a:b"), _create("c"), _create("a"), _create("b:c")])
    graph.commit("link", [_link("edge", "a:b", "c")])

    graph.commit("cleanup", [_unlink("edge", "a", "b:c")])
    assert "edge:a:b:c" in graph.relations

    with pytest.raises(OrchidPipelineGraphConflictError, match="collision"):
        graph.commit("cleanup", [_unlink("edge", "a", "b:c", strict=True)])

    graph.commit("cleanup", [_unlink("edge", "a:b", "c")])
    assert graph.relations == {}


# ── atomicity ────────────────────────────────────────────────


def test_shape_failure_rolls_back_the_whole_commit() -> None:
    def validator(shape: str, data: dict[str, Any], *, object_id: str, shape_version: str) -> dict[str, Any]:
        if shape == "Bad":
            raise OrchidPipelineShapeError("bad shape")
        return data

    graph = _graph(validator=validator)

    with pytest.raises(OrchidPipelineShapeError, match="bad shape"):
        graph.commit(
            "job",
            [_create("account:1", data={"ok": True}), _create("bad:1", object_type="Bad")],
        )

    assert graph.objects == {}
    assert graph.relations == {}
    assert graph.version == 0
    assert graph.metrics.commits == 0
    assert graph.metrics.objects_created == 0


def test_json_failure_rolls_back_the_whole_commit() -> None:
    graph = _graph()

    with pytest.raises(OrchidPipelineSerializationError):
        graph.commit("job", [_create("account:1"), _create("bad:1", data={"handle": object()})])

    assert graph.objects == {}
    assert graph.version == 0


def test_cap_failure_rolls_back_the_whole_commit() -> None:
    graph = _graph(max_objects=1)

    with pytest.raises(OrchidPipelineLimitError, match="max_objects"):
        graph.commit("job", [_create("a"), _create("b")])

    assert graph.objects == {}
    assert graph.version == 0


def test_relation_cap_failure_rolls_back_objects_created_in_the_same_commit() -> None:
    graph = _graph(max_relations=1)

    with pytest.raises(OrchidPipelineLimitError, match="max_relations"):
        graph.commit(
            "job",
            [
                _create("a"),
                _create("b"),
                _create("c"),
                _link("has_usage", "a", "b"),
                _link("has_usage", "a", "c"),
            ],
        )

    assert graph.objects == {}
    assert graph.relations == {}


def test_create_and_update_in_one_commit_are_sequential() -> None:
    graph = _graph()
    graph.commit("job", [_create("account:1", data={"v": 1}), _update("account:1", {"v": 2})])

    assert graph.objects["account:1"].revision == 2
    assert graph.objects["account:1"].data == {"v": 2}
    assert graph.version == 1
    assert graph.metrics.objects_created == 1
    assert graph.metrics.objects_updated == 1


def test_delete_then_create_the_same_id_in_one_commit() -> None:
    graph = _graph()
    graph.commit("load", [_create("account:1", data={"v": 1})])

    commit = graph.commit("reload", [_delete("account:1"), _create("account:1", data={"v": 2})])

    assert commit.deleted_object_ids == ("account:1",)
    assert commit.object_ids == ("account:1",)
    obj = graph.objects["account:1"]
    assert obj.revision == 1
    assert obj.data == {"v": 2}
    assert obj.created_by == "reload"


# ── validation hook ─────────────────────────────────────────


def test_validator_receives_context_and_can_normalise_data() -> None:
    calls: list[tuple[str, dict[str, Any], str, str]] = []

    def validator(shape: str, data: dict[str, Any], *, object_id: str, shape_version: str) -> dict[str, Any]:
        calls.append((shape, dict(data), object_id, shape_version))
        return {**data, "normalised": True}

    graph = _graph(validator=validator, shape_versions={"Account": "2"})
    graph.commit("load", [_create("account:1", data={"name": "Acme"})])

    assert calls == [("Account", {"name": "Acme"}, "account:1", "2")]
    obj = graph.objects["account:1"]
    assert obj.data == {"name": "Acme", "normalised": True}
    assert obj.shape_version == "2"

    graph.commit("analyze", [_update("account:1", {"plan": "pro"})])
    assert calls[1][3] == "2"
    assert graph.objects["account:1"].data == {"name": "Acme", "normalised": True, "plan": "pro"}


def test_validator_error_is_wrapped_with_shape_and_object_context() -> None:
    def validator(shape: str, data: dict[str, Any], *, object_id: str, shape_version: str) -> dict[str, Any]:
        raise ValueError("boom")

    graph = _graph(validator=validator)
    with pytest.raises(OrchidPipelineShapeError, match="Account.*account:1.*boom"):
        graph.commit("job", [_create("account:1")])
    assert graph.objects == {}


def test_validator_must_return_a_mapping_when_not_none() -> None:
    def validator(shape: str, data: dict[str, Any], *, object_id: str, shape_version: str) -> Any:
        return ["not", "a", "mapping"]

    graph = _graph(validator=validator)
    with pytest.raises(OrchidPipelineShapeError, match="expected a mapping"):
        graph.commit("job", [_create("account:1")])


def test_validator_returning_none_keeps_the_candidate_data() -> None:
    calls: list[str] = []

    def validator(shape: str, data: dict[str, Any], *, object_id: str, shape_version: str) -> None:
        calls.append(object_id)

    graph = _graph(validator=validator)
    graph.commit("load", [_create("account:1", data={"x": 1})])
    graph.commit("job", [_update("account:1", {"y": 2})])

    assert graph.objects["account:1"].data == {"x": 1, "y": 2}
    assert calls == ["account:1", "account:1"]


# ── provenance, version and metrics ─────────────────────────


def test_version_increments_once_per_commit_and_not_for_empty_batches() -> None:
    graph = _graph()
    graph.commit("load", [_create("a"), _create("b")])
    assert graph.version == 1

    graph.commit("link", [_link("has_usage", "a", "b")])
    assert graph.version == 2

    noop = graph.commit("noop", [])
    assert noop.version == 2
    assert graph.version == 2
    assert graph.metrics.commits == 2

    graph.commit("noop-delete", [_delete("missing")])
    assert graph.version == 3


def test_metrics_count_operations_not_batches() -> None:
    graph = _graph()
    graph.commit("load", [_create("a"), _create("b")])
    graph.commit("link", [_link("has_usage", "a", "b"), _link("has_usage", "b", "a")])
    graph.commit("update", [_update("a", {"v": 1})])
    graph.commit("cleanup", [_delete("a")])

    assert graph.metrics.to_dict() == {
        "commits": 4,
        "mutations": 6,
        "objects_created": 2,
        "objects_updated": 1,
        "objects_deleted": 1,
        "relations_created": 2,
        "relations_updated": 0,
        "relations_deleted": 2,
    }
    assert isinstance(graph.metrics, DataGraphMetrics)


def test_metrics_are_frozen_and_cannot_influence_versioning() -> None:
    graph = _graph()
    metrics = graph.metrics
    with pytest.raises(FrozenInstanceError):
        metrics.commits = 99  # type: ignore[misc]
    object.__setattr__(metrics, "commits", 99)

    graph.seed([OrchidGraphObject(id="a", type="Account", data={})])
    assert graph.version == 1

    graph.commit("job", [_create("b")])
    assert graph.version == 2
    assert graph.metrics.commits == 1


def test_failed_commit_does_not_publish_staged_timestamps() -> None:
    class FixedClock:
        def __call__(self) -> datetime:
            return START

    def validator(shape: str, data: dict[str, Any], *, object_id: str, shape_version: str) -> dict[str, Any]:
        if shape == "Bad":
            raise OrchidPipelineShapeError("bad shape")
        return data

    graph = DataGraph(clock=FixedClock(), validator=validator)
    graph.commit("load", [_create("a")])
    created = datetime.fromisoformat(graph.objects["a"].created_at)
    published = graph._last_timestamp
    assert published is not None

    with pytest.raises(OrchidPipelineShapeError, match="bad shape"):
        graph.commit("bad", [_create("b"), _create("c", object_type="Bad")])
    assert graph._last_timestamp == published

    graph.commit("good", [_update("a", {"v": 1})])
    updated = datetime.fromisoformat(graph.objects["a"].updated_at)
    assert updated == created + timedelta(microseconds=1)


def test_coarse_clock_still_produces_strictly_increasing_timestamps() -> None:
    class FixedClock:
        def __call__(self) -> datetime:
            return START

    graph = DataGraph(clock=FixedClock())
    graph.commit("job", [_create("a"), _update("a", {"v": 1})])

    obj = graph.objects["a"]
    assert datetime.fromisoformat(obj.updated_at) == datetime.fromisoformat(obj.created_at) + timedelta(microseconds=1)


# ── aliases ──────────────────────────────────────────────────


def test_commit_registers_aliases_scoped_to_the_job_and_latest_globally() -> None:
    graph = _graph()

    commit = graph.commit("load", [_create("account:1")], aliases={"account": "account:1"})
    assert commit.aliases == {"account": "account:1"}
    assert graph.alias("account") == "account:1"
    assert graph.aliases == {"account": "account:1"}
    assert graph.aliases_for("load") == {"account": "account:1"}
    assert graph.aliases_for("other") == {}

    graph.commit("analyze", [], aliases={"account": "account:1"})
    assert graph.aliases_for("analyze") == {"account": "account:1"}
    assert graph.aliases_for("load") == {"account": "account:1"}


def test_alias_target_must_exist_and_alias_registration_is_atomic() -> None:
    graph = _graph()

    with pytest.raises(OrchidPipelineGraphConflictError, match="targets missing object"):
        graph.commit("load", [_create("account:1")], aliases={"account": "missing"})

    assert graph.objects == {}
    assert graph.aliases == {}
    assert graph.version == 0


def test_aliases_are_dropped_when_their_target_is_deleted() -> None:
    graph = _graph()
    graph.commit("load", [_create("account:1")], aliases={"account": "account:1"})

    graph.commit("cleanup", [_delete("account:1")])

    assert graph.aliases == {}
    assert graph.aliases_for("load") == {}
    assert graph.alias("account") is None
    assert graph.alias_bytes == 0


def test_invalid_alias_name_rejects_the_commit() -> None:
    graph = _graph()
    with pytest.raises(ValueError, match="control characters"):
        graph.commit("load", [], aliases={"bad\nalias": "account:1"})
    with pytest.raises(ValueError, match="non-empty"):
        graph.commit("load", [], aliases={"": "account:1"})
    with pytest.raises(ValueError, match="non-empty"):
        graph.commit("load", [], aliases={"account": ""})


def test_alias_count_and_name_length_are_capped() -> None:
    graph = _graph(max_aliases=2)
    graph.commit("load", [_create("a"), _create("b"), _create("c")])
    graph.commit("aliases", [], aliases={"a": "a", "b": "b"})
    assert graph.alias_bytes > 0

    with pytest.raises(OrchidPipelineLimitError, match="max_aliases"):
        graph.commit("aliases", [], aliases={"c": "c"})
    assert graph.aliases == {"a": "a", "b": "b"}

    graph.commit("aliases", [], aliases={"a": "b"})
    assert graph.aliases == {"a": "b", "b": "b"}

    long_names = _graph()
    long_names.commit("load", [_create("a")])
    with pytest.raises(ValueError, match="at most 128"):
        long_names.commit("job", [], aliases={"x" * 129: "a"})
    long_names.commit("job", [], aliases={"x" * 128: "a"})
    assert long_names.alias("x" * 128) == "a"


def test_alias_only_commits_bump_version_only_when_state_changes() -> None:
    graph = _graph(max_aliases=5)
    graph.commit("load", [_create("a"), _create("b")])
    assert graph.version == 1

    graph.commit("aliases", [], aliases={"account": "a"})
    assert graph.version == 2

    graph.commit("aliases", [], aliases={"account": "a"})
    assert graph.version == 2

    graph.commit("aliases", [], aliases={"account": "b"})
    assert graph.version == 3

    graph.commit("other", [], aliases={"account": "b"})
    assert graph.version == 4


def test_alias_bytes_count_toward_the_graph_byte_cap() -> None:
    graph = _graph(max_graph_bytes=65)
    graph.commit("load", [_create("a")])
    graph.commit("aliases", [], aliases={"a": "a"})
    graph.commit("aliases", [], aliases={"bb": "a"})
    assert graph.alias_bytes == 52

    with pytest.raises(OrchidPipelineLimitError, match="max_graph_bytes"):
        graph.commit("aliases", [], aliases={"ccc": "a"})

    assert graph.aliases == {"a": "a", "bb": "a"}
    assert graph.alias_bytes == 52


def test_job_scoped_alias_bytes_and_count_are_bounded() -> None:
    graph = _graph(max_graph_bytes=45)
    graph.commit("load", [_create("a")])
    graph.commit("job1", [], aliases={"account": "a"})
    assert graph.alias_bytes == 34
    assert graph.job_alias_bytes == 19

    with pytest.raises(OrchidPipelineLimitError, match="max_graph_bytes"):
        graph.commit("job2", [], aliases={"account": "a"})

    assert graph.aliases == {"account": "a"}
    assert graph.aliases_for("job2") == {}

    counted = _graph(max_aliases=2)
    counted.commit("load", [_create("a")])
    counted.commit("job1", [], aliases={"account": "a"})
    counted.commit("job2", [], aliases={"account": "a"})
    with pytest.raises(OrchidPipelineLimitError, match="max_aliases"):
        counted.commit("job3", [], aliases={"account": "a"})
    assert counted.aliases_for("job3") == {}


# ── reserved input and invalid ids ──────────────────────────


def test_input_object_is_seeded_once_at_version_one_and_is_read_only() -> None:
    graph = _graph()

    seeded = graph.set_input({"account_id": "acme"})
    assert seeded.id == PIPELINE_INPUT_OBJECT_ID
    assert seeded.type == PIPELINE_INPUT_OBJECT_TYPE
    assert graph.version == 1
    assert graph.view().get(PIPELINE_INPUT_OBJECT_ID) == seeded

    mutations = [
        _create(PIPELINE_INPUT_OBJECT_ID, data={"account_id": "other"}),
        _update(PIPELINE_INPUT_OBJECT_ID, {"account_id": "other"}),
        _upsert(PIPELINE_INPUT_OBJECT_ID, {"account_id": "other"}),
        _delete(PIPELINE_INPUT_OBJECT_ID),
    ]
    for mutation in mutations:
        with pytest.raises(OrchidPipelineWriteNotAllowedError, match="reserved"):
            graph.commit("job", [mutation])

    with pytest.raises(OrchidPipelineWriteNotAllowedError, match="reserved"):
        graph.commit("job", [_create("other", object_type=PIPELINE_INPUT_OBJECT_TYPE)])

    assert graph.objects[PIPELINE_INPUT_OBJECT_ID].data == {"account_id": "acme"}
    assert graph.version == 1

    with pytest.raises(OrchidPipelineGraphConflictError, match="already seeded"):
        graph.set_input({"account_id": "second"})


def test_graph_rejects_an_object_id_corrupted_after_mutation_construction() -> None:
    graph = _graph()
    mutation = _create("account:1")
    object.__setattr__(mutation, "object_id", "bad\nid")

    with pytest.raises(ValueError, match="control characters"):
        graph.commit("job", [mutation])
    assert graph.objects == {}

    long_id = _create("account:1")
    object.__setattr__(long_id, "object_id", "x" * 257)
    with pytest.raises(ValueError, match="at most 256"):
        graph.commit("job", [long_id])


def test_seed_objects_initialise_the_graph_without_a_job_commit() -> None:
    graph = _graph()
    seed = OrchidGraphObject(id="account:seed", type="Account", data={"name": "Seed"}, created_by="seed")

    assert graph.seed([seed]) == ("account:seed",)
    assert graph.version == 1
    assert graph.objects["account:seed"].data == {"name": "Seed"}
    assert graph.metrics.commits == 0

    with pytest.raises(OrchidPipelineGraphConflictError, match="already exists"):
        graph.seed([seed])
    with pytest.raises(TypeError, match="OrchidGraphObject"):
        graph.seed([{"id": "not-an-object"}])  # type: ignore[list-item]
    with pytest.raises(OrchidPipelineWriteNotAllowedError, match="reserved"):
        graph.seed([OrchidGraphObject(id=PIPELINE_INPUT_OBJECT_ID, type="Account", data={})])
    with pytest.raises(OrchidPipelineWriteNotAllowedError, match="reserved"):
        graph.seed([OrchidGraphObject(id="fake:input", type=PIPELINE_INPUT_OBJECT_TYPE, data={})])


def test_seeding_after_a_job_commit_is_a_versioned_change() -> None:
    graph = _graph()
    graph.commit("load", [_create("a")])
    assert graph.version == 1

    graph.seed([OrchidGraphObject(id="b", type="Account", data={})])

    assert graph.version == 2
    assert set(graph.objects) == {"a", "b"}


def test_seed_batch_is_atomic() -> None:
    graph = _graph(max_objects=2)
    seeds = [
        OrchidGraphObject(id="a", type="Account", data={}),
        OrchidGraphObject(id="b", type="Account", data={}),
        OrchidGraphObject(id="c", type="Account", data={}),
    ]

    with pytest.raises(OrchidPipelineLimitError, match="max_objects"):
        graph.seed(seeds)

    assert graph.objects == {}
    assert graph.version == 0

    with pytest.raises(OrchidPipelineGraphConflictError, match="already exists"):
        graph.seed([seeds[0], OrchidGraphObject(id="a", type="Account", data={})])
    assert graph.objects == {}


def test_seed_provenance_is_validated_and_snapshot_stays_serializable() -> None:
    graph = _graph()
    seed = OrchidGraphObject(
        id="account:seed",
        type="Account",
        data={"name": "Seed"},
        created_at="2026-11-17T09:00:00+00:00",
        updated_at="2026-11-17T09:05:00+00:00",
        created_by="seed-job",
        updated_by="other-job",
    )

    graph.seed([seed])
    snapshot = graph.snapshot()
    assert snapshot["objects"]["account:seed"]["created_at"] == "2026-11-17T09:00:00+00:00"
    assert snapshot["objects"]["account:seed"]["created_by"] == "seed-job"
    assert json.loads(json.dumps(snapshot)) == snapshot

    with pytest.raises(ValueError, match="ISO-8601"):
        graph.seed([OrchidGraphObject(id="bad:date", type="Account", data={}, created_at="not-a-date")])
    with pytest.raises(TypeError, match="must be a string"):
        graph.seed([OrchidGraphObject(id="bad:type", type="Account", data={}, updated_at=123)])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="control characters"):
        graph.seed([OrchidGraphObject(id="bad:actor", type="Account", data={}, created_by="bad\njob")])

    assert set(graph.objects) == {"account:seed"}


def test_seed_accepts_empty_provenance() -> None:
    graph = _graph()
    graph.seed([OrchidGraphObject(id="a", type="Account", data={})])
    assert graph.objects["a"].created_at == ""
    assert graph.objects["a"].updated_at == ""
    assert graph.objects["a"].created_by == ""
    assert graph.objects["a"].updated_by == ""


def test_seed_shape_version_is_bounded_and_counted() -> None:
    graph = _graph(max_object_bytes=5)

    with pytest.raises(ValueError, match="at most 256"):
        graph.seed([OrchidGraphObject(id="long", type="Account", data={}, shape_version="x" * 257)])
    with pytest.raises(ValueError, match="control characters"):
        graph.seed([OrchidGraphObject(id="control", type="Account", data={}, shape_version="bad\nversion")])
    with pytest.raises(OrchidPipelineLimitError, match="max_object_bytes"):
        graph.seed([OrchidGraphObject(id="large", type="Account", data={}, shape_version="12")])

    graph.seed([OrchidGraphObject(id="ok", type="Account", data={}, shape_version="1")])
    assert graph.objects["ok"].shape_version == "1"

    input_graph = _graph(max_object_bytes=4)
    with pytest.raises(OrchidPipelineLimitError, match="max_object_bytes"):
        input_graph.set_input({})


def test_seed_revision_is_bounded_and_snapshot_is_json_safe() -> None:
    graph = _graph()

    with pytest.raises(ValueError, match="at most 9223372036854775807"):
        graph.seed([OrchidGraphObject(id="huge", type="Account", data={}, revision=10**5000)])
    assert graph.objects == {}

    graph.seed([OrchidGraphObject(id="near-max", type="Account", data={}, revision=2**63 - 1)])
    snapshot = graph.snapshot()
    assert snapshot["objects"]["near-max"]["revision"] == 2**63 - 1
    assert json.loads(json.dumps(snapshot)) == snapshot

    with pytest.raises(OrchidPipelineLimitError, match="max_revision"):
        graph.commit("job", [_update("near-max", {"v": 1})])


def test_seed_provenance_rejects_lone_surrogates() -> None:
    graph = _graph()

    with pytest.raises(ValueError, match="UTF-8"):
        graph.seed([OrchidGraphObject(id="actor", type="Account", data={}, created_by="\ud800")])
    with pytest.raises(ValueError, match="UTF-8"):
        graph.seed(
            [
                OrchidGraphObject(
                    id="timestamp",
                    type="Account",
                    data={},
                    created_at="2026-11-17T09:00:00\ud800",
                )
            ]
        )
    assert graph.objects == {}

    graph.seed([OrchidGraphObject(id="valid", type="Account", data={}, created_by="José")])
    assert json.dumps(graph.snapshot())


# ── JSON safety ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "bad_value",
    [
        b"bytes",
        bytearray(b"bytes"),
        {1, 2},
        frozenset({1}),
        object(),
        datetime.now(UTC),
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_non_json_values_are_rejected_before_mutation(bad_value: object) -> None:
    graph = _graph()
    with pytest.raises(OrchidPipelineSerializationError):
        graph.commit("job", [_create("account:1", data={"bad": bad_value})])
    assert graph.objects == {}
    assert graph.version == 0


def test_non_string_keys_circular_values_and_depth_are_rejected() -> None:
    graph = _graph()

    with pytest.raises(OrchidPipelineSerializationError, match="must be a string"):
        graph.commit("job", [_create("account:1", data={"nested": {1: "x"}})])

    circular: dict[str, Any] = {}
    circular["self"] = circular
    with pytest.raises(OrchidPipelineSerializationError, match="circular"):
        graph.commit("job", [_create("account:1", data=circular)])

    nested: dict[str, Any] = {}
    cursor = nested
    for _ in range(60):
        child: dict[str, Any] = {}
        cursor["child"] = child
        cursor = child
    with pytest.raises(OrchidPipelineSerializationError, match="nesting depth"):
        graph.commit("job", [_create("account:1", data=nested)])


def test_data_is_copied_and_tuples_become_json_lists() -> None:
    graph = _graph()
    payload = {"nested": {"x": 1}, "items": (1, 2)}
    graph.commit("job", [_create("account:1", data=payload)])

    payload["nested"]["x"] = 99
    payload["items"] = (3, 4)

    assert graph.objects["account:1"].data == {"nested": {"x": 1}, "items": [1, 2]}


def test_relation_data_must_be_json_safe() -> None:
    graph = _graph()
    _accounts(graph, "a", "b")
    with pytest.raises(OrchidPipelineSerializationError):
        graph.commit("link", [_link("has_usage", "a", "b", data={"handle": object()})])
    assert graph.relations == {}


# ── caps ─────────────────────────────────────────────────────


def test_object_cap_names_the_limit_and_keeps_previous_objects() -> None:
    graph = _graph(max_objects=1)
    graph.commit("load", [_create("a")])

    with pytest.raises(OrchidPipelineLimitError, match="max_objects") as caught:
        graph.commit("load", [_create("b")])

    assert caught.value.limit == "max_objects"
    assert caught.value.actual == 2
    assert caught.value.maximum == 1
    assert list(graph.objects) == ["a"]
    assert graph.version == 1


def test_relation_cap_keeps_previous_relations() -> None:
    graph = _graph(max_relations=1)
    _accounts(graph, "a", "b", "c")
    graph.commit("link", [_link("has_usage", "a", "b")])

    with pytest.raises(OrchidPipelineLimitError, match="max_relations"):
        graph.commit("link", [_link("has_usage", "a", "c")])

    assert list(graph.relations) == ["has_usage:a:b"]


def test_per_object_byte_cap_applies_to_objects_and_relations() -> None:
    graph = _graph(max_object_bytes=10)
    with pytest.raises(OrchidPipelineLimitError, match="max_object_bytes"):
        graph.commit("load", [_create("account:1", data={"name": "a long value"})])

    graph = _graph(max_object_bytes=10)
    graph.commit("load", [_create("a"), _create("b")])
    with pytest.raises(OrchidPipelineLimitError, match="max_object_bytes"):
        graph.commit("link", [_link("has_usage", "a", "b", data={"period": "a long value"})])


def test_total_graph_byte_cap_is_enforced() -> None:
    graph = _graph(max_graph_bytes=30)
    graph.commit("load", [_create("a", data={"v": "1234"}), _create("b", data={"v": "1234"})])

    with pytest.raises(OrchidPipelineLimitError, match="max_graph_bytes"):
        graph.commit("load", [_create("c", data={"v": "1234"})])

    assert set(graph.objects) == {"a", "b"}


def test_input_and_seed_respect_the_graph_byte_cap() -> None:
    with pytest.raises(OrchidPipelineLimitError, match="max_graph_bytes"):
        DataGraph(max_graph_bytes=5).set_input({"a": "long"})
    graph = _graph(max_graph_bytes=5)
    with pytest.raises(OrchidPipelineLimitError, match="max_graph_bytes"):
        graph.seed([OrchidGraphObject(id="a", type="Account", data={"a": "long"})])
    assert graph.objects == {}
    assert graph.version == 0


def test_relation_payload_respects_the_graph_byte_cap() -> None:
    graph = _graph(max_graph_bytes=20)
    graph.commit("load", [_create("a"), _create("b")])

    with pytest.raises(OrchidPipelineLimitError, match="max_graph_bytes"):
        graph.commit("link", [_link("edge", "a", "b", data={"v": "1234567890"})])

    assert graph.relations == {}
    assert set(graph.objects) == {"a", "b"}


def test_caps_are_validated_at_construction() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        DataGraph(max_objects=0)
    with pytest.raises(TypeError, match="integer"):
        DataGraph(max_relations=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        DataGraph(default_shape_version="")
    with pytest.raises(ValueError, match="max_neighbour_hops"):
        DataGraph(max_neighbour_hops=99)
    with pytest.raises(ValueError, match="max_neighbour_objects"):
        DataGraph(max_neighbour_objects=99_999)
    with pytest.raises(ValueError, match="at least 1"):
        DataGraph(max_aliases=0)


def test_metadata_counts_toward_the_byte_caps() -> None:
    graph = _graph(max_object_bytes=20)
    seed = OrchidGraphObject(
        id="a",
        type="Account",
        data={},
        metadata={"note": "x" * 50},
    )

    with pytest.raises(OrchidPipelineLimitError, match="max_object_bytes"):
        graph.seed([seed])
    assert graph.objects == {}

    graph = _graph(max_object_bytes=20)
    graph.commit("load", [_create("a"), _create("b")])
    with pytest.raises(OrchidPipelineLimitError, match="max_object_bytes"):
        graph.commit(
            "link",
            [
                OrchidGraphMutation(
                    op="link",
                    relation=OrchidGraphRelation(
                        id="has_usage:a:b",
                        type="has_usage",
                        source_id="a",
                        target_id="b",
                        metadata={"note": "x" * 50},
                    ),
                )
            ],
        )
    assert graph.relations == {}


def test_input_shape_version_must_be_non_empty_when_supplied() -> None:
    graph = _graph()
    with pytest.raises(ValueError, match="non-empty"):
        graph.set_input({}, shape_version="")
    assert graph._last_timestamp is None
    assert graph.objects == {}


# ── view ─────────────────────────────────────────────────────


def test_view_get_and_find_filter_by_type_and_top_level_equality() -> None:
    graph = _graph()
    graph.commit(
        "load",
        [
            _create("account:1", data={"name": "Acme", "plan": "pro"}),
            _create("account:2", data={"name": "Beta", "plan": "free"}),
            _create("usage:1", object_type="UsageSnapshot", data={"name": "Acme", "plan": None}),
        ],
    )
    view = graph.view()
    assert isinstance(view, OrchidDataGraphView)
    assert view.version == 1

    assert view.get("account:1") == graph.objects["account:1"]
    assert view.get("account:1") is not graph.objects["account:1"]
    assert view.get("missing") is None

    assert [obj.id for obj in view.find("Account")] == ["account:1", "account:2"]
    assert [obj.id for obj in view.find(where={"name": "Acme"})] == ["account:1", "usage:1"]
    assert [obj.id for obj in view.find("Account", where={"plan": "free"})] == ["account:2"]
    assert [obj.id for obj in view.find("Account", limit=1)] == ["account:1"]
    assert view.find("Account", limit=0) == []
    assert view.find(where={"missing": None}) == []
    assert [obj.id for obj in view.find(where={"plan": None})] == ["usage:1"]

    with pytest.raises(ValueError, match="non-negative"):
        view.find(limit=-1)
    with pytest.raises(TypeError, match="keys must be strings"):
        view.find(where={1: "x"})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="mapping or None"):
        view.find(where=["not", "a", "mapping"])  # type: ignore[arg-type]


def test_view_relations_filter_and_read_only_surface() -> None:
    graph = _graph()
    _accounts(graph, "a", "b", "c")
    graph.commit("link", [_link("has_usage", "a", "b"), _link("has_assessment", "b", "c")])
    view = graph.view()

    assert [rel.id for rel in view.relations()] == ["has_usage:a:b", "has_assessment:b:c"]
    assert [rel.id for rel in view.relations(type="has_usage")] == ["has_usage:a:b"]
    assert [rel.id for rel in view.relations(source_id="b")] == ["has_assessment:b:c"]
    assert [rel.id for rel in view.relations(target_id="c")] == ["has_assessment:b:c"]
    assert view.relations(type="has_usage", source_id="b") == []

    assert not hasattr(view, "commit")
    assert not hasattr(view, "set_input")
    with pytest.raises(TypeError):
        graph.objects["new"] = graph.objects["a"]  # type: ignore[index]


def test_view_is_live_across_commits() -> None:
    graph = _graph()
    view = graph.view()
    assert view.get("a") is None

    graph.commit("load", [_create("a")])

    assert view.get("a") is not None
    assert view.version == 1


def test_view_reads_are_defensive_copies() -> None:
    graph = _graph()
    graph.commit(
        "load",
        [
            _create("a", data={"nested": {"x": 1}, "list": [1]}),
            _create("b", data={"name": "B"}),
        ],
    )
    graph.commit("link", [_link("edge", "a", "b", data={"period": "2026-10"})])
    view = graph.view()

    obj = view.get("a")
    assert obj is not None
    obj.data["nested"]["x"] = 99
    obj.data["list"].append(2)
    obj.metadata["hacked"] = True
    assert graph.objects["a"].data == {"nested": {"x": 1}, "list": [1]}
    assert graph.objects["a"].metadata == {}

    found = view.find("Account")[0]
    found.data["nested"]["x"] = 77
    assert graph.objects["a"].data["nested"]["x"] == 1

    relation = view.relations()[0]
    relation.data["period"] = "hacked"
    relation.metadata["hacked"] = True
    assert graph.relations["edge:a:b"].data == {"period": "2026-10"}
    assert graph.relations["edge:a:b"].metadata == {}

    objects, relations = view.neighbours("a")
    objects[0].data["name"] = "hacked"
    relations[0].metadata["hacked"] = True
    assert graph.objects["b"].data == {"name": "B"}
    assert graph.relations["edge:a:b"].metadata == {}


def test_view_cannot_mutate_the_reserved_input_object() -> None:
    graph = _graph()
    graph.set_input({"account_id": "acme", "nested": {"x": 1}})
    view = graph.view()

    obj = view.get(PIPELINE_INPUT_OBJECT_ID)
    assert obj is not None
    obj.data["account_id"] = "hacked"
    obj.data["nested"]["x"] = 99
    obj.metadata["hacked"] = True

    assert graph.objects[PIPELINE_INPUT_OBJECT_ID].data == {"account_id": "acme", "nested": {"x": 1}}
    assert graph.objects[PIPELINE_INPUT_OBJECT_ID].metadata == {}
    assert view.find(where={"account_id": "acme"})[0].data == {"account_id": "acme", "nested": {"x": 1}}


def test_neighbours_honours_direction_hops_and_relation_filter() -> None:
    graph = _graph()
    _accounts(graph, "a", "b", "c", "d")
    graph.commit(
        "link",
        [_link("has_usage", "a", "b"), _link("has_usage", "b", "c"), _link("has_usage", "d", "a")],
    )
    view = graph.view()

    objects, relations = view.neighbours("a")
    assert [obj.id for obj in objects] == ["b"]
    assert [rel.id for rel in relations] == ["has_usage:a:b"]

    objects, _ = view.neighbours("a", max_hops=2)
    assert [obj.id for obj in objects] == ["b", "c"]

    objects, relations = view.neighbours("a", direction="in")
    assert [obj.id for obj in objects] == ["d"]
    assert [rel.id for rel in relations] == ["has_usage:d:a"]

    objects, _ = view.neighbours("a", direction="both")
    assert {obj.id for obj in objects} == {"b", "d"}

    objects, relations = view.neighbours("a", relation="other")
    assert objects == [] and relations == []

    assert view.neighbours("missing") == ([], [])
    assert view.neighbours("a", max_hops=0) == ([], [])


def test_neighbours_is_bounded_by_max_objects_and_graph_hop_bound() -> None:
    graph = _graph(max_neighbour_objects=10)
    _accounts(graph, "a", "b", "c", "d")
    graph.commit("link", [_link("has_usage", "a", "b"), _link("has_usage", "a", "c"), _link("has_usage", "a", "d")])
    view = graph.view()

    objects, relations = view.neighbours("a", max_objects=2)
    assert [obj.id for obj in objects] == ["b", "c"]
    assert len(relations) == 2

    objects, _ = view.neighbours("a", max_objects=0)
    assert objects == []

    with pytest.raises(ValueError, match="exceeds the graph's bound"):
        view.neighbours("a", max_hops=99)
    with pytest.raises(ValueError, match="non-negative"):
        view.neighbours("a", max_hops=-1)
    with pytest.raises(ValueError, match="direction"):
        view.neighbours("a", direction="sideways")  # type: ignore[arg-type]


def test_neighbours_handles_cycles_and_self_loops_without_duplicates() -> None:
    graph = _graph()
    graph.commit("load", [_create("a"), _create("b")])
    graph.commit("link", [_link("edge", "a", "b"), _link("edge", "b", "a"), _link("edge", "a", "a")])
    view = graph.view()

    objects, relations = view.neighbours("a", direction="both", max_hops=3)

    assert [obj.id for obj in objects] == ["b"]
    assert {relation.id for relation in relations} == {"edge:a:b", "edge:b:a", "edge:a:a"}
    assert len(relations) == 3


# ── serialization and stats ─────────────────────────────────


def test_snapshot_is_json_serializable_and_round_trips() -> None:
    graph = _graph()
    _accounts(graph, "account:1")
    graph.commit("link", [_link("has_usage", "account:1", "account:1", data={"period": "2026-10"})])

    snapshot = graph.snapshot()
    assert snapshot["version"] == graph.version
    assert snapshot["objects"]["account:1"]["id"] == "account:1"
    assert snapshot["objects"]["account:1"]["revision"] == 1
    assert snapshot["objects"]["account:1"]["created_by"] == "load"
    assert snapshot["relations"]["has_usage:account:1:account:1"]["data"] == {"period": "2026-10"}
    assert json.loads(json.dumps(snapshot)) == snapshot
    assert graph.to_dict() == snapshot


def test_stats_report_counts_by_type_and_size() -> None:
    graph = _graph()
    _accounts(graph, "a", "b")
    graph.commit("link", [_link("has_usage", "a", "b")])

    stats = graph.stats()
    assert stats["version"] == 2
    assert stats["objects"] == 2
    assert stats["relations"] == 1
    assert stats["bytes"] == graph.bytes > 0
    assert stats["by_type"] == {"Account": 2}


def test_commit_result_is_frozen_and_mapping_fields_are_read_only() -> None:
    graph = _graph()
    commit = graph.commit("load", [_create("a")], aliases={"account": "a"})

    with pytest.raises(FrozenInstanceError):
        commit.job_id = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        commit.aliases["account"] = "other"  # type: ignore[index]
    assert isinstance(commit, DataGraphCommit)
    assert isinstance(graph.view(), DataGraphView)
