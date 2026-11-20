"""Value-object, port, and error invariants for ``orchid_ai.core.pipeline``."""

from __future__ import annotations

import json
import logging
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pytest

from orchid_ai.core.pipeline import (
    PIPELINE_INPUT_OBJECT_ID,
    PIPELINE_INPUT_OBJECT_TYPE,
    PIPELINE_RESERVED_OBJECT_IDS,
    OrchidDataGraphView,
    OrchidGraphMutation,
    OrchidGraphObject,
    OrchidGraphRelation,
    OrchidPipelineApprovalRequiredError,
    OrchidPipelineAuthError,
    OrchidPipelineConfigError,
    OrchidPipelineDisabledError,
    OrchidPipelineError,
    OrchidPipelineEvent,
    OrchidPipelineEventsDisabledError,
    OrchidPipelineGraphConflictError,
    OrchidPipelineInputError,
    OrchidPipelineJobContext,
    OrchidPipelineLimitError,
    OrchidPipelineMappingError,
    OrchidPipelineNotFoundError,
    OrchidPipelineObserver,
    OrchidPipelineOutputError,
    OrchidPipelineResult,
    OrchidPipelineRetryableError,
    OrchidPipelineRunContext,
    OrchidPipelineSerializationError,
    OrchidPipelineShapeError,
    OrchidPipelineStep,
    OrchidPipelineStepContext,
    OrchidPipelineStepError,
    OrchidPipelineStepResult,
    OrchidPipelineTimeoutError,
    OrchidPipelineWriteNotAllowedError,
)

START = datetime(2026, 11, 17, 9, 30, tzinfo=UTC)


def _object(**overrides: object) -> OrchidGraphObject:
    values: dict[str, object] = {"id": "account:1", "type": "Account", "data": {"name": "Acme"}}
    values.update(overrides)
    return OrchidGraphObject(**values)  # type: ignore[arg-type]


def _relation(**overrides: object) -> OrchidGraphRelation:
    values: dict[str, object] = {
        "id": "has_assessment:account:1:assessment:1",
        "type": "has_assessment",
        "source_id": "account:1",
        "target_id": "assessment:1",
    }
    values.update(overrides)
    return OrchidGraphRelation(**values)  # type: ignore[arg-type]


def _run_context(**overrides: object) -> OrchidPipelineRunContext:
    values: dict[str, object] = {
        "run_id": "run-1",
        "pipeline": "assess",
        "pipeline_version": "1",
        "mode": "direct",
        "tenant_key": "tenant-1",
        "user_id": "user-1",
        "started_at": START,
    }
    values.update(overrides)
    return OrchidPipelineRunContext(**values)  # type: ignore[arg-type]


# ── Value objects ────────────────────────────────────────────


def test_graph_object_is_frozen_slotted_and_json_serializable() -> None:
    obj = _object(metadata={"source": "seed"}, created_at="2026-11-17T09:00:00+00:00")

    with pytest.raises(FrozenInstanceError):
        obj.revision = 2  # type: ignore[misc]
    assert not hasattr(obj, "__dict__")
    assert obj.revision == 1
    assert obj.shape_version == "1"

    payload = obj.to_dict()
    assert payload["data"] == {"name": "Acme"}
    assert json.loads(json.dumps(payload)) == payload


def test_graph_object_rejects_invalid_ids_and_revisions() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _object(id="")
    with pytest.raises(ValueError, match="at most 256"):
        _object(id="x" * 257)
    with pytest.raises(ValueError, match="control characters"):
        _object(id="a\nb")
    with pytest.raises(ValueError, match="at least 1"):
        _object(revision=0)
    with pytest.raises(TypeError, match="must be a mapping"):
        OrchidGraphObject(id="a", type="T", data=["not", "a", "mapping"])  # type: ignore[arg-type]


def test_reserved_input_object_is_constructible_but_not_writable() -> None:
    # The session seeds ``__input__``; the constant is what a config validator
    # rejects. The value object itself must accept the reserved ID.
    obj = OrchidGraphObject(id=PIPELINE_INPUT_OBJECT_ID, type=PIPELINE_INPUT_OBJECT_TYPE, data={"x": 1})
    assert obj.id in PIPELINE_RESERVED_OBJECT_IDS


def test_graph_relation_serializes_and_copies_mappings() -> None:
    data = {"period": "2026-10"}
    relation = _relation(data=data)
    data["period"] = "mutated"

    assert relation.data == {"period": "2026-10"}
    assert relation.to_dict()["source_id"] == "account:1"
    with pytest.raises(ValueError, match="relation type"):
        _relation(type="")


def test_graph_relation_endpoints_use_object_id_rules() -> None:
    with pytest.raises(ValueError, match="at most 256"):
        _relation(source_id="x" * 257)
    with pytest.raises(ValueError, match="control characters"):
        _relation(target_id="assessment:\n1")
    with pytest.raises(ValueError, match="non-empty"):
        _relation(source_id="")


def test_graph_relation_id_is_bounded_without_capping_endpoints() -> None:
    endpoint = "x" * 256
    relation = _relation(id=f"has_edge:{endpoint}:{endpoint}")
    assert len(relation.id) <= 1024

    with pytest.raises(ValueError, match="at most 1024"):
        _relation(id="x" * 1025)
    with pytest.raises(ValueError, match="control characters"):
        _relation(id="has_edge:a\nb")


def test_graph_value_objects_serialize_datetime_timestamps() -> None:
    obj = _object(
        created_at=START,
        updated_at=START,
        created_by="load",
        updated_by="analyze",
    )
    payload = obj.to_dict()
    assert payload["created_at"] == START.isoformat()
    assert payload["updated_at"] == START.isoformat()
    assert json.loads(json.dumps(payload)) == payload

    relation = _relation(created_at=START, created_by="analyze")
    relation_payload = relation.to_dict()
    assert relation_payload["created_at"] == START.isoformat()
    assert json.loads(json.dumps(relation_payload)) == relation_payload


def test_mutation_validates_ops_and_link_relation() -> None:
    create = OrchidGraphMutation(op="create", object_id="account:1", object_type="Account", data={"name": "Acme"})
    assert create.to_dict()["mode"] == "merge"

    # ``update`` addresses an existing object; the session already knows its
    # type, so the mutation may omit it.
    update = OrchidGraphMutation(op="update", object_id="account:1")
    assert update.object_type == ""
    with pytest.raises(ValueError, match="object_type"):
        OrchidGraphMutation(op="create", object_id="account:1")

    with pytest.raises(ValueError, match="mutation op"):
        OrchidGraphMutation(op="patch")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires a relation"):
        OrchidGraphMutation(op="link")
    with pytest.raises(ValueError, match="must not carry a relation"):
        OrchidGraphMutation(op="update", object_id="account:1", relation=_relation())
    with pytest.raises(ValueError, match="requires a relation"):
        OrchidGraphMutation(op="unlink")

    link = OrchidGraphMutation(op="link", relation=_relation())
    assert link.to_dict()["relation"] is not None
    assert link.object_id == ""

    with pytest.raises(ValueError, match="at least 1"):
        OrchidGraphMutation(op="update", object_id="account:1", object_type="Account", expected_revision=0)
    with pytest.raises(TypeError, match="expected_revision"):
        OrchidGraphMutation(
            op="update",
            object_id="account:1",
            object_type="Account",
            expected_revision=True,  # type: ignore[arg-type]
        )


def test_run_context_serializes_datetimes_and_validates_mode() -> None:
    deadline = START + timedelta(minutes=5)
    context = _run_context(mode="chat", chat_id="chat-1", deadline=deadline, metadata={"k": "v"})
    payload = context.to_dict()

    assert payload["started_at"] == START.isoformat()
    assert payload["deadline"] == deadline.isoformat()
    assert json.loads(json.dumps(payload)) == payload

    with pytest.raises(ValueError, match="run mode"):
        _run_context(mode="batch")
    with pytest.raises(ValueError, match="deadline"):
        _run_context(deadline=START - timedelta(seconds=1))


def test_job_context_defaults_and_map_bindings() -> None:
    context = OrchidPipelineJobContext(job_id="enrich", kind="agent", attempt=2, params={"x": 1})
    assert context.item is None and context.index is None and context.parent_job is None

    item_context = OrchidPipelineJobContext(
        job_id="enrich", kind="agent", item={"id": "a"}, index=0, parent_job="enrich_all"
    )
    assert item_context.to_dict()["item"] == {"id": "a"}

    with pytest.raises(ValueError, match="attempt"):
        OrchidPipelineJobContext(job_id="j", kind="llm", attempt=0)
    with pytest.raises(ValueError, match="index"):
        OrchidPipelineJobContext(job_id="j", kind="llm", index=-1)


def test_step_result_normalises_mutations_and_serializes() -> None:
    mutation = OrchidGraphMutation(op="create", object_id="account:1", object_type="Account")
    result = OrchidPipelineStepResult(output={"ok": True}, mutations=[mutation], usage={"llm_calls": 1})

    assert isinstance(result.mutations, tuple)
    assert result.mutations == (mutation,)
    assert json.loads(json.dumps(result.to_dict()))["usage"] == {"llm_calls": 1}

    with pytest.raises(TypeError, match="OrchidGraphMutation"):
        OrchidPipelineStepResult(mutations=("not-a-mutation",))  # type: ignore[arg-type]


def test_pipeline_result_to_dict_controls_graph_exposure() -> None:
    finished = START + timedelta(seconds=2)
    result = OrchidPipelineResult(
        run_id="run-1",
        pipeline="assess",
        status="succeeded",
        started_at=START,
        finished_at=finished,
        output={"summary": "done"},
        graph={"objects": {"a": {"id": "a"}}},
        jobs={"load": {"status": "succeeded"}},
        graph_stats={"objects": 1},
        summary="done",
        summary_source="template",
    )

    thin = result.to_dict()
    assert "graph" not in thin
    assert thin["status"] == "succeeded"
    assert thin["duration_ms"] == 2000.0
    assert json.loads(json.dumps(thin)) == thin

    full = result.to_dict(include_graph=True)
    assert full["graph"] == {"objects": {"a": {"id": "a"}}}

    with pytest.raises(ValueError, match="result status"):
        OrchidPipelineResult(run_id="r", pipeline="p", started_at=START, status="done")  # type: ignore[arg-type]


def test_pipeline_result_requires_status_and_started_at() -> None:
    with pytest.raises(TypeError):
        OrchidPipelineResult(run_id="r", pipeline="p", started_at=START)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        OrchidPipelineResult(run_id="r", pipeline="p", status="succeeded")  # type: ignore[call-arg]


def test_pipeline_result_rejects_finished_before_started() -> None:
    with pytest.raises(ValueError, match="must not precede"):
        OrchidPipelineResult(
            run_id="r",
            pipeline="p",
            status="succeeded",
            started_at=START,
            finished_at=START - timedelta(seconds=1),
        )


def test_pipeline_result_to_dict_rejects_live_objects() -> None:
    result = OrchidPipelineResult(
        run_id="r",
        pipeline="p",
        status="succeeded",
        started_at=START,
        output={"handle": object()},
    )
    with pytest.raises(OrchidPipelineSerializationError, match="not JSON-serializable"):
        result.to_dict()

    graph_result = OrchidPipelineResult(
        run_id="r",
        pipeline="p",
        status="succeeded",
        started_at=START,
        graph={"objects": {"a": object()}},
    )
    with pytest.raises(OrchidPipelineSerializationError, match="not JSON-serializable"):
        graph_result.to_dict(include_graph=True)

    jobs_result = OrchidPipelineResult(
        run_id="r",
        pipeline="p",
        status="succeeded",
        started_at=START,
        jobs={"load": {"handle": object()}},
    )
    with pytest.raises(OrchidPipelineSerializationError, match="not JSON-serializable"):
        jobs_result.to_dict()


def test_graph_object_and_step_result_reject_live_objects() -> None:
    obj = _object(data={"handle": object()})
    with pytest.raises(OrchidPipelineSerializationError, match="not JSON-serializable"):
        obj.to_dict()

    step_result = OrchidPipelineStepResult(output={"handle": object()})
    with pytest.raises(OrchidPipelineSerializationError, match="not JSON-serializable"):
        step_result.to_dict()

    relation = _relation(data={"handle": object()})
    with pytest.raises(OrchidPipelineSerializationError, match="not JSON-serializable"):
        relation.to_dict()


def test_pipeline_result_to_dict_normalizes_nested_datetimes() -> None:
    result = OrchidPipelineResult(
        run_id="r",
        pipeline="p",
        status="succeeded",
        started_at=START,
        jobs={"load": {"finished_at": START}},
        metadata={"at": START},
    )
    payload = result.to_dict()
    assert payload["jobs"]["load"]["finished_at"] == START.isoformat()
    assert payload["metadata"]["at"] == START.isoformat()
    json.dumps(payload)


def test_pipeline_event_to_dict_and_defaults() -> None:
    event = OrchidPipelineEvent(type="pipeline.started", run_id="run-1", data={"pipeline": "assess"})
    payload = event.to_dict()

    assert payload["type"] == "pipeline.started"
    assert "T" in payload["timestamp"]
    assert json.loads(json.dumps(payload)) == payload

    with pytest.raises(ValueError, match="event type"):
        OrchidPipelineEvent(type="", run_id="run-1")


# ── Ports ────────────────────────────────────────────────────


class _View(OrchidDataGraphView):
    def get(self, object_id: str) -> OrchidGraphObject | None:
        return None

    def find(
        self,
        object_type: str | None = None,
        where: dict | None = None,
        limit: int | None = None,
    ) -> list[OrchidGraphObject]:
        return []

    def relations(
        self,
        *,
        type: str | None = None,
        source_id: str | None = None,
        target_id: str | None = None,
    ) -> list[OrchidGraphRelation]:
        return []

    def neighbours(
        self,
        object_id: str,
        *,
        relation: str | None = None,
        direction: str = "out",
        max_hops: int = 1,
    ) -> tuple[list[OrchidGraphObject], list[OrchidGraphRelation]]:
        return [], []


def test_data_graph_view_is_abstract_and_substitutable() -> None:
    with pytest.raises(TypeError):
        OrchidDataGraphView()  # type: ignore[abstract]

    view = _View()
    assert view.get("missing") is None
    assert view.find() == []
    assert view.neighbours("missing") == ([], [])


class _EchoStep(OrchidPipelineStep):
    kind = "echo"

    async def execute(self, ctx: OrchidPipelineStepContext) -> OrchidPipelineStepResult:
        return OrchidPipelineStepResult(output={"key": ctx.idempotency_key})


class _CollectingObserver(OrchidPipelineObserver):
    def __init__(self) -> None:
        self.events: list[OrchidPipelineEvent] = []

    async def emit(self, event: OrchidPipelineEvent) -> None:
        self.events.append(event)


class _FakeStepContext:
    def __init__(self) -> None:
        self.run = _run_context()
        self.job = OrchidPipelineJobContext(job_id="echo", kind="echo")
        self.input = {"x": 1}
        self.graph = _View()
        self.auth = None
        self.chat_model = None
        self.reader = None
        self.logger = logging.getLogger("test.pipeline")
        self.idempotency_key = "run-1:echo"

    async def call_tool(self, name: str, **kwargs: object) -> object:
        return {"name": name, "kwargs": kwargs}

    async def emit_signal(self, envelope: object) -> object:
        return envelope

    def stable_id(self, *parts: str) -> str:
        return ":".join(parts)


async def test_step_and_observer_abcs_are_usable_contracts() -> None:
    with pytest.raises(TypeError):
        OrchidPipelineStep()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        OrchidPipelineObserver()  # type: ignore[abstract]

    context = _FakeStepContext()
    assert isinstance(context, OrchidPipelineStepContext)

    step = _EchoStep()
    result = await step.execute(context)
    assert result.output == {"key": "run-1:echo"}

    observer = _CollectingObserver()
    event = OrchidPipelineEvent(type="pipeline.job.started", run_id="run-1")
    await observer.emit(event)
    assert observer.events == [event]


# ── Errors ───────────────────────────────────────────────────

_PIPELINE_ERRORS = [
    OrchidPipelineConfigError,
    OrchidPipelineNotFoundError,
    OrchidPipelineDisabledError,
    OrchidPipelineAuthError,
    OrchidPipelineInputError,
    OrchidPipelineOutputError,
    OrchidPipelineShapeError,
    OrchidPipelineMappingError,
    OrchidPipelineGraphConflictError,
    OrchidPipelineWriteNotAllowedError,
    OrchidPipelineSerializationError,
    OrchidPipelineLimitError,
    OrchidPipelineTimeoutError,
    OrchidPipelineStepError,
    OrchidPipelineRetryableError,
    OrchidPipelineApprovalRequiredError,
    OrchidPipelineEventsDisabledError,
]


def test_error_hierarchy_and_stable_codes() -> None:
    assert issubclass(OrchidPipelineRetryableError, OrchidPipelineStepError)
    assert issubclass(OrchidPipelineApprovalRequiredError, OrchidPipelineStepError)
    for error_class in _PIPELINE_ERRORS:
        assert issubclass(error_class, OrchidPipelineError)
        assert error_class.code and error_class.code != OrchidPipelineError.code

    codes = [error_class.code for error_class in _PIPELINE_ERRORS]
    assert len(codes) == len(set(codes))


def test_base_error_and_result_error_are_bounded() -> None:
    error = OrchidPipelineError("x" * 5000)
    assert len(str(error)) == 2000

    result = OrchidPipelineResult(
        run_id="r",
        pipeline="p",
        status="failed",
        started_at=START,
        error="x" * 5000,
    )
    assert len(result.error or "") == 2000
    assert len(result.to_dict()["error"]) == 2000

    with pytest.raises(TypeError, match="result error"):
        OrchidPipelineResult(
            run_id="r",
            pipeline="p",
            status="failed",
            started_at=START,
            error=object(),  # type: ignore[arg-type]
        )


def test_step_error_carries_job_metadata_and_bounds_the_message() -> None:
    error = OrchidPipelineStepError("boom" * 1200, job_id="analyze", kind="agent", retryable=True)
    assert isinstance(error, OrchidPipelineError)
    assert error.job_id == "analyze"
    assert error.kind == "agent"
    assert error.retryable is True
    assert len(str(error)) == 2000

    payload = error.to_dict()
    assert payload["code"] == "pipeline_step_error"
    assert payload["job_id"] == "analyze"
    assert len(payload["message"]) == 2000

    retryable = OrchidPipelineRetryableError("transient", job_id="fetch", kind="deterministic")
    assert retryable.retryable is True
    approval = OrchidPipelineApprovalRequiredError("approve", job_id="notify", kind="emit")
    assert approval.retryable is False
