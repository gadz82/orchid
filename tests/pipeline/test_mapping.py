"""Tests for the strict JMESPath renderer (``orchid_ai.pipeline.mapping``)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pytest

from orchid_ai.config.pipeline_templates import validate_expression, validate_template
from orchid_ai.config.schema_pipeline import OrchidPipelineLimitsConfig
from orchid_ai.core.pipeline import (
    PIPELINE_INPUT_OBJECT_ID,
    PIPELINE_INPUT_OBJECT_TYPE,
    OrchidGraphObject,
    OrchidPipelineLimitError,
    OrchidPipelineMappingError,
    OrchidPipelineRunContext,
)
from orchid_ai.pipeline.graph import DataGraph
from orchid_ai.pipeline.mapping import (
    DEFAULT_MAX_RENDERED_CHARS,
    OrchidPipelineMappingContext,
    alias_snapshots,
    build_mapping_context,
    evaluate,
    evaluate_when,
    object_snapshots,
    render_template,
    render_value,
    type_snapshots,
    with_fallback,
)

START = datetime(2026, 11, 17, 9, 0, tzinfo=UTC)


# ── Fixtures / helpers ───────────────────────────────────────


def _run(**overrides: Any) -> OrchidPipelineRunContext:
    values: dict[str, Any] = {
        "run_id": "run-1",
        "pipeline": "assess",
        "pipeline_version": "3",
        "mode": "direct",
        "tenant_key": "tenant-a",
        "user_id": "user-1",
        "started_at": START,
    }
    values.update(overrides)
    return OrchidPipelineRunContext(**values)


def _context(**overrides: Any) -> dict[str, Any]:
    context: dict[str, Any] = {
        "input": {"account_id": "account:1", "period": "2026-Q4"},
        "run": {"id": "run-1", "pipeline": "assess", "mode": "direct"},
        "jobs": {
            "fetch": {"status": "succeeded", "output": {"headline": "Steady"}, "objects": {"account": "account:1"}},
            "report": {"status": "succeeded", "output": {"headline": "Grow"}},
        },
        "objects": {
            "account:1": {
                "id": "account:1",
                "type": "Account",
                "data": {"name": "Acme", "seats": 12},
                "revision": 2,
            }
        },
        "types": {
            "Account": [{"id": "account:1", "type": "Account", "data": {"name": "Acme", "seats": 12}, "revision": 2}]
        },
        "aliases": {
            "account": {"id": "account:1", "type": "Account", "data": {"name": "Acme", "seats": 12}, "revision": 2}
        },
        "summary": "Two accounts assessed.",
    }
    context.update(overrides)
    return context


def _graph() -> DataGraph:
    graph = DataGraph()
    graph.seed(
        [
            OrchidGraphObject(id="account:1", type="Account", data={"name": "Acme", "seats": 12}),
            OrchidGraphObject(id="report:1", type="Report", data={"headline": "Steady"}),
        ]
    )
    return graph


# ── Template syntax ──────────────────────────────────────────


def test_validate_template_accepts_documented_forms() -> None:
    valid = [
        "Plain text without placeholders.",
        "Account {{ input.account_id }} for {{ input.period }}.",
        "{{ aliases.account.data | to_string }}",
        "{{ missing.path || 'fallback' }}",
        "Literal \\{{ input.account_id }} stays literal.",
        "{{ to_string(aliases.account.data) }}",
    ]
    for template in valid:
        assert validate_template(template, path="prompt") == template


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        pytest.param("Account {{ input.account_id", "unbalanced", id="unbalanced-open"),
        pytest.param("Account input.account_id }}", "unbalanced", id="unbalanced-close"),
        pytest.param("Account {{ }}", "empty", id="empty"),
        pytest.param("Account {{ input.{{ nested }} }}", "nested", id="nested"),
        pytest.param("Account {{ input.account_id + 1 }}", "not a valid JMESPath expression", id="python"),
        pytest.param("Account {{ len(input) }}", "only standard JMESPath functions", id="unknown-function"),
    ],
)
def test_validate_template_rejects_bad_templates(template: str, expected: str) -> None:
    with pytest.raises(OrchidPipelineMappingError) as excinfo:
        validate_template(template, path="job 'analyze' prompt")

    message = str(excinfo.value)
    assert expected in message
    assert "job 'analyze' prompt" in message


def test_validate_template_names_the_offending_expression() -> None:
    with pytest.raises(OrchidPipelineMappingError) as excinfo:
        validate_template("Account {{ input.account_id + 1 }}", path="prompt")

    assert "input.account_id + 1" in str(excinfo.value)


def test_validate_expression_accepts_standard_functions() -> None:
    assert validate_expression("sort_by(types.Account, &data.seats)", path="map 'over'")


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        pytest.param("input.account_id + 1", "not a valid JMESPath expression", id="python"),
        pytest.param("__import__('os')", "only standard JMESPath functions", id="unknown-function"),
    ],
)
def test_validate_expression_rejects_bad_expressions(expression: str, expected: str) -> None:
    with pytest.raises(OrchidPipelineMappingError, match=expected):
        validate_expression(expression, path="when")


def test_validate_expression_rejects_empty_input() -> None:
    with pytest.raises(OrchidPipelineMappingError, match="non-empty"):
        validate_expression("   ", path="when")


def test_validate_expression_requires_a_string() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        validate_expression(3, path="when")  # type: ignore[arg-type]


def test_unknown_function_check_degrades_when_the_table_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jmespath.functions import Functions

    monkeypatch.delattr(Functions, "FUNCTION_TABLE")
    assert validate_template("{{ anything(@) }}") == "{{ anything(@) }}"


def test_validate_template_requires_a_string() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        validate_template(42, path="prompt")  # type: ignore[arg-type]


# ── Rendering ────────────────────────────────────────────────


def test_prompt_interpolates_input_and_job_output() -> None:
    template = (
        "Assess {{ input.account_id }} for {{ input.period }}. "
        "Fetch: {{ jobs.fetch.output.headline }} Report: {{ jobs.report.output.headline }}"
    )

    rendered = render_template(template, _context(), path="prompt")

    assert rendered == "Assess account:1 for 2026-Q4. Fetch: Steady Report: Grow"


def test_render_template_without_expressions_returns_the_text() -> None:
    assert render_template("No placeholders here.", _context()) == "No placeholders here."


def test_render_template_trims_expression_whitespace() -> None:
    assert render_template("{{   input.period   }}", _context()) == "2026-Q4"


def test_render_template_renders_aliases_and_types() -> None:
    template = "Name {{ aliases.account.data.name }}; first {{ types.Account[0].data.name }}"

    assert render_template(template, _context()) == "Name Acme; first Acme"


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        pytest.param("`[1, 2]`", "[1,2]", id="list"),
        pytest.param('`{"a": 1}`', '{"a":1}', id="object"),
        pytest.param("`true`", "true", id="boolean"),
        pytest.param("`12.5`", "12.5", id="number"),
        pytest.param("input.account_id", "account:1", id="string"),
    ],
)
def test_render_template_stringifies_values_as_json(expression: str, expected: str) -> None:
    assert render_template("{{ " + expression + " }}", _context()) == expected


def test_render_template_does_not_re_render_results() -> None:
    context = _context(input={"payload": "{{ input.account_id }}"})

    assert render_template("{{ input.payload }}", context) == "{{ input.account_id }}"


def test_missing_path_is_a_strict_error_naming_path_and_expression() -> None:
    with pytest.raises(OrchidPipelineMappingError) as excinfo:
        render_template("Account {{ input.missing }}", _context(), path="job 'analyze' prompt")

    message = str(excinfo.value)
    assert "job 'analyze' prompt" in message
    assert "input.missing" in message


def test_explicit_null_is_treated_as_missing() -> None:
    context = _context(input={"value": None})

    with pytest.raises(OrchidPipelineMappingError, match="resolved to no value"):
        render_template("{{ input.value }}", context)

    assert render_template("{{ input.value }}", context, strict=False, default="n/a") == "n/a"


def test_non_strict_render_returns_the_default_and_logs_a_bounded_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("pipeline-mapping-test")
    caplog.set_level(logging.WARNING, logger=logger.name)

    rendered = render_template(
        "Account {{ input.missing }}",
        _context(),
        path="when",
        strict=False,
        default="unknown",
        logger=logger,
    )

    assert rendered == "Account unknown"
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.getMessage().startswith("pipeline mapping: ")
    assert "input.missing" in record.getMessage()
    assert len(record.getMessage()) <= 300


def test_non_strict_warning_is_truncated_for_long_paths(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("pipeline-mapping-long")
    caplog.set_level(logging.WARNING, logger=logger.name)

    render_template("{{ input.missing }}", _context(), path="x" * 500, strict=False, logger=logger)

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert message.endswith("…")
    assert len(message) < 400


def test_escaped_braces_stay_literal() -> None:
    template = "Use \\{{ input.account_id }} literally, then {{ input.account_id }}"

    assert render_template(template, _context()) == "Use {{ input.account_id }} literally, then account:1"


def test_escaped_braces_are_not_evaluated_even_when_the_path_is_missing() -> None:
    assert render_template("\\{{ input.missing }}", _context()) == "{{ input.missing }}"


def test_escaped_open_without_closing_braces_is_a_literal() -> None:
    assert render_template("\\{{ alone", _context()) == "{{ alone"


def test_fallback_operator_uses_the_jmespath_or_expression() -> None:
    assert render_template("{{ input.missing || 'fallback' }}", _context()) == "fallback"
    assert render_template("{{ input.account_id || 'fallback' }}", _context()) == "account:1"


def test_with_fallback_builds_a_valid_expression() -> None:
    expression = with_fallback("input.missing", "unknown")

    assert evaluate(expression, _context(), path="prompt") == "unknown"


def test_with_fallback_result_can_be_embedded_in_a_template() -> None:
    template = "{{ " + with_fallback("input.missing", "unknown") + " }}"

    assert render_template(template, _context()) == "unknown"


@pytest.mark.parametrize(
    ("fallback", "expected"),
    [
        pytest.param("text", "text", id="string"),
        pytest.param(3, 3, id="number"),
        pytest.param(False, False, id="boolean"),
        pytest.param({"a": [1]}, {"a": [1]}, id="object"),
        pytest.param("back`tick", "back`tick", id="backtick"),
    ],
)
def test_with_fallback_supports_any_json_literal(fallback: Any, expected: Any) -> None:
    assert evaluate(with_fallback("input.missing", fallback), _context()) == expected


def test_with_fallback_rejects_an_empty_expression() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        with_fallback("  ", "x")


def test_render_template_enforces_the_size_cap() -> None:
    with pytest.raises(OrchidPipelineLimitError) as excinfo:
        render_template("{{ input.period }}", _context(), path="prompt", max_chars=4)

    message = str(excinfo.value)
    assert "prompt" in message
    assert "4-character cap" in message


def test_render_template_size_cap_counts_literal_text() -> None:
    with pytest.raises(OrchidPipelineLimitError, match="4-character cap"):
        render_template("abcdef", _context(), max_chars=4)


def test_render_template_can_opt_out_of_the_cap() -> None:
    assert render_template("abcdef", _context(), max_chars=None) == "abcdef"


def test_default_cap_matches_the_configuration_default() -> None:
    assert DEFAULT_MAX_RENDERED_CHARS == OrchidPipelineLimitsConfig().max_prompt_chars


@pytest.mark.parametrize("max_chars", [0, -1, True, 1.5])
def test_render_template_rejects_invalid_caps(max_chars: Any) -> None:
    with pytest.raises(ValueError, match="max_chars"):
        render_template("text", _context(), max_chars=max_chars)


def test_size_cap_stops_before_building_a_large_string() -> None:
    template = "{{ input.account_id }}" * 1000

    with pytest.raises(OrchidPipelineLimitError):
        render_template(template, _context(), max_chars=100)


def test_python_expressions_are_rejected_at_render_time() -> None:
    with pytest.raises(OrchidPipelineMappingError, match="not a valid JMESPath expression"):
        render_template("{{ input.account_id + 1 }}", _context(), path="prompt")


def test_no_eval_guarantee(tmp_path: Any) -> None:
    target = tmp_path / "should-not-exist"
    payload = "{{ open('" + str(target) + "', 'w') }}"

    with pytest.raises(OrchidPipelineMappingError, match="only standard JMESPath functions"):
        render_template(payload, _context(), path="prompt")

    assert not target.exists()


def test_render_template_requires_a_string_template() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        render_template(1, _context())  # type: ignore[arg-type]


def test_render_template_reports_a_non_json_value() -> None:
    with pytest.raises(OrchidPipelineMappingError, match="not JSON-serializable"):
        render_template("{{ weird }}", {"weird": {1, 2}}, path="prompt")


# ── Expression evaluation ────────────────────────────────────


def test_evaluate_returns_native_values() -> None:
    context = _context()

    assert evaluate("input.period", context) == "2026-Q4"
    assert evaluate("jobs.fetch.output", context) == {"headline": "Steady"}
    assert evaluate("types.Account[0].data.seats", context) == 12


def test_evaluate_accepts_the_context_value_object() -> None:
    context = build_mapping_context(input={"period": "2026-Q4"}, run=_run())

    assert evaluate("run.pipeline", context) == "assess"
    assert evaluate("input.period", context) == "2026-Q4"


def test_evaluate_rejects_a_non_mapping_context() -> None:
    with pytest.raises(TypeError, match="context must be"):
        evaluate("input.period", "not-a-context")  # type: ignore[arg-type]


def test_evaluate_requires_a_non_empty_expression() -> None:
    with pytest.raises(OrchidPipelineMappingError, match="non-empty"):
        evaluate("   ", _context(), path="when")


def test_evaluate_requires_a_string_expression() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        evaluate(1, _context())  # type: ignore[arg-type]


def test_evaluate_wraps_jmespath_runtime_failures() -> None:
    with pytest.raises(OrchidPipelineMappingError, match="failed"):
        evaluate("length(input.account_id, input.period)", _context(), path="when")


# ── render_value ─────────────────────────────────────────────


def test_render_value_preserves_native_types_for_single_expressions() -> None:
    rendered = render_value(
        {
            "id": "account:{{ input.account_id }}",
            "output": "{{ jobs.fetch.output }}",
            "seats": "{{ types.Account[0].data.seats }}",
            "flag": "{{ `true` }}",
            "literal": "unchanged",
        },
        _context(),
    )

    assert rendered == {
        "id": "account:account:1",
        "output": {"headline": "Steady"},
        "seats": 12,
        "flag": True,
        "literal": "unchanged",
    }


def test_render_value_recurses_through_lists_and_mappings() -> None:
    rendered = render_value(
        {"items": ["{{ input.account_id }}", {"period": "{{ input.period }}"}, 3]},
        _context(),
        path="payload",
    )

    assert rendered == {"items": ["account:1", {"period": "2026-Q4"}, 3]}


def test_render_value_renders_mixed_text() -> None:
    assert render_value("Account {{ input.account_id }}", _context()) == "Account account:1"


def test_render_value_passes_scalars_through() -> None:
    assert render_value(3, _context()) == 3
    assert render_value(None, _context()) is None
    assert render_value(b"raw", _context()) == b"raw"


def test_render_value_non_strict_uses_the_default() -> None:
    assert render_value("{{ input.missing }}", _context(), strict=False, default=None) is None


def test_render_value_is_uncapped_by_default() -> None:
    big = "x" * (DEFAULT_MAX_RENDERED_CHARS + 1)

    assert render_value("{{ input.big }}", {"input": {"big": big}}) == big


def test_render_value_enforces_an_explicit_cap_on_the_native_path() -> None:
    with pytest.raises(OrchidPipelineLimitError, match="10-character cap"):
        render_value("{{ input.big }}", {"input": {"big": "x" * 100}}, path="writes", max_chars=10)


def test_render_value_enforces_an_explicit_cap_on_the_text_path() -> None:
    with pytest.raises(OrchidPipelineLimitError, match="10-character cap"):
        render_value("v={{ input.big }}", {"input": {"big": "x" * 100}}, path="writes", max_chars=10)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param({"name": "Acme", "seats": 12}, id="object"),
        pytest.param(list(range(50)), id="array"),
    ],
)
def test_render_value_caps_native_containers_when_asked(value: Any) -> None:
    with pytest.raises(OrchidPipelineLimitError, match="10-character cap"):
        render_value("{{ input.value }}", {"input": {"value": value}}, path="writes", max_chars=10)


def test_render_value_rejects_an_invalid_cap() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        render_value("text", _context(), max_chars=0)


# ── when gates ───────────────────────────────────────────────


def test_when_missing_path_is_false_and_logs_a_bounded_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("pipeline-when-test")
    caplog.set_level(logging.WARNING, logger=logger.name)

    assert evaluate_when("input.missing", _context(), path="job 'gate' when", logger=logger) is False
    assert len(caplog.records) == 1
    assert "input.missing" in caplog.records[0].getMessage()
    assert len(caplog.records[0].getMessage()) <= 300


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        pytest.param("`true`", True, id="true"),
        pytest.param("`false`", False, id="false"),
        pytest.param("`0`", False, id="zero"),
        pytest.param("`[]`", False, id="empty-list"),
        pytest.param("`{}`", False, id="empty-object"),
        pytest.param("input.account_id", True, id="non-empty-string"),
        pytest.param("types.Account", True, id="non-empty-list"),
    ],
)
def test_when_uses_jmespath_truthiness(expression: str, expected: bool) -> None:
    assert evaluate_when(expression, _context()) is expected


def test_when_accepts_a_single_expression_template() -> None:
    assert evaluate_when("{{ input.account_id }}", _context()) is True
    assert evaluate_when("{{ input.missing }}", _context()) is False


def test_when_requires_a_string_expression() -> None:
    with pytest.raises(TypeError, match="expression must be a string"):
        evaluate_when(3, _context(), path="when")  # type: ignore[arg-type]


def test_when_rejects_a_non_jmespath_expression() -> None:
    with pytest.raises(OrchidPipelineMappingError, match="not a valid JMESPath expression"):
        evaluate_when("input.account_id + 1", _context(), path="when")


def test_when_treats_an_escaped_gate_as_a_template() -> None:
    # The configuration validator classifies ``when`` by delimiters, so the gate
    # evaluator must use the same rule or an accepted gate fails at render time.
    assert evaluate_when("\\{{ literal }}", _context(), path="when") is True


def test_when_is_not_subject_to_the_prompt_size_cap() -> None:
    assert evaluate_when("y{{ input.big }}", {"input": {"big": "x" * 40_000}}) is True


# ── Context builders ─────────────────────────────────────────


def test_build_mapping_context_exposes_the_documented_keys() -> None:
    context = build_mapping_context(input={"account_id": "account:1"}, run=_run(), jobs={"fetch": {}})

    assert set(context.to_dict()) == {"input", "run", "jobs", "objects", "types", "aliases", "summary"}
    assert context.input == {"account_id": "account:1"}
    assert context.jobs == {"fetch": {}}
    assert context.objects == {}
    assert context.types == {}
    assert context.aliases == {}
    assert context.summary == ""


def test_run_snapshot_exposes_run_id_as_id() -> None:
    context = build_mapping_context(input={}, run=_run(correlation_id="corr-1"))

    assert context.run["id"] == "run-1"
    assert context.run["pipeline"] == "assess"
    assert context.run["mode"] == "direct"
    assert context.run["started_at"] == START.isoformat()
    assert context.run["correlation_id"] == "corr-1"
    assert "run_id" not in context.run


def test_run_snapshot_ignores_non_json_metadata() -> None:
    context = build_mapping_context(input={}, run=_run(metadata={"handle": object()}))

    assert "metadata" not in context.run


def test_run_mapping_is_filtered_to_the_documented_keys() -> None:
    context = build_mapping_context(
        input={},
        run={
            "id": "run-9",
            "pipeline": "custom",
            "tenant_key": "tenant-a",
            "user_id": "user-1",
            "metadata": {"api_key": "SECRET"},
        },
    )

    assert context.run == {"id": "run-9", "pipeline": "custom"}


def test_run_snapshot_renames_a_mapping_run_id() -> None:
    context = build_mapping_context(input={}, run={"run_id": "run-9", "pipeline": "custom"})

    assert context.run == {"id": "run-9", "pipeline": "custom"}


def test_run_snapshot_hides_metadata_and_identifiers() -> None:
    context = build_mapping_context(
        input={},
        run=_run(metadata={"api_key": "SECRET"}, tenant_key="tenant-a", user_id="user-1"),
    )

    assert "metadata" not in context.run
    assert "tenant_key" not in context.run
    assert "user_id" not in context.run
    with pytest.raises(OrchidPipelineMappingError, match="resolved to no value"):
        render_template("{{ run.metadata.api_key }}", context)


def test_build_mapping_context_accepts_a_run_mapping() -> None:
    context = build_mapping_context(input={}, run={"id": "run-9", "pipeline": "custom"})

    assert context.run == {"id": "run-9", "pipeline": "custom"}


def test_objects_types_and_aliases_come_from_the_graph() -> None:
    graph = _graph()
    view = graph.view()
    graph.commit("job", aliases={"account": "account:1"})
    context = build_mapping_context(input={}, run=_run(), graph=view, aliases=graph.aliases)

    assert set(context.objects) == {"account:1", "report:1"}
    assert [snapshot["id"] for snapshot in context.types["Account"]] == ["account:1"]
    assert context.aliases["account"]["id"] == "account:1"
    assert context.aliases["account"]["revision"] == 1


def test_context_snapshots_are_detached_from_the_graph() -> None:
    context = build_mapping_context(input={}, run=_run(), graph=_graph().view())

    context.objects["account:1"]["data"]["name"] = "Changed"

    assert context.types["Account"][0]["data"]["name"] == "Acme"
    assert context.aliases == {}


def test_reserved_input_object_is_excluded_from_objects_and_types() -> None:
    graph = _graph()
    graph.set_input({"account_id": "account:1"})
    context = build_mapping_context(input={"account_id": "account:1"}, run=_run(), graph=graph.view())

    assert PIPELINE_INPUT_OBJECT_ID not in context.objects
    assert PIPELINE_INPUT_OBJECT_TYPE not in context.types
    assert evaluate("input.account_id", context) == "account:1"


def test_dangling_alias_is_skipped() -> None:
    context = build_mapping_context(
        input={},
        run=_run(),
        graph=_graph().view(),
        aliases={"ghost": "missing:1"},
    )

    assert context.aliases == {}


def test_object_snapshots_returns_empty_without_a_graph() -> None:
    assert object_snapshots(None) == {}


def test_type_snapshots_skips_snapshots_without_a_type() -> None:
    objects = {
        "a": {"id": "a", "type": "Account", "data": {}},
        "b": {"id": "b", "data": {}},
    }

    assert set(type_snapshots(objects)) == {"Account"}


def test_alias_snapshots_copies_the_snapshot() -> None:
    objects = {"a": {"id": "a", "type": "Account", "data": {}}}
    resolved = alias_snapshots({"account": "a"}, objects)

    resolved["account"]["data"]["x"] = 1

    assert objects["a"]["data"] == {}


def test_mapping_context_rejects_non_mapping_fields() -> None:
    with pytest.raises(TypeError, match="context.input"):
        OrchidPipelineMappingContext(input="no", run={})  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="context.jobs"):
        OrchidPipelineMappingContext(input={}, run={}, jobs="no")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="context.summary"):
        OrchidPipelineMappingContext(input={}, run={}, summary=1)  # type: ignore[arg-type]


def test_build_mapping_context_requires_mapping_input_and_run() -> None:
    with pytest.raises(TypeError, match="input must be a mapping"):
        build_mapping_context(input="no", run=_run())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="run must be a mapping"):
        build_mapping_context(input={}, run="no")  # type: ignore[arg-type]
