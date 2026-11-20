"""Configuration schema tests for the ``pipelines:`` block."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from orchid_ai.config.schema_agent import OrchidAgentsConfig, _default_pipeline_tool_name
from orchid_ai.config.schema_pipeline import (
    OrchidPipelineAgentJobConfig,
    OrchidPipelineDecisionJobConfig,
    OrchidPipelineDeterministicJobConfig,
    OrchidPipelineEmitJobConfig,
    OrchidPipelineFieldConfig,
    OrchidPipelineLLMJobConfig,
    OrchidPipelineMapJobConfig,
    OrchidPipelineObjectWriteConfig,
    OrchidPipelinesConfig,
    OrchidPipelineShapeConfig,
    OrchidPipelineSubPipelineJobConfig,
)
from orchid_ai.pipeline.mapping import render_template

# ── Fixtures / helpers ───────────────────────────────────────


def _base_pipeline() -> dict[str, Any]:
    """A valid pipeline exercising every job kind this prompt validates."""
    return {
        "shapes": {
            "Account": {
                "fields": {
                    "id": {"type": "string", "required": True},
                    "name": {"type": "string"},
                }
            },
            "Assessment": {
                "fields": {
                    "score": {"type": "number", "required": True},
                    "band": {"type": "string", "enum": ["low", "high"], "default": "low"},
                }
            },
            "Plan": {"fields": {"action": {"type": "string", "required": True}}},
            "Report": {"fields": {"headline": {"type": "string"}}},
            "PipelineInput": {"fields": {"account_id": {"type": "string", "required": True}}},
            "PipelineOutput": {"fields": {"summary": {"type": "string"}}},
        },
        "relations": {
            "has_assessment": {"from": ["Account"], "to": ["Assessment"]},
            "has_plan": {"from": ["Account"], "to": ["Plan"]},
        },
        "definitions": {
            "assess": {
                "input": {"shape": "PipelineInput"},
                "output": {"shape": "PipelineOutput", "mapping": {"summary": "summary"}},
                "interpretation": {"mode": "template", "template": "Run {{ run.id }}"},
                "expose": {
                    "as_tool": True,
                    "tool_name": "pipeline_assess",
                    "description": "Assess an account and propose an action.",
                },
                "jobs": {
                    "load": {
                        "kind": "deterministic",
                        "uses": "app.steps.load",
                        "writes": [
                            {
                                "id": "account:{{ input.account_id }}",
                                "shape": "Account",
                                "as": "account",
                            }
                        ],
                    },
                    "analyze": {
                        "kind": "agent",
                        "agent": "analyst",
                        "depends_on": ["load"],
                        "prompt": "Analyze {{ input.account_id }}",
                        "output": {"shape": "Assessment"},
                        "writes": [
                            {
                                "id": "assessment:{{ input.account_id }}",
                                "shape": "Assessment",
                                "from": "jobs.analyze.output",
                                "as": "assessment",
                            }
                        ],
                        "links": [
                            {
                                "type": "has_assessment",
                                "from": "account:{{ input.account_id }}",
                                "to": "assessment:{{ input.account_id }}",
                            }
                        ],
                    },
                    "choose": {
                        "kind": "decision",
                        "decision_model": "triage",
                        "depends_on": ["analyze"],
                        "question": "Which action?",
                        "labels": ["retain", "expand"],
                        "branches": {"retain": "draft_retain", "expand": "draft_expand"},
                        "fallback": "retain",
                    },
                    "draft_retain": {
                        "kind": "llm",
                        "model": "acme/model",
                        "prompt": "Draft a retention plan for {{ input.account_id }}",
                        "output": {"shape": "Plan"},
                        "writes": [
                            {
                                "id": "action:{{ input.account_id }}",
                                "shape": "Plan",
                                "from": "jobs.draft_retain.output",
                                "as": "action",
                            }
                        ],
                    },
                    "draft_expand": {
                        "kind": "llm",
                        "model": "acme/model",
                        "prompt": "Draft an expansion plan for {{ input.account_id }}",
                        "output": {"shape": "Plan"},
                        "writes": [
                            {
                                "id": "action:{{ input.account_id }}",
                                "shape": "Plan",
                                "from": "jobs.draft_expand.output",
                                "as": "action",
                            }
                        ],
                    },
                    "report": {
                        "kind": "llm",
                        "model": "acme/model",
                        "depends_on": ["draft_retain", "draft_expand"],
                        "join": "any",
                        "prompt": "Write the report.",
                        "tools": ["search"],
                        "output": {"shape": "Report"},
                        "writes": [
                            {
                                "id": "report:{{ input.account_id }}",
                                "shape": "Report",
                                "from": "jobs.report.output",
                                "as": "report",
                            }
                        ],
                    },
                    "notify": {
                        "kind": "emit",
                        "depends_on": ["report"],
                        "signal": "report.ready",
                        "payload": {"report_id": "{{ input.account_id }}"},
                        "dedupe_key": "{{ run.id }}",
                    },
                },
            }
        },
    }


def _root_config(pipelines: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "agents": {"analyst": {"description": "Analyst", "prompt": "Analyze."}},
        "tools": {"search": {"handler": "app.tools.search"}},
        "decision_models": {"triage": {"protocol": "stub"}},
        "pipelines": _base_pipeline() if pipelines is None else pipelines,
    }


def _jobs(config: dict[str, Any]) -> dict[str, Any]:
    return config["definitions"]["assess"]["jobs"]


def _rejects(config: dict[str, Any], expected: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        OrchidPipelinesConfig.model_validate(config)
    assert expected in str(excinfo.value)


def _accepts(config: dict[str, Any]) -> OrchidPipelinesConfig:
    return OrchidPipelinesConfig.model_validate(config)


# ── Acceptance ───────────────────────────────────────────────


def test_full_definition_parses_with_typed_jobs() -> None:
    config = _accepts(_base_pipeline())
    definition = config.definitions["assess"]
    jobs = definition.jobs

    assert config.enabled is True
    assert isinstance(jobs["load"], OrchidPipelineDeterministicJobConfig)
    assert isinstance(jobs["analyze"], OrchidPipelineAgentJobConfig)
    assert isinstance(jobs["choose"], OrchidPipelineDecisionJobConfig)
    assert isinstance(jobs["draft_retain"], OrchidPipelineLLMJobConfig)
    assert isinstance(jobs["report"], OrchidPipelineLLMJobConfig)
    assert isinstance(jobs["notify"], OrchidPipelineEmitJobConfig)
    assert jobs["report"].join == "any"
    assert definition.expose.tool_name == "pipeline_assess"
    assert definition.async_.enabled is False
    assert definition.interpretation.mode == "template"


def test_config_json_roundtrip_is_stable() -> None:
    first = _accepts(_base_pipeline())
    payload = first.model_dump(mode="json", by_alias=True)
    second = OrchidPipelinesConfig.model_validate(payload)
    assert second.model_dump(mode="json", by_alias=True) == payload


def test_empty_block_is_valid_and_disabled_still_validates() -> None:
    assert OrchidPipelinesConfig().definitions == {}

    config = _base_pipeline()
    config["enabled"] = False
    config["definitions"]["assess"]["jobs"]["load"].pop("uses")
    _rejects(config, "uses")


def test_mutually_exclusive_branches_may_share_an_alias() -> None:
    # ``draft_retain`` and ``draft_expand`` both bind ``action`` and are never
    # scheduled together, so the duplicate-alias rule exempts them.
    _accepts(_base_pipeline())


def test_mutually_exclusive_branches_may_write_the_same_static_object() -> None:
    config = _base_pipeline()
    _jobs(config)["draft_retain"]["writes"][0]["id"] = "action:static"
    _jobs(config)["draft_expand"]["writes"][0]["id"] = "action:static"
    _accepts(config)


def test_fallback_label_without_a_branch_is_valid() -> None:
    # A fallback may name a label with no branch entry; the router then has no
    # conditional edge and the run falls through to the finish node.
    config = _base_pipeline()
    _m_decision_fallback_label_without_branch(config)
    parsed = _accepts(config)
    assert parsed.definitions["assess"].jobs["choose"].fallback == "escalate"


def test_decision_without_branches_is_valid() -> None:
    config = _base_pipeline()
    jobs = _jobs(config)
    for job_id in ("analyze", "choose", "draft_retain", "draft_expand", "report", "notify"):
        jobs.pop(job_id)
    jobs["decide"] = {
        "kind": "decision",
        "decision_model": "triage",
        "depends_on": ["load"],
        "question": "Which?",
        "labels": ["a", "b"],
    }
    parsed = _accepts(config)
    assert parsed.definitions["assess"].jobs["decide"].branches == {}


def test_fallback_job_is_mutually_exclusive_with_branch_targets() -> None:
    config = _base_pipeline()
    jobs = _jobs(config)
    for job_id in ("draft_retain", "draft_expand"):
        jobs[job_id]["writes"][0]["id"] = "action:static"
    jobs["choose"]["fallback"] = "default_action"
    jobs["default_action"] = {
        "kind": "deterministic",
        "uses": "app.steps.default_action",
        "writes": [{"id": "action:static", "shape": "Plan", "as": "action"}],
    }
    _accepts(config)


def test_merge_from_db_reruns_pipeline_validation() -> None:
    config = OrchidAgentsConfig.model_validate(_root_config())
    before = config.agents["analyst"]

    with pytest.raises(ValueError, match="use multiple jobs or a sub-pipeline"):
        config.merge_from_db(
            [{"name": "analyst", "config": {"children": {"kid": {"description": "", "prompt": ""}}}}],
            strict=False,
        )

    assert config.agents["analyst"] is before
    assert config.agents["analyst"].children is None


def test_default_exposed_tool_name_is_sanitized_and_truncated() -> None:
    long_name = "p" + "a" * 61
    generated = _default_pipeline_tool_name(long_name)
    assert len(generated) == 64
    assert generated.startswith("pipeline_")

    config = _root_config()
    config["pipelines"]["definitions"][long_name] = {
        "jobs": {"run": {"kind": "deterministic", "uses": "app.steps.run"}},
        "expose": {"as_tool": True, "description": "A long pipeline name."},
    }
    assert OrchidAgentsConfig.model_validate(config).pipelines is not None


def test_reserved_input_substring_in_object_id_is_allowed() -> None:
    config = _base_pipeline()
    _jobs(config)["load"]["writes"][0]["id"] = "account:__input__:1"
    _accepts(config)


def test_defaults_timeout_is_not_a_ceiling() -> None:
    config = _base_pipeline()
    config["defaults"] = {"timeout_seconds": 1.0}
    config["definitions"]["assess"]["limits"] = {"timeout_seconds": 300.0}
    _jobs(config)["load"]["timeout_seconds"] = 100.0
    _accepts(config)


def test_safe_quantified_alternation_pattern_is_allowed() -> None:
    config = _base_pipeline()
    config["definitions"]["assess"]["input"] = {"fields": {"x": {"type": "string", "pattern": "(foo|bar)+"}}}
    _accepts(config)


def test_allow_untyped_relations_toggle() -> None:
    config = _base_pipeline()
    _jobs(config)["notify"]["links"] = [
        {"type": "custom_edge", "from": "account:{{ input.account_id }}", "to": "report:{{ input.account_id }}"}
    ]
    _rejects(config, "links unknown relation 'custom_edge'")

    config["allow_untyped_relations"] = True
    _accepts(config)


def test_allow_parallel_writes_toggle() -> None:
    config = _base_pipeline()
    _jobs(config)["load"]["writes"][0]["id"] = "shared:1"
    _jobs(config)["other"] = {
        "kind": "deterministic",
        "uses": "app.steps.other",
        "writes": [{"id": "shared:1", "shape": "Account"}],
    }
    _rejects(config, "both write static object 'shared:1'")

    config["allow_parallel_writes"] = True
    _accepts(config)


def test_class_shape_input_defaults_are_not_field_checked() -> None:
    # Class-based shapes resolve their fields at build time, so default-key
    # existence cannot be checked without importing the class.
    config = _base_pipeline()
    config["shapes"]["ExternalInput"] = {"class": "app.shapes.ExternalInput"}
    config["definitions"]["assess"]["input"] = {"shape": "ExternalInput", "defaults": {"anything": 1}}
    _accepts(config)


def test_map_and_subpipeline_jobs_parse() -> None:
    config = _base_pipeline()
    config["definitions"]["child"] = {"jobs": {"work": {"kind": "deterministic", "uses": "app.steps.work"}}}
    _jobs(config)["enrich"] = {
        "kind": "map",
        "over": "types.Account",
        "as": "account",
        "job": "decorate",
        "depends_on": ["load"],
    }
    _jobs(config)["decorate"] = {"kind": "deterministic", "uses": "app.steps.decorate"}
    _jobs(config)["call_child"] = {
        "kind": "subpipeline",
        "pipeline": "child",
        "input": {"account_id": "{{ input.account_id }}"},
    }
    _jobs(config)["custom"] = {"kind": "custom", "class": "app.steps.Custom", "params": {"x": 1}}

    parsed = _accepts(config)
    assert isinstance(parsed.definitions["assess"].jobs["enrich"], OrchidPipelineMapJobConfig)
    assert isinstance(parsed.definitions["assess"].jobs["call_child"], OrchidPipelineSubPipelineJobConfig)
    assert parsed.shapes["Account"].fields is not None


# ── Rejection matrix ─────────────────────────────────────────


def _m_bad_pipeline_name(config: dict[str, Any]) -> None:
    config["definitions"]["BadName"] = config["definitions"].pop("assess")


def _m_no_jobs(config: dict[str, Any]) -> None:
    _jobs(config).clear()


def _m_too_many_jobs(config: dict[str, Any]) -> None:
    config["limits"] = {"max_jobs": 2}


def _m_unknown_dependency(config: dict[str, Any]) -> None:
    _jobs(config)["report"]["depends_on"] = ["nope"]
    _jobs(config)["report"]["join"] = "all"


def _m_self_dependency(config: dict[str, Any]) -> None:
    _jobs(config)["load"]["depends_on"] = ["load"]


def _m_cycle(config: dict[str, Any]) -> None:
    _jobs(config)["load"]["depends_on"] = ["analyze"]


def _m_join_any_too_few(config: dict[str, Any]) -> None:
    _jobs(config)["report"]["depends_on"] = ["draft_retain"]
    _jobs(config)["report"]["join"] = "any"


def _m_input_both(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["input"] = {
        "shape": "PipelineInput",
        "fields": {"account_id": {"type": "string"}},
    }


def _m_input_neither(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["input"] = {}


def _m_input_unknown_shape(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["input"] = {"shape": "Nope"}


def _m_input_default_unknown_field(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["input"]["defaults"] = {"nope": 1}


def _m_output_no_projection(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["output"] = {"shape": "PipelineOutput"}


def _m_output_unknown_shape(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["output"] = {"shape": "Nope", "mapping": {"summary": "summary"}}


def _m_output_bad_mapping_expression(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["output"]["mapping"] = {"summary": "input["}


def _m_write_unknown_shape(config: dict[str, Any]) -> None:
    _jobs(config)["load"]["writes"][0]["shape"] = "Nope"


def _m_write_reserved_input(config: dict[str, Any]) -> None:
    _jobs(config)["load"]["writes"][0]["id"] = "__input__"


def _m_write_bad_alias(config: dict[str, Any]) -> None:
    _jobs(config)["load"]["writes"][0]["as"] = "1account"


def _m_write_bad_when(config: dict[str, Any]) -> None:
    _jobs(config)["load"]["writes"][0]["when"] = "input["


def _m_link_unknown_relation(config: dict[str, Any]) -> None:
    _jobs(config)["analyze"]["links"][0]["type"] = "unknown_edge"


def _m_relation_unknown_endpoint(config: dict[str, Any]) -> None:
    config["relations"]["bad"] = {"from": ["Account"], "to": ["Nope"]}


def _m_shape_cycle(config: dict[str, Any]) -> None:
    config["shapes"]["NodeA"] = {"fields": {"next": {"ref": "NodeB"}}}
    config["shapes"]["NodeB"] = {"fields": {"next": {"ref": "NodeA"}}}


def _m_shape_ref_unknown(config: dict[str, Any]) -> None:
    config["shapes"]["Account"]["fields"]["owner"] = {"ref": "Nope"}


def _m_shape_fields_and_class(config: dict[str, Any]) -> None:
    config["shapes"]["Account"]["class"] = "app.shapes.Account"


def _m_field_no_type(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["input"] = {"fields": {"x": {"description": "missing type"}}}


def _m_field_array_without_items(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["input"] = {"fields": {"x": {"type": "array"}}}


def _m_field_bad_pattern(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["input"] = {"fields": {"x": {"type": "string", "pattern": "("}}}


def _m_field_default_mismatch(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["input"] = {"fields": {"x": {"type": "integer", "default": "not-an-int"}}}


def _m_field_additional_properties_on_scalar(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["input"] = {"fields": {"x": {"type": "string", "additional_properties": True}}}


def _m_decision_no_labels(config: dict[str, Any]) -> None:
    _jobs(config)["choose"]["labels"] = []


def _m_decision_duplicate_labels(config: dict[str, Any]) -> None:
    _jobs(config)["choose"]["labels"] = ["retain", "retain"]
    _jobs(config)["choose"]["branches"] = {"retain": "draft_retain"}


def _m_decision_unknown_branch(config: dict[str, Any]) -> None:
    _jobs(config)["choose"]["branches"]["retain"] = "nope"


def _m_decision_unknown_fallback(config: dict[str, Any]) -> None:
    _jobs(config)["choose"]["fallback"] = "nope"


def _m_decision_fallback_label_without_branch(config: dict[str, Any]) -> None:
    _jobs(config)["choose"]["labels"] = ["retain", "expand", "escalate"]
    _jobs(config)["choose"]["fallback"] = "escalate"


def _m_decision_branch_self_target(config: dict[str, Any]) -> None:
    _jobs(config)["choose"]["branches"]["retain"] = "choose"


def _m_decision_join_all_over_branches(config: dict[str, Any]) -> None:
    _jobs(config)["report"]["join"] = "all"


def _m_branch_target_depends_on_decision(config: dict[str, Any]) -> None:
    _jobs(config)["draft_retain"]["depends_on"] = ["choose"]


def _m_branch_target_feeds_join_all(config: dict[str, Any]) -> None:
    _jobs(config)["after_retain"] = {
        "kind": "deterministic",
        "uses": "app.steps.after",
        "depends_on": ["draft_retain"],
    }


def _m_map_unknown_template(config: dict[str, Any]) -> None:
    _jobs(config)["enrich"] = {
        "kind": "map",
        "over": "types.Account",
        "as": "account",
        "job": "nope",
    }


def _m_map_nested(config: dict[str, Any]) -> None:
    _jobs(config)["enrich"] = {
        "kind": "map",
        "over": "types.Account",
        "as": "after",
        "job": "outer",
    }
    _jobs(config)["outer"] = {
        "kind": "map",
        "over": "types.Account",
        "as": "account",
        "job": "load",
    }


def _m_map_bad_over(config: dict[str, Any]) -> None:
    _jobs(config)["enrich"] = {
        "kind": "map",
        "over": "types[",
        "as": "account",
        "job": "load",
    }


def _m_map_bad_as(config: dict[str, Any]) -> None:
    _jobs(config)["enrich"] = {
        "kind": "map",
        "over": "types.Account",
        "as": "1account",
        "job": "load",
    }


def _m_map_reserved_as(config: dict[str, Any]) -> None:
    _jobs(config)["enrich"] = {
        "kind": "map",
        "over": "types.Account",
        "as": "input",
        "job": "load",
    }


def _m_map_decision_template(config: dict[str, Any]) -> None:
    _jobs(config)["enrich"] = {
        "kind": "map",
        "over": "types.Account",
        "as": "account",
        "job": "choose",
    }


def _m_subpipeline_unknown(config: dict[str, Any]) -> None:
    _jobs(config)["call"] = {"kind": "subpipeline", "pipeline": "nope"}


def _m_subpipeline_cycle(config: dict[str, Any]) -> None:
    config["definitions"]["alpha"] = {"jobs": {"call": {"kind": "subpipeline", "pipeline": "beta"}}}
    config["definitions"]["beta"] = {"jobs": {"call": {"kind": "subpipeline", "pipeline": "alpha"}}}


def _m_subpipeline_depth(config: dict[str, Any]) -> None:
    config["limits"] = {"max_subpipeline_depth": 2}
    for index in range(1, 5):
        config["definitions"][f"p{index}"] = {"jobs": {"run": {"kind": "deterministic", "uses": "app.steps.run"}}}
    for index in range(1, 4):
        config["definitions"][f"p{index}"]["jobs"] = {"call": {"kind": "subpipeline", "pipeline": f"p{index + 1}"}}


def _m_bad_uses(config: dict[str, Any]) -> None:
    _jobs(config)["load"]["uses"] = "notdotted"


def _m_bad_custom_class(config: dict[str, Any]) -> None:
    _jobs(config)["custom"] = {"kind": "custom", "class": "notdotted"}


def _m_duplicate_alias(config: dict[str, Any]) -> None:
    _jobs(config)["report"]["writes"][0]["as"] = "assessment"


def _m_bad_prompt_template(config: dict[str, Any]) -> None:
    _jobs(config)["analyze"]["prompt"] = "Analyze {{ input.account_id"


def _m_empty_prompt_template(config: dict[str, Any]) -> None:
    _jobs(config)["analyze"]["prompt"] = "Analyze {{ }}"


def _m_python_expression_prompt_template(config: dict[str, Any]) -> None:
    _jobs(config)["analyze"]["prompt"] = "Analyze {{ input.account_id + 1 }}"


def _m_unknown_function_prompt_template(config: dict[str, Any]) -> None:
    _jobs(config)["analyze"]["prompt"] = "Analyze {{ __import__('os') }}"


def _m_interpretation_llm_without_model(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["interpretation"] = {"mode": "llm"}


def _m_interpretation_template_without_template(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["interpretation"] = {"mode": "template"}


def _m_expose_without_description(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["expose"] = {"as_tool": True, "tool_name": "pipeline_assess"}


def _m_expose_bad_tool_name(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["expose"]["tool_name"] = "bad name"


def _m_negative_limit(config: dict[str, Any]) -> None:
    config["limits"] = {"max_jobs": 0}


def _m_async_without_trigger(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["async"] = {"enabled": True, "signal": "pipeline.assess.requested"}


def _m_retry_order(config: dict[str, Any]) -> None:
    _jobs(config)["load"]["retry"] = {"max": 1, "initial_delay_seconds": 5.0, "max_delay_seconds": 1.0}


def _m_extra_key(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["typo"] = True


def _m_definition_limit_over_ceiling(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["limits"] = {"max_llm_calls": 51}


def _m_definition_max_jobs(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["limits"] = {"max_jobs": 2}


def _m_definition_depth_override(config: dict[str, Any]) -> None:
    config["limits"] = {"max_subpipeline_depth": 4}
    for index in range(1, 5):
        config["definitions"][f"p{index}"] = {"jobs": {"run": {"kind": "deterministic", "uses": "app.steps.run"}}}
    for index in range(1, 4):
        config["definitions"][f"p{index}"]["jobs"] = {"call": {"kind": "subpipeline", "pipeline": f"p{index + 1}"}}
    config["definitions"]["p1"]["limits"] = {"max_subpipeline_depth": 2}


def _m_output_bad_from_expression(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["output"] = {"shape": "PipelineOutput", "from": "input["}


def _m_llm_without_model(config: dict[str, Any]) -> None:
    _jobs(config)["report"].pop("model")


def _m_async_signal_without_signal(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["async"] = {
        "enabled": True,
        "mode": "signal",
        "trigger_id": "some-trigger",
    }


def _m_job_timeout_zero(config: dict[str, Any]) -> None:
    _jobs(config)["load"]["timeout_seconds"] = 0


def _m_map_concurrency_over_cap(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["limits"] = {"max_parallel": 2}
    _jobs(config)["enrich"] = {
        "kind": "map",
        "over": "types.Account",
        "as": "account",
        "job": "load",
        "concurrency": 3,
    }


def _m_rag_context_over_cap(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["limits"] = {"max_context_chars": 3000}
    _jobs(config)["report"]["rag"] = {
        "enabled": True,
        "namespace": "knowledge",
        "max_context_chars": 4000,
    }


def _m_expected_revision_true(config: dict[str, Any]) -> None:
    _jobs(config)["load"]["writes"][0]["expected_revision"] = True


def _m_expected_revision_false(config: dict[str, Any]) -> None:
    _jobs(config)["load"]["writes"][0]["expected_revision"] = False


def _m_expected_revision_float(config: dict[str, Any]) -> None:
    _jobs(config)["load"]["writes"][0]["expected_revision"] = 1.0


def _m_expected_revision_bytes(config: dict[str, Any]) -> None:
    _jobs(config)["load"]["writes"][0]["expected_revision"] = b"2"


def _m_job_timeout_over_limit(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["limits"] = {"timeout_seconds": 5}
    _jobs(config)["load"]["timeout_seconds"] = 500


def _m_pattern_adjacent_literals(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["input"] = {"fields": {"x": {"type": "string", "pattern": "a*a*a*a*a*a*a*a*a*a*b"}}}


def _m_pattern_adjacent_escapes(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["input"] = {"fields": {"x": {"type": "string", "pattern": r"\w+\w+"}}}


def _m_pattern_adjacent_classes(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["input"] = {"fields": {"x": {"type": "string", "pattern": "[a-z]*[a-z]*"}}}


def _m_pattern_adjacent_groups(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["input"] = {
        "fields": {
            "x": {
                "type": "string",
                "pattern": "(a|b)*(a|b)*(a|b)*(a|b)*(a|b)*(a|b)*(a|b)*(a|b)*(a|b)*(a|b)*c",
            }
        }
    }


def _m_global_timeout_over_limit(config: dict[str, Any]) -> None:
    config["limits"] = {"timeout_seconds": 5}
    _jobs(config)["load"]["timeout_seconds"] = 25


def _m_pattern_nested_quantifier(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["input"] = {"fields": {"x": {"type": "string", "pattern": "(a+)+"}}}


def _m_pattern_alternation_quantifier(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["input"] = {"fields": {"x": {"type": "string", "pattern": "(a|aa)+"}}}


def _m_pattern_too_long(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["input"] = {"fields": {"x": {"type": "string", "pattern": "(a)" * 70}}}


def _m_field_min_gt_max(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["input"] = {"fields": {"x": {"type": "integer", "min": 5, "max": 1}}}


def _m_field_min_length_gt_max(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["input"] = {"fields": {"x": {"type": "string", "min_length": 5, "max_length": 1}}}


def _m_field_enum_duplicate(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["input"] = {"fields": {"x": {"type": "string", "enum": ["a", "a"]}}}


def _m_field_enum_type_mismatch(config: dict[str, Any]) -> None:
    config["definitions"]["assess"]["input"] = {"fields": {"x": {"type": "integer", "enum": ["a"]}}}


def _m_relation_endpoint_too_long(config: dict[str, Any]) -> None:
    _jobs(config)["analyze"]["links"][0]["to"] = "x" * 257


def _m_relation_endpoint_control_chars(config: dict[str, Any]) -> None:
    _jobs(config)["analyze"]["links"][0]["from"] = "account:\n1"


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        pytest.param(_m_bad_pipeline_name, "pipeline name 'BadName'", id="pipeline-name"),
        pytest.param(_m_no_jobs, "must declare at least one job", id="no-jobs"),
        pytest.param(_m_too_many_jobs, "exceeding limits.max_jobs=2", id="max-jobs"),
        pytest.param(_m_unknown_dependency, "depends on unknown job 'nope'", id="unknown-dependency"),
        pytest.param(_m_self_dependency, "must not depend on itself", id="self-dependency"),
        pytest.param(_m_cycle, "cycle: analyze -> load -> analyze", id="dag-cycle"),
        pytest.param(_m_join_any_too_few, "join: any but declares fewer than two", id="join-any"),
        pytest.param(_m_input_both, "exactly one of 'shape' or 'fields'", id="input-both"),
        pytest.param(_m_input_neither, "exactly one of 'shape' or 'fields'", id="input-neither"),
        pytest.param(_m_input_unknown_shape, "input references unknown shape 'Nope'", id="input-ref"),
        pytest.param(
            _m_input_default_unknown_field,
            "defaults reference unknown field 'nope'",
            id="input-default",
        ),
        pytest.param(_m_output_no_projection, "exactly one of 'mapping' or 'from'", id="output-projection"),
        pytest.param(_m_output_unknown_shape, "output references unknown shape 'Nope'", id="output-ref"),
        pytest.param(
            _m_output_bad_mapping_expression,
            "not a valid JMESPath expression",
            id="output-expression",
        ),
        pytest.param(_m_write_unknown_shape, "writes to unknown shape 'Nope'", id="write-shape"),
        pytest.param(_m_write_reserved_input, "reserved '__input__'", id="write-reserved-input"),
        pytest.param(_m_write_bad_alias, "must be an identifier", id="write-alias"),
        pytest.param(_m_write_bad_when, "not a valid JMESPath expression", id="write-when"),
        pytest.param(_m_link_unknown_relation, "links unknown relation 'unknown_edge'", id="link-relation"),
        pytest.param(
            _m_relation_unknown_endpoint,
            "relation 'bad' references unknown shape 'Nope'",
            id="relation-endpoint",
        ),
        pytest.param(_m_shape_cycle, "recursive shape references", id="shape-cycle"),
        pytest.param(_m_shape_ref_unknown, "references unknown shape 'Nope'", id="shape-ref"),
        pytest.param(_m_shape_fields_and_class, "exactly one of 'fields' or 'class'", id="shape-kind"),
        pytest.param(_m_field_no_type, "exactly one of 'type' or 'ref'", id="field-kind"),
        pytest.param(_m_field_array_without_items, "array fields require 'items'", id="field-array"),
        pytest.param(_m_field_bad_pattern, "not a valid regular expression", id="field-pattern"),
        pytest.param(_m_field_default_mismatch, "does not match field type", id="field-default"),
        pytest.param(
            _m_field_additional_properties_on_scalar,
            "'additional_properties' is only valid for object fields and shape references",
            id="field-additional-properties",
        ),
        pytest.param(_m_decision_no_labels, "'labels' must not be empty", id="decision-labels"),
        pytest.param(_m_decision_duplicate_labels, "is duplicated", id="decision-duplicate-label"),
        pytest.param(_m_decision_unknown_branch, "targets unknown job 'nope'", id="decision-branch"),
        pytest.param(_m_decision_unknown_fallback, "fallback 'nope'", id="decision-fallback"),
        pytest.param(_m_decision_branch_self_target, "must not target the decision job itself", id="decision-self"),
        pytest.param(
            _m_decision_join_all_over_branches,
            "uses join: all over mutually exclusive branches",
            id="join-all-branches",
        ),
        pytest.param(
            _m_branch_target_depends_on_decision,
            "must not list decision job 'choose'",
            id="branch-depends-decision",
        ),
        pytest.param(_m_branch_target_feeds_join_all, "which must use join: any", id="branch-join-all"),
        pytest.param(_m_map_unknown_template, "maps over unknown template job 'nope'", id="map-template"),
        pytest.param(_m_map_nested, "nested maps are not supported", id="map-nested"),
        pytest.param(_m_map_bad_over, "not a valid JMESPath expression", id="map-over"),
        pytest.param(_m_map_reserved_as, "shadows a reserved pipeline context key", id="map-reserved-as"),
        pytest.param(_m_map_bad_as, "must be an identifier", id="map-as"),
        pytest.param(_m_map_decision_template, "is a branching decision job", id="map-decision"),
        pytest.param(_m_subpipeline_unknown, "unknown sub-pipeline 'nope'", id="subpipeline-ref"),
        pytest.param(_m_subpipeline_cycle, "sub-pipeline cycle detected", id="subpipeline-cycle"),
        pytest.param(_m_subpipeline_depth, "exceeding max_subpipeline_depth=2", id="subpipeline-depth"),
        pytest.param(_m_bad_uses, "must be a dotted import path", id="deterministic-uses"),
        pytest.param(_m_bad_custom_class, "must be a dotted import path", id="custom-class"),
        pytest.param(_m_duplicate_alias, "alias 'assessment' is declared by jobs", id="duplicate-alias"),
        pytest.param(_m_bad_prompt_template, "unbalanced", id="template-unbalanced"),
        pytest.param(_m_empty_prompt_template, "empty", id="template-empty"),
        pytest.param(
            _m_python_expression_prompt_template,
            "not a valid JMESPath expression",
            id="template-python-expression",
        ),
        pytest.param(
            _m_unknown_function_prompt_template,
            "only standard JMESPath functions",
            id="template-unknown-function",
        ),
        pytest.param(
            _m_interpretation_llm_without_model,
            "requires a model",
            id="interpretation-llm",
        ),
        pytest.param(
            _m_interpretation_template_without_template,
            "requires a template string",
            id="interpretation-template",
        ),
        pytest.param(_m_expose_without_description, "requires a non-empty 'description'", id="expose-description"),
        pytest.param(_m_expose_bad_tool_name, "must match", id="expose-tool-name"),
        pytest.param(_m_negative_limit, "greater than or equal to 1", id="limit-positive"),
        pytest.param(_m_async_without_trigger, "requires 'trigger_id'", id="async-trigger"),
        pytest.param(
            _m_retry_order,
            "must be >= initial_delay_seconds",
            id="retry-order",
        ),
        pytest.param(_m_extra_key, "Extra inputs are not permitted", id="extra-key"),
        pytest.param(
            _m_definition_limit_over_ceiling,
            "exceeds the global ceiling 50",
            id="definition-limit",
        ),
        pytest.param(_m_definition_max_jobs, "exceeding limits.max_jobs=2", id="definition-max-jobs"),
        pytest.param(
            _m_definition_depth_override,
            "exceeding max_subpipeline_depth=2",
            id="definition-depth",
        ),
        pytest.param(_m_output_bad_from_expression, "not a valid JMESPath expression", id="output-from"),
        pytest.param(_m_llm_without_model, "no 'model' and no defaults.model", id="llm-no-model"),
        pytest.param(_m_async_signal_without_signal, "requires 'signal'", id="async-signal"),
        pytest.param(_m_job_timeout_zero, "greater than 0", id="job-timeout"),
        pytest.param(
            _m_map_concurrency_over_cap,
            "exceeds the effective max_parallel=2",
            id="map-concurrency",
        ),
        pytest.param(
            _m_rag_context_over_cap,
            "exceeds the effective max_context_chars=3000",
            id="rag-context",
        ),
        pytest.param(_m_expected_revision_true, "must be an integer", id="expected-revision-true"),
        pytest.param(_m_expected_revision_false, "must be an integer", id="expected-revision-false"),
        pytest.param(_m_expected_revision_float, "must be an integer", id="expected-revision-float"),
        pytest.param(_m_expected_revision_bytes, "must be an integer", id="expected-revision-bytes"),
        pytest.param(
            _m_job_timeout_over_limit,
            "exceeds the effective timeout_seconds=5",
            id="job-timeout-over-limit",
        ),
        pytest.param(_m_pattern_nested_quantifier, "catastrophic backtracking", id="pattern-nested"),
        pytest.param(
            _m_pattern_alternation_quantifier,
            "catastrophic backtracking",
            id="pattern-alternation",
        ),
        pytest.param(_m_pattern_adjacent_literals, "catastrophic backtracking", id="pattern-adjacent-literals"),
        pytest.param(_m_pattern_adjacent_escapes, "catastrophic backtracking", id="pattern-adjacent-escapes"),
        pytest.param(_m_pattern_adjacent_classes, "catastrophic backtracking", id="pattern-adjacent-classes"),
        pytest.param(
            _m_pattern_adjacent_groups,
            "field pattern risks catastrophic backtracking (heuristic check)",
            id="pattern-adjacent-groups",
        ),
        pytest.param(
            _m_global_timeout_over_limit,
            "exceeds the effective timeout_seconds=5",
            id="job-timeout-global-only",
        ),
        pytest.param(_m_pattern_too_long, "at most 200", id="pattern-length"),
        pytest.param(_m_field_min_gt_max, "'min' must not exceed 'max'", id="field-min-max"),
        pytest.param(
            _m_field_min_length_gt_max,
            "'min_length' must not exceed 'max_length'",
            id="field-length-order",
        ),
        pytest.param(_m_field_enum_duplicate, "duplicate value", id="field-enum-duplicate"),
        pytest.param(_m_field_enum_type_mismatch, "does not match field type", id="field-enum-type"),
        pytest.param(_m_relation_endpoint_too_long, "at most 256", id="relation-endpoint-length"),
        pytest.param(
            _m_relation_endpoint_control_chars,
            "control characters",
            id="relation-endpoint-control",
        ),
    ],
)
def test_rejects_invalid_configuration(mutate: Callable[[dict[str, Any]], None], expected: str) -> None:
    config = _base_pipeline()
    mutate(config)
    _rejects(config, expected)


def test_escaped_template_braces_are_accepted_and_render_literally() -> None:
    config = _base_pipeline()
    template = "Use \\{{ input.account_id }} literally, then {{ input.account_id }}"
    _jobs(config)["analyze"]["prompt"] = template

    accepted = _accepts(config).definitions["assess"].jobs["analyze"].prompt

    assert accepted == template
    assert render_template(accepted, {"input": {"account_id": "account:1"}}) == (
        "Use {{ input.account_id }} literally, then account:1"
    )


def test_stray_closing_delimiters_are_rejected() -> None:
    config = _base_pipeline()
    _jobs(config)["analyze"]["prompt"] = "Analyze input.account_id }}"

    _rejects(config, "unbalanced")


def test_object_write_defaults_and_modes() -> None:
    write = OrchidPipelineObjectWriteConfig(id="account:1", shape="Account")
    assert write.mode == "upsert"
    assert write.from_ is None

    replaced = OrchidPipelineObjectWriteConfig(id="account:1", shape="Account", mode="replace")
    assert replaced.mode == "replace"


def test_shape_allow_version_mismatch_defaults_to_false_and_can_opt_in() -> None:
    default = OrchidPipelineShapeConfig.model_validate({"fields": {"id": {"type": "string"}}})
    assert default.allow_shape_version_mismatch is False

    opted_in = OrchidPipelineShapeConfig.model_validate(
        {"allow_shape_version_mismatch": True, "fields": {"id": {"type": "string"}}}
    )
    assert opted_in.allow_shape_version_mismatch is True


def test_pipeline_block_allow_version_mismatch_defaults_to_false_and_can_opt_in() -> None:
    config = _base_pipeline()
    assert _accepts(config).allow_shape_version_mismatch is False

    config["allow_shape_version_mismatch"] = True
    assert _accepts(config).allow_shape_version_mismatch is True


def test_additional_properties_is_allowed_on_object_and_shape_reference_fields() -> None:
    object_field = OrchidPipelineFieldConfig.model_validate({"type": "object", "additional_properties": True})
    assert object_field.additional_properties is True

    reference = OrchidPipelineFieldConfig.model_validate({"ref": "Account", "additional_properties": True})
    assert reference.additional_properties is True


# ── Cross-block validation (schema_agent) ────────────────────


def _root_rejects(config: dict[str, Any], expected: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        OrchidAgentsConfig.model_validate(config)
    assert expected in str(excinfo.value)


def test_absent_pipelines_block_keeps_existing_behaviour() -> None:
    config = OrchidAgentsConfig.model_validate({"agents": {"analyst": {"description": "", "prompt": ""}}})
    assert config.pipelines is None


def test_root_config_accepts_valid_pipeline_bindings() -> None:
    config = OrchidAgentsConfig.model_validate(_root_config())
    assert config.pipelines is not None
    assert isinstance(config.pipelines.definitions["assess"].jobs["analyze"], OrchidPipelineAgentJobConfig)


def test_exposed_pipeline_tool_is_available_inside_the_fleet() -> None:
    config = _root_config()
    _jobs(config["pipelines"])["report"]["tools"] = ["pipeline_assess", "search"]
    assert OrchidAgentsConfig.model_validate(config).pipelines is not None


def test_unknown_agent_reference_is_rejected() -> None:
    config = _root_config()
    _jobs(config["pipelines"])["analyze"]["agent"] = "ghost"
    _root_rejects(config, "references unknown agent 'ghost'")


def test_parent_agent_reference_is_rejected() -> None:
    config = _root_config()
    config["agents"]["analyst"]["children"] = {"kid": {"description": "", "prompt": ""}}
    _root_rejects(config, "use multiple jobs or a sub-pipeline")


def test_nested_short_name_is_rejected() -> None:
    config = _root_config()
    config["agents"].pop("analyst")
    config["agents"]["team"] = {
        "description": "",
        "prompt": "",
        "children": {"analyst": {"description": "", "prompt": ""}},
    }
    _root_rejects(config, "references nested agent 'analyst'")


def test_nested_exact_path_is_rejected() -> None:
    config = _root_config()
    config["agents"]["analyst"]["children"] = {"kid": {"description": "", "prompt": ""}}
    _jobs(config["pipelines"])["analyze"]["agent"] = "analyst.kid"
    _root_rejects(config, "references nested agent 'analyst.kid'")


def test_top_level_agent_name_with_a_dot_is_addressable() -> None:
    config = _root_config()
    config["agents"]["a.b"] = {"description": "", "prompt": ""}
    _jobs(config["pipelines"])["analyze"]["agent"] = "a.b"

    parsed = OrchidAgentsConfig.model_validate(config)

    assert parsed.pipelines is not None


def test_unknown_tool_reference_is_rejected() -> None:
    config = _root_config()
    _jobs(config["pipelines"])["report"]["tools"] = ["ghost"]
    _root_rejects(config, "references unknown tool 'ghost'")


def test_unknown_decision_model_is_rejected() -> None:
    config = _root_config()
    _jobs(config["pipelines"])["choose"]["decision_model"] = "ghost"
    _root_rejects(config, "references unknown decision model 'ghost'")


def test_exposed_tool_name_collision_is_rejected() -> None:
    config = _root_config()
    config["tools"]["pipeline_assess"] = {"handler": "app.tools.collision"}
    _root_rejects(config, "collides with a declared built-in tool")


def test_disabled_pipelines_skip_exposed_tool_validation() -> None:
    config = _root_config()
    config["pipelines"]["enabled"] = False
    config["tools"]["pipeline_assess"] = {"handler": "app.tools.collision"}
    assert OrchidAgentsConfig.model_validate(config).pipelines is not None


def test_disabled_definition_skips_exposed_tool_validation() -> None:
    config = _root_config()
    config["pipelines"]["definitions"]["assess"]["enabled"] = False
    config["tools"]["pipeline_assess"] = {"handler": "app.tools.collision"}
    assert OrchidAgentsConfig.model_validate(config).pipelines is not None


def test_duplicate_exposed_tool_names_are_rejected() -> None:
    config = _root_config()
    config["pipelines"]["definitions"]["other"] = {
        "jobs": {"work": {"kind": "deterministic", "uses": "app.steps.work"}},
        "expose": {"as_tool": True, "tool_name": "pipeline_assess", "description": "Other."},
    }
    _root_rejects(config, "expose the same tool name 'pipeline_assess'")


def _events_block(trigger_id: str = "real") -> dict[str, Any]:
    return {
        "enabled": True,
        "processors": [{"class": "app.events.Processor"}],
        "triggers": [
            {
                "id": trigger_id,
                "on": {"signal": "custom.event"},
                "emits": {
                    "agent": "analyst",
                    "prompt_template": "Go.",
                    "identity": {"mode": "service_account", "name": "bot"},
                },
            }
        ],
    }


def test_async_trigger_is_validated_when_events_are_enabled() -> None:
    config = _root_config()
    config["events"] = _events_block()
    config["pipelines"]["definitions"]["assess"]["async"] = {
        "enabled": True,
        "signal": "pipeline.assess.requested",
        "trigger_id": "missing",
    }
    _root_rejects(config, "does not match an enabled events trigger")


def test_async_trigger_reference_accepts_a_declared_trigger() -> None:
    config = _root_config()
    config["events"] = _events_block()
    config["pipelines"]["definitions"]["assess"]["async"] = {
        "enabled": True,
        "signal": "pipeline.assess.requested",
        "trigger_id": "real",
    }
    assert OrchidAgentsConfig.model_validate(config).pipelines is not None


def test_async_is_not_cross_checked_without_events() -> None:
    config = _root_config()
    config["pipelines"]["definitions"]["assess"]["async"] = {
        "enabled": True,
        "signal": "pipeline.assess.requested",
        "trigger_id": "missing",
    }
    assert OrchidAgentsConfig.model_validate(config).pipelines is not None
