"""Tests for the decision-point registry, core spec, and observability helper."""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest

import orchid_ai
from orchid_ai.config.schema_decision import OrchidDecisionPointConfig
from orchid_ai.core.decision import (
    OrchidDecisionFallbackPolicy,
    OrchidDecisionPointBinding,
    OrchidDecisionPointSpec,
    OrchidDecisionQuestion,
    OrchidDecisionState,
)
from orchid_ai.decision.observability import (
    DECISION_LOG_FIELDS,
    DECISION_LOGGER_NAME,
    build_decision_record,
    log_decision,
)
from orchid_ai.decision.registry import (
    DECISION_POINT_REGISTRY,
    get_decision_point,
    iter_decision_points,
    list_decision_points,
    register_decision_point,
)
from orchid_ai.decision.stub import StubDecisionModel
from orchid_ai.graph.decision_bindings import DecisionBinding

PACKAGE_ROOT = Path(orchid_ai.__file__).resolve().parent


@pytest.fixture
def registry_state():
    """Restore the process-wide registry after a test mutates it."""
    saved = dict(DECISION_POINT_REGISTRY)
    yield
    DECISION_POINT_REGISTRY.clear()
    DECISION_POINT_REGISTRY.update(saved)


@pytest.mark.parametrize(
    ("name", "scope"),
    [
        ("routing", "fleet"),
        ("skills", "agent"),
        ("tools", "agent"),
        ("rag", "agent"),
    ],
)
def test_legacy_points_are_registered_as_builtins(name: str, scope: str) -> None:
    spec = get_decision_point(name)
    assert spec.builtin is True
    assert spec.scope == scope


@pytest.mark.parametrize(
    ("name", "fallback"),
    [
        ("routing", "llm"),
        ("skills", "llm"),
        ("tools", "all_tools"),
        ("rag", "keep_original"),
    ],
)
def test_legacy_points_declare_their_existing_fallback(name: str, fallback: str) -> None:
    spec = get_decision_point(name)
    assert spec.fallback.mode == fallback


def test_tools_point_declares_the_existing_candidate_ceiling() -> None:
    assert get_decision_point("tools").max_candidates == 64


def test_consumer_point_can_be_registered_and_looked_up(registry_state) -> None:
    spec = OrchidDecisionPointSpec(
        name="tool_choice",
        scope="agent",
        max_candidates=8,
        fallback="all_tools",
        description="Choose tools for one turn.",
    )
    assert register_decision_point(spec) is spec
    assert get_decision_point("tool_choice") is spec
    assert "tool_choice" in list_decision_points()
    assert "tool_choice" in list_decision_points("agent")


def test_duplicate_registration_is_rejected(registry_state) -> None:
    register_decision_point(OrchidDecisionPointSpec(name="tool_choice"))
    with pytest.raises(ValueError, match="already registered"):
        register_decision_point(OrchidDecisionPointSpec(name="tool_choice"))


def test_builtin_point_cannot_be_replaced_even_with_override(registry_state) -> None:
    with pytest.raises(ValueError, match="built-in"):
        register_decision_point(OrchidDecisionPointSpec(name="routing"), override=True)


def test_consumer_point_can_be_replaced_with_override(registry_state) -> None:
    register_decision_point(OrchidDecisionPointSpec(name="tool_choice", description="first"))
    replacement = OrchidDecisionPointSpec(name="tool_choice", description="second")
    register_decision_point(replacement, override=True)
    assert get_decision_point("tool_choice") is replacement


def test_unknown_point_lists_every_registered_name() -> None:
    with pytest.raises(KeyError) as excinfo:
        get_decision_point("does_not_exist")
    message = excinfo.value.args[0]
    assert "unknown decision point" in message
    assert "registered points" in message
    for name in list_decision_points():
        assert name in message


def test_list_decision_points_filters_by_scope() -> None:
    assert list_decision_points("fleet") == ("routing",)
    assert list_decision_points("agent") == ("rag", "skills", "tools")
    assert set(list_decision_points()) == {"routing", "skills", "tools", "rag"}


def test_iter_decision_points_is_name_sorted() -> None:
    assert [spec.name for spec in iter_decision_points()] == list(list_decision_points())


def test_invalid_scope_filter_is_rejected() -> None:
    with pytest.raises(ValueError, match="scope"):
        list_decision_points("global")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": ""},
        {"name": "Tool Choice"},
        {"name": "model-routing"},
        {"name": "x" * 65},
        {"name": "tool_choice\n"},
        {"name": "tool_choice", "scope": "global"},
        {"name": "tool_choice", "max_candidates": 0},
        {"name": "tool_choice", "max_candidates": True},
        {"name": "tool_choice", "fallback": "All Tools"},
        {"name": "tool_choice", "fallback": "all_tools\n"},
        {"name": "tool_choice", "builtin": "yes"},
        {"name": "tool_choice", "validate_config": "not callable"},
    ],
)
def test_invalid_specs_are_rejected(kwargs: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        OrchidDecisionPointSpec(**kwargs)


def test_fallback_mode_string_is_normalised() -> None:
    spec = OrchidDecisionPointSpec(name="tool_choice", fallback="all_tools")
    assert isinstance(spec.fallback, OrchidDecisionFallbackPolicy)
    assert spec.fallback.mode == "all_tools"
    assert spec.fallback.description == ""


def test_registry_does_not_import_transport_or_schema_libraries() -> None:
    module = PACKAGE_ROOT / "decision" / "registry.py"
    tree = ast.parse(module.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.add(node.module.split(".", 1)[0])
    assert not imported & {"pydantic", "httpx", "litellm"}


def test_core_decision_imports_only_stdlib() -> None:
    module = PACKAGE_ROOT / "core" / "decision.py"
    tree = ast.parse(module.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.add(node.module.split(".", 1)[0])
    assert not imported - sys.stdlib_module_names


def test_binding_implements_the_consumer_protocol() -> None:
    binding = DecisionBinding(
        spec=get_decision_point("tools"),
        config=OrchidDecisionPointConfig(model="stub"),
        model=StubDecisionModel(),
    )
    assert isinstance(binding, OrchidDecisionPointBinding)


def test_consumer_only_needs_the_protocol() -> None:
    binding = DecisionBinding(
        spec=get_decision_point("tools"),
        config=OrchidDecisionPointConfig(model="stub", options={"mode": "fast"}),
        model=StubDecisionModel(),
    )

    def consume(candidate: OrchidDecisionPointBinding) -> tuple[str, str, dict]:
        return candidate.point, candidate.provider, dict(candidate.options)

    assert consume(binding) == ("tools", "stub", {"mode": "fast"})
    assert not hasattr(binding, "config")
    assert not hasattr(binding, "model")


def test_binding_options_are_read_only() -> None:
    binding = DecisionBinding(
        spec=get_decision_point("tools"),
        config=OrchidDecisionPointConfig(model="stub", options={"mode": "fast"}),
        model=StubDecisionModel(),
    )
    with pytest.raises(TypeError):
        binding.options["mode"] = "slow"  # type: ignore[index]


def test_binding_metadata_sanitises_hostile_tokens() -> None:
    binding = DecisionBinding(
        spec=get_decision_point("tools"),
        config=OrchidDecisionPointConfig(model="evil\nkey=1"),
        model=StubDecisionModel(),
    )
    metadata = binding.metadata()
    assert "\n" not in metadata["provider"]
    assert "=" not in metadata["provider"]
    assert metadata["provider"] == "evil_key_1"


def test_top_level_package_exports_the_extension_surface() -> None:
    import orchid_ai

    assert orchid_ai.register_decision_point is register_decision_point
    assert orchid_ai.OrchidDecisionPointSpec is OrchidDecisionPointSpec
    assert orchid_ai.OrchidDecisionPointBinding is OrchidDecisionPointBinding
    assert orchid_ai.OrchidDecisionFallbackPolicy is OrchidDecisionFallbackPolicy


# ── Observability ────────────────────────────────────────────


def test_record_has_exactly_the_public_fields() -> None:
    record = build_decision_record(
        point="tool_choice",
        path="decision",
        provider="stub",
        latency_ms=12.3456,
        confidence=0.87654,
        outcome="accepted",
    )
    assert tuple(record) == DECISION_LOG_FIELDS
    assert record == {
        "point": "tool_choice",
        "path": "decision",
        "provider": "stub",
        "latency_ms": 12.346,
        "confidence": 0.8765,
        "outcome": "accepted",
        "fallback_reason": None,
    }


def test_fallback_record_keeps_the_stable_reason() -> None:
    record = build_decision_record(
        point="rag",
        path="llm_fallback",
        provider="stub",
        outcome="fallback",
        fallback_reason="low_confidence",
    )
    assert record["path"] == "llm_fallback"
    assert record["outcome"] == "fallback"
    assert record["fallback_reason"] == "low_confidence"


def test_log_decision_emits_one_structured_line(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("INFO", logger=DECISION_LOGGER_NAME):
        record = log_decision(
            point="tool_choice",
            path="llm_fallback",
            provider="stub",
            latency_ms=4.5,
            confidence=0.42,
            outcome="fallback",
            fallback_reason="low_confidence",
        )

    lines = [entry.getMessage() for entry in caplog.records if entry.name == DECISION_LOGGER_NAME]
    assert len(lines) == 1
    line = lines[0]
    for fragment in (
        "decision point=tool_choice",
        "path=llm_fallback",
        "provider=stub",
        "latency_ms=4.5000",
        "confidence=0.4200",
        "outcome=fallback",
        "fallback_reason=low_confidence",
    ):
        assert fragment in line
    assert "\n" not in line
    assert record["point"] == "tool_choice"


def test_log_decision_has_no_payload_parameter() -> None:
    parameters = set(inspect.signature(log_decision).parameters)
    assert parameters == {
        "point",
        "outcome",
        "path",
        "provider",
        "latency_ms",
        "confidence",
        "fallback_reason",
        "logger",
    }
    assert not parameters & {"state", "payload", "text", "content", "prompt"}


def test_hostile_values_cannot_inject_extra_log_lines(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("INFO", logger=DECISION_LOGGER_NAME):
        record = log_decision(
            point="tool_choice\nforged=1",
            provider="evil\r\npoint=other",
            outcome="fallback",
            fallback_reason="bad reason\nwith newline",
        )
    assert "\n" not in record["point"]
    assert "\n" not in record["provider"]
    assert "\n" not in str(record["fallback_reason"])
    lines = [entry.getMessage() for entry in caplog.records if entry.name == DECISION_LOGGER_NAME]
    assert len(lines) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"point": "", "outcome": "accepted"},
        {"point": "p", "outcome": "accepted", "path": "other"},
        {"point": "p", "outcome": "maybe"},
        {"point": "p", "outcome": "accepted", "latency_ms": -1.0},
        {"point": "p", "outcome": "accepted", "confidence": 1.5},
    ],
)
def test_invalid_observations_are_rejected(kwargs: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_decision_record(**kwargs)


def test_binding_decide_uses_the_stub_provider() -> None:
    import asyncio

    binding = DecisionBinding(
        spec=get_decision_point("tools"),
        config=OrchidDecisionPointConfig(model="stub"),
        model=StubDecisionModel(answers={"q": {"probability": 0.9}}),
    )
    result = asyncio.run(
        binding.decide(
            OrchidDecisionState(text="x"),
            [OrchidDecisionQuestion(name="q", kind="boolean")],
        )
    )
    assert result.answers["q"].value is True
