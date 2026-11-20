"""Tests for the pipeline shape compiler (``orchid_ai.pipeline.schema``)."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from orchid_ai.config.schema_pipeline import OrchidPipelineFieldConfig, OrchidPipelineShapeConfig
from orchid_ai.core.pipeline import OrchidPipelineShapeError
from orchid_ai.pipeline.schema import (
    _MAX_SHAPE_REF_DEPTH,
    OrchidCompiledShape,
    OrchidShapeRegistry,
    ShapeCompiler,
    _inline_root_ref,
    compile_shapes,
)

# ── Fixtures / helpers ───────────────────────────────────────


class _RiskModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float
    band: str = "low"


@dataclass
class _UsageRecord:
    account_id: str
    events: int = 0
    tags: list[str] = field(default_factory=list)


class _NotAShape:
    pass


class _RecursiveNode(BaseModel):
    name: str = ""
    child: _RecursiveNode | None = None


class _NodeWrapper(BaseModel):
    node: _RecursiveNode


@pytest.fixture
def shape_module(monkeypatch: pytest.MonkeyPatch) -> str:
    """Register a fake module holding class-based shapes for dotted-path resolution."""
    module_name = "pipeline_shape_compiler_fakes"
    module = ModuleType(module_name)
    module.Risk = _RiskModel  # type: ignore[attr-defined]
    module.Usage = _UsageRecord  # type: ignore[attr-defined]
    module.Node = _RecursiveNode  # type: ignore[attr-defined]
    module.NodeWrapper = _NodeWrapper  # type: ignore[attr-defined]
    module.NotAShape = _NotAShape  # type: ignore[attr-defined]
    module.VALUE = 42  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, module)
    return module_name


def _declarations(**specs: dict[str, Any]) -> dict[str, OrchidPipelineShapeConfig]:
    return {name: OrchidPipelineShapeConfig.model_validate(spec) for name, spec in specs.items()}


def _fields(**specs: dict[str, Any]) -> dict[str, OrchidPipelineFieldConfig]:
    return {name: OrchidPipelineFieldConfig.model_validate(spec) for name, spec in specs.items()}


def _compile(**specs: dict[str, Any]) -> OrchidShapeRegistry:
    return compile_shapes(_declarations(**specs))


# ── Field types and required/default ────────────────────────


def test_required_and_optional_fields() -> None:
    registry = _compile(
        Account={
            "fields": {
                "id": {"type": "string", "required": True},
                "name": {"type": "string"},
            }
        }
    )
    shape = registry.get("Account")

    model = shape.validate({"id": "account-1"})
    assert isinstance(model, BaseModel)
    assert model.id == "account-1"
    assert model.name is None

    with pytest.raises(OrchidPipelineShapeError) as missing:
        shape.validate({})
    assert missing.value.field_path == "$.id"
    assert "shape 'Account'" in str(missing.value)
    assert "Field required" in str(missing.value)

    accepted = shape.validate({"id": "account-1", "name": None})
    assert accepted.name is None


@pytest.mark.parametrize(
    ("type_name", "value"),
    [
        ("string", "value"),
        ("integer", 3),
        ("number", 2.5),
        ("boolean", True),
        ("null", None),
    ],
)
def test_scalar_types_accept_and_reject(type_name: str, value: Any) -> None:
    registry = _compile(Shape={"fields": {"value": {"type": type_name, "required": True}}})
    shape = registry.get("Shape")

    assert shape.validate({"value": value}).value == value

    with pytest.raises(OrchidPipelineShapeError) as rejected:
        shape.validate({"value": []})
    assert rejected.value.field_path == "$.value"


def test_null_type_rejects_non_null() -> None:
    shape = _compile(Shape={"fields": {"value": {"type": "null", "required": True}}}).get("Shape")
    assert shape.validate({"value": None}).value is None
    with pytest.raises(OrchidPipelineShapeError):
        shape.validate({"value": "x"})

    optional = _compile(Shape={"fields": {"value": {"type": "null"}}}).get("Shape")
    assert optional.validate({}).value is None


def test_defaults_are_applied_and_validated() -> None:
    shape = _compile(
        Account={
            "fields": {
                "plan": {"type": "string", "enum": ["free", "pro"], "default": "free"},
                "mrr": {"type": "number", "default": 0},
                "tags": {"type": "array", "items": {"type": "string"}},
            }
        }
    ).get("Account")

    model = shape.validate({})
    assert model.plan == "free"
    # ``validate_default`` coerces the integer default to the declared number type.
    assert model.mrr == 0.0
    assert model.tags is None
    assert shape.dump({}) == {"plan": "free", "mrr": 0.0, "tags": None}


def test_required_wins_over_declared_default() -> None:
    shape = _compile(Shape={"fields": {"value": {"type": "string", "required": True, "default": "x"}}}).get("Shape")
    with pytest.raises(OrchidPipelineShapeError):
        shape.validate({})
    assert shape.validate({"value": "x"}).value == "x"


def test_optional_without_default_is_nullable() -> None:
    shape = _compile(Shape={"fields": {"note": {"type": "string"}}}).get("Shape")
    assert shape.validate({"note": None}).note is None
    assert shape.validate({}).note is None
    assert shape.dump({}) == {"note": None}


def test_optional_with_default_is_not_nullable() -> None:
    shape = _compile(Shape={"fields": {"note": {"type": "string", "default": "n/a"}}}).get("Shape")
    assert shape.validate({}).note == "n/a"
    with pytest.raises(OrchidPipelineShapeError) as rejected:
        shape.validate({"note": None})
    assert rejected.value.field_path == "$.note"


# ── Declared defaults are compile-checked ───────────────────


def test_default_below_min_fails_compile() -> None:
    with pytest.raises(OrchidPipelineShapeError) as rejected:
        _compile(Shape={"fields": {"score": {"type": "number", "min": 5, "default": 0}}})
    error = rejected.value
    message = str(error)
    assert error.shape == "Shape"
    assert "shape 'Shape'" in message
    assert "default for field 'score'" in message
    assert "$.score" in message
    assert "greater_than_equal" in message


def test_default_above_max_fails_compile() -> None:
    with pytest.raises(OrchidPipelineShapeError) as rejected:
        _compile(Shape={"fields": {"score": {"type": "integer", "max": 10, "default": 11}}})
    assert "less_than_equal" in str(rejected.value)


def test_default_violating_pattern_fails_compile() -> None:
    with pytest.raises(OrchidPipelineShapeError) as rejected:
        _compile(Shape={"fields": {"code": {"type": "string", "pattern": "^[a-z]+$", "default": "ABC"}}})
    message = str(rejected.value)
    assert "default for field 'code'" in message
    assert "$.code" in message
    assert "string_pattern_mismatch" in message


def test_default_is_checked_even_when_required() -> None:
    with pytest.raises(OrchidPipelineShapeError) as rejected:
        _compile(Shape={"fields": {"score": {"type": "number", "min": 5, "required": True, "default": 0}}})
    assert "default for field 'score'" in str(rejected.value)


def test_default_is_validated_against_ref_and_array_shapes() -> None:
    with pytest.raises(OrchidPipelineShapeError) as nested:
        _compile(
            Account={"fields": {"id": {"type": "string", "required": True}}},
            Plan={"fields": {"account": {"ref": "Account", "default": {"id": 5}}}},
        )
    message = str(nested.value)
    assert "default for field 'account'" in message
    assert "$.account.id" in message

    with pytest.raises(OrchidPipelineShapeError) as array:
        _compile(Shape={"fields": {"ids": {"type": "array", "items": {"type": "integer"}, "default": ["x"]}}})
    assert "$.ids[0]" in str(array.value)


def test_reserved_pydantic_field_names_are_rejected() -> None:
    for reserved in ("model_config", "model_dump", "model_fields", "dict", "validate"):
        with pytest.raises(OrchidPipelineShapeError) as rejected:
            _compile(Shape={"fields": {reserved: {"type": "string"}}})
        message = str(rejected.value)
        assert "shape 'Shape'" in message
        assert "reserved Pydantic name" in message
        assert reserved in message


def test_reserved_pydantic_field_name_inside_nested_object_is_rejected() -> None:
    with pytest.raises(OrchidPipelineShapeError) as rejected:
        _compile(
            Shape={
                "fields": {
                    "payload": {
                        "type": "object",
                        "properties": {"model_config": {"type": "string"}},
                    }
                }
            }
        )
    assert "field 'model_config'" in str(rejected.value)
    assert "reserved Pydantic name" in str(rejected.value)


def test_enum_rejection_and_json_schema() -> None:
    shape = _compile(Account={"fields": {"plan": {"type": "string", "enum": ["free", "pro"], "required": True}}}).get(
        "Account"
    )
    assert shape.validate({"plan": "pro"}).plan == "pro"

    with pytest.raises(OrchidPipelineShapeError) as rejected:
        shape.validate({"plan": "enterprise"})
    assert rejected.value.field_path == "$.plan"
    assert "pro" in str(rejected.value)

    schema = shape.model_json_schema()
    assert schema["properties"]["plan"]["enum"] == ["free", "pro"]


# ── Nested shapes: ref, items, properties ───────────────────


def test_ref_shapes_compile_recursively() -> None:
    registry = _compile(
        Account={
            "fields": {
                "id": {"type": "string", "required": True},
                "tags": {"type": "array", "items": {"type": "string"}},
            }
        },
        Plan={
            "fields": {
                "account": {"ref": "Account", "required": True},
                "score": {"type": "number", "default": 0},
            }
        },
    )
    plan = registry.get("Plan")

    model = plan.validate({"account": {"id": "account-1"}})
    assert model.account.id == "account-1"
    assert model.account.tags is None
    assert model.score == 0

    with pytest.raises(OrchidPipelineShapeError) as nested:
        plan.validate({"account": {"id": 5}})
    assert nested.value.field_path == "$.account.id"
    assert "shape 'Plan'" in str(nested.value)

    assert plan.dump({"account": {"id": "account-1"}}) == {
        "account": {"id": "account-1", "tags": None},
        "score": 0.0,
    }


def test_array_items_and_index_paths() -> None:
    shape = _compile(
        Account={
            "fields": {
                "tags": {"type": "array", "items": {"type": "string"}, "required": True},
            }
        }
    ).get("Account")

    assert shape.validate({"tags": ["a", "b"]}).tags == ["a", "b"]

    with pytest.raises(OrchidPipelineShapeError) as indexed:
        shape.validate({"tags": ["a", 5]})
    assert indexed.value.field_path == "$.tags[1]"
    assert "$.tags[1]" in str(indexed.value)


def test_array_of_refs() -> None:
    registry = _compile(
        Item={"fields": {"id": {"type": "string", "required": True}}},
        Cart={"fields": {"items": {"type": "array", "items": {"ref": "Item"}, "required": True}}},
    )
    cart = registry.get("Cart")
    assert [item.id for item in cart.validate({"items": [{"id": "a"}, {"id": "b"}]}).items] == ["a", "b"]

    with pytest.raises(OrchidPipelineShapeError) as rejected:
        cart.validate({"items": [{"id": "a"}, {}]})
    assert rejected.value.field_path == "$.items[1].id"


def test_inline_object_properties() -> None:
    shape = _compile(
        Account={
            "fields": {
                "owner": {
                    "type": "object",
                    "required": True,
                    "properties": {
                        "name": {"type": "string", "required": True},
                        "age": {"type": "integer"},
                    },
                }
            }
        }
    ).get("Account")

    model = shape.validate({"owner": {"name": "Ada"}})
    assert model.owner.name == "Ada"
    assert model.owner.age is None

    with pytest.raises(OrchidPipelineShapeError) as missing:
        shape.validate({"owner": {}})
    assert missing.value.field_path == "$.owner.name"

    with pytest.raises(OrchidPipelineShapeError) as extra:
        shape.validate({"owner": {"name": "Ada", "role": "admin"}})
    assert extra.value.field_path == "$.owner.role"
    assert "extra_forbidden" in str(extra.value)


# ── additional_properties ───────────────────────────────────


def test_additional_properties_defaults_to_false() -> None:
    shape = _compile(Account={"fields": {"id": {"type": "string", "required": True}}}).get("Account")
    with pytest.raises(OrchidPipelineShapeError) as extra:
        shape.validate({"id": "a", "unknown": 1})
    assert extra.value.field_path == "$.unknown"
    assert shape.model_json_schema()["additionalProperties"] is False


def test_additional_properties_true_allows_and_serializes_extras() -> None:
    shape = _compile(
        Account={"additional_properties": True, "fields": {"id": {"type": "string", "required": True}}}
    ).get("Account")
    model = shape.validate({"id": "a", "unknown": 1})
    assert model.unknown == 1
    assert shape.dump({"id": "a", "unknown": 1}) == {"id": "a", "unknown": 1}


def test_object_without_properties_is_closed_unless_extras_allowed() -> None:
    closed = _compile(Shape={"fields": {"payload": {"type": "object", "required": True}}}).get("Shape")
    assert closed.validate({"payload": {}}).payload.model_dump() == {}
    with pytest.raises(OrchidPipelineShapeError):
        closed.validate({"payload": {"anything": 1}})

    open_shape = _compile(
        Shape={"fields": {"payload": {"type": "object", "additional_properties": True, "required": True}}}
    ).get("Shape")
    assert open_shape.validate({"payload": {"anything": 1}}).payload == {"anything": 1}


def test_inline_object_additional_properties() -> None:
    shape = _compile(
        Shape={
            "fields": {
                "payload": {
                    "type": "object",
                    "additional_properties": True,
                    "properties": {"kind": {"type": "string", "required": True}},
                    "required": True,
                }
            }
        }
    ).get("Shape")
    model = shape.validate({"payload": {"kind": "x", "extra": 2}})
    assert model.payload.kind == "x"
    assert model.payload.extra == 2


# ── Bounds, lengths and patterns ────────────────────────────


def test_numeric_bounds() -> None:
    shape = _compile(Shape={"fields": {"score": {"type": "number", "min": 0, "max": 10, "required": True}}}).get(
        "Shape"
    )
    assert shape.validate({"score": 0}).score == 0
    assert shape.validate({"score": 10}).score == 10

    with pytest.raises(OrchidPipelineShapeError) as low:
        shape.validate({"score": -1})
    assert low.value.field_path == "$.score"

    with pytest.raises(OrchidPipelineShapeError):
        shape.validate({"score": 11})

    schema = shape.model_json_schema()
    assert schema["properties"]["score"]["minimum"] == 0
    assert schema["properties"]["score"]["maximum"] == 10


def test_string_and_array_length_bounds() -> None:
    shape = _compile(
        Shape={
            "fields": {
                "code": {"type": "string", "min_length": 2, "max_length": 3, "required": True},
                "tags": {"type": "array", "items": {"type": "string"}, "min_length": 1, "max_length": 2},
            }
        }
    ).get("Shape")

    assert shape.validate({"code": "abc", "tags": ["a"]}).code == "abc"

    with pytest.raises(OrchidPipelineShapeError) as short:
        shape.validate({"code": "a"})
    assert short.value.field_path == "$.code"

    with pytest.raises(OrchidPipelineShapeError) as long:
        shape.validate({"code": "abcd"})
    assert long.value.field_path == "$.code"

    with pytest.raises(OrchidPipelineShapeError) as too_many:
        shape.validate({"code": "abc", "tags": ["a", "b", "c"]})
    assert too_many.value.field_path == "$.tags"


def test_pattern() -> None:
    shape = _compile(Shape={"fields": {"code": {"type": "string", "pattern": "^[a-z]+$", "required": True}}}).get(
        "Shape"
    )

    assert shape.validate({"code": "abc"}).code == "abc"
    with pytest.raises(OrchidPipelineShapeError) as rejected:
        shape.validate({"code": "ABC"})
    assert rejected.value.field_path == "$.code"
    assert shape.model_json_schema()["properties"]["code"]["pattern"] == "^[a-z]+$"


def test_pattern_rejected_on_non_string_field() -> None:
    with pytest.raises(OrchidPipelineShapeError) as rejected:
        _compile(Shape={"fields": {"count": {"type": "integer", "pattern": "^[0-9]+$"}}})
    message = str(rejected.value)
    assert "pattern" in message
    assert "only valid for string fields" in message


def test_pattern_unsupported_by_pydantic_engine_fails_compile() -> None:
    pattern = "^(?=.*[a-z]).+$"
    with pytest.raises(OrchidPipelineShapeError) as rejected:
        _compile(Shape={"fields": {"code": {"type": "string", "pattern": pattern}}})
    message = str(rejected.value)
    assert "shape 'Shape'" in message
    assert "field 'code'" in message
    assert pattern in message
    assert "not supported by Pydantic's regex engine" in message
    assert "look-around" in message


def test_description_reaches_json_schema() -> None:
    shape = _compile(
        Account={
            "description": "One customer account",
            "fields": {"id": {"type": "string", "required": True, "description": "Stable identifier"}},
        }
    ).get("Account")

    schema = shape.model_json_schema()
    assert schema["description"] == "One customer account"
    assert schema["properties"]["id"]["description"] == "Stable identifier"
    assert shape.description == "One customer account"


# ── JSON Schema ─────────────────────────────────────────────


def test_json_schema_shape() -> None:
    registry = _compile(
        Account={"fields": {"id": {"type": "string", "required": True}}},
        Plan={
            "fields": {
                "account": {"ref": "Account", "required": True},
                "plan": {"type": "string", "enum": ["free", "pro"], "default": "free"},
            }
        },
    )
    schema = registry.model_json_schema("Plan")

    assert schema["title"] == "Plan"
    assert schema["type"] == "object"
    assert schema["required"] == ["account"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["plan"]["enum"] == ["free", "pro"]
    assert schema["properties"]["plan"]["default"] == "free"
    assert schema["properties"]["account"]["$ref"] == "#/$defs/Account"
    assert schema["$defs"]["Account"]["properties"]["id"]["type"] == "string"

    # The alias returns the same document, and the copy is defensive.
    assert registry.json_schema("Plan") == schema
    schema["title"] = "mutated"
    assert registry.model_json_schema("Plan")["title"] == "Plan"


def _assert_local_refs_resolve(schema: Any) -> None:
    def resolves(reference: str) -> bool:
        if not reference.startswith("#/"):
            return False
        current = schema
        for raw_token in reference[2:].split("/"):
            token = raw_token.replace("~1", "/").replace("~0", "~")
            if isinstance(current, dict) and token in current:
                current = current[token]
            elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
                current = current[int(token)]
            else:
                return False
        return True

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            reference = node.get("$ref")
            if isinstance(reference, str):
                assert resolves(reference), f"unresolved $ref {reference!r}"
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)


def test_every_emitted_schema_ref_resolves(shape_module: str) -> None:
    registry = _compile(
        Account={"fields": {"id": {"type": "string", "required": True}}},
        Plan={
            "fields": {
                "account": {"ref": "Account", "required": True},
                "history": {"type": "array", "items": {"ref": "Account"}},
            }
        },
        Usage={"class": f"{shape_module}.Usage"},
    )
    for name in registry.names:
        _assert_local_refs_resolve(registry.model_json_schema(name))


def test_inline_root_ref_flattens_and_preserves_other_defs() -> None:
    schema = {
        "$ref": "#/$defs/Thing",
        "$defs": {
            "Thing": {"type": "object", "properties": {"id": {"type": "string"}}},
            "Other": {"type": "string"},
        },
    }
    result = _inline_root_ref(schema, "Thing")
    assert result["title"] == "Thing"
    assert result["properties"] == {"id": {"type": "string"}}
    assert result["$defs"] == {"Other": {"type": "string"}}


def test_inline_root_ref_keeps_inner_defs() -> None:
    schema = {
        "$ref": "#/$defs/Thing",
        "$defs": {"Thing": {"type": "object", "$defs": {"Inner": {"type": "string"}}}},
    }
    result = _inline_root_ref(schema, "Thing")
    assert result["$defs"] == {"Inner": {"type": "string"}}
    assert result["title"] == "Thing"


def test_inline_root_ref_without_ref_only_sets_title() -> None:
    assert _inline_root_ref({"type": "object"}, "Thing") == {"type": "object", "title": "Thing"}


def test_inline_root_ref_leaves_a_dangling_ref_unchanged() -> None:
    assert _inline_root_ref({"$ref": "#/$defs/Missing"}, "Thing") == {
        "$ref": "#/$defs/Missing",
        "title": "Thing",
    }


def test_shape_json_schema_alias_and_repr() -> None:
    shape = _compile(Account={"fields": {"id": {"type": "string"}}}).get("Account")
    assert shape.json_schema() == shape.model_json_schema()
    assert repr(shape) == "OrchidCompiledShape(name='Account', version='1')"


# ── Class-based shapes ─────────────────────────────────────


def test_class_based_pydantic_shape(shape_module: str) -> None:
    registry = _compile(
        CustomerRisk={
            "class": f"{shape_module}.Risk",
            "version": "2",
            "description": "Assessed customer risk",
        }
    )
    shape = registry.get("CustomerRisk")

    assert shape.version == "2"
    assert shape.description == "Assessed customer risk"
    assert shape.model is _RiskModel

    model = shape.validate({"score": 1})
    assert model.score == 1.0
    assert model.band == "low"
    assert shape.dump({"score": 1}) == {"score": 1.0, "band": "low"}

    with pytest.raises(OrchidPipelineShapeError) as rejected:
        shape.validate({"score": "high"})
    assert rejected.value.field_path == "$.score"

    # The class's own strictness is preserved.
    with pytest.raises(OrchidPipelineShapeError) as extra:
        shape.validate({"score": 1, "unknown": True})
    assert extra.value.field_path == "$.unknown"

    schema = shape.model_json_schema()
    assert schema["title"] == "CustomerRisk"
    assert schema["description"] == "Assessed customer risk"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["score"]["type"] == "number"


def test_class_based_dataclass_shape(shape_module: str) -> None:
    shape = _compile(Usage={"class": f"{shape_module}.Usage"}).get("Usage")

    model = shape.validate({"account_id": "account-1"})
    assert model == _UsageRecord(account_id="account-1")
    assert shape.dump({"account_id": "account-1", "events": 2}) == {
        "account_id": "account-1",
        "events": 2,
        "tags": [],
    }

    with pytest.raises(OrchidPipelineShapeError) as missing:
        shape.validate({})
    assert missing.value.field_path == "$.account_id"

    schema = shape.model_json_schema()
    assert schema["title"] == "Usage"
    assert schema["type"] == "object"
    assert schema["required"] == ["account_id"]


def test_class_based_import_failure_fails_fast() -> None:
    with pytest.raises(OrchidPipelineShapeError) as rejected:
        _compile(Broken={"class": "pipeline_shape_compiler_missing.Model"})
    message = str(rejected.value)
    assert "cannot import shape class" in message
    assert "pipeline_shape_compiler_missing.Model" in message


def test_class_based_shape_must_be_a_model_or_dataclass(shape_module: str) -> None:
    with pytest.raises(OrchidPipelineShapeError) as rejected:
        _compile(Bad={"class": f"{shape_module}.NotAShape"})
    assert "must be a Pydantic BaseModel subclass or a dataclass" in str(rejected.value)


def test_class_path_must_resolve_to_a_class(shape_module: str) -> None:
    with pytest.raises(OrchidPipelineShapeError) as rejected:
        _compile(Bad={"class": f"{shape_module}.VALUE"})
    assert "resolved to int, not a class" in str(rejected.value)


def test_recursive_class_shape_is_rejected(shape_module: str) -> None:
    with pytest.raises(OrchidPipelineShapeError) as rejected:
        _compile(Node={"class": f"{shape_module}.Node"})
    error = rejected.value
    message = str(error)
    assert error.shape == "Node"
    assert "shape 'Node'" in message
    assert "recursive" in message
    assert "not supported in v1" in message


def test_class_shape_with_a_recursive_nested_type_is_rejected(shape_module: str) -> None:
    with pytest.raises(OrchidPipelineShapeError) as rejected:
        _compile(Wrapper={"class": f"{shape_module}.NodeWrapper"})
    message = str(rejected.value)
    assert "shape 'Wrapper'" in message
    assert "recursive schema reference" in message
    assert "not supported in v1" in message


# ── Recursive references ────────────────────────────────────


@pytest.mark.parametrize(
    "field",
    [
        {"ref": "Node"},
        {"type": "array", "items": {"ref": "Node"}},
        {"type": "object", "properties": {"nested": {"ref": "Node"}}},
    ],
)
def test_direct_recursion_is_rejected(field: dict[str, Any]) -> None:
    with pytest.raises(OrchidPipelineShapeError) as rejected:
        compile_shapes(_declarations(Node={"fields": {"child": field}}))
    message = str(rejected.value)
    assert "recursive shape references are not supported in v1" in message
    assert "Node -> Node" in message
    assert rejected.value.code == "pipeline_shape_error"


def test_indirect_recursion_is_rejected_with_the_cycle() -> None:
    with pytest.raises(OrchidPipelineShapeError) as rejected:
        compile_shapes(
            _declarations(
                A={"fields": {"b": {"ref": "B"}}},
                B={"fields": {"a": {"ref": "A"}}},
            )
        )
    assert "A -> B -> A" in str(rejected.value)


def test_unknown_ref_is_rejected() -> None:
    with pytest.raises(OrchidPipelineShapeError) as rejected:
        _compile(Plan={"fields": {"account": {"ref": "Missing"}}})
    message = str(rejected.value)
    assert "unknown shape reference 'Missing'" in message
    assert "declared shapes: Plan" in message


def _chain_shapes(count: int) -> dict[str, Any]:
    return {
        f"S{index}": (
            {"fields": {"next": {"ref": f"S{index + 1}"}}}
            if index + 1 < count
            else {"fields": {"value": {"type": "string"}}}
        )
        for index in range(count)
    }


def test_ref_chain_at_the_depth_limit_compiles() -> None:
    registry = compile_shapes(_declarations(**_chain_shapes(_MAX_SHAPE_REF_DEPTH)))
    assert len(registry) == _MAX_SHAPE_REF_DEPTH


def test_ref_chain_above_the_depth_limit_is_rejected() -> None:
    with pytest.raises(OrchidPipelineShapeError) as rejected:
        compile_shapes(_declarations(**_chain_shapes(_MAX_SHAPE_REF_DEPTH + 1)))
    error = rejected.value
    message = str(error)
    assert error.shape == "S0"
    assert "shape 'S0'" in message
    assert f"depth {_MAX_SHAPE_REF_DEPTH + 1}" in message
    assert f"maximum of {_MAX_SHAPE_REF_DEPTH}" in message


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ({"ref": "Account", "enum": ["a"]}, "must not combine 'ref' with 'enum'"),
        ({"ref": "Account", "pattern": "^a$"}, "must not combine 'ref' with 'pattern'"),
    ],
)
def test_ref_cannot_combine_with_enum_or_pattern(field: dict[str, Any], expected: str) -> None:
    with pytest.raises(OrchidPipelineShapeError) as rejected:
        _compile(
            Account={"fields": {"id": {"type": "string", "required": True}}},
            Plan={"fields": {"account": field}},
        )
    assert expected in str(rejected.value)


# ── Error context ───────────────────────────────────────────


def test_error_carries_shape_object_id_and_field_path() -> None:
    shape = _compile(Account={"fields": {"id": {"type": "string", "required": True}, "mrr": {"type": "number"}}}).get(
        "Account"
    )

    with pytest.raises(OrchidPipelineShapeError) as rejected:
        shape.validate({"id": "a", "mrr": "not-a-number"}, object_id="account:1")

    error = rejected.value
    assert error.code == "pipeline_shape_error"
    assert error.shape == "Account"
    assert error.object_id == "account:1"
    assert error.field_path == "$.mrr"
    message = str(error)
    assert "shape 'Account'" in message
    assert "object 'account:1'" in message
    assert "$.mrr" in message
    assert error.errors[0]["path"] == "$.mrr"
    assert error.errors[0]["type"] == "float_parsing"


def test_error_without_object_id_omits_it() -> None:
    shape = _compile(Account={"fields": {"id": {"type": "string", "required": True}}}).get("Account")
    with pytest.raises(OrchidPipelineShapeError) as rejected:
        shape.validate({})
    assert rejected.value.object_id is None
    assert "object" not in str(rejected.value)


def test_multiple_errors_report_every_path() -> None:
    shape = _compile(
        Account={
            "fields": {
                "id": {"type": "string", "required": True},
                "name": {"type": "string", "required": True},
            }
        }
    ).get("Account")

    with pytest.raises(OrchidPipelineShapeError) as rejected:
        shape.validate({}, object_id="account:1")
    message = str(rejected.value)
    assert "$.id" in message
    assert "$.name" in message
    assert rejected.value.field_path == "$.id"
    assert {entry["path"] for entry in rejected.value.errors} == {"$.id", "$.name"}


def test_validation_message_truncates_after_ten_errors() -> None:
    shape = _compile(Shape={"fields": {f"f{index}": {"type": "string", "required": True} for index in range(12)}}).get(
        "Shape"
    )

    with pytest.raises(OrchidPipelineShapeError) as rejected:
        shape.validate({})
    message = str(rejected.value)
    assert "... and 2 more error(s)" in message
    assert len(rejected.value.errors) == 10


def test_root_level_path_for_non_object_input() -> None:
    shape = _compile(Account={"fields": {"id": {"type": "string", "required": True}}}).get("Account")
    with pytest.raises(OrchidPipelineShapeError) as rejected:
        shape.validate(["not", "an", "object"])
    assert rejected.value.field_path == "$"
    assert "validation failed at $" in str(rejected.value)


def test_non_identifier_location_uses_bracket_notation() -> None:
    shape = _compile(Account={"fields": {"id": {"type": "string"}}}).get("Account")
    with pytest.raises(OrchidPipelineShapeError) as rejected:
        shape.validate({"not-an-id": 1})
    assert rejected.value.field_path == "$['not-an-id']"


# ── Shape versions ─────────────────────────────────────────


def test_version_metadata_and_mismatch(shape_module: str) -> None:
    shape = _compile(Account={"version": "2", "fields": {"id": {"type": "string", "required": True}}}).get("Account")
    assert shape.version == "2"
    shape.check_version("2")

    with pytest.raises(OrchidPipelineShapeError) as stale:
        shape.check_version("1", object_id="account:1")
    message = str(stale.value)
    assert "shape version '1'" in message
    assert "configured shape version is '2'" in message
    assert "allow_shape_version_mismatch: true" in message
    assert stale.value.object_id == "account:1"

    shape.check_version("1", allow_mismatch=True)
    assert shape.validate({"id": "a"}, shape_version="1", allow_version_mismatch=True).id == "a"
    with pytest.raises(OrchidPipelineShapeError):
        shape.validate({"id": "a"}, object_id="account:1", shape_version="1")

    class_shape = _compile(CustomerRisk={"class": f"{shape_module}.Risk", "version": "3"}).get("CustomerRisk")
    assert class_shape.version == "3"


def test_shape_level_allow_version_mismatch_suppresses_error() -> None:
    shape = _compile(
        Account={
            "version": "2",
            "allow_shape_version_mismatch": True,
            "fields": {"id": {"type": "string"}},
        }
    ).get("Account")

    assert shape.allow_shape_version_mismatch is True
    shape.check_version("1")
    assert shape.validate({"id": "a"}, shape_version="1").id == "a"
    assert shape.dump({"id": "a"}, shape_version="1") == {"id": "a"}


def test_pipeline_level_allow_version_mismatch_is_the_default() -> None:
    registry = compile_shapes(
        _declarations(Account={"version": "2", "fields": {"id": {"type": "string"}}}),
        allow_shape_version_mismatch=True,
    )
    shape = registry.get("Account")
    assert shape.allow_shape_version_mismatch is True
    shape.check_version("1")


def test_class_shape_honours_allow_version_mismatch(shape_module: str) -> None:
    shape = _compile(
        CustomerRisk={
            "class": f"{shape_module}.Risk",
            "version": "3",
            "allow_shape_version_mismatch": True,
        }
    ).get("CustomerRisk")
    shape.check_version("2")


def test_version_mismatch_still_raises_without_opt_in(shape_module: str) -> None:
    shape = _compile(Account={"version": "2", "fields": {"id": {"type": "string"}}}).get("Account")
    assert shape.allow_shape_version_mismatch is False
    with pytest.raises(OrchidPipelineShapeError):
        shape.check_version("1")


def test_version_must_be_a_non_empty_string() -> None:
    shape = _compile(Account={"fields": {"id": {"type": "string"}}}).get("Account")
    with pytest.raises(OrchidPipelineShapeError) as rejected:
        shape.check_version("")
    assert "shape_version must be a non-empty string" in str(rejected.value)


def test_shape_compiler_rejects_a_non_mapping() -> None:
    with pytest.raises(TypeError, match="must be a mapping"):
        ShapeCompiler(["not", "a", "mapping"])  # type: ignore[arg-type]


# ── Registry ───────────────────────────────────────────────


def test_registry_lookup_and_unknown_shape() -> None:
    registry = _compile(A={"fields": {"x": {"type": "string"}}}, B={"fields": {"y": {"type": "string"}}})
    assert registry.names == ("A", "B")
    assert "A" in registry
    assert len(registry) == 2
    assert list(registry) == ["A", "B"]
    assert isinstance(registry.get("A"), OrchidCompiledShape)

    with pytest.raises(OrchidPipelineShapeError) as unknown:
        registry.get("C")
    assert "unknown shape 'C'" in str(unknown.value)
    assert unknown.value.shape == "C"


def test_registry_validate_and_dump_delegate() -> None:
    registry = _compile(A={"fields": {"x": {"type": "string", "required": True}}})
    assert registry.validate("A", {"x": "v"}).x == "v"
    assert registry.dump("A", {"x": "v"}) == {"x": "v"}
    with pytest.raises(OrchidPipelineShapeError):
        registry.validate("A", {})


# ── Inline field maps (pipeline input) ────────────────────


def test_inline_shape_compiles_against_declared_refs() -> None:
    compiler = ShapeCompiler(_declarations(Account={"fields": {"id": {"type": "string", "required": True}}}))
    compiler.compile()
    inline = compiler.compile_inline(
        "PipelineInput",
        _fields(
            account={"ref": "Account", "required": True},
            note={"type": "string"},
        ),
        additional_properties=True,
    )

    assert inline.dump({"account": {"id": "a"}, "extra": 1}) == {
        "account": {"id": "a"},
        "note": None,
        "extra": 1,
    }
    with pytest.raises(OrchidPipelineShapeError) as missing:
        inline.validate({})
    assert missing.value.field_path == "$.account"
    assert inline.model_json_schema()["title"] == "PipelineInput"


def test_inline_shape_rejects_unknown_ref_and_empty_fields() -> None:
    compiler = ShapeCompiler(_declarations(Account={"fields": {"id": {"type": "string"}}}))
    compiler.compile()

    with pytest.raises(OrchidPipelineShapeError) as unknown:
        compiler.compile_inline("Input", _fields(x={"ref": "Nope"}))
    assert "unknown shape reference 'Nope'" in str(unknown.value)

    with pytest.raises(OrchidPipelineShapeError):
        compiler.compile_inline("Input", {})


def test_field_compilation_failure_is_wrapped() -> None:
    compiler = ShapeCompiler(_declarations(Account={"fields": {"id": {"type": "string"}}}))
    compiler.compile()

    with pytest.raises(OrchidPipelineShapeError) as rejected:
        # A raw dict instead of an ``OrchidPipelineFieldConfig`` stands in for
        # any unexpected failure while building a field definition.
        compiler.compile_inline("Input", {"broken": {"type": "string"}})  # type: ignore[dict-item]

    error = rejected.value
    assert error.shape == "Input"
    assert "shape 'Input'" in str(error)
    assert "cannot compile field 'broken'" in str(error)
