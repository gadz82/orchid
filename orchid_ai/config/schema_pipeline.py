"""YAML schema for the ``pipelines:`` configuration block.

The block is optional. When it is absent, :class:`orchid_ai.config.schema_agent.OrchidAgentsConfig`
parses exactly as before and no pipeline work happens. When present, every
model forbids unknown keys so a typo fails at load time instead of surfacing as
a confusing runtime error.

The models below mirror the pipeline definition model (shapes, relations,
definitions, jobs) and enforce the *static* rules from the specification:

* names and identifiers,
* reference existence (shapes, relations, jobs, sub-pipelines),
* acyclicity and sub-pipeline depth,
* decision branch/fallback/join consistency,
* map template constraints,
* template and JMESPath syntax,
* the reserved ``__input__`` object,
* duplicate aliases and parallel static writes,
* positive limits and timeouts.

Checks that need knowledge of the wider agent configuration — leaf-agent
references, tool references, decision-model keys, exposed tool collisions, and
async triggers — run in :mod:`orchid_ai.config.schema_agent` after this block
validates on its own. No consumer class is imported at validation time.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..core.pipeline import OrchidPipelineMappingError
from .pipeline_templates import validate_expression, validate_template
from .schema_events import ChatBinding, IdentityClaim

# ── Names and syntax ─────────────────────────────────────────

_PIPELINE_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{1,62}$")
_JOB_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")
_SHAPE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_RELATION_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DOTTED_PATH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+$")
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

_FIELD_TYPES = frozenset({"string", "integer", "number", "boolean", "object", "array", "null"})
_NUMERIC_TYPES = frozenset({"integer", "number"})
_LENGTH_TYPES = frozenset({"string", "array"})
_MAX_PATTERN_CHARS = 200
_MAX_OBJECT_ID_CHARS = 256
_BRACE_QUANTIFIER_RE = re.compile(r"\{\d+(?:,\d*)?\}")

_PIPELINE_INPUT_OBJECT_ID = "__input__"
_PIPELINE_INPUT_OBJECT_TYPE = "__pipeline_input__"


def _find_group_end(pattern: str, start: int) -> int | None:
    """Return the index of the ``)`` matching the ``(`` at ``start``."""
    depth = 0
    in_class = False
    escaped = False
    for index in range(start, len(pattern)):
        char = pattern[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if in_class:
            if char == "]":
                in_class = False
            continue
        if char == "[":
            in_class = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _split_top_level_branches(inner: str) -> list[str]:
    branches: list[str] = []
    start = 0
    depth = 0
    in_class = False
    escaped = False
    for index, char in enumerate(inner):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if in_class:
            if char == "]":
                in_class = False
            continue
        if char == "[":
            in_class = True
        elif char == "(":
            depth += 1
        elif char == ")" and depth:
            depth -= 1
        elif char == "|" and depth == 0:
            branches.append(inner[start:index])
            start = index + 1
    branches.append(inner[start:])
    return branches


def _has_inner_quantifier(text: str) -> bool:
    in_class = False
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if in_class:
            if char == "]":
                in_class = False
            continue
        if char == "[":
            in_class = True
        elif char in "*+" or (char == "{" and _BRACE_QUANTIFIER_RE.match(text, index) is not None):
            return True
    return False


def _has_overlapping_alternatives(inner: str) -> bool:
    if inner.startswith("?"):
        # Drop a non-capturing / lookaround / named-group prefix so
        # ``(?:a|aa)+`` is analysed like ``(a|aa)+``.
        separator = inner.find(":")
        if separator == -1:
            separator = inner.find(">")
        if separator != -1:
            inner = inner[separator + 1 :]
    branches = _split_top_level_branches(inner)
    if len(branches) < 2:
        return False
    for index, left in enumerate(branches):
        for right in branches[index + 1 :]:
            if left == right or left.startswith(right) or right.startswith(left):
                return True
    return False


def _quantifier_at(pattern: str, index: int) -> tuple[bool, int]:
    """Return ``(is_quantified, next_index)`` for a quantifier at ``index``."""
    if index >= len(pattern):
        return False, index
    char = pattern[index]
    if char in "*+?":
        end = index + 1
        if end < len(pattern) and pattern[end] in "?+":
            end += 1  # lazy / possessive modifier
        return True, end
    if char == "{":
        match = _BRACE_QUANTIFIER_RE.match(pattern, index)
        if match is not None:
            end = match.end()
            if end < len(pattern) and pattern[end] in "?+":
                end += 1
            return True, end
    return False, index


def _has_adjacent_quantified_atoms(pattern: str) -> bool:
    """Detect runs like ``a*a*``, ``\\w+\\w+``, or repeated quantified groups."""
    previous_atom: str | None = None
    previous_quantified = False
    index = 0
    length = len(pattern)
    while index < length:
        char = pattern[index]
        if char in "|)^$":
            previous_atom = None
            previous_quantified = False
            index += 1
            continue
        if char in "*+?" and previous_atom is None:
            # A group prefix (``(?:``) or a quantifier after a group; neither
            # is an atom for adjacency purposes.
            index += 1
            continue

        if char == "(":
            end = _find_group_end(pattern, index)
            if end is None:
                index += 1
                continue
            # The group body is its own adjacency scope. Recursing here keeps
            # inner runs detectable while the group itself acts as one atom in
            # the enclosing scope (``(a|b)*(a|b)*``).
            if _has_adjacent_quantified_atoms(pattern[index + 1 : end]):
                return True
            atom = pattern[index : end + 1]
            index = end + 1
        elif char == "[":
            start = index
            index += 1
            while index < length:
                if pattern[index] == "\\":
                    index += 2
                    continue
                if pattern[index] == "]":
                    index += 1
                    break
                index += 1
            atom = pattern[start:index]
        elif char == "\\":
            start = index
            index = min(length, index + 2)
            atom = pattern[start:index]
        else:
            atom = char
            index += 1

        quantified, index = _quantifier_at(pattern, index)
        if quantified and previous_quantified and previous_atom == atom:
            return True
        previous_atom = atom
        previous_quantified = quantified
    return False


def _iter_group_starts(pattern: str) -> list[int]:
    starts: list[int] = []
    in_class = False
    escaped = False
    for index, char in enumerate(pattern):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if in_class:
            if char == "]":
                in_class = False
            continue
        if char == "[":
            in_class = True
        elif char == "(":
            starts.append(index)
    return starts


def _looks_catastrophic(pattern: str) -> bool:
    """Conservative heuristic for catastrophic-backtracking patterns.

    This is deliberately *not* a full safe-regex engine. It rejects two
    families that dominate practical ReDoS reports:

    * adjacent quantified atoms of the same literal or character class
      (``a*a*``, ``\\w+\\w+``, ``[a-z]*[a-z]*``), including runs of three or
      more;
    * a quantified group whose body already contains a quantifier (``(a+)+``)
      or whose alternatives overlap by prefix (``(a|aa)+``).

    A quantified alternation without inner quantifiers and without overlapping
    prefixes is allowed (``(foo|bar)+``). Everything else is left to the
    bounded pattern length and the interpreter's compiler, so false negatives
    remain possible.
    """
    if _has_adjacent_quantified_atoms(pattern):
        return True

    for index in _iter_group_starts(pattern):
        end = _find_group_end(pattern, index)
        if end is None:
            continue
        next_index = end + 1
        if next_index >= len(pattern):
            continue
        next_char = pattern[next_index]
        quantified = next_char in "*+" or (
            next_char == "{" and _BRACE_QUANTIFIER_RE.match(pattern, next_index) is not None
        )
        if not quantified:
            continue
        inner = pattern[index + 1 : end]
        if _has_inner_quantifier(inner) or _has_overlapping_alternatives(inner):
            return True
    return False


def _require_object_id_template(value: str, *, label: str) -> str:
    """Validate an object-id template that the session expands at commit time."""
    if not value.strip():
        raise ValueError(f"{label} must be non-empty")
    if len(value) > _MAX_OBJECT_ID_CHARS:
        raise ValueError(f"{label} must be at most {_MAX_OBJECT_ID_CHARS} characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{label} must not contain control characters")
    if "{{" in value or "}}" in value:
        _require_template(value, label=label)
    return value


def _require_template(value: str, *, label: str) -> str:
    """Reject unbalanced delimiters and non-JMESPath expressions.

    The grammar and compiler live in :mod:`orchid_ai.config.pipeline_templates` so
    a template accepted here is guaranteed to render, and a template the renderer
    would reject is a configuration error instead of a first-run failure.
    """
    try:
        return validate_template(value, path=label)
    except OrchidPipelineMappingError as exc:
        raise ValueError(str(exc)) from exc


def _require_jmespath(value: str, *, label: str) -> str:
    try:
        return validate_expression(value, path=label)
    except OrchidPipelineMappingError as exc:
        raise ValueError(str(exc)) from exc


def _walk_strings(value: Any, *, path: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, str):
        found.append((path, value))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            found.extend(_walk_strings(item, path=f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_walk_strings(item, path=f"{path}[{index}]"))
    return found


def _require_templates(value: Any, *, label: str) -> None:
    for path, text in _walk_strings(value, path=label):
        _require_template(text, label=path)


def _matches_field_type(value: Any, type_name: str) -> bool:
    if type_name == "null":
        return value is None
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "array":
        return isinstance(value, (list, tuple))
    if type_name == "object":
        return isinstance(value, Mapping)
    return True


def _is_identifier(value: str) -> bool:
    return bool(_IDENTIFIER_RE.fullmatch(value))


# ── Shapes ───────────────────────────────────────────────────


class OrchidPipelineFieldConfig(BaseModel):
    """One field inside a shape (or an inline nested object schema)."""

    type: Literal["string", "integer", "number", "boolean", "object", "array", "null"] | None = None
    ref: str | None = None
    required: bool = False
    default: Any = None
    enum: list[Any] | None = None
    description: str = ""
    items: OrchidPipelineFieldConfig | None = None
    properties: dict[str, OrchidPipelineFieldConfig] | None = None
    additional_properties: bool = False
    min: float | None = None
    max: float | None = None
    min_length: int | None = Field(default=None, ge=0)
    max_length: int | None = Field(default=None, ge=0)
    pattern: str | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("ref")
    @classmethod
    def _validate_ref(cls, value: str | None) -> str | None:
        if value is not None and not _SHAPE_NAME_RE.fullmatch(value):
            raise ValueError(f"shape reference {value!r} is not a valid shape name")
        return value

    @field_validator("pattern")
    @classmethod
    def _validate_pattern(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if len(value) > _MAX_PATTERN_CHARS:
            raise ValueError(f"field pattern must be at most {_MAX_PATTERN_CHARS} characters")
        if _looks_catastrophic(value):
            raise ValueError(
                "field pattern risks catastrophic backtracking (heuristic check); "
                "avoid adjacent quantified atoms and quantified groups with inner quantifiers "
                "or overlapping alternatives"
            )
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"field pattern is not a valid regular expression: {exc}") from exc
        return value

    @model_validator(mode="after")
    def _validate_shape_of_field(self) -> OrchidPipelineFieldConfig:
        has_type = self.type is not None
        has_ref = self.ref is not None
        if has_type == has_ref:
            raise ValueError("field must set exactly one of 'type' or 'ref'")

        if self.type == "array":
            if self.items is None:
                raise ValueError("array fields require 'items'")
        elif self.items is not None:
            raise ValueError("'items' is only valid for array fields")

        if self.type == "object":
            if self.properties is not None:
                for name in self.properties:
                    if not _is_identifier(name):
                        raise ValueError(f"nested property name {name!r} must be an identifier")
        elif self.properties is not None:
            raise ValueError("'properties' is only valid for object fields")

        if self.additional_properties and self.type != "object" and self.ref is None:
            raise ValueError("'additional_properties' is only valid for object fields and shape references")

        if self.min is not None or self.max is not None:
            if self.type not in _NUMERIC_TYPES:
                raise ValueError("'min'/'max' are only valid for integer or number fields")
            if self.min is not None and self.max is not None and self.min > self.max:
                raise ValueError("field 'min' must not exceed 'max'")
        if self.min_length is not None or self.max_length is not None:
            if self.type not in _LENGTH_TYPES:
                raise ValueError("'min_length'/'max_length' are only valid for string or array fields")
            if self.min_length is not None and self.max_length is not None and self.min_length > self.max_length:
                raise ValueError("field 'min_length' must not exceed 'max_length'")

        if self.enum is not None:
            if not self.enum:
                raise ValueError("field 'enum' must not be empty")
            seen: list[Any] = []
            for item in self.enum:
                if any(item == existing for existing in seen):
                    raise ValueError(f"field 'enum' contains duplicate value {item!r}")
                seen.append(item)
                if self.type is not None and self.type != "null" and not _matches_field_type(item, self.type):
                    raise ValueError(f"enum value {item!r} does not match field type {self.type!r}")

        if self.default is not None:
            if self.type is not None and not _matches_field_type(self.default, self.type):
                raise ValueError(f"field default {self.default!r} does not match field type {self.type!r}")
            if self.enum is not None and not any(self.default == item for item in self.enum):
                raise ValueError(f"field default {self.default!r} is not one of the declared enum values")
        return self

    def iter_refs(self) -> list[str]:
        refs = [self.ref] if self.ref is not None else []
        if self.items is not None:
            refs.extend(self.items.iter_refs())
        for field in (self.properties or {}).values():
            refs.extend(field.iter_refs())
        return refs


class OrchidPipelineShapeConfig(BaseModel):
    """A named record definition, either field-based or class-based."""

    description: str = ""
    version: str = "1"
    allow_shape_version_mismatch: bool = False
    additional_properties: bool = False
    fields: dict[str, OrchidPipelineFieldConfig] | None = None
    class_path: str | None = Field(default=None, alias="class")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("shape version must be non-empty")
        return value

    @field_validator("class_path")
    @classmethod
    def _validate_class_path(cls, value: str | None) -> str | None:
        if value is not None and not _DOTTED_PATH_RE.fullmatch(value):
            raise ValueError(f"shape class {value!r} must be a dotted import path")
        return value

    @model_validator(mode="after")
    def _fields_xor_class(self) -> OrchidPipelineShapeConfig:
        if (self.fields is None) == (self.class_path is None):
            raise ValueError("shape must set exactly one of 'fields' or 'class'")
        if self.fields is not None:
            if not self.fields:
                raise ValueError("shape 'fields' must declare at least one field")
            for name in self.fields:
                if not _is_identifier(name):
                    raise ValueError(f"shape field name {name!r} must be an identifier")
        return self

    def iter_refs(self) -> list[str]:
        refs: list[str] = []
        for field in (self.fields or {}).values():
            refs.extend(field.iter_refs())
        return refs


class OrchidPipelineRelationConfig(BaseModel):
    """A typed relation: allowed endpoint shapes plus optional edge data."""

    from_: list[str] = Field(alias="from")
    to: list[str]
    description: str = ""
    version: str = "1"
    fields: dict[str, OrchidPipelineFieldConfig] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("from_", "to")
    @classmethod
    def _endpoints_non_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("relation endpoints must not be empty")
        for name in value:
            if not isinstance(name, str) or not _SHAPE_NAME_RE.fullmatch(name):
                raise ValueError(f"relation endpoint {name!r} is not a valid shape name")
        return value

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("relation version must be non-empty")
        return value

    @model_validator(mode="after")
    def _unique_endpoints(self) -> OrchidPipelineRelationConfig:
        if len(set(self.from_)) != len(self.from_):
            raise ValueError("relation 'from' endpoints must be unique")
        if len(set(self.to)) != len(self.to):
            raise ValueError("relation 'to' endpoints must be unique")
        for name in self.fields:
            if not _is_identifier(name):
                raise ValueError(f"relation field name {name!r} must be an identifier")
        return self

    def iter_refs(self) -> list[str]:
        refs: list[str] = []
        for field in self.fields.values():
            refs.extend(field.iter_refs())
        return refs


# ── Definition input / output ────────────────────────────────


class OrchidPipelineInputConfig(BaseModel):
    """Caller input: a declared shape or an inline field map, plus defaults."""

    shape: str | None = None
    fields: dict[str, OrchidPipelineFieldConfig] | None = None
    defaults: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @field_validator("shape")
    @classmethod
    def _validate_shape(cls, value: str | None) -> str | None:
        if value is not None and not _SHAPE_NAME_RE.fullmatch(value):
            raise ValueError(f"input shape {value!r} is not a valid shape name")
        return value

    @model_validator(mode="after")
    def _shape_xor_fields(self) -> OrchidPipelineInputConfig:
        if (self.shape is None) == (self.fields is None):
            raise ValueError("pipeline input must set exactly one of 'shape' or 'fields'")
        if self.fields is not None:
            if not self.fields:
                raise ValueError("pipeline input 'fields' must declare at least one field")
            for name in self.fields:
                if not _is_identifier(name):
                    raise ValueError(f"input field name {name!r} must be an identifier")
        return self


class OrchidPipelineOutputConfig(BaseModel):
    """Definition output projection (shape plus mapping or a single ``from``)."""

    shape: str | None = None
    mapping: dict[str, str] | None = None
    from_: str | None = Field(default=None, alias="from")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("shape")
    @classmethod
    def _validate_shape(cls, value: str | None) -> str | None:
        if value is not None and not _SHAPE_NAME_RE.fullmatch(value):
            raise ValueError(f"output shape {value!r} is not a valid shape name")
        return value

    @field_validator("from_")
    @classmethod
    def _validate_from(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _require_jmespath(value, label="pipeline output 'from'")

    @model_validator(mode="after")
    def _shape_requires_projection(self) -> OrchidPipelineOutputConfig:
        if self.shape is None:
            if self.mapping is not None or self.from_ is not None:
                raise ValueError("pipeline output mapping/from requires 'shape'")
            return self
        if (self.mapping is None) == (self.from_ is None):
            raise ValueError("pipeline output with 'shape' requires exactly one of 'mapping' or 'from'")
        if self.mapping is not None:
            if not self.mapping:
                raise ValueError("pipeline output 'mapping' must not be empty")
            for name in self.mapping:
                if not _is_identifier(name):
                    raise ValueError(f"output mapping key {name!r} must be an identifier")
        return self


class OrchidPipelineJobOutputConfig(BaseModel):
    """Structured or free-text output requested from an LLM/agent job."""

    shape: str | None = None
    format: Literal["json", "text"] = "json"

    model_config = ConfigDict(extra="forbid")

    @field_validator("shape")
    @classmethod
    def _validate_shape(cls, value: str | None) -> str | None:
        if value is not None and not _SHAPE_NAME_RE.fullmatch(value):
            raise ValueError(f"output shape {value!r} is not a valid shape name")
        return value


# ── Retry, limits, defaults ──────────────────────────────────


class OrchidPipelineRetryConfig(BaseModel):
    """Per-job (or default) retry policy."""

    max: int = Field(default=0, ge=0)
    backoff: Literal["fixed", "linear", "exponential"] = "exponential"
    jitter: bool = True
    initial_delay_seconds: float = Field(default=1.0, gt=0.0)
    max_delay_seconds: float = Field(default=300.0, gt=0.0)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _max_ge_initial(self) -> OrchidPipelineRetryConfig:
        if self.max_delay_seconds < self.initial_delay_seconds:
            raise ValueError("retry.max_delay_seconds must be >= initial_delay_seconds")
        return self


class OrchidPipelineLimitsConfig(BaseModel):
    """Hard run caps. Per-definition values may only tighten these."""

    timeout_seconds: float | None = Field(default=None, gt=0.0)
    max_jobs: int = Field(default=100, ge=1)
    max_objects: int = Field(default=10_000, ge=1)
    max_relations: int = Field(default=50_000, ge=1)
    max_object_bytes: int = Field(default=1_000_000, ge=1)
    max_graph_bytes: int = Field(default=20_000_000, ge=1)
    max_llm_calls: int = Field(default=50, ge=1)
    max_prompt_chars: int = Field(default=32_000, ge=1)
    max_context_chars: int = Field(default=8_000, ge=1)
    max_map_items: int = Field(default=100, ge=1)
    max_parallel: int = Field(default=8, ge=1)
    max_subpipeline_depth: int = Field(default=3, ge=1)

    model_config = ConfigDict(extra="forbid")


class OrchidPipelineDefinitionLimitsConfig(BaseModel):
    """Optional per-definition overrides; every field is a ceiling."""

    timeout_seconds: float | None = Field(default=None, gt=0.0)
    max_jobs: int | None = Field(default=None, ge=1)
    max_objects: int | None = Field(default=None, ge=1)
    max_relations: int | None = Field(default=None, ge=1)
    max_object_bytes: int | None = Field(default=None, ge=1)
    max_graph_bytes: int | None = Field(default=None, ge=1)
    max_llm_calls: int | None = Field(default=None, ge=1)
    max_prompt_chars: int | None = Field(default=None, ge=1)
    max_context_chars: int | None = Field(default=None, ge=1)
    max_map_items: int | None = Field(default=None, ge=1)
    max_parallel: int | None = Field(default=None, ge=1)
    max_subpipeline_depth: int | None = Field(default=None, ge=1)

    model_config = ConfigDict(extra="forbid")


class OrchidPipelineDefaultsConfig(BaseModel):
    """Values inherited by every job; an explicit job value wins."""

    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    retry: OrchidPipelineRetryConfig = Field(default_factory=OrchidPipelineRetryConfig)
    timeout_seconds: float | None = Field(default=None, gt=0.0)

    model_config = ConfigDict(extra="forbid")

    @field_validator("model")
    @classmethod
    def _model_non_empty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("defaults.model must be non-empty")
        return value


# ── Object / relation writes ─────────────────────────────────


class OrchidPipelineObjectWriteConfig(BaseModel):
    """Declarative object write — also the job's write allowlist entry."""

    id: str
    shape: str
    from_: str | None = Field(default=None, alias="from")
    mode: Literal["create", "update", "upsert", "replace"] = "upsert"
    expected_revision: int | str | None = None
    when: str | None = None
    as_: str | None = Field(default=None, alias="as")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _require_object_id_template(value, label="object write id")

    @field_validator("shape")
    @classmethod
    def _validate_shape(cls, value: str) -> str:
        if not _SHAPE_NAME_RE.fullmatch(value):
            raise ValueError(f"object write shape {value!r} is not a valid shape name")
        return value

    @field_validator("as_")
    @classmethod
    def _validate_alias(cls, value: str | None) -> str | None:
        if value is not None and not _is_identifier(value):
            raise ValueError(f"object write alias {value!r} must be an identifier")
        return value

    @field_validator("expected_revision", mode="before")
    @classmethod
    def _reject_non_integral_revision(cls, value: Any) -> Any:
        # ``bool`` is an ``int`` subclass and Pydantic's smart union would
        # coerce ``1.0`` to ``1``; both hide a config mistake.
        if isinstance(value, (bool, float, bytes, bytearray)):
            # Pydantic only converts ValueError/AssertionError into a
            # ValidationError, so this must not be a TypeError.
            raise ValueError(  # noqa: TRY004
                "object write expected_revision must be an integer or JMESPath expression"
            )
        return value

    @field_validator("expected_revision")
    @classmethod
    def _validate_expected_revision(cls, value: int | str | None) -> int | str | None:
        if value is None:
            return value
        if isinstance(value, int):
            if value < 1:
                raise ValueError("object write expected_revision must be at least 1")
            return value
        if not value.strip():
            raise ValueError("object write expected_revision must not be empty")
        if value.isdigit():
            if int(value) < 1:
                raise ValueError("object write expected_revision must be at least 1")
            return value
        return _require_jmespath(value, label="object write expected_revision")

    @field_validator("when")
    @classmethod
    def _validate_when(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if "{{" in value or "}}" in value:
            return _require_template(value, label="object write when")
        return _require_jmespath(value, label="object write when")

    @field_validator("from_")
    @classmethod
    def _validate_from(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _require_jmespath(value, label="object write from")

    @model_validator(mode="after")
    def _reserved_input(self) -> OrchidPipelineObjectWriteConfig:
        if self.id == _PIPELINE_INPUT_OBJECT_ID:
            raise ValueError(f"object write id must not target the reserved {_PIPELINE_INPUT_OBJECT_ID!r} object")
        if self.shape == _PIPELINE_INPUT_OBJECT_TYPE:
            raise ValueError(f"object write shape must not be the reserved {_PIPELINE_INPUT_OBJECT_TYPE!r} type")
        return self


class OrchidPipelineRelationWriteConfig(BaseModel):
    """Declarative relation write — also the job's link allowlist entry."""

    type: str
    from_: str = Field(alias="from")
    to: str
    data: dict[str, Any] | None = None
    when: str | None = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        if not _RELATION_NAME_RE.fullmatch(value):
            raise ValueError(f"relation type {value!r} is not a valid relation name")
        return value

    @field_validator("from_", "to")
    @classmethod
    def _endpoint_non_empty(cls, value: str) -> str:
        return _require_object_id_template(value, label="relation write endpoint")

    @field_validator("when")
    @classmethod
    def _validate_when(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if "{{" in value or "}}" in value:
            return _require_template(value, label="relation write when")
        return _require_jmespath(value, label="relation write when")

    @model_validator(mode="after")
    def _validate_templates(self) -> OrchidPipelineRelationWriteConfig:
        _require_templates({"from": self.from_, "to": self.to, "data": self.data}, label="relation write")
        return self


# ── Job kinds ────────────────────────────────────────────────


class _OrchidPipelineJobBase(BaseModel):
    description: str = ""
    depends_on: list[str] = Field(default_factory=list)
    join: Literal["all", "any"] = "all"
    when: str | None = None
    writes: list[OrchidPipelineObjectWriteConfig] = Field(default_factory=list)
    links: list[OrchidPipelineRelationWriteConfig] = Field(default_factory=list)
    retry: OrchidPipelineRetryConfig | None = None
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    on_error: Literal["fail", "continue", "skip"] = "fail"
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @field_validator("when")
    @classmethod
    def _validate_when(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if "{{" in value or "}}" in value:
            return _require_template(value, label="job when")
        return _require_jmespath(value, label="job when")

    @model_validator(mode="after")
    def _duplicate_aliases(self) -> _OrchidPipelineJobBase:
        aliases = [write.as_ for write in self.writes if write.as_ is not None]
        if len(set(aliases)) != len(aliases):
            raise ValueError("job declares duplicate write aliases")
        return self


class OrchidPipelineDeterministicJobConfig(_OrchidPipelineJobBase):
    kind: Literal["deterministic"] = "deterministic"
    uses: str
    params: dict[str, Any] = Field(default_factory=dict)
    allow_dynamic_writes: bool = True

    @field_validator("uses")
    @classmethod
    def _validate_uses(cls, value: str) -> str:
        if not _DOTTED_PATH_RE.fullmatch(value):
            raise ValueError(f"deterministic 'uses' {value!r} must be a dotted import path")
        return value


class OrchidPipelineLLMRAGConfig(BaseModel):
    enabled: bool = False
    namespace: str | None = None
    k: int = Field(default=5, ge=1)
    strategy: str = "simple"
    max_context_chars: int = Field(default=3000, ge=1)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _namespace_when_enabled(self) -> OrchidPipelineLLMRAGConfig:
        if self.enabled and (self.namespace is None or not self.namespace.strip()):
            raise ValueError("llm rag.namespace is required when rag is enabled")
        return self


class OrchidPipelineLLMJobConfig(_OrchidPipelineJobBase):
    kind: Literal["llm"] = "llm"
    model: str | None = None
    system_prompt: str | None = None
    prompt: str
    output: OrchidPipelineJobOutputConfig | None = None
    rag: OrchidPipelineLLMRAGConfig | None = None
    tools: list[str] = Field(default_factory=list)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)

    @field_validator("model")
    @classmethod
    def _model_non_empty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("llm job 'model' must be non-empty")
        return value

    @field_validator("prompt")
    @classmethod
    def _validate_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("llm job 'prompt' must be non-empty")
        return _require_template(value, label="llm job prompt")

    @field_validator("system_prompt")
    @classmethod
    def _validate_system_prompt(cls, value: str | None) -> str | None:
        if value is not None:
            _require_template(value, label="llm job system_prompt")
        return value

    @field_validator("tools")
    @classmethod
    def _validate_tools(cls, value: list[str]) -> list[str]:
        for name in value:
            if not name.strip():
                raise ValueError("llm job tool names must be non-empty")
        return value


class OrchidPipelineAgentJobConfig(_OrchidPipelineJobBase):
    kind: Literal["agent"] = "agent"
    agent: str
    prompt: str
    input: dict[str, Any] | None = None
    output: OrchidPipelineJobOutputConfig | None = None
    persist_memory: bool = False

    @field_validator("agent")
    @classmethod
    def _agent_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("agent job 'agent' must be non-empty")
        return value

    @field_validator("prompt")
    @classmethod
    def _validate_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("agent job 'prompt' must be non-empty")
        return _require_template(value, label="agent job prompt")

    @model_validator(mode="after")
    def _validate_input_templates(self) -> OrchidPipelineAgentJobConfig:
        if self.input is not None:
            _require_templates(self.input, label="agent job input")
        return self


class OrchidPipelineDecisionJobConfig(_OrchidPipelineJobBase):
    kind: Literal["decision"] = "decision"
    decision_model: str
    question: str
    labels: list[str]
    branches: dict[str, str] = Field(default_factory=dict)
    fallback: str | None = None
    context_from: str | None = None
    on_low_confidence: Literal["fallback", "llm", "fail"] = "fallback"

    @field_validator("decision_model")
    @classmethod
    def _model_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("decision job 'decision_model' must be non-empty")
        return value

    @field_validator("question")
    @classmethod
    def _question_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("decision job 'question' must be non-empty")
        return _require_template(value, label="decision job question")

    @field_validator("labels")
    @classmethod
    def _validate_labels(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("decision job 'labels' must not be empty")
        seen: set[str] = set()
        for label in value:
            if not label.strip():
                raise ValueError("decision job labels must be non-empty")
            if label in seen:
                raise ValueError(f"decision job label {label!r} is duplicated")
            seen.add(label)
        return value

    @field_validator("context_from")
    @classmethod
    def _validate_context_from(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _require_jmespath(value, label="decision job context_from")

    @model_validator(mode="after")
    def _branches_match_labels(self) -> OrchidPipelineDecisionJobConfig:
        known = set(self.labels)
        for label, target in self.branches.items():
            if label not in known:
                raise ValueError(f"decision branch label {label!r} is not declared in 'labels'")
            if not isinstance(target, str) or not target.strip():
                raise ValueError(f"decision branch {label!r} must name a target job")
        return self


_RESERVED_MAP_BINDINGS = frozenset(
    {"input", "run", "jobs", "objects", "types", "aliases", "summary", "index", "parent_job"}
)


class OrchidPipelineMapJobConfig(_OrchidPipelineJobBase):
    kind: Literal["map"] = "map"
    over: str
    as_: str = Field(alias="as")
    job: str
    concurrency: int = Field(default=1, ge=1)
    on_item_error: Literal["fail", "skip"] = "fail"

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("over")
    @classmethod
    def _validate_over(cls, value: str) -> str:
        return _require_jmespath(value, label="map job 'over'")

    @field_validator("as_")
    @classmethod
    def _validate_as(cls, value: str) -> str:
        if not _is_identifier(value):
            raise ValueError(f"map job 'as' binding {value!r} must be an identifier")
        if value in _RESERVED_MAP_BINDINGS:
            raise ValueError(f"map job 'as' binding {value!r} shadows a reserved pipeline context key")
        return value

    @field_validator("job")
    @classmethod
    def _validate_job(cls, value: str) -> str:
        if not _JOB_ID_RE.fullmatch(value):
            raise ValueError(f"map job template {value!r} is not a valid job id")
        return value


class OrchidPipelineSubPipelineJobConfig(_OrchidPipelineJobBase):
    kind: Literal["subpipeline"] = "subpipeline"
    pipeline: str
    input: dict[str, Any] | None = None
    merge_graph: bool = False
    output_from: str = "output"

    @field_validator("pipeline")
    @classmethod
    def _validate_pipeline(cls, value: str) -> str:
        if not _PIPELINE_NAME_RE.fullmatch(value):
            raise ValueError(f"sub-pipeline reference {value!r} is not a valid pipeline name")
        return value

    @field_validator("output_from")
    @classmethod
    def _validate_output_from(cls, value: str) -> str:
        return _require_jmespath(value, label="sub-pipeline job output_from")

    @model_validator(mode="after")
    def _validate_input_templates(self) -> OrchidPipelineSubPipelineJobConfig:
        if self.input is not None:
            _require_templates(self.input, label="sub-pipeline job input")
        return self


class OrchidPipelineEmitJobConfig(_OrchidPipelineJobBase):
    kind: Literal["emit"] = "emit"
    signal: str
    payload: dict[str, Any] = Field(default_factory=dict)
    dedupe_key: str | None = None
    identity: IdentityClaim | None = None
    chat_binding: ChatBinding | None = None

    @field_validator("signal")
    @classmethod
    def _validate_signal(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("emit job 'signal' must be non-empty")
        return value

    @model_validator(mode="after")
    def _validate_templates(self) -> OrchidPipelineEmitJobConfig:
        _require_templates(self.payload, label="emit job payload")
        if self.dedupe_key is not None:
            if not self.dedupe_key.strip():
                raise ValueError("emit job 'dedupe_key' must be non-empty")
            _require_template(self.dedupe_key, label="emit job dedupe_key")
        return self


class OrchidPipelineCustomJobConfig(_OrchidPipelineJobBase):
    kind: Literal["custom"] = "custom"
    class_path: str = Field(alias="class")
    params: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("class_path")
    @classmethod
    def _validate_class_path(cls, value: str) -> str:
        if not _DOTTED_PATH_RE.fullmatch(value):
            raise ValueError(f"custom job 'class' {value!r} must be a dotted import path")
        return value


OrchidPipelineJobConfig = Annotated[
    OrchidPipelineDeterministicJobConfig
    | OrchidPipelineLLMJobConfig
    | OrchidPipelineAgentJobConfig
    | OrchidPipelineDecisionJobConfig
    | OrchidPipelineMapJobConfig
    | OrchidPipelineSubPipelineJobConfig
    | OrchidPipelineEmitJobConfig
    | OrchidPipelineCustomJobConfig,
    Field(discriminator="kind"),
]


# ── Exposure, async, interpretation ─────────────────────────


class OrchidPipelineExposeConfig(BaseModel):
    """Opt-in exposure of a definition as an agent-callable tool."""

    as_tool: bool = False
    tool_name: str | None = None
    description: str | None = None
    require_role: str | None = None
    requires_approval: bool = False
    parallel_safe: bool = False
    max_tool_result_chars: int = Field(default=8000, ge=1)
    interpretation: Literal["none", "template", "llm", "caller"] | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("tool_name")
    @classmethod
    def _validate_tool_name(cls, value: str | None) -> str | None:
        if value is not None and not _TOOL_NAME_RE.fullmatch(value):
            raise ValueError(f"exposed tool name {value!r} must match [A-Za-z0-9][A-Za-z0-9_-]{{0,63}}")
        return value

    @model_validator(mode="after")
    def _as_tool_requires_description(self) -> OrchidPipelineExposeConfig:
        if self.as_tool and (self.description is None or not self.description.strip()):
            raise ValueError("expose.as_tool requires a non-empty 'description'")
        return self


class OrchidPipelineAsyncConfig(BaseModel):
    """Async launch binding; the trigger existence check runs cross-block."""

    enabled: bool = False
    signal: str | None = None
    trigger_id: str | None = None
    mode: Literal["signal", "background"] = "signal"
    timeout_seconds: float | None = Field(default=None, gt=0.0)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _required_when_enabled(self) -> OrchidPipelineAsyncConfig:
        if not self.enabled:
            return self
        if not self.trigger_id or not self.trigger_id.strip():
            raise ValueError("async.enabled requires 'trigger_id'")
        if self.mode == "signal" and (not self.signal or not self.signal.strip()):
            raise ValueError("async.mode 'signal' requires 'signal'")
        return self


class OrchidPipelineInterpretationConfig(BaseModel):
    """How a run's human-readable summary is produced."""

    mode: Literal["none", "template", "llm", "caller"] = "none"
    template: str | None = None
    model: str | None = None
    max_objects: int = Field(default=20, ge=1)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _mode_requirements(self) -> OrchidPipelineInterpretationConfig:
        if self.mode == "template":
            if self.template is None or not self.template.strip():
                raise ValueError("interpretation.mode 'template' requires a template string")
            _require_template(self.template, label="interpretation template")
        if self.mode == "llm":
            if self.model is None or not self.model.strip():
                raise ValueError("interpretation.mode 'llm' requires a model")
            if self.template is not None:
                _require_template(self.template, label="interpretation template")
        return self


# ── Definition ───────────────────────────────────────────────


class OrchidPipelineDefinitionConfig(BaseModel):
    """One named pipeline: input/output, jobs, exposure, async, limits."""

    description: str = ""
    version: str = "1"
    enabled: bool = True
    input: OrchidPipelineInputConfig | None = None
    output: OrchidPipelineOutputConfig | None = None
    interpretation: OrchidPipelineInterpretationConfig = Field(default_factory=OrchidPipelineInterpretationConfig)
    jobs: dict[str, OrchidPipelineJobConfig] = Field(default_factory=dict)
    expose: OrchidPipelineExposeConfig = Field(default_factory=OrchidPipelineExposeConfig)
    async_: OrchidPipelineAsyncConfig = Field(default_factory=OrchidPipelineAsyncConfig, alias="async")
    limits: OrchidPipelineDefinitionLimitsConfig = Field(default_factory=OrchidPipelineDefinitionLimitsConfig)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("pipeline version must be non-empty")
        return value


# ── Top-level block ──────────────────────────────────────────


# ORCHID-MEM: 01a1e991-b9d2-7a8f-a021-ad7cdaa6a3fe — Root pipelines configuration; parses and validates while keeping an absent block zero-impact.
class OrchidPipelinesConfig(BaseModel):
    """Root ``pipelines:`` configuration.

    ``enabled: false`` still parses and validates the block (early error
    detection) but the framework builds no registry, tools, or graph. An absent
    block leaves ``OrchidAgentsConfig.pipelines`` as ``None`` and changes
    nothing.
    """

    enabled: bool = True
    allow_untyped_relations: bool = False
    allow_parallel_writes: bool = False
    allow_shape_version_mismatch: bool = False
    defaults: OrchidPipelineDefaultsConfig = Field(default_factory=OrchidPipelineDefaultsConfig)
    limits: OrchidPipelineLimitsConfig = Field(default_factory=OrchidPipelineLimitsConfig)
    shapes: dict[str, OrchidPipelineShapeConfig] = Field(default_factory=dict)
    relations: dict[str, OrchidPipelineRelationConfig] = Field(default_factory=dict)
    definitions: dict[str, OrchidPipelineDefinitionConfig] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_block(self) -> OrchidPipelinesConfig:
        errors = _validate_shapes(self.shapes, self.relations)
        errors.extend(_validate_subpipeline_graph(self.definitions, self.limits.max_subpipeline_depth))
        for name, definition in self.definitions.items():
            if not _PIPELINE_NAME_RE.fullmatch(name):
                errors.append(f"pipeline name {name!r} must match ^[a-z][a-z0-9_-]{{1,62}}$")
                continue
            errors.extend(_validate_definition(name, definition, self))
        if errors:
            raise ValueError("invalid pipelines configuration: " + "; ".join(errors))
        return self


# ── Validation helpers ───────────────────────────────────────


def _validate_shapes(
    shapes: dict[str, OrchidPipelineShapeConfig],
    relations: dict[str, OrchidPipelineRelationConfig],
) -> list[str]:
    errors: list[str] = []
    for name, shape in shapes.items():
        if not _SHAPE_NAME_RE.fullmatch(name):
            errors.append(f"shape name {name!r} must be a valid identifier")

    # Every ref must resolve, and v1 forbids recursive shape references.
    ref_graph: dict[str, set[str]] = {}
    for name, shape in shapes.items():
        refs = set(shape.iter_refs())
        for ref in refs:
            if ref not in shapes:
                errors.append(f"shape {name!r} references unknown shape {ref!r}")
        ref_graph[name] = {ref for ref in refs if ref in shapes}

    cycle = _find_cycle(ref_graph)
    if cycle:
        errors.append("recursive shape references are not supported in v1: " + " -> ".join(cycle))

    for name, relation in relations.items():
        if not _RELATION_NAME_RE.fullmatch(name):
            errors.append(f"relation name {name!r} must match ^[a-z][a-z0-9_]{{0,63}}$")
        for endpoint in [*relation.from_, *relation.to]:
            if endpoint not in shapes:
                errors.append(f"relation {name!r} references unknown shape {endpoint!r}")
        for ref in relation.iter_refs():
            if ref not in shapes:
                errors.append(f"relation {name!r} references unknown shape {ref!r}")
    return errors


def _find_cycle(graph: Mapping[str, set[str]]) -> list[str] | None:
    """Return one cycle as a node path, or ``None`` when the graph is acyclic."""
    visited: set[str] = set()
    stack: list[str] = []
    on_stack: set[str] = set()

    def visit(node: str) -> list[str] | None:
        if node in on_stack:
            start = stack.index(node)
            return [*stack[start:], node]
        if node in visited:
            return None
        visited.add(node)
        on_stack.add(node)
        stack.append(node)
        for neighbour in sorted(graph.get(node, set())):
            cycle = visit(neighbour)
            if cycle:
                return cycle
        stack.pop()
        on_stack.remove(node)
        return None

    for node in sorted(graph):
        cycle = visit(node)
        if cycle:
            return cycle
    return None


def _validate_subpipeline_graph(
    definitions: dict[str, OrchidPipelineDefinitionConfig],
    max_depth: int,
) -> list[str]:
    errors: list[str] = []
    graph: dict[str, set[str]] = {name: set() for name in definitions}
    for name, definition in definitions.items():
        for job in definition.jobs.values():
            if isinstance(job, OrchidPipelineSubPipelineJobConfig):
                graph[name].add(job.pipeline)
                if job.pipeline not in definitions:
                    errors.append(f"pipeline {name!r} job references unknown sub-pipeline {job.pipeline!r}")

    cycle = _find_cycle(graph)
    if cycle:
        return [*errors, "sub-pipeline cycle detected: " + " -> ".join(cycle)]

    depths: dict[str, int] = {}

    def depth(name: str) -> int:
        if name in depths:
            return depths[name]
        depths[name] = 1
        children = [child for child in graph.get(name, set()) if child in definitions]
        if children:
            depths[name] = 1 + max(depth(child) for child in children)
        return depths[name]

    for name, definition in definitions.items():
        effective_max_depth = definition.limits.max_subpipeline_depth or max_depth
        if depth(name) > effective_max_depth:
            errors.append(
                f"pipeline {name!r} nests sub-pipelines {depth(name)} deep, "
                f"exceeding max_subpipeline_depth={effective_max_depth}"
            )
    return errors


def _validate_definition(
    name: str,
    definition: OrchidPipelineDefinitionConfig,
    config: OrchidPipelinesConfig,
) -> list[str]:
    errors: list[str] = []
    jobs = definition.jobs
    shapes = config.shapes
    relations = config.relations
    label = f"pipeline {name!r}"

    if not jobs:
        return [f"{label} must declare at least one job"]
    effective_max_jobs = definition.limits.max_jobs or config.limits.max_jobs
    if len(jobs) > effective_max_jobs:
        errors.append(f"{label} declares {len(jobs)} jobs, exceeding limits.max_jobs={effective_max_jobs}")

    for job_id in jobs:
        if not _JOB_ID_RE.fullmatch(job_id):
            errors.append(f"{label} job id {job_id!r} must match ^[a-z][a-z0-9_-]{{0,62}}$")

    errors.extend(_validate_definition_limits(label, definition, config))
    errors.extend(_validate_input_output(label, definition, shapes))
    errors.extend(_validate_dag(label, jobs))

    effective_max_parallel = definition.limits.max_parallel or config.limits.max_parallel
    effective_max_context_chars = definition.limits.max_context_chars or config.limits.max_context_chars
    effective_timeout_seconds = definition.limits.timeout_seconds
    if effective_timeout_seconds is None:
        effective_timeout_seconds = config.limits.timeout_seconds

    branch_membership: dict[str, list[tuple[str, str]]] = {}
    for job_id, job in jobs.items():
        job_label = f"{label} job {job_id!r}"
        errors.extend(
            _validate_job(
                job_label,
                job,
                jobs,
                shapes,
                relations,
                config,
                max_parallel=effective_max_parallel,
                max_context_chars=effective_max_context_chars,
                max_timeout_seconds=effective_timeout_seconds,
            )
        )
        if isinstance(job, OrchidPipelineDecisionJobConfig):
            for branch_label, target in job.branches.items():
                if target == job_id:
                    errors.append(f"{job_label} branch {branch_label!r} must not target the decision job itself")
                elif target not in jobs:
                    errors.append(f"{job_label} branch {branch_label!r} targets unknown job {target!r}")
                else:
                    branch_membership.setdefault(target, []).append((job_id, branch_label))
            if job.fallback is not None and job.fallback not in job.labels and job.fallback not in jobs:
                errors.append(f"{job_label} fallback {job.fallback!r} must be a declared label or a job id")

    # A fallback that names a job id is the target selected when no branch
    # matches, so it is mutually exclusive with the declared branch targets for
    # alias / static-write purposes. It is deliberately not added to
    # ``branch_membership``: fallback jobs need not obey branch-target topology.
    exclusive_membership = {target: list(owners) for target, owners in branch_membership.items()}
    for job_id, job in jobs.items():
        if (
            isinstance(job, OrchidPipelineDecisionJobConfig)
            and job.fallback is not None
            and job.fallback not in job.labels
            and job.fallback in jobs
        ):
            exclusive_membership.setdefault(job.fallback, []).append((job_id, "__fallback__"))

    errors.extend(_validate_branch_topology(label, jobs, branch_membership))
    errors.extend(_validate_aliases(label, jobs, exclusive_membership))
    errors.extend(_validate_parallel_writes(label, jobs, config.allow_parallel_writes, exclusive_membership))
    return errors


def _validate_definition_limits(
    label: str,
    definition: OrchidPipelineDefinitionConfig,
    config: OrchidPipelinesConfig,
) -> list[str]:
    errors: list[str] = []
    overrides = definition.limits
    for field_name in (
        "timeout_seconds",
        "max_jobs",
        "max_objects",
        "max_relations",
        "max_object_bytes",
        "max_graph_bytes",
        "max_llm_calls",
        "max_prompt_chars",
        "max_context_chars",
        "max_map_items",
        "max_parallel",
        "max_subpipeline_depth",
    ):
        override = getattr(overrides, field_name)
        ceiling = getattr(config.limits, field_name)
        if override is not None and ceiling is not None and override > ceiling:
            errors.append(f"{label} limits.{field_name}={override} exceeds the global ceiling {ceiling}")
    return errors


def _validate_input_output(
    label: str,
    definition: OrchidPipelineDefinitionConfig,
    shapes: dict[str, OrchidPipelineShapeConfig],
) -> list[str]:
    errors: list[str] = []
    input_cfg = definition.input
    if input_cfg is not None:
        if input_cfg.shape is not None and input_cfg.shape not in shapes:
            errors.append(f"{label} input references unknown shape {input_cfg.shape!r}")
        if input_cfg.fields is not None:
            errors.extend(_validate_field_refs(f"{label} input", input_cfg.fields, shapes))
        declared: set[str] | None = set(input_cfg.fields or {})
        if input_cfg.shape is not None and input_cfg.shape in shapes:
            declared_fields = shapes[input_cfg.shape].fields
            # A class-based shape's fields are resolved at build time, so its
            # default keys cannot be checked here without importing the class.
            declared = set(declared_fields) if declared_fields is not None else None
        if declared is not None:
            for key in input_cfg.defaults:
                if key not in declared:
                    errors.append(f"{label} input defaults reference unknown field {key!r}")

    output_cfg = definition.output
    if output_cfg is not None and output_cfg.shape is not None:
        if output_cfg.shape not in shapes:
            errors.append(f"{label} output references unknown shape {output_cfg.shape!r}")
        if output_cfg.mapping is not None:
            for key, expression in output_cfg.mapping.items():
                if "{{" in expression or "}}" in expression:
                    try:
                        _require_template(expression, label=f"{label} output mapping {key!r}")
                    except ValueError as exc:
                        errors.append(str(exc))
                else:
                    try:
                        _require_jmespath(expression, label=f"{label} output mapping {key!r}")
                    except ValueError as exc:
                        errors.append(str(exc))
    return errors


def _validate_field_refs(
    label: str,
    fields: dict[str, OrchidPipelineFieldConfig],
    shapes: dict[str, OrchidPipelineShapeConfig],
) -> list[str]:
    errors: list[str] = []
    for field_name, field in fields.items():
        for ref in field.iter_refs():
            if ref not in shapes:
                errors.append(f"{label} field {field_name!r} references unknown shape {ref!r}")
    return errors


def _validate_dag(label: str, jobs: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    indegree: dict[str, int] = {job_id: 0 for job_id in jobs}
    dependents: dict[str, list[str]] = {job_id: [] for job_id in jobs}

    for job_id, job in jobs.items():
        seen: set[str] = set()
        for dependency in job.depends_on:
            if dependency == job_id:
                errors.append(f"{label} job {job_id!r} must not depend on itself")
                continue
            if dependency not in jobs:
                errors.append(f"{label} job {job_id!r} depends on unknown job {dependency!r}")
                continue
            if dependency in seen:
                errors.append(f"{label} job {job_id!r} declares duplicate dependency {dependency!r}")
                continue
            seen.add(dependency)
            indegree[job_id] += 1
            dependents[dependency].append(job_id)
        if job.join == "any" and len(job.depends_on) < 2:
            errors.append(f"{label} job {job_id!r} uses join: any but declares fewer than two dependencies")

    queue = sorted(job_id for job_id, count in indegree.items() if count == 0)
    processed: list[str] = []
    while queue:
        current = queue.pop(0)
        processed.append(current)
        for dependent in dependents[current]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                queue.append(dependent)

    if len(processed) != len(jobs):
        cycle = _find_cycle(
            {
                job_id: {dependency for dependency in job.depends_on if dependency in jobs}
                for job_id, job in jobs.items()
            }
        )
        if cycle:
            errors.append(f"{label} dependency graph contains a cycle: " + " -> ".join(cycle))
        else:  # pragma: no cover — Kahn and DFS should agree
            remaining = sorted(job_id for job_id, count in indegree.items() if count > 0)
            errors.append(f"{label} dependency graph contains a cycle involving: {', '.join(remaining)}")
    return errors


def _reachable(start: str, target: str, jobs: dict[str, Any]) -> bool:
    stack = [start]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current == target:
            return True
        if current in seen:
            continue
        seen.add(current)
        job = jobs.get(current)
        if job is not None:
            stack.extend(job.depends_on)
    return False


def _validate_job(
    label: str,
    job: Any,
    jobs: dict[str, Any],
    shapes: dict[str, OrchidPipelineShapeConfig],
    relations: dict[str, OrchidPipelineRelationConfig],
    config: OrchidPipelinesConfig,
    *,
    max_parallel: int,
    max_context_chars: int,
    max_timeout_seconds: float | None,
) -> list[str]:
    errors: list[str] = []

    if (
        job.timeout_seconds is not None
        and max_timeout_seconds is not None
        and job.timeout_seconds > max_timeout_seconds
    ):
        errors.append(
            f"{label} timeout_seconds={job.timeout_seconds} exceeds the effective timeout_seconds={max_timeout_seconds}"
        )

    for write in job.writes:
        if write.shape not in shapes:
            errors.append(f"{label} writes to unknown shape {write.shape!r}")
    for link in job.links:
        if link.type not in relations and not config.allow_untyped_relations:
            errors.append(
                f"{label} links unknown relation {link.type!r} "
                "(set pipelines.allow_untyped_relations=true to allow untyped relations)"
            )

    if isinstance(job, OrchidPipelineLLMJobConfig):
        if job.model is None and config.defaults.model is None:
            errors.append(f"{label} is an llm job with no 'model' and no defaults.model")
        if job.output is not None and job.output.shape is not None and job.output.shape not in shapes:
            errors.append(f"{label} output references unknown shape {job.output.shape!r}")
        if job.rag is not None and job.rag.max_context_chars > max_context_chars:
            errors.append(
                f"{label} rag.max_context_chars={job.rag.max_context_chars} "
                f"exceeds the effective max_context_chars={max_context_chars}"
            )
        for tool_name in job.tools:
            if not _TOOL_NAME_RE.fullmatch(tool_name):
                errors.append(f"{label} declares invalid tool name {tool_name!r}")
    elif isinstance(job, OrchidPipelineAgentJobConfig):
        if job.output is not None and job.output.shape is not None and job.output.shape not in shapes:
            errors.append(f"{label} output references unknown shape {job.output.shape!r}")
    elif isinstance(job, OrchidPipelineMapJobConfig):
        if job.concurrency > max_parallel:
            errors.append(
                f"{label} map.concurrency={job.concurrency} exceeds the effective max_parallel={max_parallel}"
            )
        template = jobs.get(job.job)
        if template is None:
            errors.append(f"{label} maps over unknown template job {job.job!r}")
        else:
            if isinstance(template, OrchidPipelineMapJobConfig):
                errors.append(f"{label} template job {job.job!r} is itself a map job; nested maps are not supported")
            if isinstance(template, OrchidPipelineDecisionJobConfig) and template.branches:
                errors.append(f"{label} template job {job.job!r} is a branching decision job")
    elif isinstance(job, OrchidPipelineSubPipelineJobConfig) and job.pipeline not in config.definitions:
        errors.append(f"{label} references unknown sub-pipeline {job.pipeline!r}")
    return errors


def _validate_branch_topology(
    label: str,
    jobs: dict[str, Any],
    branch_membership: dict[str, list[tuple[str, str]]],
) -> list[str]:
    errors: list[str] = []

    for target, owners in branch_membership.items():
        target_job = jobs.get(target)
        if target_job is None:
            continue
        for decision_id, _branch_label in owners:
            if decision_id in target_job.depends_on:
                errors.append(
                    f"{label} branch target {target!r} must not list decision job {decision_id!r} in depends_on"
                )

    for job_id, job in jobs.items():
        if job.join == "all":
            owners_by_decision: dict[str, set[str]] = {}
            for dependency in job.depends_on:
                for decision_id, branch_label in branch_membership.get(dependency, []):
                    owners_by_decision.setdefault(decision_id, set()).add(branch_label)
            for decision_id, branch_labels in owners_by_decision.items():
                if len(branch_labels) > 1:
                    errors.append(
                        f"{label} job {job_id!r} uses join: all over mutually exclusive branches "
                        f"of decision {decision_id!r}; use join: any"
                    )

    for target in branch_membership:
        if target not in jobs:
            continue
        for dependent_id, dependent in jobs.items():
            if target in dependent.depends_on and dependent.join != "any":
                errors.append(f"{label} branch target {target!r} feeds job {dependent_id!r} which must use join: any")
    return errors


def _mutually_exclusive(left: str, right: str, branch_membership: dict[str, list[tuple[str, str]]]) -> bool:
    for decision_id, label in branch_membership.get(left, []):
        for other_decision, other_label in branch_membership.get(right, []):
            if decision_id == other_decision and label != other_label:
                return True
    return False


def _validate_aliases(
    label: str,
    jobs: dict[str, Any],
    branch_membership: dict[str, list[tuple[str, str]]],
) -> list[str]:
    errors: list[str] = []
    owners: dict[str, list[str]] = {}
    for job_id, job in jobs.items():
        for write in job.writes:
            if write.as_ is not None:
                owners.setdefault(write.as_, []).append(job_id)

    for alias, job_ids in owners.items():
        for index, left in enumerate(job_ids):
            for right in job_ids[index + 1 :]:
                if not _mutually_exclusive(left, right, branch_membership):
                    errors.append(
                        f"{label} alias {alias!r} is declared by jobs {left!r} and {right!r} "
                        "that are not mutually exclusive"
                    )
    return errors


def _validate_parallel_writes(
    label: str,
    jobs: dict[str, Any],
    allow_parallel: bool,
    branch_membership: dict[str, list[tuple[str, str]]],
) -> list[str]:
    if allow_parallel:
        return []
    errors: list[str] = []
    static_writes: dict[str, list[str]] = {}
    for job_id, job in jobs.items():
        for write in job.writes:
            if "{{" not in write.id and "}}" not in write.id:
                static_writes.setdefault(write.id, []).append(job_id)

    for object_id, job_ids in static_writes.items():
        for index, left in enumerate(job_ids):
            for right in job_ids[index + 1 :]:
                if _reachable(left, right, jobs) or _reachable(right, left, jobs):
                    continue
                if _mutually_exclusive(left, right, branch_membership):
                    continue
                errors.append(
                    f"{label} parallel jobs {left!r} and {right!r} both write static object "
                    f"{object_id!r}; set pipelines.allow_parallel_writes=true to allow it"
                )
    return errors


__all__ = [
    "OrchidPipelineAgentJobConfig",
    "OrchidPipelineAsyncConfig",
    "OrchidPipelineCustomJobConfig",
    "OrchidPipelineDecisionJobConfig",
    "OrchidPipelineDefaultsConfig",
    "OrchidPipelineDefinitionConfig",
    "OrchidPipelineDefinitionLimitsConfig",
    "OrchidPipelineDeterministicJobConfig",
    "OrchidPipelineEmitJobConfig",
    "OrchidPipelineExposeConfig",
    "OrchidPipelineFieldConfig",
    "OrchidPipelineInputConfig",
    "OrchidPipelineInterpretationConfig",
    "OrchidPipelineJobConfig",
    "OrchidPipelineJobOutputConfig",
    "OrchidPipelineLLMJobConfig",
    "OrchidPipelineLLMRAGConfig",
    "OrchidPipelineLimitsConfig",
    "OrchidPipelineMapJobConfig",
    "OrchidPipelineObjectWriteConfig",
    "OrchidPipelineOutputConfig",
    "OrchidPipelineRelationConfig",
    "OrchidPipelineRelationWriteConfig",
    "OrchidPipelineRetryConfig",
    "OrchidPipelineShapeConfig",
    "OrchidPipelineSubPipelineJobConfig",
    "OrchidPipelinesConfig",
]
