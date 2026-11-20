"""Shape compiler: YAML shape declarations to typed Pydantic validation models.

A *shape* is a named record definition from the ``pipelines.shapes`` block. This
module is the single place that turns a shape declaration into something the
runtime can validate with:

* field-based shapes compile through :func:`pydantic.create_model`;
* class-based shapes resolve a dotted path at build time (a Pydantic
  ``BaseModel`` subclass, or a dataclass through a ``TypeAdapter``);
* nested ``ref`` shapes compile recursively, and recursive references are
  rejected in v1 (ADR-033) with a message naming the cycle;
* declared-shape reference chains are bounded (``_MAX_SHAPE_REF_DEPTH``) so a
  pathological graph fails at build time instead of in the interpreter;
* every compiled shape exposes a JSON Schema for the API, tool parameter
  schemas and LLM structured output, with every ``$ref`` resolvable inside the
  emitted document;
* validation failures are wrapped in
  :class:`orchid_ai.core.pipeline.OrchidPipelineShapeError` carrying the shape
  name, the object id (when the caller knows it) and the field path.

Validation is purely structural. Content, safety and business rules belong to
the steps that consume the value.

Optionality is uniform:

* a ``required: true`` field must be supplied; a declared default is validated
  at compile time but never applied;
* a non-required field **without** a declared default is nullable (``T | None``)
  and defaults to ``None``;
* a non-required field **with** a declared default takes that value and is not
  nullable (passing ``null`` is rejected unless the field type is ``null``).

Declared defaults are validated against the compiled annotation and its
constraints when the shape compiles, so a ``default`` below ``min`` or outside a
``pattern`` fails at build time rather than on first use. ``pattern`` values are
also checked against the regex engine Pydantic actually uses, so a Python
``re``-only construct (look-around, backreference) fails with a readable shape
error instead of an opaque schema build failure.

``additional_properties`` defaults to ``False``, so a field-based shape compiles
with ``extra="forbid"`` unless the shape opts in. A shape's ``version`` is
carried so a write can be checked against the object's ``shape_version``; the
shape (or the pipeline block) may set ``allow_shape_version_mismatch: true`` to
tolerate a mismatch. v1 ships no migration (ADR-033).

Nothing here imports a consumer class at module import time: a dotted ``class``
path is resolved only when :meth:`ShapeCompiler.compile` runs.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import is_dataclass
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    create_model,
)
from pydantic.fields import FieldInfo

from ..config.schema_pipeline import OrchidPipelineFieldConfig, OrchidPipelineShapeConfig
from ..core.pipeline import OrchidPipelineShapeError
from ..utils import import_class

__all__ = [
    "OrchidCompiledShape",
    "OrchidShapeRegistry",
    "ShapeCompiler",
    "compile_shapes",
]

_MAX_REPORTED_ERRORS = 10
_MAX_SHAPE_REF_DEPTH = 32
_PATH_ROOT = "$"

#: Names Pydantic owns on every model. ``create_model`` silently absorbs some of
#: them (``model_config`` becomes configuration), so a shape field using one can
#: never round-trip; reject the whole set at compile time.
_RESERVED_FIELD_NAMES = frozenset(
    {
        "model_config",
        "model_fields",
        "model_json_schema",
        "model_dump",
        "model_validate",
        "model_copy",
        "dict",
        "json",
        "copy",
        "schema",
        "validate",
        "construct",
    }
)

_OBJECT_FORBID = ConfigDict(extra="forbid", validate_default=True)
_OBJECT_ALLOW = ConfigDict(extra="allow", validate_default=True)


# ── Error formatting ─────────────────────────────────────────


def _format_location(location: Sequence[Any]) -> str:
    """Render a Pydantic ``loc`` tuple as a ``$.field[0].nested`` path."""
    path = _PATH_ROOT
    for part in location:
        if isinstance(part, int) and not isinstance(part, bool):
            path += f"[{part}]"
        elif isinstance(part, str) and part.isidentifier():
            path += f".{part}"
        else:
            path += f"[{part!r}]"
    return path


def _shape_error(message: str, *, shape: str, object_id: str | None = None) -> OrchidPipelineShapeError:
    """Build a shape error with the context the API and CLI map to a response."""
    error = OrchidPipelineShapeError(message)
    error.shape = shape
    error.object_id = object_id
    error.field_path = ""
    error.errors = ()
    return error


def _validation_error(
    shape: OrchidCompiledShape,
    object_id: str | None,
    exc: ValidationError,
) -> OrchidPipelineShapeError:
    """Wrap a Pydantic ``ValidationError`` with shape, object and path context."""
    raw_errors = exc.errors()
    rendered: list[str] = []
    entries: list[dict[str, str]] = []
    for raw in raw_errors[:_MAX_REPORTED_ERRORS]:
        path = _format_location(tuple(raw.get("loc") or ()))
        kind = str(raw.get("type") or "value_error")
        message = str(raw.get("msg") or "invalid value")
        rendered.append(f"{path}: {kind}: {message}")
        entries.append({"path": path, "type": kind, "message": message})

    remaining = len(raw_errors) - len(entries)
    if remaining > 0:
        rendered.append(f"... and {remaining} more error(s)")

    prefix = f"shape {shape.name!r}"
    if object_id is not None:
        prefix += f" object {object_id!r}"
    error = OrchidPipelineShapeError(f"{prefix} validation failed at " + "; ".join(rendered))
    error.shape = shape.name
    error.object_id = object_id
    error.field_path = entries[0]["path"] if entries else _PATH_ROOT
    error.errors = tuple(entries)
    return error


def _inline_root_ref(schema: dict[str, Any], title: str) -> dict[str, Any]:
    """Flatten a ``$ref``-rooted schema (dataclass adapters) and pin the title."""
    reference = schema.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/$defs/"):
        definition_key = reference[len("#/$defs/") :]
        definitions = schema.get("$defs") or {}
        target = definitions.get(definition_key)
        if isinstance(target, dict):
            flattened = dict(target)
            if "$defs" not in flattened:
                others = {key: value for key, value in definitions.items() if key != definition_key}
                if others:
                    flattened["$defs"] = others
            schema = flattened
    schema["title"] = title
    return schema


def _resolve_json_pointer(root: Any, reference: str) -> Any | None:
    """Resolve a local ``#/...`` JSON pointer, or return ``None``."""
    if not reference.startswith("#/"):
        return None
    current = root
    for raw_token in reference[2:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
            current = current[int(token)]
        else:
            return None
    return current


def _unresolved_schema_refs(schema: Mapping[str, Any]) -> list[str]:
    """Return every ``$ref`` in ``schema`` that does not resolve locally.

    Pydantic emits ``#/$defs/...`` references; a recursive model flattened at
    the root leaves a reference to the definition that was removed. ADR-033
    rejects those shapes, so the compiler treats any unresolved reference as a
    build-time error rather than emitting a schema a consumer cannot follow.
    """
    unresolved: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            reference = node.get("$ref")
            if isinstance(reference, str) and _resolve_json_pointer(schema, reference) is None:
                unresolved.append(reference)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(schema)
    return unresolved


def _schema_ref_targets(node: Any) -> set[str]:
    """Collect the local ``$defs`` names referenced anywhere under ``node``."""
    targets: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            reference = value.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                targets.add(reference[len("#/$defs/") :])
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(node)
    return targets


def _schema_ref_cycle(schema: Mapping[str, Any]) -> list[str] | None:
    """Return one cycle among ``$defs`` references, or ``None``.

    A recursive class emits a schema that references itself; ADR-033 rejects
    recursive shapes in v1 even when the document happens to be self-contained.
    """
    definitions = schema.get("$defs")
    if not isinstance(definitions, Mapping):
        return None
    graph: dict[str, list[str]] = {
        name: sorted(_schema_ref_targets(definition))
        for name, definition in definitions.items()
        if isinstance(name, str)
    }
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        status = state.get(node, 0)
        if status == 1:
            start = stack.index(node)
            return [*stack[start:], node]
        if status == 2:
            return None
        state[node] = 1
        stack.append(node)
        for target in graph.get(node, ()):
            if target in graph:
                cycle = visit(target)
                if cycle is not None:
                    return cycle
        stack.pop()
        state[node] = 2
        return None

    for node in graph:
        cycle = visit(node)
        if cycle is not None:
            return cycle
    return None


# ── Compiled shape ───────────────────────────────────────────


# ORCHID-MEM: 01a1ea22-813e-76d3-a395-ae8659f15beb — Compiled shape; validates, dumps JSON-safe data, emits JSON Schema and enforces shape versions.
class OrchidCompiledShape:
    """A shape compiled to a validation model plus its JSON Schema.

    ``model`` is the generated Pydantic model for a field-based shape, or the
    imported class for a class-based shape. Validation returns the model (or
    dataclass) instance; :meth:`dump` serializes it with ``model_dump(mode="json")``
    so dates, enums and nested models become JSON-safe.

    For a class-based shape ``additional_properties`` mirrors the declaration
    only: the imported class's own configuration decides whether extras are
    accepted.

    ``allow_shape_version_mismatch`` comes from the shape declaration (or the
    pipeline block) and makes :meth:`check_version` tolerate a different
    ``shape_version``. A caller may also allow a single mismatch explicitly.
    """

    def __init__(
        self,
        *,
        name: str,
        version: str,
        description: str,
        model: type[Any],
        adapter: TypeAdapter[Any],
        additional_properties: bool,
        allow_shape_version_mismatch: bool = False,
    ) -> None:
        self.name = name
        self.version = version
        self.description = description
        self.model = model
        self.additional_properties = additional_properties
        self.allow_shape_version_mismatch = allow_shape_version_mismatch
        self._adapter = adapter
        self._schema_cache: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return f"OrchidCompiledShape(name={self.name!r}, version={self.version!r})"

    def check_version(
        self,
        shape_version: str,
        *,
        object_id: str | None = None,
        allow_mismatch: bool = False,
    ) -> None:
        """Raise when an object was written with a different shape version.

        ADR-033 keeps versioning strict: a mismatch fails unless the shape (or
        pipeline block) opted in with ``allow_shape_version_mismatch: true`` or
        the caller tolerates it for one operation.
        """
        if not isinstance(shape_version, str) or not shape_version.strip():
            raise _shape_error(
                f"shape {self.name!r}: shape_version must be a non-empty string",
                shape=self.name,
                object_id=object_id,
            )
        if shape_version == self.version or allow_mismatch or self.allow_shape_version_mismatch:
            return
        prefix = f"shape {self.name!r}"
        if object_id is not None:
            prefix += f" object {object_id!r}"
        raise _shape_error(
            f"{prefix} was written with shape version {shape_version!r} "
            f"but the configured shape version is {self.version!r}; re-seed or migrate the object, "
            "or set allow_shape_version_mismatch: true",
            shape=self.name,
            object_id=object_id,
        )

    def validate(
        self,
        data: Any,
        *,
        object_id: str | None = None,
        shape_version: str | None = None,
        allow_version_mismatch: bool = False,
    ) -> Any:
        """Validate ``data`` and return the typed model (or dataclass) instance.

        ``shape_version`` is the version the incoming object claims, when the
        caller knows it. A mismatch fails before any field is validated.
        """
        if shape_version is not None:
            self.check_version(
                shape_version,
                object_id=object_id,
                allow_mismatch=allow_version_mismatch,
            )
        try:
            return self._adapter.validate_python(data)
        except ValidationError as exc:
            raise _validation_error(self, object_id, exc) from exc

    def dump(
        self,
        data: Any,
        *,
        object_id: str | None = None,
        shape_version: str | None = None,
        allow_version_mismatch: bool = False,
    ) -> dict[str, Any]:
        """Validate ``data`` and serialize it with ``model_dump(mode="json")``."""
        value = self.validate(
            data,
            object_id=object_id,
            shape_version=shape_version,
            allow_version_mismatch=allow_version_mismatch,
        )
        try:
            if isinstance(value, BaseModel):
                dumped = value.model_dump(mode="json")
            else:
                dumped = self._adapter.dump_python(value, mode="json")
        except Exception as exc:  # serialization failures are shape errors too
            raise _shape_error(
                f"shape {self.name!r}: validated value is not JSON-serializable: {exc}",
                shape=self.name,
                object_id=object_id,
            ) from exc
        if not isinstance(dumped, dict):
            raise _shape_error(
                f"shape {self.name!r} must serialize to a JSON object, got {type(dumped).__name__}",
                shape=self.name,
                object_id=object_id,
            )
        return dumped

    def model_json_schema(self) -> dict[str, Any]:
        """Return the JSON Schema for this shape (a defensive copy)."""
        if self._schema_cache is None:
            try:
                raw = self._adapter.json_schema()
            except Exception as exc:
                raise _shape_error(
                    f"shape {self.name!r}: cannot generate a JSON Schema: {exc}",
                    shape=self.name,
                ) from exc
            schema = _inline_root_ref(dict(raw), self.name)
            if self.description:
                schema["description"] = self.description
            unresolved = _unresolved_schema_refs(schema)
            if unresolved:
                raise _shape_error(
                    f"shape {self.name!r}: JSON Schema has unresolved reference(s) "
                    f"{', '.join(unresolved[:3])}; recursive shape references are not supported in v1",
                    shape=self.name,
                )
            self._schema_cache = schema
        return copy.deepcopy(self._schema_cache)

    def json_schema(self) -> dict[str, Any]:
        """Alias for :meth:`model_json_schema`."""
        return self.model_json_schema()


# ── Registry ─────────────────────────────────────────────────


class OrchidShapeRegistry:
    """Immutable name-to-shape lookup produced by :class:`ShapeCompiler`."""

    def __init__(self, shapes: Mapping[str, OrchidCompiledShape]) -> None:
        self._shapes: dict[str, OrchidCompiledShape] = dict(shapes)

    def __contains__(self, name: object) -> bool:
        return name in self._shapes

    def __len__(self) -> int:
        return len(self._shapes)

    def __iter__(self) -> Iterator[str]:
        return iter(self._shapes)

    @property
    def names(self) -> tuple[str, ...]:
        """Declared shape names in configuration order."""
        return tuple(self._shapes)

    def get(self, name: str) -> OrchidCompiledShape:
        """Return a compiled shape or raise a clear unknown-shape error."""
        try:
            return self._shapes[name]
        except KeyError:
            declared = ", ".join(self._shapes) or "(none)"
            raise _shape_error(
                f"unknown shape {name!r}; declared shapes: {declared}",
                shape=name,
            ) from None

    def validate(
        self,
        name: str,
        data: Any,
        *,
        object_id: str | None = None,
        shape_version: str | None = None,
        allow_version_mismatch: bool = False,
    ) -> Any:
        """Validate ``data`` against the named shape."""
        return self.get(name).validate(
            data,
            object_id=object_id,
            shape_version=shape_version,
            allow_version_mismatch=allow_version_mismatch,
        )

    def dump(
        self,
        name: str,
        data: Any,
        *,
        object_id: str | None = None,
        shape_version: str | None = None,
        allow_version_mismatch: bool = False,
    ) -> dict[str, Any]:
        """Validate and JSON-serialize ``data`` against the named shape."""
        return self.get(name).dump(
            data,
            object_id=object_id,
            shape_version=shape_version,
            allow_version_mismatch=allow_version_mismatch,
        )

    def model_json_schema(self, name: str) -> dict[str, Any]:
        """Return the JSON Schema of the named shape."""
        return self.get(name).model_json_schema()

    def json_schema(self, name: str) -> dict[str, Any]:
        """Alias for :meth:`model_json_schema`."""
        return self.model_json_schema(name)


# ── Compiler ─────────────────────────────────────────────────


# ORCHID-MEM: 01a1ea22-816d-7b24-ad9e-e839813722ef — Shape compiler; turns YAML fields or class paths into validated Pydantic models with JSON Schema.
class ShapeCompiler:
    """Compile declared shapes (and inline field maps) into validation models.

    ``resolve_class`` is injectable so a test can resolve a fake module without
    touching the import system; production code uses
    :func:`orchid_ai.utils.import_class`.

    Optionality: a non-required field without a default is nullable and
    defaults to ``None``; a non-required field with a default takes that value
    and rejects ``None``. ``required`` fields must always be supplied.
    """

    def __init__(
        self,
        shapes: Mapping[str, OrchidPipelineShapeConfig] | None = None,
        *,
        resolve_class: Callable[[str], type[Any]] | None = None,
        allow_shape_version_mismatch: bool = False,
    ) -> None:
        if shapes is not None and not isinstance(shapes, Mapping):
            raise TypeError("shapes must be a mapping of shape name to declaration")
        self._shapes: dict[str, OrchidPipelineShapeConfig] = dict(shapes or {})
        self._resolve_class = resolve_class or import_class
        self._allow_shape_version_mismatch = allow_shape_version_mismatch
        self._compiled: dict[str, OrchidCompiledShape] = {}
        self._depths: dict[str, int] = {}
        self._in_progress: list[str] = []
        self._registry: OrchidShapeRegistry | None = None

    def compile(self) -> OrchidShapeRegistry:
        """Compile every declared shape; recursion and imports fail fast."""
        if self._registry is None:
            for name in self._shapes:
                self._compile_named(name)
            self._registry = OrchidShapeRegistry(self._compiled)
        return self._registry

    def compile_inline(
        self,
        name: str,
        fields: Mapping[str, OrchidPipelineFieldConfig],
        *,
        description: str = "",
        version: str = "1",
        additional_properties: bool = False,
    ) -> OrchidCompiledShape:
        """Compile an inline field map (pipeline input) against declared refs.

        Inline shapes are not added to the registry: only declared shapes are
        addressable by name. Declared shapes are compiled first so a ``ref``
        inside the inline map resolves.
        """
        self.compile()
        if not fields:
            raise _shape_error(f"inline shape {name!r} must declare at least one field", shape=name)
        compiled = self._compile_fields_shape(
            name=name,
            fields=fields,
            description=description,
            version=version,
            additional_properties=additional_properties,
            allow_shape_version_mismatch=False,
        )
        self._check_ref_depth(
            name,
            [ref for field in fields.values() for ref in field.iter_refs()],
            root=name,
        )
        return compiled

    # ── Named shapes ─────────────────────────────────────────

    def _compile_named(self, name: str, *, owner: str | None = None) -> OrchidCompiledShape:
        compiled = self._compiled.get(name)
        if compiled is not None:
            return compiled
        if name in self._in_progress:
            start = self._in_progress.index(name)
            cycle = [*self._in_progress[start:], name]
            raise _shape_error(
                "recursive shape references are not supported in v1: " + " -> ".join(cycle),
                shape=owner or name,
            )
        shape = self._shapes.get(name)
        if shape is None:
            declared = ", ".join(self._shapes) or "(none)"
            raise _shape_error(
                f"unknown shape reference {name!r}; declared shapes: {declared}",
                shape=owner or name,
            )

        self._in_progress.append(name)
        try:
            compiled = self._compile_shape(name, shape)
            depth = self._check_ref_depth(name, shape.iter_refs(), root=self._in_progress[0])
        finally:
            self._in_progress.pop()
        self._depths[name] = depth
        self._compiled[name] = compiled
        return compiled

    def _check_ref_depth(self, name: str, refs: Iterable[str], *, root: str) -> int:
        """Return this shape's declared-ref depth, rejecting an over-long chain.

        Depth is computed from the children's already-computed depth, not from
        the current traversal, so declaration order cannot hide a long chain.
        """
        depth = 1 + max((self._depths.get(ref, 0) for ref in refs), default=0)
        if depth > _MAX_SHAPE_REF_DEPTH:
            raise _shape_error(
                f"shape {root!r}: declared-shape reference depth {depth} exceeds the maximum of "
                f"{_MAX_SHAPE_REF_DEPTH} (chain through {name!r})",
                shape=root,
            )
        return depth

    def _compile_shape(self, name: str, shape: OrchidPipelineShapeConfig) -> OrchidCompiledShape:
        allow_mismatch = shape.allow_shape_version_mismatch or self._allow_shape_version_mismatch
        if shape.class_path is not None:
            return self._compile_class_shape(name, shape, allow_shape_version_mismatch=allow_mismatch)
        if shape.fields is None:  # pragma: no cover - guarded by the config schema
            raise _shape_error(f"shape {name!r} must declare 'fields' or 'class'", shape=name)
        return self._compile_fields_shape(
            name=name,
            fields=shape.fields,
            description=shape.description,
            version=shape.version,
            additional_properties=shape.additional_properties,
            allow_shape_version_mismatch=allow_mismatch,
        )

    def _compile_class_shape(
        self,
        name: str,
        shape: OrchidPipelineShapeConfig,
        *,
        allow_shape_version_mismatch: bool,
    ) -> OrchidCompiledShape:
        class_path = shape.class_path or ""
        try:
            imported = self._resolve_class(class_path)
        except Exception as exc:
            raise _shape_error(
                f"shape {name!r}: cannot import shape class {class_path!r}: {exc}",
                shape=name,
            ) from exc
        if not isinstance(imported, type):
            raise _shape_error(
                f"shape {name!r}: {class_path!r} resolved to {type(imported).__name__}, not a class",
                shape=name,
            )

        is_model = issubclass(imported, BaseModel)
        is_dataclass_shape = is_dataclass(imported)
        if not is_model and not is_dataclass_shape:
            raise _shape_error(
                f"shape {name!r}: class {class_path!r} must be a Pydantic BaseModel subclass or a dataclass",
                shape=name,
            )

        try:
            adapter = TypeAdapter(imported)
            # A class-based shape must be usable wherever a JSON Schema is
            # needed (API response, tool parameters, LLM output), so generate
            # it now rather than failing on first use.
            raw_schema = adapter.json_schema()
        except Exception as exc:
            raise _shape_error(
                f"shape {name!r}: class {class_path!r} is not JSON-Schema-compatible: {exc}",
                shape=name,
            ) from exc

        flattened = _inline_root_ref(dict(raw_schema), name)
        cycle = _schema_ref_cycle(raw_schema)
        if cycle is not None:
            raise _shape_error(
                f"shape {name!r}: class {class_path!r} contains a recursive schema reference "
                f"({' -> '.join(cycle)}); recursive shape references are not supported in v1",
                shape=name,
            )
        unresolved = _unresolved_schema_refs(flattened)
        if unresolved:
            raise _shape_error(
                f"shape {name!r}: class {class_path!r} is recursive (unresolved JSON Schema reference "
                f"{unresolved[0]!r}); recursive shape references are not supported in v1",
                shape=name,
            )

        compiled = OrchidCompiledShape(
            name=name,
            version=shape.version,
            description=shape.description,
            model=imported,
            adapter=adapter,
            additional_properties=shape.additional_properties,
            allow_shape_version_mismatch=allow_shape_version_mismatch,
        )
        # Cache the schema (with the declaration's description) at build time.
        compiled.model_json_schema()
        return compiled

    # ── Field-based shapes ───────────────────────────────────

    def _compile_fields_shape(
        self,
        *,
        name: str,
        fields: Mapping[str, OrchidPipelineFieldConfig],
        description: str,
        version: str,
        additional_properties: bool,
        allow_shape_version_mismatch: bool,
    ) -> OrchidCompiledShape:
        definitions: dict[str, tuple[Any, FieldInfo]] = {}
        for field_name, field in fields.items():
            try:
                definitions[field_name] = self._field_definition(name, field_name, field)
            except OrchidPipelineShapeError:
                raise
            except Exception as exc:
                raise _shape_error(
                    f"shape {name!r}: cannot compile field {field_name!r}: {exc}",
                    shape=name,
                ) from exc
        config = _OBJECT_ALLOW if additional_properties else _OBJECT_FORBID
        try:
            model = create_model(name, __config__=config, **definitions)
        except OrchidPipelineShapeError:
            raise
        except Exception as exc:
            raise _shape_error(
                f"shape {name!r}: cannot compile its fields: {exc}",
                shape=name,
            ) from exc

        compiled = OrchidCompiledShape(
            name=name,
            version=version,
            description=description,
            model=model,
            adapter=TypeAdapter(model),
            additional_properties=additional_properties,
            allow_shape_version_mismatch=allow_shape_version_mismatch,
        )
        # Every configured shape compiles with a JSON Schema; generating it now
        # surfaces an unusable schema at build time, not on the first API call.
        compiled.model_json_schema()
        return compiled

    def _field_definition(
        self,
        model_name: str,
        field_name: str,
        field: OrchidPipelineFieldConfig,
    ) -> tuple[Any, FieldInfo]:
        if field_name in _RESERVED_FIELD_NAMES:
            raise _shape_error(
                f"shape {model_name!r}: field {field_name!r} is a reserved Pydantic name "
                "and cannot be used as a shape field",
                shape=model_name,
            )
        annotation = self._base_annotation(model_name, field_name, field)
        constraints = self._field_constraints(field)

        if field.default is not None:
            self._check_declared_default(model_name, field_name, field.default, annotation, constraints)

        if field.required:
            # ``required`` wins over a declared default: a field the shape calls
            # required must be supplied, and silently filling it would hide a
            # caller mistake.
            return annotation, Field(**constraints)
        if field.default is None:
            annotation = _optional(annotation)
        return annotation, Field(**constraints, default=field.default)

    def _check_declared_default(
        self,
        model_name: str,
        field_name: str,
        default: Any,
        annotation: Any,
        constraints: Mapping[str, Any],
    ) -> None:
        """Validate a declared default against the compiled field constraints.

        ``create_model`` does not validate defaults; without this check a typo
        such as ``min: 5`` with ``default: 0`` compiles and only fails when the
        field is omitted at runtime.
        """
        adapter = TypeAdapter(Annotated[annotation, Field(**dict(constraints))])
        try:
            adapter.validate_python(default)
        except ValidationError as exc:
            rendered = []
            for raw in exc.errors()[:3]:
                location = (field_name, *(raw.get("loc") or ()))
                kind = str(raw.get("type") or "value_error")
                message = str(raw.get("msg") or "invalid value")
                rendered.append(f"{_format_location(location)}: {kind}: {message}")
            raise _shape_error(
                f"shape {model_name!r}: default for field {field_name!r} is invalid at " + "; ".join(rendered),
                shape=model_name,
            ) from exc

    def _field_constraints(self, field: OrchidPipelineFieldConfig) -> dict[str, Any]:
        constraints: dict[str, Any] = {}
        if field.description:
            constraints["description"] = field.description
        if field.min is not None:
            constraints["ge"] = field.min
        if field.max is not None:
            constraints["le"] = field.max
        if field.min_length is not None:
            constraints["min_length"] = field.min_length
        if field.max_length is not None:
            constraints["max_length"] = field.max_length
        if field.pattern is not None:
            constraints["pattern"] = field.pattern
        return constraints

    def _base_annotation(
        self,
        model_name: str,
        field_name: str,
        field: OrchidPipelineFieldConfig,
    ) -> Any:
        if field.ref is not None:
            if field.enum is not None:
                raise _shape_error(
                    f"shape {model_name!r}: field {field_name!r} must not combine 'ref' with 'enum'",
                    shape=model_name,
                )
            if field.pattern is not None:
                raise _shape_error(
                    f"shape {model_name!r}: field {field_name!r} must not combine 'ref' with 'pattern'",
                    shape=model_name,
                )
            return self._compile_named(field.ref, owner=model_name).model

        if field.type is None:  # pragma: no cover - guarded by the config schema
            raise _shape_error(
                f"shape {model_name!r}: field {field_name!r} must set 'type' or 'ref'",
                shape=model_name,
            )
        if field.pattern is not None and field.type != "string":
            raise _shape_error(
                f"shape {model_name!r}: field {field_name!r} sets 'pattern' but its type is "
                f"{field.type!r}; 'pattern' is only valid for string fields",
                shape=model_name,
            )
        if field.pattern is not None:
            self._check_pattern_engine(model_name, field_name, field.pattern)

        if field.enum is not None:
            return Literal[tuple(field.enum)]

        if field.type == "array":
            if field.items is None:  # pragma: no cover - guarded by the config schema
                raise _shape_error(
                    f"shape {model_name!r}: array field {field_name!r} requires 'items'",
                    shape=model_name,
                )
            item_annotation = self._base_annotation(model_name, f"{field_name}_item", field.items)
            return list[item_annotation]
        if field.type == "object":
            return self._object_model(f"{model_name}__{field_name}", field)

        return {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "null": type(None),
        }[field.type]

    def _object_model(self, model_name: str, field: OrchidPipelineFieldConfig) -> Any:
        if field.properties is None:
            if field.additional_properties:
                return dict[str, Any]
            # ``additional_properties: false`` with no properties is a closed
            # empty object; a free-form map must opt in explicitly.
            return create_model(model_name, __config__=_OBJECT_FORBID)

        definitions = {
            property_name: self._field_definition(model_name, property_name, property_field)
            for property_name, property_field in field.properties.items()
        }
        config = _OBJECT_ALLOW if field.additional_properties else _OBJECT_FORBID
        return create_model(model_name, __config__=config, **definitions)

    def _check_pattern_engine(self, model_name: str, field_name: str, pattern: str) -> None:
        """Reject a pattern Pydantic's Rust regex engine cannot compile.

        P-001 validates patterns with Python ``re`` (plus a ReDoS heuristic);
        look-around and backreferences pass there but fail inside Pydantic with
        an opaque schema error. Compiling a throwaway adapter here turns that
        into a shape error naming the shape, field and pattern.
        """
        try:
            TypeAdapter(Annotated[str, StringConstraints(pattern=pattern)])
        except Exception as exc:
            reason = " ".join(str(exc).split())
            raise _shape_error(
                f"shape {model_name!r}: field {field_name!r} pattern {pattern!r} is not supported by "
                f"Pydantic's regex engine: {reason}",
                shape=model_name,
            ) from exc


def _optional(annotation: Any) -> Any:
    if annotation is type(None):
        return annotation
    return annotation | None


def compile_shapes(
    shapes: Mapping[str, OrchidPipelineShapeConfig],
    *,
    resolve_class: Callable[[str], type[Any]] | None = None,
    allow_shape_version_mismatch: bool = False,
) -> OrchidShapeRegistry:
    """Compile declared shapes into an immutable registry.

    ``allow_shape_version_mismatch`` is the pipeline-level default; a shape's
    own flag wins when either is ``True``. Callers that also need
    :meth:`ShapeCompiler.compile_inline` should keep the compiler instance.
    """
    return ShapeCompiler(
        shapes,
        resolve_class=resolve_class,
        allow_shape_version_mismatch=allow_shape_version_mismatch,
    ).compile()
