"""Mapping and templating: one strict JMESPath renderer for every pipeline string.

Prompts, ``when`` gates, object-id templates, ``writes.from``, output mappings
and trigger inputs all render through this module. Keeping a single renderer is
what makes a template behave identically at configuration time, at job time and
in a Bloom trigger.

Template grammar (v1):

* ``{{ expr }}`` evaluates ``expr`` as standard JMESPath over the documented
  context (:class:`OrchidPipelineMappingContext`);
* ``{{ expr || 'fallback' }}`` is plain JMESPath — the ``||`` operator is part
  of the language, so no fallback syntax is invented here;
* ``\\{{`` starts a literal ``{{`` run: the backslash is dropped and the text
  up to the matching ``}}`` is emitted verbatim, never evaluated;
* every other character is literal text.

The grammar lives in :mod:`orchid_ai.config.pipeline_templates`, which the
configuration loader also uses, so a template accepted at load time is one the
renderer can evaluate.

Failure behaviour is explicit, never silent:

* a missing path is a strict error (:class:`OrchidPipelineMappingError`) naming
  the template path and the expression — the default for prompts, mappings and
  ``writes.from``;
* a non-strict render (``when`` gates, :func:`evaluate_when`) returns a
  documented default and logs one bounded warning;
* a rendered prompt above the size cap raises
  :class:`OrchidPipelineLimitError` before the value can reach a model;
* a template whose expression is not valid JMESPath, or calls a function that
  is not part of the standard library, is rejected at configuration load.

Nothing here uses ``eval``, Python expressions or arbitrary function calls:
JMESPath is the only expression language, and v1 registers no custom functions.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from jmespath.exceptions import JMESPathError

from ..config.pipeline_templates import compile_expression, split_template
from ..core.pipeline import (
    PIPELINE_INPUT_OBJECT_ID,
    PIPELINE_INPUT_OBJECT_TYPE,
    OrchidDataGraphView,
    OrchidGraphObject,
    OrchidPipelineLimitError,
    OrchidPipelineMappingError,
    OrchidPipelineRunContext,
)

__all__ = [
    "DEFAULT_MAX_RENDERED_CHARS",
    "OrchidPipelineMappingContext",
    "alias_snapshots",
    "build_mapping_context",
    "evaluate",
    "evaluate_when",
    "object_snapshots",
    "references_context_key",
    "render_template",
    "render_value",
    "type_snapshots",
    "with_fallback",
]

#: Default rendered-text cap, mirroring ``pipelines.limits.max_prompt_chars``.
DEFAULT_MAX_RENDERED_CHARS = 32_000

_MAX_WARNING_CHARS = 300
_LOG = logging.getLogger(__name__)

#: Run keys a template may read. The run metadata bag, the tenant key and the
#: user id are deliberately absent: metadata can carry integrator secrets and
#: templates are model-visible.
_RUN_TEMPLATE_KEYS = frozenset(
    {
        "id",
        "pipeline",
        "pipeline_version",
        "mode",
        "started_at",
        "chat_id",
        "correlation_id",
        "parent_run_id",
        "trigger_id",
        "signal_id",
        "deadline",
    }
)


# ORCHID-MEM: 01a1eaf2-3046-7175-886d-f8026f8850af — Mapping context value object; the documented JMESPath shape for prompts, gates, writes and mappings.
@dataclass(frozen=True, slots=True)
class OrchidPipelineMappingContext:
    """The documented JMESPath context every template is evaluated against.

    ``input`` is the validated run input, ``run`` the run identity (``run.id``
    is the run identifier), ``jobs`` the per-job results, ``objects`` every
    object keyed by id, ``types`` the same objects grouped by type (the natural
    source for ``map.over``), ``aliases`` the latest object written under each
    alias, and ``summary`` the interpretation text when one already exists.

    The reserved ``__input__`` object is never part of ``objects`` or ``types``:
    input is reachable through ``input``, and exposing the raw graph node would
    duplicate it under a name operators should not depend on.
    """

    input: Mapping[str, Any]
    run: Mapping[str, Any]
    jobs: Mapping[str, Any] = field(default_factory=dict)
    objects: Mapping[str, Any] = field(default_factory=dict)
    types: Mapping[str, Any] = field(default_factory=dict)
    aliases: Mapping[str, Any] = field(default_factory=dict)
    summary: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "input", _require_mapping(self.input, "context.input"))
        object.__setattr__(self, "run", _require_mapping(self.run, "context.run"))
        for field_name in ("jobs", "objects", "types", "aliases"):
            object.__setattr__(
                self,
                field_name,
                _require_mapping(getattr(self, field_name), f"context.{field_name}"),
            )
        if not isinstance(self.summary, str):
            raise TypeError("context.summary must be a string")

    def to_dict(self) -> dict[str, Any]:
        """Return the plain mapping JMESPath evaluates against."""
        return {
            "input": self.input,
            "run": self.run,
            "jobs": self.jobs,
            "objects": self.objects,
            "types": self.types,
            "aliases": self.aliases,
            "summary": self.summary,
        }


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


# ── Evaluation and rendering ─────────────────────────────────


def _as_context(context: OrchidPipelineMappingContext | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(context, OrchidPipelineMappingContext):
        return context.to_dict()
    if isinstance(context, Mapping):
        return context
    raise TypeError("context must be an OrchidPipelineMappingContext or a mapping")


def _warn(logger: logging.Logger | None, message: str) -> None:
    target = logger or _LOG
    if len(message) > _MAX_WARNING_CHARS:
        message = message[: _MAX_WARNING_CHARS - 1] + "…"
    target.warning("pipeline mapping: %s", message)


# ORCHID-MEM: 01a1eaf2-30ae-7121-9626-315bffe26265 — JMESPath evaluation; strict errors for mappings, false plus bounded warning for gates.
_SCOPE_CHANGING_NODES = frozenset(
    {"subexpression", "projection", "value_projection", "filter_projection", "index_expression"}
)


# ORCHID-MEM: 01a1ef78-6ea7-7fb0-9a40-fa55bb803432 — Mapping expression guard; detects a top-level context key with the JMESPath tree, not a substring.
def references_context_key(expression: str, key: str) -> bool:
    """Whether a mapping expression reads the top-level context ``key``.

    The check uses the template tokenizer and the JMESPath parse tree, not a
    substring: escaped ``\\{{ ... }}`` literals and nested fields such as
    ``jobs.summary`` do not count, while ``summary``, ``length(summary)``,
    ``{summary: summary}`` and ``{{  summary }}`` do.
    """
    if not isinstance(expression, str):
        return False
    if "{{" in expression or "}}" in expression:
        try:
            tokens = split_template(expression, path="expression")
        except Exception:
            return False
        return any(token.is_expression and _expression_reads_key(token.text, key) for token in tokens)
    return _expression_reads_key(expression, key)


def _expression_reads_key(expression: str, key: str) -> bool:
    try:
        parsed = compile_expression(expression, path="expression").parsed
    except Exception:
        return False
    return _node_reads_key(parsed, key)


def _node_reads_key(node: Any, key: str, *, root: bool = True) -> bool:
    if not isinstance(node, Mapping):
        return False
    kind = node.get("type")
    if kind == "field":
        return root and node.get("value") == key
    children = node.get("children") or []
    if kind in {"current", "identity"}:
        return False
    if kind == "subexpression":
        current_root = root
        for index, child in enumerate(children):
            if _node_reads_key(child, key, root=current_root):
                return True
            current_root = current_root and index == 0 and _preserves_scope(child)
        return False
    if kind == "pipe":
        if not children:
            return False
        left = children[0]
        if _node_reads_key(left, key, root=root):
            return True
        if len(children) < 2:
            return False
        return _node_reads_key(children[1], key, root=root and _preserves_scope(left))
    if kind in _SCOPE_CHANGING_NODES:
        if not children:
            return False
        if _node_reads_key(children[0], key, root=root):
            return True
        return any(_node_reads_key(child, key, root=False) for child in children[1:])
    if kind == "expref":
        return any(_node_reads_key(child, key, root=False) for child in children)
    return any(_node_reads_key(child, key, root=root) for child in children)


def _preserves_scope(node: Any) -> bool:
    """Whether ``node`` returns its input unchanged (``@`` / ``identity``)."""
    if not isinstance(node, Mapping):
        return False
    return node.get("type") in {"current", "identity"}


def evaluate(
    expression: str,
    context: OrchidPipelineMappingContext | Mapping[str, Any],
    *,
    path: str = "expression",
    strict: bool = True,
    default: Any = None,
    logger: logging.Logger | None = None,
) -> Any:
    """Evaluate one JMESPath ``expression`` against ``context``.

    A result of ``None`` — JMESPath cannot distinguish a missing path from an
    explicit ``null`` — is a :class:`OrchidPipelineMappingError` in strict mode.
    In non-strict mode it logs one bounded warning and returns ``default``; that
    is the ``when`` contract, where an unresolved path means "gate closed".

    Returned containers are the values inside ``context``. Callers that hand a
    result to code which might mutate it should copy it first.
    """
    if not isinstance(expression, str):
        raise TypeError(f"{path} expression must be a string")
    if not expression.strip():
        raise OrchidPipelineMappingError(f"{path} must be a non-empty JMESPath expression")

    compiled = compile_expression(expression.strip(), path=path)
    context_mapping = _as_context(context)
    try:
        result = compiled.search(context_mapping)
    except JMESPathError as exc:
        # A compiled expression can still fail at search time: wrong argument
        # types, wrong arity, or a function missing from an older interpreter.
        raise OrchidPipelineMappingError(f"{path}: JMESPath expression {expression!r} failed: {exc}") from exc

    if result is None:
        message = f"{path}: JMESPath expression {expression!r} resolved to no value"
        if strict:
            raise OrchidPipelineMappingError(message)
        _warn(logger, message)
        return default
    return result


def _stringify(value: Any, *, path: str, expression: str) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise OrchidPipelineMappingError(
            f"{path}: JMESPath expression {expression!r} produced a value that is not JSON-serializable: {exc}"
        ) from exc


def _raise_size_error(path: str, rendered: int, max_chars: int) -> None:
    raise OrchidPipelineLimitError(
        f"{path}: rendered template is at least {rendered} characters, exceeding the {max_chars}-character cap"
    )


def _check_size(value: Any, *, path: str, expression: str, max_chars: int | None) -> None:
    if max_chars is None:
        return
    size = len(value) if isinstance(value, str) else len(_stringify(value, path=path, expression=expression))
    if size > max_chars:
        _raise_size_error(path, size, max_chars)


# ORCHID-MEM: 01a1eaf2-30e3-71da-9f73-b7188638e6cc — Text renderer; escapes literal braces, single-pass, size-capped.
def render_template(
    template: str,
    context: OrchidPipelineMappingContext | Mapping[str, Any],
    *,
    path: str = "template",
    strict: bool = True,
    default: str = "",
    max_chars: int | None = DEFAULT_MAX_RENDERED_CHARS,
    logger: logging.Logger | None = None,
) -> str:
    """Render ``{{ expr }}`` placeholders in ``template`` to text.

    Expression results are inserted as text: strings verbatim, every other JSON
    value as compact JSON (so a list or object renders predictably instead of
    as a Python repr). ``max_chars=None`` opts out of the cap; the default
    mirrors ``limits.max_prompt_chars`` because rendered templates are prompts.
    """
    _check_max_chars(max_chars)

    tokens = split_template(template, path=path)
    chunks: list[str] = []
    rendered = 0
    for token in tokens:
        if token.is_expression:
            value = evaluate(token.text, context, path=path, strict=strict, default=None, logger=logger)
            chunk = default if value is None else _stringify(value, path=path, expression=token.text)
        else:
            chunk = token.text
        rendered += len(chunk)
        if max_chars is not None and rendered > max_chars:
            _raise_size_error(path, rendered, max_chars)
        chunks.append(chunk)
    return "".join(chunks)


def _check_max_chars(max_chars: int | None) -> None:
    if max_chars is not None and (isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1):
        raise ValueError("max_chars must be a positive integer or None")


def render_value(
    value: Any,
    context: OrchidPipelineMappingContext | Mapping[str, Any],
    *,
    path: str = "value",
    strict: bool = True,
    default: Any = None,
    max_chars: int | None = None,
    logger: logging.Logger | None = None,
) -> Any:
    """Render templates inside a nested value, preserving JSON types.

    A string that is exactly one expression keeps the expression's native
    result (a list stays a list, an object an object, a number a number) and
    uses ``default`` when non-strict; a string that mixes text and expressions
    renders to text, where a non-strict missing value becomes an empty string.
    Mappings and sequences are rendered recursively, so ``writes.data``, an
    ``emit.payload`` and a trigger ``input`` need no per-field plumbing.

    ``max_chars`` defaults to no cap because structured values are bounded by
    the graph limits, not by prompt characters. Pass a cap when the rendered
    value is text destined for a model; when set, it applies to both the native
    and the text path.
    """
    _check_max_chars(max_chars)

    if isinstance(value, str):
        tokens = split_template(value, path=path)
        if len(tokens) == 1 and tokens[0].is_expression:
            result = evaluate(tokens[0].text, context, path=path, strict=strict, default=default, logger=logger)
            _check_size(result, path=path, expression=tokens[0].text, max_chars=max_chars)
            return result
        return render_template(
            value,
            context,
            path=path,
            strict=strict,
            default="",
            max_chars=max_chars,
            logger=logger,
        )
    if isinstance(value, Mapping):
        return {
            key: render_value(
                item,
                context,
                path=f"{path}.{key}",
                strict=strict,
                default=default,
                max_chars=max_chars,
                logger=logger,
            )
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [
            render_value(
                item,
                context,
                path=f"{path}[{index}]",
                strict=strict,
                default=default,
                max_chars=max_chars,
                logger=logger,
            )
            for index, item in enumerate(value)
        ]
    return value


def evaluate_when(
    expression: str,
    context: OrchidPipelineMappingContext | Mapping[str, Any],
    *,
    path: str = "when",
    logger: logging.Logger | None = None,
) -> bool:
    """Evaluate a ``when`` gate: false instead of an error for a missing path.

    Gates are the one documented non-strict context. A missing path logs a
    bounded warning and closes the gate, so one absent optional value never
    fails an entire run. An invalid expression is still an error: the
    configuration loader rejects it first, and silently closing a gate on a typo
    would hide the mistake. A gate is classified as a template when it contains
    template delimiters — the same rule the configuration validator uses — so an
    escaped-only gate renders as text instead of being parsed as JMESPath.

    Gates are not prompts: no rendered-size cap applies to them.
    """
    if not isinstance(expression, str):
        raise TypeError(f"{path} expression must be a string")
    if "{{" in expression or "}}" in expression:
        value = render_value(expression, context, path=path, strict=False, default="", logger=logger)
        return bool(value) and value != ""
    value = evaluate(expression, context, path=path, strict=False, default=False, logger=logger)
    return _truthy(value)


def _truthy(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value != ""
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    return True


def with_fallback(expression: str, fallback: Any) -> str:
    """Return a bare JMESPath ``expression`` with a literal ``fallback`` after ``||``.

    The result is an expression, not a template: pass it to :func:`evaluate`, or
    embed it in a template yourself (``"{{ " + with_fallback(expr, 'unknown') + " }}"``).
    The fallback is embedded as a JMESPath raw-JSON literal, so any JSON value
    works and nothing is interpolated as code.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression must be a non-empty string")
    literal = json.dumps(fallback, ensure_ascii=False, separators=(",", ":")).replace("`", "\\`")
    return f"{expression.strip()} || `{literal}`"


# ── Context builders ─────────────────────────────────────────


def object_snapshots(graph: OrchidDataGraphView | None) -> dict[str, dict[str, Any]]:
    """Return every graph object as a JSON-safe snapshot keyed by id.

    The reserved pipeline input object is excluded: ``input`` already exposes
    it, and its internal id is not part of the operator-facing contract.
    """
    if graph is None:
        return {}
    snapshots: dict[str, dict[str, Any]] = {}
    for obj in graph.find():
        if obj.id == PIPELINE_INPUT_OBJECT_ID or obj.type == PIPELINE_INPUT_OBJECT_TYPE:
            continue
        snapshots[obj.id] = _snapshot(obj)
    return snapshots


def _snapshot(obj: OrchidGraphObject) -> dict[str, Any]:
    return obj.to_dict()


def type_snapshots(objects: Mapping[str, Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group object snapshots by type, preserving graph insertion order.

    Entries are deep copies: a step that reads and then mutates ``types``
    cannot corrupt the matching entry in ``objects`` for the rest of the render.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for snapshot in objects.values():
        object_type = snapshot.get("type")
        if not isinstance(object_type, str):
            continue
        grouped.setdefault(object_type, []).append(copy.deepcopy(snapshot))
    return grouped


def alias_snapshots(
    aliases: Mapping[str, str],
    objects: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Resolve alias → latest object snapshot; a dangling alias is skipped."""
    resolved: dict[str, dict[str, Any]] = {}
    for alias, object_id in aliases.items():
        snapshot = objects.get(object_id)
        if snapshot is not None:
            resolved[alias] = copy.deepcopy(snapshot)
    return resolved


def _run_snapshot(run: OrchidPipelineRunContext | Mapping[str, Any]) -> Mapping[str, Any]:
    """Project a run context to the documented ``run`` mapping.

    The JMESPath name is ``run.id`` (not ``run.run_id``) because that is the
    name the specification and every example template use. The projection reuses
    ``OrchidPipelineRunContext.to_dict`` so values are JSON-normalised, and then
    drops the tenant/user identifiers and the free-form metadata bag: templates
    are model-visible, and metadata is integrator-owned data that may hold
    secrets.
    """
    if isinstance(run, OrchidPipelineRunContext):
        # Built field by field rather than via ``to_dict()``: metadata is dropped
        # below, so a non-JSON value in it must not fail a render that never
        # exposes it.
        return {
            "id": run.run_id,
            "pipeline": run.pipeline,
            "pipeline_version": run.pipeline_version,
            "mode": run.mode,
            "started_at": run.started_at.isoformat(),
            "chat_id": run.chat_id,
            "correlation_id": run.correlation_id,
            "parent_run_id": run.parent_run_id,
            "trigger_id": run.trigger_id,
            "signal_id": run.signal_id,
            "deadline": run.deadline.isoformat() if run.deadline is not None else None,
        }
    mapping = _require_mapping(run, "run")
    snapshot = dict(mapping)
    if "run_id" in snapshot:
        snapshot.setdefault("id", snapshot.pop("run_id"))
    return {key: value for key, value in snapshot.items() if key in _RUN_TEMPLATE_KEYS}


def build_mapping_context(
    *,
    input: Mapping[str, Any],
    run: OrchidPipelineRunContext | Mapping[str, Any],
    jobs: Mapping[str, Any] | None = None,
    graph: OrchidDataGraphView | None = None,
    aliases: Mapping[str, str] | None = None,
    summary: str = "",
) -> OrchidPipelineMappingContext:
    """Assemble the documented JMESPath context for one render.

    ``jobs`` is the per-job result mapping (status, output, objects). ``aliases``
    maps an alias name to the object id the session recorded, so the renderer
    resolves "latest object written under this alias" without knowing the graph's
    alias store. Build the context once per job render and reuse it: it
    serialises every graph object once and copies the grouped ``types`` and the
    aliased entries, which is bounded by the run's graph caps but not free.
    """
    snapshot_input = _require_mapping(input, "input")
    snapshots = object_snapshots(graph)
    return OrchidPipelineMappingContext(
        input=dict(snapshot_input),
        run=dict(_run_snapshot(run)),
        jobs=dict(jobs or {}),
        objects=snapshots,
        types=type_snapshots(snapshots),
        aliases=alias_snapshots(aliases or {}, snapshots),
        summary=summary,
    )
