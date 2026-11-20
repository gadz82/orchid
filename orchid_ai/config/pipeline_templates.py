"""Template grammar and JMESPath validation shared by configuration and rendering.

A pipeline template is literal text with ``{{ expr }}`` placeholders. This module
owns the grammar — delimiters, the ``\\{{`` escape, expression compilation and
the standard-functions-only rule — so the configuration loader and the runtime
renderer accept exactly the same language.

It lives in ``config/`` because a template is part of the ``pipelines:``
contract: ``config/schema_pipeline.py`` validates it at load time and
``pipeline/mapping.py`` renders it, and both depend on this module instead of on
each other. That keeps the package dependency direction one-way.

Nothing here uses ``eval``, Python expressions or arbitrary function calls:
JMESPath is the only expression language, and v1 registers no custom functions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jmespath
from jmespath.exceptions import JMESPathError
from jmespath.functions import Functions
from jmespath.parser import ParsedResult

from ..core.pipeline import OrchidPipelineMappingError

__all__ = [
    "TemplateToken",
    "compile_expression",
    "split_template",
    "validate_expression",
    "validate_template",
]

_ESCAPED_OPEN = "\\{{"


@dataclass(frozen=True, slots=True)
class TemplateToken:
    """One parsed template piece: literal text, or a JMESPath expression."""

    is_expression: bool
    text: str


def split_template(template: str, *, path: str) -> list[TemplateToken]:
    """Split a template into literal and expression tokens.

    ``\\{{`` opens a literal run that extends to the matching ``}}``; the
    backslash and the delimiters are dropped so the rendered text carries the
    original braces. Every other ``{{`` opens an expression, and a stray ``}}``
    is a syntax error.
    """
    if not isinstance(template, str):
        raise TypeError(f"{path} must be a string")

    tokens: list[TemplateToken] = []
    literal: list[str] = []
    index = 0
    length = len(template)

    while index < length:
        if template.startswith(_ESCAPED_OPEN, index):
            end = template.find("}}", index + len(_ESCAPED_OPEN))
            if end == -1:
                literal.append("{{")
                index += len(_ESCAPED_OPEN)
            else:
                literal.append(template[index + 1 : end + 2])
                index = end + 2
            continue

        if template.startswith("{{", index):
            end = template.find("}}", index + 2)
            if end == -1:
                raise OrchidPipelineMappingError(f"{path} has unbalanced '{{{{ }}}}' delimiters")
            expression = template[index + 2 : end]
            if not expression.strip():
                raise OrchidPipelineMappingError(f"{path} contains an empty '{{{{ }}}}' expression")
            if "{{" in expression:
                raise OrchidPipelineMappingError(f"{path} contains nested '{{{{ }}}}' delimiters")
            if literal:
                tokens.append(TemplateToken(False, "".join(literal)))
                literal = []
            tokens.append(TemplateToken(True, expression.strip()))
            index = end + 2
            continue

        if template.startswith("}}", index):
            raise OrchidPipelineMappingError(f"{path} has unbalanced '{{{{ }}}}' delimiters")

        literal.append(template[index])
        index += 1

    if literal:
        tokens.append(TemplateToken(False, "".join(literal)))
    return tokens


def compile_expression(expression: str, *, path: str) -> ParsedResult:
    """Compile ``expression`` and reject non-standard function calls."""
    try:
        compiled = jmespath.compile(expression)
    except JMESPathError as exc:
        raise OrchidPipelineMappingError(
            f"{path} contains {expression!r}, which is not a valid JMESPath expression: {exc}"
        ) from exc

    unknown = _unknown_function(compiled.parsed)
    if unknown is not None:
        raise OrchidPipelineMappingError(
            f"{path} contains {expression!r}, which calls {unknown!r}; only standard JMESPath functions are allowed"
        )
    return compiled


def _unknown_function(node: Any) -> str | None:
    """Return the first non-standard function name in a parsed JMESPath tree.

    ``jmespath`` resolves function names at evaluation time, so a template such
    as ``{{ __import__('os') }}`` compiles and only fails on the first render.
    Checking the parse tree against the interpreter's own function table moves
    that failure to configuration load. When the table is unavailable (an
    unexpected ``jmespath`` release) the check degrades to a no-op rather than
    rejecting valid templates.
    """
    table = getattr(Functions, "FUNCTION_TABLE", None)
    if not isinstance(table, Mapping):
        return None

    stack = [node]
    while stack:
        current = stack.pop()
        if not isinstance(current, Mapping):
            continue
        if current.get("type") == "function_expression":
            name = current.get("value")
            if isinstance(name, str) and name not in table:
                return name
        children = current.get("children")
        if isinstance(children, list):
            stack.extend(children)
    return None


# ORCHID-MEM: 01a1eaf2-3079-75e1-8755-851c40bbf87c — Template validator; configuration load and render share one JMESPath syntax rule.
def validate_template(template: str, *, path: str = "template") -> str:
    """Validate ``template`` and return it unchanged.

    Raises :class:`OrchidPipelineMappingError` for unbalanced or nested
    delimiters, an empty expression, an expression that is not valid JMESPath,
    or a call to a function outside the standard JMESPath set. It is the only
    syntax rule: configuration models call it at load time and
    :func:`orchid_ai.pipeline.mapping.render_template` parses with the same
    grammar at render time.
    """
    for token in split_template(template, path=path):
        if token.is_expression:
            compile_expression(token.text, path=path)
    return template


def validate_expression(expression: str, *, path: str = "expression") -> str:
    """Validate a bare JMESPath ``expression`` and return it unchanged.

    Bare expressions (``when``, ``writes.from``, ``output.from``, ``map.over``)
    never interpolate, but they must still be valid standard JMESPath, so they
    share the compiler check with :func:`validate_template` instead of a second
    syntax rule.
    """
    if not isinstance(expression, str):
        raise TypeError(f"{path} must be a string")
    if not expression.strip():
        raise OrchidPipelineMappingError(f"{path} must be a non-empty JMESPath expression")
    compile_expression(expression.strip(), path=path)
    return expression
