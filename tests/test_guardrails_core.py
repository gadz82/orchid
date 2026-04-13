"""Tests for core guardrail ABCs, data classes, and GuardrailChain."""

from __future__ import annotations

import pytest

from orchid_ai.core.guardrails import (
    Guardrail,
    GuardrailAction,
    GuardrailChain,
    GuardrailContext,
    GuardrailDirection,
    GuardrailResult,
)


# ── Test helpers ─────────────────────────────────────────────


class AlwaysPassGuardrail(Guardrail):
    @property
    def name(self) -> str:
        return "always_pass"

    async def check(self, content: str, context: GuardrailContext) -> GuardrailResult:
        return GuardrailResult.passed(self.name)


class AlwaysBlockGuardrail(Guardrail):
    def __init__(self, message: str = "Blocked"):
        self._message = message

    @property
    def name(self) -> str:
        return "always_block"

    async def check(self, content: str, context: GuardrailContext) -> GuardrailResult:
        return GuardrailResult(
            triggered=True,
            action=GuardrailAction.BLOCK,
            guardrail_name=self.name,
            message=self._message,
        )


class RedactGuardrail(Guardrail):
    @property
    def name(self) -> str:
        return "redactor"

    async def check(self, content: str, context: GuardrailContext) -> GuardrailResult:
        if "secret" in content.lower():
            return GuardrailResult(
                triggered=True,
                action=GuardrailAction.REDACT,
                guardrail_name=self.name,
                redacted_content=content.replace("secret", "[REDACTED]"),
            )
        return GuardrailResult.passed(self.name)


class WarnGuardrail(Guardrail):
    @property
    def name(self) -> str:
        return "warner"

    async def check(self, content: str, context: GuardrailContext) -> GuardrailResult:
        return GuardrailResult(
            triggered=True,
            action=GuardrailAction.WARN,
            guardrail_name=self.name,
            message="Warning!",
        )


# ── GuardrailResult tests ────────────────────────────────────


class TestGuardrailResult:
    def test_passed_factory(self):
        result = GuardrailResult.passed("test")
        assert not result.triggered
        assert not result.blocked
        assert result.guardrail_name == "test"

    def test_blocked_property(self):
        result = GuardrailResult(
            triggered=True,
            action=GuardrailAction.BLOCK,
            guardrail_name="test",
        )
        assert result.blocked

    def test_triggered_but_not_blocked(self):
        result = GuardrailResult(
            triggered=True,
            action=GuardrailAction.WARN,
            guardrail_name="test",
        )
        assert result.triggered
        assert not result.blocked


# ── GuardrailContext tests ───────────────────────────────────


class TestGuardrailContext:
    def test_defaults(self):
        ctx = GuardrailContext(direction=GuardrailDirection.INPUT)
        assert ctx.agent_name == ""
        assert ctx.tenant_key == "default"
        assert ctx.user_id == ""
        assert ctx.metadata == {}

    def test_frozen(self):
        ctx = GuardrailContext(direction=GuardrailDirection.INPUT)
        with pytest.raises(AttributeError):
            ctx.agent_name = "test"  # type: ignore[misc]


# ── GuardrailAction tests ───────────────────────────────────


class TestGuardrailAction:
    def test_enum_values(self):
        assert GuardrailAction.ALLOW.value == "allow"
        assert GuardrailAction.BLOCK.value == "block"
        assert GuardrailAction.REDACT.value == "redact"
        assert GuardrailAction.WARN.value == "warn"
        assert GuardrailAction.LOG.value == "log"


# ── GuardrailChain tests ────────────────────────────────────


class TestGuardrailChain:
    @pytest.fixture
    def input_context(self):
        return GuardrailContext(direction=GuardrailDirection.INPUT)

    @pytest.mark.asyncio
    async def test_empty_chain_passes(self, input_context):
        chain = GuardrailChain()
        result = await chain.evaluate("any content", input_context)
        assert not result.triggered
        assert chain.empty

    @pytest.mark.asyncio
    async def test_all_pass(self, input_context):
        chain = GuardrailChain([AlwaysPassGuardrail(), AlwaysPassGuardrail()])
        result = await chain.evaluate("clean content", input_context)
        assert not result.triggered

    @pytest.mark.asyncio
    async def test_block_short_circuits(self, input_context):
        chain = GuardrailChain(
            [
                AlwaysBlockGuardrail("First block"),
                AlwaysBlockGuardrail("Second block"),
            ]
        )
        result = await chain.evaluate("bad content", input_context)
        assert result.blocked
        assert result.message == "First block"

    @pytest.mark.asyncio
    async def test_pass_then_block(self, input_context):
        chain = GuardrailChain(
            [
                AlwaysPassGuardrail(),
                AlwaysBlockGuardrail("Blocked!"),
            ]
        )
        result = await chain.evaluate("content", input_context)
        assert result.blocked
        assert result.message == "Blocked!"

    @pytest.mark.asyncio
    async def test_redact_propagates(self, input_context):
        chain = GuardrailChain([RedactGuardrail()])
        result = await chain.evaluate("my secret data", input_context)
        assert result.triggered
        assert result.action == GuardrailAction.REDACT
        assert result.redacted_content == "my [REDACTED] data"

    @pytest.mark.asyncio
    async def test_redact_content_passed_to_next(self, input_context):
        """Redacted content should be passed to subsequent guardrails."""
        chain = GuardrailChain([RedactGuardrail(), AlwaysPassGuardrail()])
        result = await chain.evaluate("my secret data", input_context)
        assert result.redacted_content == "my [REDACTED] data"

    @pytest.mark.asyncio
    async def test_warn_collected(self, input_context):
        chain = GuardrailChain([WarnGuardrail()])
        result = await chain.evaluate("content", input_context)
        assert result.triggered
        assert result.action == GuardrailAction.WARN
        assert result.message == "Warning!"

    def test_add_and_len(self):
        chain = GuardrailChain()
        assert len(chain) == 0
        chain.add(AlwaysPassGuardrail())
        assert len(chain) == 1

    def test_repr(self):
        chain = GuardrailChain([AlwaysPassGuardrail(), AlwaysBlockGuardrail()])
        assert "always_pass" in repr(chain)
        assert "always_block" in repr(chain)

    def test_guardrails_property_returns_copy(self):
        chain = GuardrailChain([AlwaysPassGuardrail()])
        guards = chain.guardrails
        guards.append(AlwaysBlockGuardrail())
        assert len(chain) == 1  # original unchanged
