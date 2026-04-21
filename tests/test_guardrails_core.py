"""Tests for core guardrail ABCs, data classes, and OrchidGuardrailChain."""

from __future__ import annotations

import pytest

from orchid_ai.core.guardrails import (
    OrchidGuardrail,
    OrchidGuardrailAction,
    OrchidGuardrailChain,
    OrchidGuardrailContext,
    OrchidGuardrailDirection,
    OrchidGuardrailResult,
)


# ── Test helpers ─────────────────────────────────────────────


class AlwaysPassGuardrail(OrchidGuardrail):
    @property
    def name(self) -> str:
        return "always_pass"

    async def check(self, content: str, context: OrchidGuardrailContext) -> OrchidGuardrailResult:
        return OrchidGuardrailResult.passed(self.name)


class AlwaysBlockGuardrail(OrchidGuardrail):
    def __init__(self, message: str = "Blocked"):
        self._message = message

    @property
    def name(self) -> str:
        return "always_block"

    async def check(self, content: str, context: OrchidGuardrailContext) -> OrchidGuardrailResult:
        return OrchidGuardrailResult(
            triggered=True,
            action=OrchidGuardrailAction.BLOCK,
            guardrail_name=self.name,
            message=self._message,
        )


class RedactGuardrail(OrchidGuardrail):
    @property
    def name(self) -> str:
        return "redactor"

    async def check(self, content: str, context: OrchidGuardrailContext) -> OrchidGuardrailResult:
        if "secret" in content.lower():
            return OrchidGuardrailResult(
                triggered=True,
                action=OrchidGuardrailAction.REDACT,
                guardrail_name=self.name,
                redacted_content=content.replace("secret", "[REDACTED]"),
            )
        return OrchidGuardrailResult.passed(self.name)


class WarnGuardrail(OrchidGuardrail):
    @property
    def name(self) -> str:
        return "warner"

    async def check(self, content: str, context: OrchidGuardrailContext) -> OrchidGuardrailResult:
        return OrchidGuardrailResult(
            triggered=True,
            action=OrchidGuardrailAction.WARN,
            guardrail_name=self.name,
            message="Warning!",
        )


# ── OrchidGuardrailResult tests ────────────────────────────────────


class TestGuardrailResult:
    def test_passed_factory(self):
        result = OrchidGuardrailResult.passed("test")
        assert not result.triggered
        assert not result.blocked
        assert result.guardrail_name == "test"

    def test_blocked_property(self):
        result = OrchidGuardrailResult(
            triggered=True,
            action=OrchidGuardrailAction.BLOCK,
            guardrail_name="test",
        )
        assert result.blocked

    def test_triggered_but_not_blocked(self):
        result = OrchidGuardrailResult(
            triggered=True,
            action=OrchidGuardrailAction.WARN,
            guardrail_name="test",
        )
        assert result.triggered
        assert not result.blocked


# ── OrchidGuardrailContext tests ───────────────────────────────────


class TestGuardrailContext:
    def test_defaults(self):
        ctx = OrchidGuardrailContext(direction=OrchidGuardrailDirection.INPUT)
        assert ctx.agent_name == ""
        assert ctx.tenant_key == "default"
        assert ctx.user_id == ""
        assert ctx.metadata == {}

    def test_frozen(self):
        ctx = OrchidGuardrailContext(direction=OrchidGuardrailDirection.INPUT)
        with pytest.raises(AttributeError):
            ctx.agent_name = "test"  # type: ignore[misc]


# ── OrchidGuardrailAction tests ───────────────────────────────────


class TestGuardrailAction:
    def test_enum_values(self):
        assert OrchidGuardrailAction.ALLOW.value == "allow"
        assert OrchidGuardrailAction.BLOCK.value == "block"
        assert OrchidGuardrailAction.REDACT.value == "redact"
        assert OrchidGuardrailAction.WARN.value == "warn"
        assert OrchidGuardrailAction.LOG.value == "log"


# ── OrchidGuardrailChain tests ────────────────────────────────────


class TestGuardrailChain:
    @pytest.fixture
    def input_context(self):
        return OrchidGuardrailContext(direction=OrchidGuardrailDirection.INPUT)

    @pytest.mark.asyncio
    async def test_empty_chain_passes(self, input_context):
        chain = OrchidGuardrailChain()
        result = await chain.evaluate("any content", input_context)
        assert not result.triggered
        assert chain.empty

    @pytest.mark.asyncio
    async def test_all_pass(self, input_context):
        chain = OrchidGuardrailChain([AlwaysPassGuardrail(), AlwaysPassGuardrail()])
        result = await chain.evaluate("clean content", input_context)
        assert not result.triggered

    @pytest.mark.asyncio
    async def test_block_short_circuits(self, input_context):
        chain = OrchidGuardrailChain(
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
        chain = OrchidGuardrailChain(
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
        chain = OrchidGuardrailChain([RedactGuardrail()])
        result = await chain.evaluate("my secret data", input_context)
        assert result.triggered
        assert result.action == OrchidGuardrailAction.REDACT
        assert result.redacted_content == "my [REDACTED] data"

    @pytest.mark.asyncio
    async def test_redact_content_passed_to_next(self, input_context):
        """Redacted content should be passed to subsequent guardrails."""
        chain = OrchidGuardrailChain([RedactGuardrail(), AlwaysPassGuardrail()])
        result = await chain.evaluate("my secret data", input_context)
        assert result.redacted_content == "my [REDACTED] data"

    @pytest.mark.asyncio
    async def test_warn_collected(self, input_context):
        chain = OrchidGuardrailChain([WarnGuardrail()])
        result = await chain.evaluate("content", input_context)
        assert result.triggered
        assert result.action == OrchidGuardrailAction.WARN
        assert result.message == "Warning!"

    def test_add_and_len(self):
        chain = OrchidGuardrailChain()
        assert len(chain) == 0
        chain.add(AlwaysPassGuardrail())
        assert len(chain) == 1

    def test_repr(self):
        chain = OrchidGuardrailChain([AlwaysPassGuardrail(), AlwaysBlockGuardrail()])
        assert "always_pass" in repr(chain)
        assert "always_block" in repr(chain)

    def test_guardrails_property_returns_copy(self):
        chain = OrchidGuardrailChain([AlwaysPassGuardrail()])
        guards = chain.guardrails
        guards.append(AlwaysBlockGuardrail())
        assert len(chain) == 1  # original unchanged
