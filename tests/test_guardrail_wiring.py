from __future__ import annotations


import pytest
from langchain_core.messages import AIMessage, HumanMessage

from orchid_ai.config.schema import OrchidGuardrailRuleConfig, OrchidGuardrailsConfig
from orchid_ai.core.guardrails import (
    OrchidGuardrail,
    OrchidGuardrailAction,
    OrchidGuardrailChain,
    OrchidGuardrailContext,
    OrchidGuardrailResult,
)
from orchid_ai.graph.guardrail_wiring import _GuardrailWiring
from orchid_ai.core.state import OrchidAuthContext


class PassGuardrail(OrchidGuardrail):
    @property
    def name(self) -> str:
        return "pass"

    async def check(self, content: str, context: OrchidGuardrailContext) -> OrchidGuardrailResult:
        return OrchidGuardrailResult.passed(self.name)


class BlockGuardrail(OrchidGuardrail):
    @property
    def name(self) -> str:
        return "block"

    async def check(self, content: str, context: OrchidGuardrailContext) -> OrchidGuardrailResult:
        return OrchidGuardrailResult(
            triggered=True,
            action=OrchidGuardrailAction.BLOCK,
            guardrail_name=self.name,
            message="Blocked by guardrail",
        )


class RedactGuardrail(OrchidGuardrail):
    @property
    def name(self) -> str:
        return "redact"

    async def check(self, content: str, context: OrchidGuardrailContext) -> OrchidGuardrailResult:
        return OrchidGuardrailResult(
            triggered=True,
            action=OrchidGuardrailAction.REDACT,
            guardrail_name=self.name,
            redacted_content="[REDACTED]",
        )


class WarnGuardrail(OrchidGuardrail):
    @property
    def name(self) -> str:
        return "warn"

    async def check(self, content: str, context: OrchidGuardrailContext) -> OrchidGuardrailResult:
        return OrchidGuardrailResult(
            triggered=True,
            action=OrchidGuardrailAction.WARN,
            guardrail_name=self.name,
            message="Warning!",
        )


def _make_state(
    messages=None,
    final_response=None,
    auth=None,
    chat_id="c-1",
):
    return {
        "messages": messages or [HumanMessage(content="hello")],
        "chat_id": chat_id,
        "auth_context": auth or OrchidAuthContext(access_token="tok", tenant_key="t-1", user_id="u-1"),
        "final_response": final_response,
        "active_agents": [],
        "pending_agents": [],
    }


class TestBuildChains:
    def test_build_chains_returns_two_chains(self):
        cfg = OrchidGuardrailsConfig(
            input=[OrchidGuardrailRuleConfig(type="max_length", fail_action="block", config={"max_characters": 100})],
            output=[OrchidGuardrailRuleConfig(type="content_safety", fail_action="block")],
        )
        input_chain, output_chain = _GuardrailWiring.build_chains(cfg)
        assert len(input_chain) >= 1
        assert len(output_chain) >= 1

    def test_empty_configs(self):
        cfg = OrchidGuardrailsConfig()
        input_chain, output_chain = _GuardrailWiring.build_chains(cfg)
        assert input_chain is not None
        assert output_chain is not None


class TestInputGuardrailsNode:
    @pytest.mark.asyncio
    async def test_input_guardrail_passes(self):
        chain = OrchidGuardrailChain([PassGuardrail()])
        node = _GuardrailWiring.create_global_input_node(chain)
        state = _make_state()
        result = await node(state)
        assert result == state

    @pytest.mark.asyncio
    async def test_input_guardrail_blocks(self):
        chain = OrchidGuardrailChain([BlockGuardrail()])
        node = _GuardrailWiring.create_global_input_node(chain)
        state = _make_state()
        result = await node(state)
        assert result["final_response"] == "Blocked by guardrail"
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "Blocked by guardrail"
        assert result["active_agents"] == []
        assert result["pending_agents"] == []

    @pytest.mark.asyncio
    async def test_input_guardrail_redacts(self):
        chain = OrchidGuardrailChain([RedactGuardrail()])
        node = _GuardrailWiring.create_global_input_node(chain)
        state = _make_state()
        result = await node(state)
        assert result != state
        messages = result.get("messages", [])
        assert len(messages) >= 1
        assert messages[-1].content == "[REDACTED]"

    @pytest.mark.asyncio
    async def test_input_guardrail_warn_passes_through(self):
        chain = OrchidGuardrailChain([WarnGuardrail()])
        node = _GuardrailWiring.create_global_input_node(chain)
        state = _make_state()
        result = await node(state)
        assert result == state

    @pytest.mark.asyncio
    async def test_input_no_query_returns_state_unchanged(self):
        chain = OrchidGuardrailChain([BlockGuardrail()])
        node = _GuardrailWiring.create_global_input_node(chain)
        state = _make_state(messages=[])
        state["messages"] = []  # override the falsy [] default
        result = await node(state)
        assert result is state or result == state

    @pytest.mark.asyncio
    async def test_input_no_auth_context(self):
        chain = OrchidGuardrailChain([PassGuardrail()])
        node = _GuardrailWiring.create_global_input_node(chain)
        state = dict(_make_state())
        state.pop("auth_context", None)
        result = await node(state)
        assert result == state

    @pytest.mark.asyncio
    async def test_input_redact_without_last_human_message(self):
        """When no HumanMessage exists in state, redact guardrail still runs but does not crash."""
        chain = OrchidGuardrailChain([RedactGuardrail()])
        node = _GuardrailWiring.create_global_input_node(chain)
        state = _make_state(messages=[])
        state["messages"] = [AIMessage(content="secret info")]
        result = await node(state)
        messages = result.get("messages", [])
        assert len(messages) >= 1


class TestOutputGuardrailsNode:
    @pytest.mark.asyncio
    async def test_output_no_final_response_returns_unchanged(self):
        chain = OrchidGuardrailChain([BlockGuardrail()])
        node = _GuardrailWiring.create_global_output_node(chain)
        state = _make_state(final_response=None)
        result = await node(state)
        assert result == state

    @pytest.mark.asyncio
    async def test_output_passes(self):
        chain = OrchidGuardrailChain([PassGuardrail()])
        node = _GuardrailWiring.create_global_output_node(chain)
        state = _make_state(final_response="good response")
        result = await node(state)
        assert result == state

    @pytest.mark.asyncio
    async def test_output_blocks(self):
        chain = OrchidGuardrailChain([BlockGuardrail()])
        node = _GuardrailWiring.create_global_output_node(chain)
        state = _make_state(final_response="bad response")
        result = await node(state)
        assert result["final_response"] == "Blocked by guardrail"
        assert result["messages"][0].content == "Blocked by guardrail"

    @pytest.mark.asyncio
    async def test_output_redacts(self):
        chain = OrchidGuardrailChain([RedactGuardrail()])
        node = _GuardrailWiring.create_global_output_node(chain)
        state = _make_state(final_response="contains sensitive data")
        result = await node(state)
        assert result["final_response"] == "[REDACTED]"
        assert result["messages"][0].content == "[REDACTED]"

    @pytest.mark.asyncio
    async def test_output_warn_passes_through(self):
        chain = OrchidGuardrailChain([WarnGuardrail()])
        node = _GuardrailWiring.create_global_output_node(chain)
        state = _make_state(final_response="warn but pass")
        result = await node(state)
        assert result == state

    @pytest.mark.asyncio
    async def test_output_without_auth(self):
        chain = OrchidGuardrailChain([PassGuardrail()])
        node = _GuardrailWiring.create_global_output_node(chain)
        state = dict(_make_state(final_response="response"))
        state.pop("auth_context", None)
        result = await node(state)
        assert result == state
