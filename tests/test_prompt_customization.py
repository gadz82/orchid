"""Tests for the customisable internal prompts (Phase 1 prompt customisation).

Covers four extension points:

  1. ``OrchidAgentPromptConfig`` — agentic-loop section templates.
  2. ``OrchidMiniAgentConfig.system_prompt_template`` — mini-agent
     focused prompt assembly.
  3. ``OrchidQueryTransformerPromptsConfig`` — per-transformer prompt
     overrides for the four built-in query transformers.
  4. The defaults-merger pass on ``rag.retrieval.transformer_prompts``.

For each extension point we assert two things: defaults still produce
the legacy strings (regression guard), and overrides are threaded
through end-to-end.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchid_ai.agents.mini_agent_node import _build_mini_system_prompt
from orchid_ai.config.schema import (
    OrchidAgentConfig,
    OrchidAgentPromptConfig,
    OrchidAgentsConfig,
    OrchidDefaultsConfig,
    OrchidMiniAgentConfig,
    OrchidQueryTransformerPromptsConfig,
    OrchidRAGConfig,
    OrchidRAGDefaultsConfig,
    OrchidRetrievalConfig,
)
from orchid_ai.config.schema_prompts import (
    DEFAULT_MCP_PROMPT_TEMPLATE,
    DEFAULT_PRIOR_RESULTS_HEADER,
    DEFAULT_RAG_HEADER,
    DEFAULT_RESOURCE_TEMPLATE,
    DEFAULT_RESOURCES_HEADER,
    DEFAULT_SKIPPED_PROMPT_TEMPLATE,
    DEFAULT_SUMMARISE_HISTORY_REMINDER,
    DEFAULT_SUMMARISE_PRIOR_RESULTS_HEADER,
    DEFAULT_SUMMARISE_RAG_HEADER,
    DEFAULT_SUMMARISE_USER_TEMPLATE,
)
from orchid_ai.core.helpers import summarise as helpers_summarise
from orchid_ai.rag.transformers import (
    DecomposeTransformer,
    HyDETransformer,
    MultiQueryTransformer,
    ReformulateTransformer,
    clear_query_transformers,
    get_query_transformer,
    resolve_transformer_kwargs,
)
from orchid_ai.rag.transformers.decompose import DEFAULT_DECOMPOSE_PROMPT
from orchid_ai.rag.transformers.hyde import DEFAULT_MULTI_PROMPT, DEFAULT_SINGLE_PROMPT
from orchid_ai.rag.transformers.multi_query import DEFAULT_MULTI_QUERY_PROMPT
from orchid_ai.rag.transformers.reformulate import DEFAULT_REFORMULATE_PROMPT


# ── 1. OrchidAgentPromptConfig ─────────────────────────────────


class TestOrchidAgentPromptConfig:
    def test_defaults_match_legacy_strings(self) -> None:
        cfg = OrchidAgentPromptConfig()
        # The defaults are sourced from module-level constants so the
        # test guarantees both paths agree.
        assert cfg.prior_results_header == DEFAULT_PRIOR_RESULTS_HEADER
        assert cfg.mcp_prompt_template == DEFAULT_MCP_PROMPT_TEMPLATE
        assert cfg.skipped_prompt_template == DEFAULT_SKIPPED_PROMPT_TEMPLATE
        assert cfg.resources_header == DEFAULT_RESOURCES_HEADER
        assert cfg.resource_template == DEFAULT_RESOURCE_TEMPLATE
        assert cfg.rag_header == DEFAULT_RAG_HEADER
        assert cfg.prior_results_max_chars == 4000
        assert cfg.resource_max_chars == 2000
        # Summarise-helper defaults.
        assert cfg.summarise_history_reminder == DEFAULT_SUMMARISE_HISTORY_REMINDER
        assert cfg.summarise_prior_results_header == DEFAULT_SUMMARISE_PRIOR_RESULTS_HEADER
        assert cfg.summarise_rag_section_header == DEFAULT_SUMMARISE_RAG_HEADER
        assert cfg.summarise_user_template == DEFAULT_SUMMARISE_USER_TEMPLATE
        assert cfg.summarise_prior_results_max_chars == 4000

    def test_overrides_are_honoured(self) -> None:
        cfg = OrchidAgentPromptConfig(
            prior_results_header="\n## PRIOR",
            mcp_prompt_template="\n>>> {name}: {text}",
            resource_max_chars=512,
        )
        assert cfg.prior_results_header == "\n## PRIOR"
        assert cfg.mcp_prompt_template == "\n>>> {name}: {text}"
        # Untouched fields fall back to defaults.
        assert cfg.rag_header == DEFAULT_RAG_HEADER
        assert cfg.resource_max_chars == 512

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(Exception):
            OrchidAgentPromptConfig(unknown_field="x")  # type: ignore[arg-type]


class TestAgenticSystemPromptAssembly:
    """End-to-end check: the agent uses prompt_sections to build the prompt."""

    def _build(self, config: OrchidAgentConfig, *, caps: Any, rag_data: Any, state: Any) -> str:
        # Construct a GenericAgent without invoking ``__init__`` so we
        # do not need a chat model / vector reader for what is a pure
        # method test.
        from orchid_ai.agents.generic_agent import GenericAgent

        agent = GenericAgent.__new__(GenericAgent)
        agent._config = config
        agent._agent_peers = {}
        return agent._build_agentic_system_prompt(caps=caps, rag_data=rag_data, state=state)

    def _caps(
        self,
        *,
        rendered: list[dict[str, str]] | None = None,
        skipped: list[dict[str, Any]] | None = None,
        resources: dict[str, str] | None = None,
    ) -> Any:
        c = MagicMock()
        c.rendered_prompts = rendered or []
        c.skipped_prompts = skipped or []
        c.resource_contents = resources or {}
        return c

    def test_default_assembly_contains_legacy_headers(self) -> None:
        cfg = OrchidAgentConfig(description="d", prompt="SYS")
        state = {"mcp_context": {cfg.name: {"prior": "value"}}}
        prompt = self._build(
            cfg,
            caps=self._caps(resources={"docs": "hello"}),
            rag_data=[{"k": "v"}],
            state=state,
        )
        assert "--- Previous Tool Results (from prior turns) ---" in prompt
        assert "--- Available Resources ---" in prompt
        assert "--- Background Knowledge (RAG) ---" in prompt
        # Resource template renders with the section content.
        assert "[docs]" in prompt
        assert "hello" in prompt

    def test_custom_section_headers_used(self) -> None:
        cfg = OrchidAgentConfig(
            description="d",
            prompt="SYS",
            prompt_sections=OrchidAgentPromptConfig(
                prior_results_header="\n#### prior",
                resources_header="\n#### resources",
                rag_header="\n#### rag",
                resource_template="\n* {name} :: {content}",
            ),
        )
        state = {"mcp_context": {cfg.name: {"prior": "value"}}}
        prompt = self._build(
            cfg,
            caps=self._caps(resources={"docs": "hello"}),
            rag_data=[{"k": "v"}],
            state=state,
        )
        assert "#### prior" in prompt
        assert "#### resources" in prompt
        assert "#### rag" in prompt
        assert "* docs :: hello" in prompt
        # Legacy default markers must not leak in when overridden.
        assert "--- Previous Tool Results (from prior turns) ---" not in prompt

    def test_resource_truncation_respects_config(self) -> None:
        cfg = OrchidAgentConfig(
            description="d",
            prompt="SYS",
            prompt_sections=OrchidAgentPromptConfig(resource_max_chars=8),
        )
        prompt = self._build(
            cfg,
            caps=self._caps(resources={"big": "x" * 100}),
            rag_data=[],
            state=None,
        )
        # 8 chars of x come through, the 9th onward is dropped.
        assert "xxxxxxxx" in prompt
        assert "x" * 9 not in prompt


# ── 2. Mini-agent system prompt template ───────────────────────


class TestMiniSystemPromptTemplate:
    def test_default_assembly_unchanged(self) -> None:
        rendered = _build_mini_system_prompt(
            parent_prompt="PARENT",
            instruction="DO THIS",
            tool_subset=["alpha", "beta"],
            caps_descriptions={"alpha": "Alpha tool", "beta": "Beta tool"},
        )
        assert rendered.startswith("PARENT")
        assert "DO THIS" in rendered
        assert "Tools available to you:" in rendered
        assert "- alpha: Alpha tool" in rendered
        assert "- beta: Beta tool" in rendered

    def test_template_placeholders_resolved(self) -> None:
        template = "[parent]\n{parent_prompt}\n[instruction]\n{instruction}\n[tools]\n{tool_list}"
        rendered = _build_mini_system_prompt(
            parent_prompt="PARENT",
            instruction="DO THIS",
            tool_subset=["alpha"],
            caps_descriptions={"alpha": "Alpha tool"},
            template=template,
        )
        assert rendered == "[parent]\nPARENT\n[instruction]\nDO THIS\n[tools]\n- alpha: Alpha tool"

    def test_template_with_empty_tool_subset(self) -> None:
        template = "P:{parent_prompt}|I:{instruction}|T:{tool_list}"
        rendered = _build_mini_system_prompt(
            parent_prompt="P",
            instruction="I",
            tool_subset=[],
            caps_descriptions={},
            template=template,
        )
        assert rendered == "P:P|I:I|T:"

    def test_mini_agent_config_field_default(self) -> None:
        cfg = OrchidMiniAgentConfig()
        assert cfg.system_prompt_template is None

    def test_mini_agent_config_extra_keys_rejected(self) -> None:
        with pytest.raises(Exception):
            OrchidMiniAgentConfig(unknown="value")  # type: ignore[arg-type]


# ── 3. Query transformer prompts ───────────────────────────────


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_query_transformers()
    yield
    clear_query_transformers()


class TestTransformerDefaults:
    """Default behaviour: legacy prompt strings are still emitted."""

    @pytest.mark.asyncio
    async def test_multi_query_uses_default_prompt(self) -> None:
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=MagicMock(content="alt 1\nalt 2\nalt 3"))

        await MultiQueryTransformer(num_queries=3).transform("query", chat_model=chat_model)

        sys_msg = chat_model.ainvoke.await_args.args[0][0]
        assert sys_msg.content == DEFAULT_MULTI_QUERY_PROMPT.format(n=3)

    @pytest.mark.asyncio
    async def test_decompose_uses_default_prompt(self) -> None:
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=MagicMock(content="q1\nq2"))

        await DecomposeTransformer(max_sub_queries=4).transform("query", chat_model=chat_model)

        sys_msg = chat_model.ainvoke.await_args.args[0][0]
        assert sys_msg.content == DEFAULT_DECOMPOSE_PROMPT.format(n=4)

    @pytest.mark.asyncio
    async def test_hyde_single_default(self) -> None:
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=MagicMock(content="paragraph"))

        await HyDETransformer(n_hypothetical=1).transform("query", chat_model=chat_model)

        sys_msg = chat_model.ainvoke.await_args.args[0][0]
        assert sys_msg.content == DEFAULT_SINGLE_PROMPT

    @pytest.mark.asyncio
    async def test_hyde_multi_default(self) -> None:
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=MagicMock(content="p1\np2"))

        await HyDETransformer(n_hypothetical=2).transform("query", chat_model=chat_model)

        sys_msg = chat_model.ainvoke.await_args.args[0][0]
        assert sys_msg.content == DEFAULT_MULTI_PROMPT.format(n=2)

    @pytest.mark.asyncio
    async def test_reformulate_default(self) -> None:
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=MagicMock(content="rewrite"))

        await ReformulateTransformer().transform(
            "the first one",
            chat_model=chat_model,
            history=[{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}],
        )

        # Reformulate threads system + history + user via dict messages.
        sent = chat_model.ainvoke.await_args.args[0]
        assert sent[0]["content"] == DEFAULT_REFORMULATE_PROMPT


class TestTransformerCustomPrompts:
    """Custom prompts: constructor kwargs override the defaults."""

    @pytest.mark.asyncio
    async def test_multi_query_custom_prompt(self) -> None:
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=MagicMock(content=""))

        custom = "Make {n} queries"
        await MultiQueryTransformer(num_queries=2, system_prompt=custom).transform(
            "query",
            chat_model=chat_model,
        )

        sys_msg = chat_model.ainvoke.await_args.args[0][0]
        assert sys_msg.content == "Make 2 queries"

    @pytest.mark.asyncio
    async def test_hyde_custom_single_and_multi(self) -> None:
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=MagicMock(content="x"))

        await HyDETransformer(n_hypothetical=1, single_prompt="ONE").transform(
            "q",
            chat_model=chat_model,
        )
        assert chat_model.ainvoke.await_args.args[0][0].content == "ONE"

        chat_model.ainvoke.reset_mock()
        chat_model.ainvoke = AsyncMock(return_value=MagicMock(content="x\ny"))
        await HyDETransformer(n_hypothetical=2, multi_prompt="MULTI {n}").transform(
            "q",
            chat_model=chat_model,
        )
        assert chat_model.ainvoke.await_args.args[0][0].content == "MULTI 2"

    @pytest.mark.asyncio
    async def test_decompose_custom_prompt(self) -> None:
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=MagicMock(content=""))
        await DecomposeTransformer(max_sub_queries=3, system_prompt="split {n}").transform(
            "q",
            chat_model=chat_model,
        )
        assert chat_model.ainvoke.await_args.args[0][0].content == "split 3"

    @pytest.mark.asyncio
    async def test_reformulate_custom_prompt(self) -> None:
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=MagicMock(content="rewrite"))
        await ReformulateTransformer(system_prompt="REWRITE").transform(
            "ambiguous",
            chat_model=chat_model,
            history=[{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}],
        )
        assert chat_model.ainvoke.await_args.args[0][0]["content"] == "REWRITE"


class TestRegistryKwargForwarding:
    def test_get_query_transformer_forwards_system_prompt(self) -> None:
        t = get_query_transformer("multi_query", system_prompt="custom {n}")
        assert isinstance(t, MultiQueryTransformer)
        assert t._system_prompt == "custom {n}"

    def test_get_query_transformer_no_kwargs_uses_defaults(self) -> None:
        t = get_query_transformer("multi_query")
        assert t._system_prompt == DEFAULT_MULTI_QUERY_PROMPT

    def test_resolve_kwargs_for_each_builtin(self) -> None:
        prompts = OrchidQueryTransformerPromptsConfig(
            multi_query="MQ",
            decompose="DC",
            reformulate="RF",
        )
        prompts.hyde.single = "HS"
        prompts.hyde.multi = "HM"

        assert resolve_transformer_kwargs("multi_query", prompts) == {"system_prompt": "MQ"}
        assert resolve_transformer_kwargs("decompose", prompts) == {"system_prompt": "DC"}
        assert resolve_transformer_kwargs("reformulate", prompts) == {"system_prompt": "RF"}
        assert resolve_transformer_kwargs("hyde", prompts) == {
            "single_prompt": "HS",
            "multi_prompt": "HM",
        }

    def test_resolve_kwargs_returns_empty_when_unset(self) -> None:
        empty = OrchidQueryTransformerPromptsConfig()
        for name in ("multi_query", "decompose", "reformulate", "hyde"):
            assert resolve_transformer_kwargs(name, empty) == {}

    def test_resolve_kwargs_none_safe(self) -> None:
        assert resolve_transformer_kwargs("multi_query", None) == {}

    def test_resolve_kwargs_unknown_name_returns_empty(self) -> None:
        # Custom transformers fall back to their own defaults.
        assert resolve_transformer_kwargs("custom", OrchidQueryTransformerPromptsConfig()) == {}


# ── 4. Summarise() helper overrides ─────────────────────────────


class TestSummariseHelperOverrides:
    """``core.helpers.summarise`` honours the per-agent prompt overrides."""

    @pytest.mark.asyncio
    async def test_default_reminder_and_headers(self) -> None:
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=MagicMock(content="ok"))

        await helpers_summarise(
            "What's the weather?",
            {"weather": "sunny"},
            [{"k": "v"}],
            system_prompt="You are an assistant.",
            chat_model=chat_model,
            conversation_history=[{"role": "user", "content": "Hi"}],
            prior_tool_context={"prev": "value"},
        )

        sent = chat_model.ainvoke.await_args.args[0]
        sys_content = sent[0]["content"]
        # Default reminder + prior-results header are preserved.
        assert "IMPORTANT: The conversation history below" in sys_content
        assert "--- Previous Tool Results (from prior turns) ---" in sys_content
        # User content uses default template.
        user_content = sent[-1]["content"]
        assert "User query: What's the weather?" in user_content
        assert "Background knowledge (from RAG):" in user_content
        assert "Live data (from API):" in user_content

    @pytest.mark.asyncio
    async def test_overrides_replace_defaults(self) -> None:
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=MagicMock(content="ok"))

        await helpers_summarise(
            "weather?",
            {"k": "v"},
            [{"r": "v"}],
            system_prompt="SYS",
            chat_model=chat_model,
            conversation_history=[{"role": "user", "content": "Hi"}],
            prior_tool_context={"p": "v"},
            history_reminder="\n\n[REMINDER]",
            prior_results_header="\n\n=== PRIOR ===\n",
            rag_section_header="=== RAG ===\n",
            user_content_template="Q: {query}\n{rag_section}DATA: {mcp_data}",
        )

        sent = chat_model.ainvoke.await_args.args[0]
        sys_content = sent[0]["content"]
        assert "[REMINDER]" in sys_content
        assert "=== PRIOR ===" in sys_content
        # Default reminder text must NOT leak in.
        assert "IMPORTANT: The conversation history below" not in sys_content

        user_content = sent[-1]["content"]
        assert user_content.startswith("Q: weather?")
        assert "=== RAG ===" in user_content
        assert "DATA:" in user_content
        # Default template strings absent.
        assert "User query:" not in user_content
        assert "Live data (from API):" not in user_content

    @pytest.mark.asyncio
    async def test_prior_results_max_chars_truncation(self) -> None:
        chat_model = MagicMock()
        chat_model.ainvoke = AsyncMock(return_value=MagicMock(content="ok"))
        big = "x" * 1000

        await helpers_summarise(
            "q",
            {},
            [],
            system_prompt="SYS",
            chat_model=chat_model,
            prior_tool_context={"big": big},
            prior_results_max_chars=20,
        )

        sys_content = chat_model.ainvoke.await_args.args[0][0]["content"]
        # The JSON dump is truncated to 20 chars after the header.
        # Find the dump and check its length.
        idx = sys_content.find('"big"')
        assert idx >= 0
        # The total dump length is bounded.
        dump_segment = sys_content[idx:]
        assert len(dump_segment) <= 22  # "big": " + a few chars


# ── 5. Defaults merging ─────────────────────────────────────────


class TestDefaultsMergeTransformerPrompts:
    def test_agent_inherits_unset_overrides_from_defaults(self) -> None:
        defaults_retrieval = OrchidRetrievalConfig(
            transformer_prompts=OrchidQueryTransformerPromptsConfig(
                multi_query="MQ-default",
                decompose="DC-default",
            ),
        )
        cfg = OrchidAgentsConfig(
            defaults=OrchidDefaultsConfig(
                rag=OrchidRAGDefaultsConfig(retrieval=defaults_retrieval),
            ),
            agents={
                "a": OrchidAgentConfig(description="d", prompt="P"),
            },
        )
        merged = cfg.agents["a"].rag.retrieval.transformer_prompts
        assert merged.multi_query == "MQ-default"
        assert merged.decompose == "DC-default"

    def test_agent_overrides_take_precedence(self) -> None:
        defaults_retrieval = OrchidRetrievalConfig(
            transformer_prompts=OrchidQueryTransformerPromptsConfig(
                multi_query="MQ-default",
            ),
        )
        cfg = OrchidAgentsConfig(
            defaults=OrchidDefaultsConfig(
                rag=OrchidRAGDefaultsConfig(retrieval=defaults_retrieval),
            ),
            agents={
                "a": OrchidAgentConfig(
                    description="d",
                    prompt="P",
                    rag=OrchidRAGConfig(
                        retrieval=OrchidRetrievalConfig(
                            transformer_prompts=OrchidQueryTransformerPromptsConfig(
                                multi_query="MQ-agent",
                            ),
                        ),
                    ),
                ),
            },
        )
        merged = cfg.agents["a"].rag.retrieval.transformer_prompts
        assert merged.multi_query == "MQ-agent"

    def test_hyde_nested_inherits_independently(self) -> None:
        defaults_retrieval = OrchidRetrievalConfig(
            transformer_prompts=OrchidQueryTransformerPromptsConfig(),
        )
        defaults_retrieval.transformer_prompts.hyde.single = "HS-default"
        defaults_retrieval.transformer_prompts.hyde.multi = "HM-default"

        agent_retrieval = OrchidRetrievalConfig(
            transformer_prompts=OrchidQueryTransformerPromptsConfig(),
        )
        # Agent overrides only ``single``.
        agent_retrieval.transformer_prompts.hyde.single = "HS-agent"

        cfg = OrchidAgentsConfig(
            defaults=OrchidDefaultsConfig(
                rag=OrchidRAGDefaultsConfig(retrieval=defaults_retrieval),
            ),
            agents={
                "a": OrchidAgentConfig(
                    description="d",
                    prompt="P",
                    rag=OrchidRAGConfig(retrieval=agent_retrieval),
                ),
            },
        )
        merged = cfg.agents["a"].rag.retrieval.transformer_prompts.hyde
        assert merged.single == "HS-agent"
        assert merged.multi == "HM-default"
