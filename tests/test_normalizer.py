from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from orchid_ai.tools.normalizer import (
    _NORMALIZER_REGISTRY,
    LLMNormalizer,
    PassthroughNormalizer,
    PromptNormalizer,
    get_normalizer,
    register_normalizer,
)


class _FakeChatModel:
    def __init__(self, response: str = "normalised output") -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, messages: list[dict[str, Any]], **kwargs: Any) -> MagicMock:
        self.calls.append({"messages": messages, **kwargs})
        result = MagicMock()
        result.content = self.response
        return result


class TestPassthroughNormalizer:
    @pytest.mark.asyncio
    async def test_returns_input_unchanged(self) -> None:
        normalizer = PassthroughNormalizer()
        result = await normalizer.normalize("hello world")
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_returns_empty_string_unchanged(self) -> None:
        normalizer = PassthroughNormalizer()
        result = await normalizer.normalize("")
        assert result == ""

    @pytest.mark.asyncio
    async def test_ignores_context(self) -> None:
        normalizer = PassthroughNormalizer()
        result = await normalizer.normalize("prompt", context={"key": "val"})
        assert result == "prompt"


class TestLLMNormalizer:
    @pytest.mark.asyncio
    async def test_returns_canned_response(self) -> None:
        llm = _FakeChatModel(response="reformulated prompt")
        normalizer = LLMNormalizer(llm, instruction="Simplify user requests.")
        result = await normalizer.normalize("do this")
        assert result == "reformulated prompt"

    @pytest.mark.asyncio
    async def test_passes_instruction_as_system(self) -> None:
        llm = _FakeChatModel()
        normalizer = LLMNormalizer(llm, instruction="Translate to French.")
        await normalizer.normalize("hello")
        assert len(llm.calls) == 1
        messages = llm.calls[0]["messages"]
        assert messages[0].content == "Translate to French."

    @pytest.mark.asyncio
    async def test_passes_prompt_as_user(self) -> None:
        llm = _FakeChatModel()
        normalizer = LLMNormalizer(llm, instruction="Fix grammar.")
        await normalizer.normalize("helo world")
        assert len(llm.calls) == 1
        messages = llm.calls[0]["messages"]
        assert messages[1].content == "helo world"

    @pytest.mark.asyncio
    async def test_ignores_context(self) -> None:
        llm = _FakeChatModel()
        normalizer = LLMNormalizer(llm, instruction="Rephrase.")
        await normalizer.normalize("test", context={"extra": "data"})
        assert len(llm.calls) == 1


class TestNormalizerRegistry:
    def test_passthrough_registered_at_import(self) -> None:
        assert "passthrough" in _NORMALIZER_REGISTRY
        assert _NORMALIZER_REGISTRY["passthrough"] is PassthroughNormalizer

    def test_register_and_get(self) -> None:
        register_normalizer("my_passthrough", PassthroughNormalizer)
        inst = get_normalizer("my_passthrough")
        assert isinstance(inst, PassthroughNormalizer)

    def test_unknown_name_falls_back_to_passthrough(self) -> None:
        inst = get_normalizer("nonexistent")
        assert isinstance(inst, PassthroughNormalizer)

    def test_uninstantiable_class_falls_back_to_passthrough(self) -> None:
        class NoDefaultConstructor(PromptNormalizer):
            def __init__(self, required_arg: str) -> None:
                self._arg = required_arg

            async def normalize(self, prompt: str, *, context: dict[str, Any] | None = None) -> str:
                return prompt

        register_normalizer("no_default", NoDefaultConstructor)
        inst = get_normalizer("no_default")
        assert isinstance(inst, PassthroughNormalizer)
