from __future__ import annotations

import pytest

from orchid_ai.core.tool import OrchidToolInput
from orchid_ai.tools.function_tool import FunctionTool


@pytest.mark.asyncio
async def test_wraps_sync_function():
    def add(a: int, b: int) -> int:
        return a + b

    tool = FunctionTool(add, name="add")

    output = await tool.invoke(OrchidToolInput(parameters={"a": 1, "b": 2}))

    assert output.result == 3


@pytest.mark.asyncio
async def test_wraps_async_function():
    async def add(a: int, b: int) -> int:
        return a + b

    tool = FunctionTool(add, name="add_async")

    output = await tool.invoke(OrchidToolInput(parameters={"a": 3, "b": 4}))

    assert output.result == 7


def test_auto_extract_schema():
    def greet(name: str, count: int = 1, active: bool = True) -> str:
        return f"{name}:{count}:{active}"

    tool = FunctionTool(greet, name="greet")
    schema = tool.get_parameters_schema()

    assert schema["properties"]["name"]["type"] == "string"
    assert schema["properties"]["count"]["type"] == "integer"
    assert schema["properties"]["active"]["type"] == "boolean"
    assert schema["required"] == ["name"]


@pytest.mark.asyncio
async def test_filters_extra_params():
    received: dict[str, int] = {}

    def add(a: int, b: int) -> int:
        received["a"] = a
        received["b"] = b
        return a + b

    tool = FunctionTool(add, name="strict_add")

    output = await tool.invoke(OrchidToolInput(parameters={"a": 5, "b": 6, "ignored": 7}))

    assert output.result == 11
    assert received == {"a": 5, "b": 6}
