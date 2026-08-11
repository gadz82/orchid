from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import ClassVar

import pytest

from orchid_ai.core.tool import OrchidTool, OrchidToolInput, OrchidToolOutput


class SampleTool(OrchidTool):
    name = "sample"
    description = "Sample tool"
    parameters_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "value": {"type": "string", "description": "Input value"},
        },
        "required": ["value"],
    }

    async def invoke(self, tool_input: OrchidToolInput) -> OrchidToolOutput:
        return OrchidToolOutput(result=tool_input.parameters["value"])


def test_orchid_tool_input_frozen():
    tool_input = OrchidToolInput(parameters={"value": "x"})

    with pytest.raises(FrozenInstanceError):
        tool_input.query = "updated"


def test_orchid_tool_output_defaults():
    output = OrchidToolOutput()

    assert output.result is None
    assert output.metadata == {}


def test_orchid_tool_abc_cannot_instantiate():
    with pytest.raises(TypeError):
        OrchidTool()


def test_orchid_tool_get_llm_function_schema():
    schema = SampleTool().get_llm_function_schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "sample"
    assert schema["function"]["description"] == "Sample tool"
    assert schema["function"]["parameters"]["required"] == ["value"]
