"""Tests for OrchidMCPToolResult and OrchidMCPClient ABC from src/core/mcp.py."""

from __future__ import annotations

import pytest

from orchid_ai.core.mcp import OrchidMCPClient, OrchidMCPToolResult


# ── OrchidMCPToolResult defaults ──


def test_tool_result_defaults():
    r = OrchidMCPToolResult()
    assert r.content == []
    assert r.is_error is False


# ── .text property ──


def test_text_concatenates_text_blocks():
    r = OrchidMCPToolResult(
        content=[
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]
    )
    assert r.text == "hello\nworld"


def test_text_ignores_non_text_blocks():
    r = OrchidMCPToolResult(
        content=[
            {"type": "text", "text": "keep"},
            {"type": "image", "url": "http://img.png"},
            {"type": "text", "text": "this"},
        ]
    )
    assert r.text == "keep\nthis"


def test_text_empty_for_empty_content():
    r = OrchidMCPToolResult(content=[])
    assert r.text == ""


def test_text_empty_for_only_non_text_blocks():
    r = OrchidMCPToolResult(content=[{"type": "image", "url": "x"}])
    assert r.text == ""


# ── OrchidMCPClient is abstract ──


def test_mcp_client_is_abstract():
    with pytest.raises(TypeError):
        OrchidMCPClient()


# ── Mock MCP client works ──


@pytest.mark.asyncio
async def test_mock_mcp_call_tool(mock_mcp, auth):
    result = await mock_mcp.call_tool("my_tool", {"arg": 1}, auth)
    assert "result_my_tool" in result.text
    assert len(mock_mcp.tool_calls) == 1
    assert mock_mcp.tool_calls[0]["tool"] == "my_tool"


@pytest.mark.asyncio
async def test_mock_mcp_server_url(mock_mcp):
    assert mock_mcp.server_url == "http://mock-mcp"
