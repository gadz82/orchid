"""Shared test fixtures for the Orchid framework."""

from __future__ import annotations

import pytest
from datetime import datetime

from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.core.repository import OrchidVectorReader, OrchidVectorWriter, OrchidVectorStoreAdmin
from orchid_ai.core.mcp import OrchidMCPClient, OrchidMCPToolResult
from orchid_ai.rag.scopes import OrchidRAGScope
from orchid_ai.rag.backends.null import NullVectorReader
from orchid_ai.persistence.models import OrchidChatSession, OrchidChatMessage


# ── OrchidAuthContext fixtures ──


@pytest.fixture
def auth():
    return OrchidAuthContext(access_token="test-token", tenant_key="t-123", user_id="u-456")


@pytest.fixture
def expired_auth():
    return OrchidAuthContext(access_token="expired", tenant_key="t-123", user_id="u-456", expires_at=1.0)


@pytest.fixture
def scope():
    return OrchidRAGScope(tenant_id="t-123", user_id="u-456", chat_id="c-789", agent_id="test")


# ── Mock vector reader that records calls ──


class MockVectorReader(OrchidVectorReader):
    def __init__(self, results=None):
        self.calls = []
        self._results = results or []

    async def retrieve(self, query, namespace, k=5, scope=None):
        self.calls.append({"query": query, "namespace": namespace, "k": k, "scope": scope})
        return self._results


class MockVectorWriter(OrchidVectorWriter):
    def __init__(self):
        self.indexed = []
        self.upserted = []
        self.deleted = []

    async def index(self, documents, namespace):
        self.indexed.append((documents, namespace))

    async def upsert(self, documents, namespace):
        self.upserted.append((documents, namespace))

    async def delete(self, document_ids, namespace):
        self.deleted.append((document_ids, namespace))


class MockVectorRepository(MockVectorReader, MockVectorWriter, OrchidVectorStoreAdmin):
    def __init__(self, results=None):
        MockVectorReader.__init__(self, results)
        MockVectorWriter.__init__(self)

    async def ensure_collections(self, namespaces):
        pass


@pytest.fixture
def null_reader():
    return NullVectorReader()


@pytest.fixture
def mock_reader():
    return MockVectorReader()


@pytest.fixture
def mock_writer():
    return MockVectorWriter()


@pytest.fixture
def mock_repository():
    return MockVectorRepository()


# ── Mock MCP client ──


class MockMCPClient(OrchidMCPClient):
    def __init__(self, tool_results=None):
        self._tool_results = tool_results or {}
        self.tool_calls = []

    async def call_tool(self, tool_name, arguments, auth):
        self.tool_calls.append({"tool": tool_name, "args": arguments})
        result = self._tool_results.get(
            tool_name,
            OrchidMCPToolResult(content=[{"type": "text", "text": f"result_{tool_name}"}]),
        )
        return result

    async def list_tools(self, auth):
        return [{"name": k, "description": f"Tool {k}"} for k in self._tool_results]

    async def list_prompts(self, auth):
        return []

    async def list_resources(self, auth):
        return []

    async def get_prompt(self, name, arguments, auth):
        return []

    async def read_resource(self, uri, auth):
        return ""

    @property
    def server_url(self):
        return "http://mock-mcp"


@pytest.fixture
def mock_mcp():
    return MockMCPClient()


# ── Chat fixtures ──


@pytest.fixture
def chat_session():
    return OrchidChatSession(
        id="chat-1",
        tenant_id="t-123",
        user_id="u-456",
        title="Test Chat",
        created_at=datetime(2025, 1, 1),
        updated_at=datetime(2025, 1, 1),
    )


@pytest.fixture
def chat_message():
    return OrchidChatMessage(id="msg-1", chat_id="chat-1", role="user", content="Hello")
