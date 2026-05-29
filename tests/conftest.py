"""Shared test fixtures for the Orchid framework."""

from __future__ import annotations

import os
import pytest
from datetime import datetime

from orchid_ai.agents.skill_executor import _skill_depth_var
from orchid_ai.core.agent import _EMPTY_RUN_CONTEXT, _run_ctx_var
from orchid_ai.core.state import OrchidAuthContext
from orchid_ai.core.repository import OrchidVectorReader, OrchidVectorWriter, OrchidVectorStoreAdmin
from orchid_ai.core.mcp import OrchidMCPClient, OrchidMCPToolResult
from orchid_ai.rag.scopes import OrchidRAGScope
from orchid_ai.rag.backends.null import NullVectorReader
from orchid_ai.persistence.models import OrchidChatSession, OrchidChatMessage


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Snapshot and restore ``os.environ`` around every test.

    Orchid's ``from_config_path`` and ``apply_yaml_to_env`` mutate
    ``os.environ`` to flatten YAML values for downstream
    pydantic-settings consumers.  Without this fixture, tests that
    trigger config loading leak env vars into subsequent tests,
    causing order-dependent failures.

    Uses ``monkeypatch.setenv`` / ``monkeypatch.delenv`` semantics:
    pytest automatically restores the original env on teardown.
    """
    # monkeypatch already snapshots env on first setenv/delenv call,
    # but we need to ensure ALL mutations (including direct
    # os.environ[k] = v) are reverted.  Capture the full snapshot
    # and restore it in the finalizer.
    original_env = dict(os.environ)
    yield
    # Restore: remove keys that were added, restore keys that changed
    current_keys = set(os.environ.keys())
    original_keys = set(original_env.keys())
    for key in current_keys - original_keys:
        del os.environ[key]
    for key, value in original_env.items():
        if os.environ.get(key) != value:
            os.environ[key] = value


@pytest.fixture(autouse=True)
def _reset_agent_context_vars():
    """Reset per-request ContextVars between tests.

    The H4 fix moved per-request agent state (auth, chat_id,
    message_id, correlation_id) and skill recursion depth onto
    asyncio-task-local ContextVars.  pytest-asyncio's task model
    does not always isolate ContextVar bindings between tests in
    the same module, so a test that sets ``agent._current_chat_id
    = "C-7"`` would leak that binding into a sibling test that
    relies on the default ``None``.  Snapshotting around every
    test restores isolation regardless of asyncio_mode.
    """
    ctx_token = _run_ctx_var.set(_EMPTY_RUN_CONTEXT)
    depth_token = _skill_depth_var.set(0)
    try:
        yield
    finally:
        _run_ctx_var.reset(ctx_token)
        _skill_depth_var.reset(depth_token)


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

    async def retrieve(self, query, namespace, k=5, scope=None, metadata_filters=None):
        self.calls.append(
            {
                "query": query,
                "namespace": namespace,
                "k": k,
                "scope": scope,
                "metadata_filters": metadata_filters,
            }
        )
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
        self._cache_warmed = False

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

    async def warm_cache(self, auth):
        self._cache_warmed = True

    def invalidate_cache(self):
        self._cache_warmed = False

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
