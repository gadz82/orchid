# Tests — AI Agent Instructions

## Overview

Unit and integration tests for the `orchid-ai` library. Tests live alongside the source in the same repository and run via `pytest`.

## Running

```bash
cd orchid && source .venv/bin/activate
pytest                          # all tests (with coverage)
pytest tests/ -x                # stop on first failure
pytest tests/ -k "test_scopes"  # specific pattern
```

## Layout

```
tests/
├── conftest.py          Shared fixtures (MockMCPClient, MockVectorReader, auth, env isolation)
├── mcp/                 MCP client, dispatcher, discovery, cache tests
├── events/              Pollen + Bloom event subsystem tests
├── core/                Core ABC conformance tests
├── tools/               Built-in tool registry tests
├── test_*.py            Flat test files (to be reorganised into subdirs)
```

## Conventions

1. **Every code change must include tests.** New features: happy path + edge cases. Bug fixes: regression test first.
2. **Use `pytest.mark.asyncio`** for async tests. `asyncio_mode = "auto"` is set in `pyproject.toml`.
3. **Use fixtures from `conftest.py`** — `auth`, `mock_reader`, `mock_mcp`, `null_reader`, etc.
4. **Environment isolation is automatic.** The `_isolate_env` autouse fixture snapshots/restores `os.environ` around every test. The `_reset_agent_context_vars` fixture resets ContextVars.
5. **Don't reference files outside `orchid/`.** Tests that need example configs should use `tests/fixtures/` or skip with `pytest.skip()` when the workspace isn't available.
6. **Mock external services.** Never make real network calls. Use `MockMCPClient`, `MockVectorReader`, or `unittest.mock.patch`.
7. **Coverage minimum is 79%.** Enforced via `--cov-fail-under=79` in `pyproject.toml`.

## Key Fixtures (conftest.py)

| Fixture | Type | Purpose |
|---------|------|---------|
| `auth` | `OrchidAuthContext` | Standard test auth (tenant=t-123, user=u-456) |
| `expired_auth` | `OrchidAuthContext` | Expired token for auth tests |
| `scope` | `OrchidRAGScope` | Standard RAG scope |
| `mock_reader` | `MockVectorReader` | Records retrieve() calls |
| `mock_mcp` | `MockMCPClient` | Records tool calls, supports warm_cache/invalidate_cache |
| `null_reader` | `NullVectorReader` | Returns empty results |

## Adding Tests

- Place new test files as `tests/test_<module>.py`
- For MCP-related tests, use `tests/mcp/`
- For events-related tests, use `tests/events/`
- Import the module under test directly: `from orchid_ai.agents.strategies import CallAllStrategy`
