"""Tests for LangGraph checkpointer factory and graph integration (#6)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from orchid_ai.config.schema import (
    AgentConfig,
    AgentsConfig,
    DefaultsConfig,
    LLMConfig,
    RAGConfig,
)
from orchid_ai.runtime import OrchidRuntime


# ── Factory tests ───────────────────────────────────────────


class TestBuildCheckpointerMemory:
    """build_checkpointer('memory') returns MemorySaver."""

    @pytest.mark.asyncio
    async def test_memory_type(self):
        from orchid_ai.checkpointing import build_checkpointer

        saver = await build_checkpointer("memory")
        assert isinstance(saver, MemorySaver)

    @pytest.mark.asyncio
    async def test_memory_ignores_dsn(self):
        from orchid_ai.checkpointing import build_checkpointer

        saver = await build_checkpointer("memory", dsn="ignored.db")
        assert isinstance(saver, MemorySaver)


class TestBuildCheckpointerSQLite:
    """build_checkpointer('sqlite') requires langgraph-checkpoint-sqlite."""

    @pytest.mark.asyncio
    async def test_sqlite_missing_package(self):
        from orchid_ai.checkpointing import build_checkpointer

        with patch.dict("sys.modules", {"langgraph.checkpoint.sqlite": None, "langgraph.checkpoint.sqlite.aio": None}):
            with pytest.raises(ImportError, match="langgraph-checkpoint-sqlite"):
                await build_checkpointer("sqlite", dsn="test.db")

    @pytest.mark.asyncio
    async def test_sqlite_missing_dsn(self):
        from orchid_ai.checkpointing import build_checkpointer

        with pytest.raises(ValueError, match="DSN"):
            await build_checkpointer("sqlite", dsn="")


class TestBuildCheckpointerPostgres:
    """build_checkpointer('postgres') requires langgraph-checkpoint-postgres."""

    @pytest.mark.asyncio
    async def test_postgres_missing_package(self):
        from orchid_ai.checkpointing import build_checkpointer

        with patch.dict(
            "sys.modules",
            {"langgraph.checkpoint.postgres": None, "langgraph.checkpoint.postgres.aio": None},
        ):
            with pytest.raises(ImportError, match="langgraph-checkpoint-postgres"):
                await build_checkpointer("postgres", dsn="postgresql://localhost/test")

    @pytest.mark.asyncio
    async def test_postgres_missing_dsn(self):
        from orchid_ai.checkpointing import build_checkpointer

        with pytest.raises(ValueError, match="DSN"):
            await build_checkpointer("postgres", dsn="")


class TestBuildCheckpointerCustom:
    """build_checkpointer() with dotted class path."""

    @pytest.mark.asyncio
    async def test_invalid_path_raises_import_error(self):
        from orchid_ai.checkpointing import build_checkpointer

        with pytest.raises(ImportError, match="Cannot resolve"):
            await build_checkpointer("nonexistent.module.Foo")

    @pytest.mark.asyncio
    async def test_non_subclass_raises_type_error(self):
        from orchid_ai.checkpointing import build_checkpointer

        # str is not a BaseCheckpointSaver subclass
        with pytest.raises(TypeError, match="BaseCheckpointSaver"):
            await build_checkpointer("builtins.str")


# ── Shutdown tests ──────────────────────────────────────────


class TestShutdownCheckpointer:
    """shutdown_checkpointer() handles various cleanup patterns."""

    @pytest.mark.asyncio
    async def test_none_is_noop(self):
        from orchid_ai.checkpointing import shutdown_checkpointer

        await shutdown_checkpointer(None)  # should not raise

    @pytest.mark.asyncio
    async def test_memory_saver_shutdown(self):
        from orchid_ai.checkpointing import shutdown_checkpointer

        saver = MemorySaver()
        await shutdown_checkpointer(saver)  # should not raise

    @pytest.mark.asyncio
    async def test_saver_with_aclose(self):
        from orchid_ai.checkpointing import shutdown_checkpointer

        saver = MagicMock(spec=BaseCheckpointSaver)
        saver.aclose = AsyncMock()

        await shutdown_checkpointer(saver)
        saver.aclose.assert_called_once()


# ── Runtime field tests ─────────────────────────────────────


class TestRuntimeCheckpointerField:
    """OrchidRuntime.checkpointer field."""

    def test_defaults_to_none(self):
        runtime = OrchidRuntime()
        assert runtime.checkpointer is None

    def test_accepts_memory_saver(self):
        saver = MemorySaver()
        runtime = OrchidRuntime(checkpointer=saver)
        assert runtime.checkpointer is saver


# ── Graph compilation tests ─────────────────────────────────


class TestGraphCompileWithCheckpointer:
    """build_graph() passes checkpointer to compile()."""

    def test_with_checkpointer(self):
        from orchid_ai.graph.graph import build_graph

        saver = MemorySaver()
        config = AgentsConfig(
            defaults=DefaultsConfig(llm=LLMConfig(model="ollama/llama3.2")),
            agents={
                "test": AgentConfig(
                    description="test",
                    prompt="test",
                    rag=RAGConfig(enabled=False),
                ),
            },
        )
        runtime = OrchidRuntime(default_model="ollama/llama3.2", checkpointer=saver)
        graph = build_graph(config=config, runtime=runtime)
        assert graph is not None
        # The compiled graph should have a checkpointer
        assert graph.checkpointer is saver

    def test_without_checkpointer(self):
        from orchid_ai.graph.graph import build_graph

        config = AgentsConfig(
            agents={
                "test": AgentConfig(
                    description="test",
                    prompt="test",
                    rag=RAGConfig(enabled=False),
                    llm=LLMConfig(model="ollama/llama3.2"),
                ),
            },
        )
        runtime = OrchidRuntime(default_model="ollama/llama3.2")
        graph = build_graph(config=config, runtime=runtime)
        assert graph is not None
        assert graph.checkpointer is None


# ── SDK import tests ────────────────────────────────────────


class TestSDKImports:
    """build_checkpointer is importable from the SDK surface."""

    def test_import_from_orchid_ai(self):
        from orchid_ai import build_checkpointer, shutdown_checkpointer

        assert callable(build_checkpointer)
        assert callable(shutdown_checkpointer)

    def test_import_from_checkpointing(self):
        from orchid_ai.checkpointing import build_checkpointer, shutdown_checkpointer

        assert callable(build_checkpointer)
        assert callable(shutdown_checkpointer)
