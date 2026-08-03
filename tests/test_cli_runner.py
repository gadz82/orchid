from __future__ import annotations

import pytest

from orchid_ai.tools.cli_runner import AsyncSubprocessCLIRunner


@pytest.fixture
def runner() -> AsyncSubprocessCLIRunner:
    return AsyncSubprocessCLIRunner()


class TestAsyncSubprocessCLIRunner:
    @pytest.mark.asyncio
    async def test_arg_mode_echo(self, runner: AsyncSubprocessCLIRunner) -> None:
        result = await runner.run(
            command=["/bin/echo", "hello"],
            prompt="",
            cwd=None,
            timeout=10,
            env=None,
            stdin_mode="arg",
        )
        assert result.exit_code == 0
        assert result.stdout.strip() == "hello"

    @pytest.mark.asyncio
    async def test_stdin_mode_cat(self, runner: AsyncSubprocessCLIRunner) -> None:
        result = await runner.run(
            command=["/bin/cat"],
            prompt="hello from stdin",
            cwd=None,
            timeout=10,
            env=None,
            stdin_mode="stdin",
        )
        assert result.exit_code == 0
        assert "hello from stdin" in result.stdout

    @pytest.mark.asyncio
    async def test_timeout_kills_process(self, runner: AsyncSubprocessCLIRunner) -> None:
        result = await runner.run(
            command=["/bin/sleep", "10"],
            prompt="",
            cwd=None,
            timeout=0.1,
            env=None,
            stdin_mode="arg",
        )
        assert result.timed_out is True
        assert result.exit_code == -1

    @pytest.mark.asyncio
    async def test_stderr_captured(self, runner: AsyncSubprocessCLIRunner) -> None:
        result = await runner.run(
            command=["/bin/sh", "-c", "echo ok && echo err >&2"],
            prompt="",
            cwd=None,
            timeout=10,
            env=None,
            stdin_mode="arg",
        )
        assert result.exit_code == 0
        assert "ok" in result.stdout
        assert "err" in result.stderr

    @pytest.mark.asyncio
    async def test_env_merged(self) -> None:
        custom_runner = AsyncSubprocessCLIRunner()
        result = await custom_runner.run(
            command=["/bin/sh", "-c", "echo $CUSTOM_VAR"],
            prompt="",
            cwd=None,
            timeout=10,
            env={"CUSTOM_VAR": "custom_value"},
            stdin_mode="arg",
        )
        assert result.exit_code == 0
        assert "custom_value" in result.stdout

    @pytest.mark.asyncio
    async def test_nonzero_exit_code(self, runner: AsyncSubprocessCLIRunner) -> None:
        result = await runner.run(
            command=["/bin/sh", "-c", "exit 42"],
            prompt="",
            cwd=None,
            timeout=10,
            env=None,
            stdin_mode="arg",
        )
        assert result.exit_code == 42
