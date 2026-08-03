from __future__ import annotations

from typing import Any

import pytest

from orchid_ai.core.tool import OrchidToolInput
from orchid_ai.tools.cli_runner import AsyncSubprocessCLIRunner, CLIResult, CLIRunner
from orchid_ai.tools.external_cli import ExternalAgentCLITool
from orchid_ai.tools.normalizer import PromptNormalizer


class FakeCLIRunner(CLIRunner):
    def __init__(self, response: CLIResult | None = None) -> None:
        self.response = response or CLIResult(exit_code=0, stdout="fake output", stderr="")
        self.last_call: dict[str, Any] = {}

    async def run(
        self,
        *,
        command: list[str],
        prompt: str,
        cwd: str | None,
        timeout: float,
        env: dict[str, str] | None,
        stdin_mode: str,
    ) -> CLIResult:
        self.last_call = {
            "command": command,
            "prompt": prompt,
            "cwd": cwd,
            "timeout": timeout,
            "env": env,
            "stdin_mode": stdin_mode,
        }
        return self.response


class FakeNormalizer(PromptNormalizer):
    def __init__(self) -> None:
        self.last_prompt: str | None = None

    async def normalize(self, prompt: str, *, context: dict[str, Any] | None = None) -> str:
        self.last_prompt = prompt
        return f"[normalised] {prompt}"


@pytest.fixture
def tool() -> ExternalAgentCLITool:
    return ExternalAgentCLITool(
        name="ask_external",
        description="Delegate to an external agent.",
        command=["mockcli"],
        args=("--verbose",),
        timeout=30.0,
    )


class TestExternalAgentCLIToolDefaults:
    def test_requires_approval_default(self) -> None:
        tool = ExternalAgentCLITool(name="t", command=["cmd"])
        assert tool.requires_approval is True

    def test_parallel_safe_default(self) -> None:
        tool = ExternalAgentCLITool(name="t", command=["cmd"])
        assert tool.parallel_safe is False

    def test_parameters_schema_only_exposes_prompt(self) -> None:
        tool = ExternalAgentCLITool(name="t", command=["cmd"])
        schema = tool.get_parameters_schema()
        assert list(schema["properties"].keys()) == ["prompt"]
        assert schema["required"] == ["prompt"]

    def test_llm_function_schema_exposes_only_prompt(self) -> None:
        tool = ExternalAgentCLITool(name="t", command=["cmd"])
        func_schema = tool.get_llm_function_schema()
        assert func_schema["function"]["name"] == "t"
        props = func_schema["function"]["parameters"]["properties"]
        assert list(props.keys()) == ["prompt"]

    def test_name_and_description_propagated(self) -> None:
        tool = ExternalAgentCLITool(name="my_tool", description="does things", command=["cmd"])
        assert tool.name == "my_tool"
        assert tool.description == "does things"


class TestExternalAgentCLIToolInvoke:
    @pytest.mark.asyncio
    async def test_success_returns_stdout_and_exit_code(self) -> None:
        runner = FakeCLIRunner()
        tool = ExternalAgentCLITool(name="ask", command=["cmd"], runner=runner)
        output = await tool.invoke(OrchidToolInput(parameters={"prompt": "hello"}))
        assert output.result == {"stdout": "fake output", "exit_code": 0}

    @pytest.mark.asyncio
    async def test_timeout_returns_error(self) -> None:
        runner = FakeCLIRunner(CLIResult(exit_code=-1, stdout="", stderr="", timed_out=True))
        tool = ExternalAgentCLITool(name="ask", command=["cmd"], timeout=5.0, runner=runner)
        output = await tool.invoke(OrchidToolInput(parameters={"prompt": "hello"}))
        assert output.result == {"error": "timeout", "timeout_s": 5.0}

    @pytest.mark.asyncio
    async def test_stderr_included_when_present(self) -> None:
        runner = FakeCLIRunner(CLIResult(exit_code=0, stdout="ok", stderr="some warning"))
        tool = ExternalAgentCLITool(name="ask", command=["cmd"], runner=runner)
        output = await tool.invoke(OrchidToolInput(parameters={"prompt": "hello"}))
        assert output.result == {"stdout": "ok", "exit_code": 0, "stderr": "some warning"}

    @pytest.mark.asyncio
    async def test_stderr_absent_when_empty(self) -> None:
        runner = FakeCLIRunner(CLIResult(exit_code=0, stdout="ok", stderr=""))
        tool = ExternalAgentCLITool(name="ask", command=["cmd"], runner=runner)
        output = await tool.invoke(OrchidToolInput(parameters={"prompt": "hello"}))
        assert "stderr" not in output.result

    @pytest.mark.asyncio
    async def test_normalizer_applied_before_runner(self) -> None:
        normalizer = FakeNormalizer()
        runner = FakeCLIRunner()
        tool = ExternalAgentCLITool(name="ask", command=["cmd"], runner=runner, normalizer=normalizer)
        await tool.invoke(OrchidToolInput(parameters={"prompt": "hello"}))
        assert normalizer.last_prompt == "hello"
        assert runner.last_call["prompt"] == "[normalised] hello"

    @pytest.mark.asyncio
    async def test_command_and_args_merged(self) -> None:
        runner = FakeCLIRunner()
        tool = ExternalAgentCLITool(name="ask", command=["mockcli"], args=("--verbose",), runner=runner)
        await tool.invoke(OrchidToolInput(parameters={"prompt": "hello"}))
        assert runner.last_call["command"] == ["mockcli", "--verbose"]

    @pytest.mark.asyncio
    async def test_cwd_timeout_env_forwarded(self) -> None:
        runner = FakeCLIRunner()
        tool = ExternalAgentCLITool(
            name="ask",
            command=["cmd"],
            cwd="/tmp",
            timeout=60.0,
            env={"FOO": "bar"},
            stdin_mode="stdin",
            runner=runner,
        )
        await tool.invoke(OrchidToolInput(parameters={"prompt": "hi"}))
        assert runner.last_call["cwd"] == "/tmp"
        assert runner.last_call["timeout"] == 60.0
        assert runner.last_call["env"] == {"FOO": "bar"}
        assert runner.last_call["stdin_mode"] == "stdin"

    @pytest.mark.asyncio
    async def test_default_runner_is_async_subprocess(self) -> None:
        tool = ExternalAgentCLITool(name="ask", command=["/bin/echo"])
        assert tool._runner is not None
        assert isinstance(tool._runner, AsyncSubprocessCLIRunner)

    @pytest.mark.asyncio
    async def test_default_normalizer_is_passthrough(self) -> None:
        tool = ExternalAgentCLITool(name="ask", command=["cmd"])
        result = await tool._normalizer.normalize("unchanged")
        assert result == "unchanged"
