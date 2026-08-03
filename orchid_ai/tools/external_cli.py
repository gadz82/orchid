from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.tool import OrchidTool, OrchidToolInput, OrchidToolOutput
from .cli_runner import AsyncSubprocessCLIRunner, CLIRunner
from .normalizer import PassthroughNormalizer, PromptNormalizer


class ExternalAgentCLITool(OrchidTool):
    """Delegates a sub-task to an external AI-agent CLI subprocess."""

    requires_approval = True
    parallel_safe = False

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        command: list[str],
        args: Sequence[str] = (),
        cwd: str | None = None,
        timeout: float = 600.0,
        env: dict[str, str] | None = None,
        stdin_mode: str = "arg",
        normalizer: PromptNormalizer | None = None,
        runner: CLIRunner | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self._command = [*command, *args]
        self._cwd = cwd
        self._timeout = timeout
        self._env = env
        self._stdin_mode = stdin_mode
        self._normalizer = normalizer or PassthroughNormalizer()
        self._runner = runner or AsyncSubprocessCLIRunner()
        self.parameters_schema = {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The sub-task to delegate to the external agent.",
                },
            },
            "required": ["prompt"],
        }

    async def invoke(self, tool_input: OrchidToolInput) -> OrchidToolOutput:
        prompt = tool_input.parameters.get("prompt", "")
        normalised = await self._normalizer.normalize(prompt, context=tool_input.context)
        result = await self._runner.run(
            command=self._command,
            prompt=normalised,
            cwd=self._cwd,
            timeout=self._timeout,
            env=self._env,
            stdin_mode=self._stdin_mode,
        )
        if result.timed_out:
            return OrchidToolOutput(result={"error": "timeout", "timeout_s": self._timeout})
        payload: dict[str, Any] = {
            "stdout": result.stdout,
            "exit_code": result.exit_code,
        }
        if result.stderr:
            payload["stderr"] = result.stderr
        return OrchidToolOutput(result=payload)
