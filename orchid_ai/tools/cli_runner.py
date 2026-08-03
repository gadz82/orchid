from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class CLIResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class CLIRunner(ABC):
    """Executes an external CLI command. Injected into ExternalAgentCLITool."""

    @abstractmethod
    async def run(
        self,
        *,
        command: list[str],
        prompt: str,
        cwd: str | None,
        timeout: float,
        env: dict[str, str] | None,
        stdin_mode: str,
    ) -> CLIResult: ...


class AsyncSubprocessCLIRunner(CLIRunner):
    """Default runner: asyncio.create_subprocess_exec (no shell=True)."""

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
        argv = list(command)
        if stdin_mode == "stdin":
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd or None,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=_merge_env(env),
            )
            data = prompt.encode()
        else:
            if prompt:
                argv = [*argv, prompt]
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd or None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=_merge_env(env),
            )
            data = None
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(input=data), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return CLIResult(exit_code=-1, stdout="", stderr="", timed_out=True)
        return CLIResult(
            proc.returncode if proc.returncode is not None else 0,
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
        )


def _merge_env(env: dict[str, str] | None) -> dict[str, str] | None:
    if env is None:
        return None
    merged = dict(os.environ)
    merged.update(env)
    return merged
