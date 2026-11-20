"""Target dispatcher for Bloom job runs.

One :class:`JobRun` carries one :class:`JobSpec`; the spec's ``target``
discriminator decides which concrete runner handles it.  The dispatcher keeps
that rule in one place so the processor never grows a target lookup, and an
unknown target fails terminally instead of silently falling back to an agent.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Callable, Mapping
from typing import Any

from ...core.events.errors import JobRunnerError
from ...core.events.job import JobRun, JobStatus
from ...core.events.runner import OrchidJobRunner


# ORCHID-MEM: 01a1f342-64db-7cda-b4a7-c08b81be79a0 — Bloom target dispatch; the job spec target selects the runner and an unknown target fails without retry.
class DispatchJobRunner(OrchidJobRunner):
    """Route a run to the runner registered for ``run.spec.target``."""

    def __init__(
        self,
        *,
        runners: Mapping[str, OrchidJobRunner],
        clock: Callable[[], _dt.datetime] | None = None,
    ) -> None:
        if not runners:
            raise ValueError("DispatchJobRunner requires at least one target runner")
        self._runners = dict(runners)
        self._clock = clock or (lambda: _dt.datetime.now(tz=_dt.UTC))

    async def run(self, run: JobRun, *, auth: Any) -> None:
        target = run.spec.target
        runner = self._runners.get(target)
        if runner is None:
            message = f"unknown job target {target!r} for run {run.run_id} (known targets: {sorted(self._runners)})"
            run.status = JobStatus.FAILED
            run.error = message
            run.finished_at = self._clock()
            raise JobRunnerError(message, retryable=False)
        await runner.run(run, auth=auth)
