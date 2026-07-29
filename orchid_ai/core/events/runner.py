"""ABC for the component that actually executes a :class:`JobRun`.

The processor pulls a signal, matches a trigger, builds a
:class:`JobSpec`, instantiates a :class:`JobRun`, and hands the pair
to a runner.  The runner mutates the run in place — this ABC's only
contract is that every code path leaves the run in a *terminal* status
(``SUCCEEDED``, ``FAILED``, ``RETRY_SCHEDULED``, ``CANCELLED``) before
it returns.

Concrete runners:

- ``orchid_ai.events.runners.graph_runner.GraphJobRunner`` — invokes
  the existing LangGraph supervisor under the synthesised auth.
- Integrators can wire alternative runners (e.g. a thin runner that
  dispatches to a separate batch system) without touching the rest of
  the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .job import JobRun


class OrchidJobRunner(ABC):
    """Executes one attempt of a :class:`JobRun`."""

    @abstractmethod
    async def run(self, run: JobRun, *, auth: Any) -> None:
        """Execute the job under ``auth``.  Mutates ``run`` in place.

        On success: ``run.status = SUCCEEDED`` and ``run.result`` set.
        On retryable failure: ``run.status = RETRY_SCHEDULED`` and
        ``run.next_retry_at`` set.
        On terminal failure: ``run.status = FAILED`` and ``run.error``
        set.

        The method should not raise — failures are reported through the
        run object so the processor can persist the row regardless.
        ``auth`` is typed as ``Any`` here to keep ``core/events/`` from
        importing :class:`OrchidAuthContext` directly; concrete runners
        narrow the type at the call boundary.
        """
