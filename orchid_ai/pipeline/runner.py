"""Direct pipeline runner: session, execution config, invocation and result.

The runner is the thin orchestration layer above the graph. It validates the
caller's input, creates the run session, builds the ``RunnableConfig`` (auth,
run name, tags, metadata, recursion limit, concurrency, session), invokes the
compiled graph, and turns the final state plus the session into one
:class:`OrchidPipelineResult`.

What the runner deliberately does **not** do:

* touch chat storage — pipeline runs never create a chat or persist a message;
* pass a checkpointer — pipeline runs are not resumable;
* put the information graph or the session in graph state — both travel in
  ``RunnableConfig`` only.

Result assembly projects the declared output shape (or returns the graph
snapshot when no output block is declared), validates it, and asks
:class:`OrchidPipelineInterpreter` for the configured summary. Job metrics and
graph statistics are always present; the full graph is opt-in and size-capped.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from ..core.pipeline import (
    OrchidPipelineAuthError,
    OrchidPipelineError,
    OrchidPipelineEvent,
    OrchidPipelineInputError,
    OrchidPipelineLimitError,
    OrchidPipelineObserver,
    OrchidPipelineOutputError,
    OrchidPipelineResult,
    OrchidPipelineRunContext,
    OrchidPipelineTimeoutError,
)
from ..core.run_config import with_auth
from ..core.state import OrchidAuthContext
from .builder import PipelineExecution, build_pipeline_graph, final_status
from .definition import CompiledPipeline, PipelineDependencies
from .interpretation import OrchidPipelineInterpreter
from .mapping import evaluate, references_context_key, render_value
from .session import PipelineRunSession, _redact

__all__ = ["OrchidPipelineRunner"]

_MAX_ERROR_CHARS = 2000
_LOG = logging.getLogger(__name__)


# ORCHID-MEM: 01a1ed8b-2435-71bb-bf71-bfd64fa67712 — Direct pipeline runner; validates input, builds the run config and session, invokes the graph and assembles the typed result.
class OrchidPipelineRunner:
    """Run one compiled pipeline directly (no HTTP, no chat, no scheduler)."""

    def __init__(
        self,
        pipeline: CompiledPipeline,
        *,
        deps: PipelineDependencies | None = None,
        execution: PipelineExecution | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if not isinstance(pipeline, CompiledPipeline):
            raise TypeError("OrchidPipelineRunner requires a CompiledPipeline")
        self._pipeline = pipeline
        self._deps = deps
        self._execution = execution or PipelineExecution()
        self._logger = logger or logging.getLogger(__name__)

    @property
    def pipeline(self) -> CompiledPipeline:
        return self._pipeline

    async def run(
        self,
        input: Mapping[str, Any],
        *,
        auth: OrchidAuthContext | None = None,
        user_id: str = "",
        tenant_id: str = "default",
        access_token: str = "",
        chat_id: str | None = None,
        correlation_id: str | None = None,
        seed: Sequence[Mapping[str, Any]] = (),
        mode: Literal["direct", "chat"] = "direct",
        timeout: float | None = None,
        include_graph: bool = False,
        interpretation: Literal["none", "template", "llm", "caller"] | None = None,
        observer: OrchidPipelineObserver | None = None,
        run_id: str | None = None,
    ) -> OrchidPipelineResult:
        """Run this runner's pipeline to completion and return its typed result."""
        return await self._run_pipeline(
            self._pipeline,
            input,
            auth=auth,
            user_id=user_id,
            tenant_id=tenant_id,
            access_token=access_token,
            chat_id=chat_id,
            correlation_id=correlation_id,
            seed=seed,
            mode=mode,
            timeout=timeout,
            include_graph=include_graph,
            interpretation=interpretation,
            observer=observer,
            run_id=run_id,
        )

    # ORCHID-MEM: 01a1f2fe-ccea-729f-ba18-d09d5db99d65 — Child runner entry; reuses the instance, deps and execution, with nested depth and inherited deadline.
    async def run_pipeline(
        self,
        pipeline: CompiledPipeline,
        input: Mapping[str, Any],
        **kwargs: Any,
    ) -> OrchidPipelineResult:
        """Run another compiled pipeline with the same runner instance.

        Sub-pipeline steps use this entry point: it reuses the parent's
        dependencies, execution collaborators and child-runner wiring instead
        of constructing a second orchestration path. Child events are not
        forwarded to the parent observer.
        """
        if not isinstance(pipeline, CompiledPipeline):
            raise TypeError("run_pipeline requires a CompiledPipeline")
        return await self._run_pipeline(pipeline, input, **kwargs)

    async def _run_pipeline(
        self,
        pipeline: CompiledPipeline,
        input: Mapping[str, Any],
        *,
        auth: OrchidAuthContext | None = None,
        user_id: str = "",
        tenant_id: str = "default",
        access_token: str = "",
        chat_id: str | None = None,
        correlation_id: str | None = None,
        seed: Sequence[Mapping[str, Any]] = (),
        mode: Literal["direct", "chat"] = "direct",
        timeout: float | None = None,
        include_graph: bool = False,
        interpretation: Literal["none", "template", "llm", "caller"] | None = None,
        observer: OrchidPipelineObserver | None = None,
        run_id: str | None = None,
        parent_run_id: str | None = None,
        subpipeline_depth: int = 0,
        subpipeline_max_depth: int | None = None,
        usage_out: dict[str, int] | None = None,
    ) -> OrchidPipelineResult:
        """Run ``pipeline`` on this runner instance."""
        effective_max_depth = (
            subpipeline_max_depth if subpipeline_max_depth is not None else pipeline.limits.max_subpipeline_depth
        )
        if subpipeline_depth >= effective_max_depth:
            raise OrchidPipelineLimitError(
                f"pipeline {pipeline.name!r} sub-pipeline depth {subpipeline_depth} exceeds "
                f"max_subpipeline_depth={effective_max_depth}"
            )
        auth_context = self._resolve_auth(
            pipeline,
            auth=auth,
            user_id=user_id,
            tenant_id=tenant_id,
            access_token=access_token,
        )
        validated = self._validate_input(pipeline, input)
        run = self._build_run(
            pipeline,
            auth_context,
            chat_id=chat_id,
            correlation_id=correlation_id,
            mode=mode,
            timeout=timeout,
            run_id=run_id,
            parent_run_id=parent_run_id,
            subpipeline_depth=subpipeline_depth,
            subpipeline_max_depth=effective_max_depth,
        )
        session = PipelineRunSession(
            definition=pipeline,
            run=run,
            input=validated,
            observer=observer,
            logger=self._logger,
            usage_out=usage_out,
        )
        execution = self._execution
        if execution.child_runner is None:
            execution = replace(execution, child_runner=self.run_pipeline)
        graph = build_pipeline_graph(pipeline, deps=self._deps, execution=execution)
        state = await self._invoke(pipeline, graph, session, auth_context, run, seed)
        result = await self._assemble(
            pipeline,
            session,
            state,
            include_graph=include_graph,
            interpretation=self._interpretation_for(pipeline, interpretation, mode),
        )
        await session.emit(
            "pipeline.finished",
            status=result.status,
            duration_ms=result.duration_ms,
            summary=result.summary,
            graph_stats=dict(result.graph_stats),
            result=result.to_dict(include_graph=False),
        )
        return result

    async def stream(
        self,
        input: Mapping[str, Any],
        **kwargs: Any,
    ) -> AsyncIterator[OrchidPipelineEvent]:
        """Run the pipeline and yield its events; the last one carries the result.

        The run happens in a background task so the caller can consume events
        while jobs execute. If the run fails before it can emit
        ``pipeline.finished``, the exception surfaces after the delivered events.
        """
        queue_observer = _QueueObserver()
        caller_observer = kwargs.pop("observer", None)
        observer = _FanoutObserver(queue_observer, caller_observer)
        task = asyncio.create_task(self.run(input, observer=observer, **kwargs))
        try:
            while True:
                getter: asyncio.Future[OrchidPipelineEvent] = asyncio.ensure_future(queue_observer.queue.get())
                done, _ = await asyncio.wait({getter, task}, return_when=asyncio.FIRST_COMPLETED)
                if getter in done:
                    event = getter.result()
                    yield event
                    if event.type == "pipeline.finished":
                        break
                    continue
                getter.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await getter
                # The runner exited without a finished event: surface the failure.
                await task
                break
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    # ── Internals ────────────────────────────────────────────

    def _resolve_auth(
        self,
        pipeline: CompiledPipeline,
        *,
        auth: OrchidAuthContext | None,
        user_id: str,
        tenant_id: str,
        access_token: str,
    ) -> OrchidAuthContext:
        if auth is not None:
            return auth
        if not user_id and not tenant_id and not access_token:
            raise OrchidPipelineAuthError(
                f"pipeline {pipeline.name!r} requires an authenticated caller; "
                "pass an OrchidAuthContext or user_id/tenant_id"
            )
        return OrchidAuthContext(
            access_token=access_token,
            tenant_key=tenant_id or "default",
            user_id=user_id,
        )

    def _validate_input(self, pipeline: CompiledPipeline, input: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(input, Mapping):
            raise OrchidPipelineInputError("pipeline input must be a mapping")
        payload = dict(input)
        input_config = pipeline.definition.input
        if input_config is not None:
            for key, value in input_config.defaults.items():
                payload.setdefault(key, value)
        shape = pipeline.input_shape
        if shape is None:
            return payload
        try:
            return shape.dump(payload, shape_version=shape.version)
        except Exception as exc:
            raise OrchidPipelineInputError(f"pipeline {pipeline.name!r} input is invalid: {exc}") from exc

    def _build_run(
        self,
        pipeline: CompiledPipeline,
        auth: OrchidAuthContext,
        *,
        chat_id: str | None,
        correlation_id: str | None,
        mode: Literal["direct", "chat"],
        timeout: float | None,
        run_id: str | None,
        parent_run_id: str | None = None,
        subpipeline_depth: int = 0,
        subpipeline_max_depth: int | None = None,
    ) -> OrchidPipelineRunContext:
        started_at = datetime.now(UTC)
        effective_timeout = timeout if timeout is not None else pipeline.limits.timeout_seconds
        if timeout is not None and pipeline.limits.timeout_seconds is not None:
            effective_timeout = min(timeout, pipeline.limits.timeout_seconds)
        deadline = started_at + timedelta(seconds=effective_timeout) if effective_timeout else None
        return OrchidPipelineRunContext(
            run_id=run_id or str(uuid.uuid4()),
            pipeline=pipeline.name,
            pipeline_version=pipeline.version,
            mode="chat" if mode == "chat" else "direct",
            tenant_key=auth.tenant_key,
            user_id=auth.user_id,
            started_at=started_at,
            chat_id=chat_id if mode == "chat" else None,
            correlation_id=correlation_id,
            parent_run_id=parent_run_id,
            deadline=deadline,
            subpipeline_depth=subpipeline_depth,
            subpipeline_max_depth=(
                subpipeline_max_depth if subpipeline_max_depth is not None else pipeline.limits.max_subpipeline_depth
            ),
        )

    async def _invoke(
        self,
        pipeline: CompiledPipeline,
        graph: Any,
        session: PipelineRunSession,
        auth: OrchidAuthContext,
        run: OrchidPipelineRunContext,
        seed: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        config = with_auth(
            auth,
            thread_id=run.run_id,
            base={
                "run_name": f"pipeline:{pipeline.name}",
                "tags": ["orchid", "pipeline", pipeline.name],
                "metadata": {
                    "pipeline": pipeline.name,
                    "run_id": run.run_id,
                    "tenant_key": auth.tenant_key,
                },
                "recursion_limit": max(25, 4 * len(pipeline.jobs) + 10),
                "max_concurrency": pipeline.limits.max_parallel,
                "configurable": {"pipeline_session": session},
            },
        )
        seed_items: list[dict[str, Any]] = []
        for index, item in enumerate(seed):
            if not isinstance(item, Mapping):
                raise OrchidPipelineInputError(f"seed[{index}] must be a mapping")
            seed_items.append(dict(item))
        state = {
            "run": run.to_dict(),
            "input": dict(session.input),
            "seed": seed_items,
            "job_results": {},
            "errors": [],
        }
        try:
            return await graph.ainvoke(state, config=config)
        except (OrchidPipelineInputError, OrchidPipelineAuthError):
            # Caller errors are not run outcomes: surface them to the caller.
            raise
        except asyncio.CancelledError:
            session.cancel()
            raise
        except Exception as exc:
            # The exception may wrap a transport error whose message carries
            # request headers; never store, log or emit it raw.
            message = _redact(str(exc))[:_MAX_ERROR_CHARS] or "pipeline graph failed"
            self._logger.error("[Pipeline] run %s failed: %s", run.run_id, message)
            return {
                "status": "failed",
                "job_results": {},
                "errors": [],
                "output": None,
                "summary": "",
                "error": message,
            }

    def _interpretation_for(
        self,
        pipeline: CompiledPipeline,
        requested: Literal["none", "template", "llm", "caller"] | None,
        mode: Literal["direct", "chat"],
    ) -> Literal["none", "template", "llm", "caller"] | None:
        """Resolve the run's interpretation mode.

        An explicit caller request wins, then the definition's exposed-tool
        override. A chat-scoped run defaults to the deterministic caller digest;
        a direct run uses the definition's configured mode.
        """
        if requested is not None:
            return requested
        if mode != "chat":
            return None
        exposed = pipeline.definition.expose.interpretation
        if exposed is not None:
            return exposed
        return "caller"

    # ORCHID-MEM: 01a1ef3d-7a54-7fc0-b74f-7135141f2d0a — Result assembly; output projection, interpretation, job metrics and opt-in graph.
    async def _assemble(
        self,
        pipeline: CompiledPipeline,
        session: PipelineRunSession,
        state: Mapping[str, Any],
        *,
        include_graph: bool,
        interpretation: Literal["none", "template", "llm", "caller"] | None = None,
    ) -> OrchidPipelineResult:
        if state.get("__interrupt__"):
            status = "failed"
        elif state.get("status") == "failed":
            # The graph failed outside the job nodes (or a step interrupted);
            # the session may still hold successful jobs, but the run did not
            # complete.
            status = "failed"
        else:
            # A decision that ends on END never reaches the finish node, so the
            # deadline is re-checked here for every run.
            try:
                session.check_deadline()
            except OrchidPipelineTimeoutError:
                session.cancel()
            status = final_status(session)
        error = _result_error(session, state)
        metadata: dict[str, Any] = {}
        mode = interpretation or pipeline.definition.interpretation.mode
        interpreter = OrchidPipelineInterpreter(pipeline, session, logger=self._logger)
        graph_stats = session.stats()

        # A deterministic summary can be computed before projection, so an
        # output mapping field may read `summary` (spec §7.12.4). Template and
        # LLM summaries depend on the projected output, so they are produced
        # after it and a `summary` mapping reference is rejected at build time.
        pre_summary = ""
        if mode in ("none", "caller"):
            pre_summary = (
                await interpreter.interpret(
                    output=None,
                    status=status,
                    graph_stats=graph_stats,
                    mode=mode,
                    error=error,
                )
            ).text

        output: dict[str, Any] | None = None
        if status in ("failed", "cancelled"):
            # A failed/cancelled run must not expose a projection built from
            # partial state, and a projection failure must not hide the
            # original run error.
            output = None
        else:
            try:
                output = _project_output(
                    pipeline,
                    session,
                    summary=pre_summary,
                    mode=mode,
                )
            except OrchidPipelineOutputError as exc:
                status = "failed"
                error = str(exc)
                output = None
                metadata["error_code"] = exc.code

        graph: dict[str, Any] = {}
        if include_graph:
            snapshot = session.snapshot()
            payload_size = len(json.dumps(snapshot, ensure_ascii=False, default=str))
            if payload_size <= session.limits.max_graph_bytes:
                graph = snapshot
            else:
                metadata["graph_omitted"] = "size"

        interpretation_output = output
        if mode == "llm" and pipeline.definition.output is None:
            # The graph snapshot is the result output for a pipeline without an
            # output block, but it must never become the LLM prompt.
            interpretation_output = None
        summary = await interpreter.interpret(
            output=interpretation_output,
            status=status,
            graph_stats=graph_stats,
            mode=interpretation,
            error=error,
        )
        for key, value in summary.metadata.items():
            metadata.setdefault(key, value)
        # The interpretation call (if any) is part of the run cost.
        graph_stats = session.stats()
        return OrchidPipelineResult(
            run_id=session.run.run_id,
            pipeline=pipeline.name,
            status=status,
            output=output,
            graph=graph,
            summary=summary.text,
            summary_source=summary.source,
            jobs={job_id: result.to_dict() for job_id, result in session.job_results.items()},
            graph_stats=graph_stats,
            error=error,
            started_at=session.run.started_at,
            finished_at=datetime.now(UTC),
            metadata=metadata,
        )


def _project_output(
    pipeline: CompiledPipeline,
    session: PipelineRunSession,
    *,
    summary: str = "",
    mode: str = "none",
) -> dict[str, Any]:
    """Build the declared output, or the graph snapshot when none is declared.

    A declared shape always validates the projection; a shape or mapping failure
    becomes :class:`OrchidPipelineOutputError` so the run can fail with a stable
    output error instead of returning unvalidated data. ``summary`` is the
    already computed deterministic summary when the interpretation mode permits
    one; ``summary_available=False`` rejects a mapping reference that cannot be
    satisfied at this point.
    """
    config = pipeline.definition.output
    if config is None:
        return dict(session.snapshot())
    try:
        shape = pipeline.output_shape
        if shape is None:
            raise OrchidPipelineOutputError(f"pipeline {pipeline.name!r} output shape is not compiled")
        if mode in ("template", "llm") and _output_reads_summary(config):
            raise OrchidPipelineOutputError(
                f"pipeline {pipeline.name!r} output references 'summary', which is unavailable "
                f"for interpretation mode {mode!r}"
            )
        context = session.mapping_context(summary=summary)
        if config.from_ is not None:
            value = evaluate(config.from_, context, path="pipeline output 'from'")
            if not isinstance(value, Mapping):
                raise OrchidPipelineOutputError("pipeline output 'from' must resolve to an object")
            payload = dict(value)
        else:
            if not config.mapping:
                raise OrchidPipelineOutputError("pipeline output mapping is empty")
            payload = {
                key: _render_output_expression(expression, context, key) for key, expression in config.mapping.items()
            }
        return shape.dump(payload, shape_version=shape.version)
    except OrchidPipelineOutputError:
        raise
    except OrchidPipelineError as exc:
        detail = _redact(str(exc))[:500]
        raise OrchidPipelineOutputError(f"pipeline {pipeline.name!r} output projection failed: {detail}") from exc
    except Exception as exc:
        raise OrchidPipelineOutputError(
            f"pipeline {pipeline.name!r} output projection failed ({type(exc).__name__})"
        ) from exc


def _render_output_expression(expression: str, context: Any, field: str) -> Any:
    path = f"pipeline output field {field!r}"
    if "{{" in expression or "}}" in expression:
        return render_value(expression, context, path=path)
    return evaluate(expression, context, path=path)


def _output_reads_summary(config: Any) -> bool:
    """Whether the output mapping or ``from`` reads the top-level ``summary``."""
    if config.mapping and any(references_context_key(expression, "summary") for expression in config.mapping.values()):
        return True
    return config.from_ is not None and references_context_key(config.from_, "summary")


class _QueueObserver(OrchidPipelineObserver):
    """Collect events into a queue for :meth:`OrchidPipelineRunner.stream`."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[OrchidPipelineEvent] = asyncio.Queue()

    async def emit(self, event: OrchidPipelineEvent) -> None:
        await self.queue.put(event)


class _FanoutObserver(OrchidPipelineObserver):
    """Send each event to the stream queue and the caller's observer."""

    def __init__(self, *observers: OrchidPipelineObserver | None) -> None:
        self._observers = [observer for observer in observers if observer is not None]

    async def emit(self, event: OrchidPipelineEvent) -> None:
        for observer in self._observers:
            try:
                await observer.emit(event)
            except Exception:
                _LOG.warning("pipeline observer failed", exc_info=True)


def _result_error(session: PipelineRunSession, state: Mapping[str, Any]) -> str | None:
    if state.get("__interrupt__"):
        return "pipeline interrupts are not supported in v1"
    explicit = state.get("error")
    if isinstance(explicit, str) and explicit:
        return explicit[:_MAX_ERROR_CHARS]
    if state.get("status") == "failed" and not session.errors:
        return "pipeline run failed"
    if session.errors:
        return str(session.errors[0].get("message") or "")[:_MAX_ERROR_CHARS] or None
    if session.cancelled:
        return "pipeline run was cancelled"
    for result in session.job_results.values():
        if result.status == "failed" and result.error:
            return result.error[:_MAX_ERROR_CHARS]
    return None
