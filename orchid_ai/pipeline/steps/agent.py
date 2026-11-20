"""Agent pipeline step: one leaf agent through the shared executor.

An ``kind: agent`` job renders its prompt (and optional structured input) from
the documented JMESPath context, runs one **top-level leaf** agent through
:class:`OrchidAgentExecutor` with the pipeline auth, the run correlation id and
``persist_memory`` from the job (default ``False``), then turns the agent's last
``AIMessage`` into a shape-validated object or ``{"text": ...}``.

The graph builder supplies one executor per top-level leaf agent through
``PipelineDependencies.agent_executors``. That executor carries the same
global + per-agent guardrail chains, model router and inline mini-agent mode as
the chat graph; this step only resolves the agent, builds the state and maps the
result. When no graph-built executor is available (programmatic/test use), the
fallback executor carries the agent config and inline mini-agents but no
guardrail or routing policy. The step never creates a chat row or writes
conversation memory unless the job opts in.

Bounded metadata: MCP/RAG/skills summaries and mini-agent outcomes are
normalized to JSON-safe values and truncated
(``{"truncated": true, "preview": ...}``) instead of copying full payloads or
prompts.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphBubbleUp

from ...config.schema_pipeline import OrchidPipelineAgentJobConfig
from ...core.agent import OrchidAgent
from ...core.pipeline import (
    OrchidPipelineAuthError,
    OrchidPipelineConfigError,
    OrchidPipelineError,
    OrchidPipelineLimitError,
    OrchidPipelineShapeError,
    OrchidPipelineStep,
    OrchidPipelineStepError,
    OrchidPipelineStepResult,
    OrchidPipelineTimeoutError,
)
from ...graph.agent_executor import OrchidAgentExecutor
from ..definition import CompiledJob
from ..mapping import DEFAULT_MAX_RENDERED_CHARS, build_mapping_context, render_template, render_value

__all__ = ["AgentStep"]

_MAX_METADATA_CHARS = 2000
_MAX_METADATA_PREVIEW_CHARS = 500
_AUTH_ERROR_FRAGMENT = "Error: no auth context"


# ORCHID-MEM: 01a1eec8-8a24-723f-8e47-39a895ba786d — Agent step; top-level leaf agent via the graph-built shared executor, pipeline auth, no memory persistence by default.
class AgentStep(OrchidPipelineStep):
    """Execute a ``kind: agent`` job."""

    kind = "agent"

    def __init__(
        self,
        job: CompiledJob,
        *,
        agents: Mapping[str, OrchidAgent] | None = None,
        agent_configs: Mapping[str, Any] | None = None,
        agent_executors: Mapping[str, Any] | None = None,
        executor_factory: Callable[[OrchidAgent, Any], Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if not isinstance(job, CompiledJob):
            raise TypeError("AgentStep requires a CompiledJob")
        if not isinstance(job.config, OrchidPipelineAgentJobConfig):
            raise TypeError("AgentStep requires an OrchidPipelineAgentJobConfig")
        self._job = job
        self._config = job.config
        self._agents = dict(agents or {})
        self._agent_configs = dict(agent_configs or {})
        self._agent_executors = dict(agent_executors or {})
        self._executor_factory = executor_factory or _default_executor
        self._logger = logger or logging.getLogger(__name__)

    async def execute(self, ctx: Any) -> OrchidPipelineStepResult:
        """Render, run the leaf agent and map its answer to the job output."""
        timeout = self._effective_timeout(ctx)
        usage: dict[str, Any] = {"llm_calls": 0, "job_id": self._job.id, "kind": self.kind}
        try:
            if timeout is None:
                return await self._execute(ctx, usage)
            try:
                return await asyncio.wait_for(self._execute(ctx, usage), timeout=timeout)
            except TimeoutError as exc:
                raise OrchidPipelineTimeoutError(
                    f"agent job {self._job.id!r} timed out after {timeout:.3f} seconds"
                ) from exc
        except OrchidPipelineError as exc:
            if getattr(exc, "usage", None) is None:
                exc.usage = dict(usage)
            raise

    def _effective_timeout(self, ctx: Any) -> float | None:
        timeout = self._job.timeout_seconds
        deadline = getattr(ctx.run, "deadline", None)
        if deadline is None:
            return timeout
        remaining = (deadline - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            raise OrchidPipelineTimeoutError(f"agent job {self._job.id!r} cannot start: the run deadline has passed")
        return remaining if timeout is None else min(timeout, remaining)

    async def _execute(self, ctx: Any, usage: dict[str, Any]) -> OrchidPipelineStepResult:
        agent = self._resolve_agent(ctx)
        pipeline = getattr(ctx, "pipeline", None)
        limits = getattr(pipeline, "limits", None)
        context = getattr(ctx, "mapping_context", None)
        if context is None:
            context = build_mapping_context(input=ctx.input, run=ctx.run)
        max_prompt_chars = getattr(limits, "max_prompt_chars", DEFAULT_MAX_RENDERED_CHARS)
        max_context_chars = getattr(limits, "max_context_chars", 8000)

        prompt = render_template(
            self._config.prompt,
            context,
            path=f"job {self._job.id!r} prompt",
            max_chars=max_prompt_chars,
        )
        if self._config.input:
            structured = render_value(self._config.input, context, path=f"job {self._job.id!r} input")
            structured_text = json.dumps(structured, ensure_ascii=False, default=str)
            if len(structured_text) > max_context_chars:
                raise OrchidPipelineLimitError(
                    f"agent job {self._job.id!r} structured input is {len(structured_text)} characters, "
                    f"exceeding max_context_chars={max_context_chars}"
                )
            block = "Structured input (untrusted JSON data): " + structured_text
            budget = max_prompt_chars - len(prompt) - 2
            if len(block) > budget:
                raise OrchidPipelineLimitError(
                    f"agent job {self._job.id!r} prompt and structured input exceed max_prompt_chars={max_prompt_chars}"
                )
            prompt = f"{prompt}\n\n{block}"

        chat_id = ctx.run.chat_id if ctx.run.mode == "chat" else None
        state: dict[str, Any] = {
            "messages": [HumanMessage(content=prompt)],
            "chat_id": chat_id or "",
            "mcp_context": {},
            "rag_context": {},
        }
        trace: dict[str, Any] = {}
        # The shared executor does not expose provider round counts. One call
        # is the lower bound for an agent activation; the fork path replaces it
        # with decomposer + mini + aggregator activations below.
        usage["llm_calls"] = 1
        try:
            executor = self._agent_executors.get(self._config.agent)
            if executor is None:
                executor = self._executor_factory(agent, self._agent_configs.get(self._config.agent))
            run_kwargs: dict[str, Any] = {
                "auth": ctx.auth,
                "chat_id": chat_id,
                "message_id": None,
                "correlation_id": ctx.run.run_id,
                "persist_memory": self._config.persist_memory,
            }
            if _accepts_trace(executor):
                run_kwargs["trace_out"] = trace
            result = await executor.run(state, **run_kwargs)
        except GraphBubbleUp:
            raise
        except OrchidPipelineError:
            raise
        except Exception as exc:
            self._logger.error(
                "[pipeline:%s] agent %r failed (%s)", self._job.id, self._config.agent, type(exc).__name__
            )
            raise OrchidPipelineStepError(
                f"agent job {self._job.id!r} failed while running agent {self._config.agent!r} ({type(exc).__name__})",
                job_id=self._job.id,
                kind=self.kind,
            ) from exc

        if not isinstance(result, Mapping):
            raise OrchidPipelineStepError(
                f"agent job {self._job.id!r} agent {self._config.agent!r} returned an invalid result",
                job_id=self._job.id,
                kind=self.kind,
            )
        if result.get("error"):
            # The executor stores the raw exception text; never surface it.
            raise OrchidPipelineStepError(
                f"agent job {self._job.id!r} agent {self._config.agent!r} failed",
                job_id=self._job.id,
                kind=self.kind,
            )
        try:
            text = _last_ai_text(result)
        except Exception as exc:
            raise OrchidPipelineStepError(
                f"agent job {self._job.id!r} agent {self._config.agent!r} returned an invalid result",
                job_id=self._job.id,
                kind=self.kind,
            ) from exc
        if text is None:
            raise OrchidPipelineStepError(
                f"agent job {self._job.id!r} agent {self._config.agent!r} returned no assistant message",
                job_id=self._job.id,
                kind=self.kind,
            )
        if _AUTH_ERROR_FRAGMENT in text:
            raise OrchidPipelineAuthError(
                f"agent job {self._job.id!r} has no auth context for agent {self._config.agent!r}"
            )

        try:
            usage["llm_calls"] = _llm_call_count(result, trace)
            output = self._output(text, pipeline)
            metadata = _bounded_metadata(result)
            if trace.get("model_routing"):
                metadata["model_routing"] = _bounded_json(trace["model_routing"])
        except GraphBubbleUp:
            raise
        except OrchidPipelineError:
            raise
        except Exception as exc:
            raise OrchidPipelineStepError(
                f"agent job {self._job.id!r} agent {self._config.agent!r} returned an invalid result",
                job_id=self._job.id,
                kind=self.kind,
            ) from exc
        return OrchidPipelineStepResult(output=output, usage=usage, metadata=metadata)

    # ── Resolution ──────────────────────────────────────────

    def _resolve_agent(self, ctx: Any) -> OrchidAgent:
        agent = self._agents.get(self._config.agent)
        if agent is None:
            raise OrchidPipelineConfigError(
                f"agent job {self._job.id!r} references unknown agent {self._config.agent!r}"
            )
        config = self._agent_configs.get(self._config.agent)
        if config is not None and getattr(config, "children", None):
            raise OrchidPipelineConfigError(
                f"agent job {self._job.id!r} references agent {self._config.agent!r}, which has children; "
                "only top-level leaf agents are addressable — use multiple jobs or a sub-pipeline"
            )
        if ctx.auth is None:
            raise OrchidPipelineAuthError(f"agent job {self._job.id!r} requires an authenticated caller")
        return agent

    def _output(self, text: str, pipeline: Any) -> dict[str, Any]:
        output_config = self._config.output
        if output_config is None or output_config.shape is None or output_config.format != "json":
            return {"text": text}
        shapes = getattr(pipeline, "shapes", None)
        if shapes is None:
            raise OrchidPipelineConfigError(
                f"agent job {self._job.id!r} references shape {output_config.shape!r}, "
                "but the pipeline declares no shapes"
            )
        try:
            shape = shapes.get(output_config.shape)
        except Exception as exc:
            raise OrchidPipelineConfigError(
                f"agent job {self._job.id!r} references unknown shape {output_config.shape!r}"
            ) from exc
        try:
            payload = _parse_json(text)
            return shape.dump(payload)
        except OrchidPipelineShapeError as exc:
            raise _shape_failure(
                self._job.id,
                shape.name,
                f"output did not validate against shape {shape.name!r}: {_bounded(str(exc), 300)}",
                source=exc,
            ) from exc
        except (ValueError, TypeError) as exc:
            raise _shape_failure(
                self._job.id,
                shape.name,
                f"output did not validate against shape {shape.name!r}: {_bounded(str(exc), 300)}",
            ) from exc


# ── Helpers ─────────────────────────────────────────────────


def _default_executor(agent: OrchidAgent, agent_config: Any) -> OrchidAgentExecutor:
    """Build the fallback executor for callers without graph-wired policies.

    It carries the agent config and inline mini-agents. Guardrails and model
    routing come from the graph-built executor in ``PipelineDependencies``;
    this fallback cannot invent chains or a router without the runtime.
    """
    return OrchidAgentExecutor(agent, agent_config=agent_config, inline_mini_agents=True)


def _accepts_trace(executor: Any) -> bool:
    """Whether an injected executor understands the optional routing trace.

    ``OrchidAgentExecutor.run`` accepts ``trace_out``; older/custom executors
    with an explicit signature may not. Feature-detect instead of turning a
    compatible executor into a sanitized ``TypeError``.
    """
    try:
        parameters = inspect.signature(executor.run).parameters
    except (TypeError, ValueError):
        return False
    if "trace_out" in parameters:
        return True
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())


def _last_ai_text(result: Mapping[str, Any]) -> str | None:
    messages = result.get("messages") or []
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return _text(message.content)
    return None


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return str(content)


def _llm_call_count(result: Mapping[str, Any], trace: Mapping[str, Any]) -> int:
    """Lower-bound LLM activations for the job's usage counter.

    The shared executor does not expose provider round counts, so the job
    counts known activations: one for the agent, one for a decomposer that ran,
    and on a fork one per mini plus the aggregator (the parent run is bypassed).
    An injected executor that does not report a trace keeps the historical
    one-call lower bound; a guardrail block reports zero.
    """
    if trace.get("guardrails_blocked"):
        return 0
    decisions = result.get("mini_agent_decisions") or {}
    if decisions:
        outcomes = result.get("mini_agent_outcomes") or {}
        return 1 + len(outcomes) + 1
    agent_invoked = trace.get("agent_invoked")
    decomposer_ran = trace.get("decomposer_ran")
    if agent_invoked is None and decomposer_ran is None:
        return 1
    return int(bool(agent_invoked)) + int(bool(decomposer_ran))


def _bounded_metadata(result: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("mcp_context", "rag_context", "skills_metadata"):
        if key in result:
            metadata[key] = _bounded_json(result[key])
    if result.get("mini_agent_outcomes"):
        metadata["mini_agent_outcomes"] = _bounded_json(result["mini_agent_outcomes"])
    return metadata


def _bounded_json(value: Any, limit: int = _MAX_METADATA_CHARS) -> Any:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str, allow_nan=False)
    except (TypeError, ValueError):
        return {"truncated": True, "preview": _preview(value)}
    if len(text) <= limit:
        return json.loads(text)
    return {"truncated": True, "preview": text[:_MAX_METADATA_PREVIEW_CHARS]}


def _preview(value: Any) -> str:
    try:
        rendered = repr(value)
    except Exception:
        rendered = f"<{type(value).__name__}>"
    return rendered[:_MAX_METADATA_PREVIEW_CHARS]


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.removesuffix("```")
        cleaned = cleaned.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("response does not contain a JSON object")
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("response is not valid JSON") from exc
    if not isinstance(data, dict):
        raise TypeError("response is not a JSON object")
    return data


def _shape_failure(
    job_id: str,
    shape_name: str,
    message: str,
    *,
    source: BaseException | None = None,
) -> OrchidPipelineShapeError:
    error = OrchidPipelineShapeError(f"agent job {job_id!r} {message}")
    error.shape = getattr(source, "shape", shape_name)
    error.object_id = getattr(source, "object_id", None)
    error.field_path = getattr(source, "field_path", "")
    error.errors = getattr(source, "errors", ())
    return error


def _bounded(message: str, limit: int) -> str:
    if len(message) <= limit:
        return message
    return message[: limit - 1] + "…"
