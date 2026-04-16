"""
LangChain callback handler for per-request observability.

``OrchidMetricsHandler`` collects token usage, LLM call counts, tool call
counts, retry counts, and per-agent latency.  Create a fresh instance per
graph invocation and pass it via the LangGraph ``config`` dict.

Example::

    from orchid_ai.observability import OrchidMetricsHandler

    handler = OrchidMetricsHandler()
    result = await graph.ainvoke(state, config={"callbacks": [handler]})
    metrics = handler.get_metrics()
    # {
    #     "total_tokens": 1247,
    #     "llm_calls": 3,
    #     "tool_calls": 2,
    #     "retries": 0,
    #     "avg_llm_latency_s": 0.842,
    #     "agent_latencies_s": {"learning": 1.23, "notifications": 0.95},
    #     ...
    # }
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


class OrchidMetricsHandler(BaseCallbackHandler):
    """Tracks LLM usage, tool calls, and agent latency per request.

    Create a fresh instance per ``graph.ainvoke()`` call to get isolated
    metrics.  After the graph completes, call :meth:`get_metrics` to
    retrieve the aggregated stats.

    Thread-safe: all counters are protected by a lock so concurrent
    agent execution (parallel mode) doesn't corrupt state.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()

        # ── Token tracking ──
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0

        # ── Call counts ──
        self.llm_calls: int = 0
        self.llm_errors: int = 0
        self.tool_calls: int = 0
        self.retries: int = 0

        # ── LLM latency tracking (seconds) ──
        self._llm_start_times: dict[UUID, float] = {}
        self.llm_latencies: list[float] = []

        # ── Agent-level tracking ──
        self._agent_start_times: dict[UUID, tuple[str, float]] = {}
        self.agent_latencies: dict[str, list[float]] = {}
        self.agent_call_counts: dict[str, int] = {}

    # ── LLM callbacks ──────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._llm_start_times[run_id] = time.monotonic()

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            self.llm_calls += 1

            start = self._llm_start_times.pop(run_id, None)
            if start is not None:
                self.llm_latencies.append(time.monotonic() - start)

            if response.llm_output:
                usage = response.llm_output.get("token_usage", {})
                self.prompt_tokens += usage.get("prompt_tokens", 0)
                self.completion_tokens += usage.get("completion_tokens", 0)
                self.total_tokens += usage.get("total_tokens", 0)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            self.llm_errors += 1
            self._llm_start_times.pop(run_id, None)

    # ── Tool callbacks ─────────────────────────────────────────

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            self.tool_calls += 1

    # ── Chain callbacks (agent tracking) ───────────────────────

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", "")
        if name.endswith("_agent"):
            agent_name = name.removesuffix("_agent")
            self._agent_start_times[run_id] = (agent_name, time.monotonic())

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        entry = self._agent_start_times.pop(run_id, None)
        if entry is not None:
            agent_name, start = entry
            latency = time.monotonic() - start
            with self._lock:
                self.agent_latencies.setdefault(agent_name, []).append(latency)
                self.agent_call_counts[agent_name] = self.agent_call_counts.get(agent_name, 0) + 1

    # ── Retry callback ─────────────────────────────────────────

    def on_retry(
        self,
        retry_state: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            self.retries += 1

    # ── Metrics retrieval ──────────────────────────────────────

    def get_metrics(self) -> dict[str, Any]:
        """Return aggregated metrics from this handler's lifecycle.

        Returns a dict with:
        - ``prompt_tokens``, ``completion_tokens``, ``total_tokens``
        - ``llm_calls``, ``llm_errors``
        - ``tool_calls``, ``retries``
        - ``avg_llm_latency_s`` — average LLM call latency in seconds
        - ``agent_latencies_s`` — per-agent average latency in seconds
        - ``agent_call_counts`` — per-agent invocation count
        """
        with self._lock:
            avg_llm = sum(self.llm_latencies) / len(self.llm_latencies) if self.llm_latencies else 0.0
            agent_avg = {name: sum(lats) / len(lats) if lats else 0.0 for name, lats in self.agent_latencies.items()}
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "llm_calls": self.llm_calls,
                "llm_errors": self.llm_errors,
                "tool_calls": self.tool_calls,
                "retries": self.retries,
                "avg_llm_latency_s": round(avg_llm, 3),
                "agent_latencies_s": {name: round(avg, 3) for name, avg in agent_avg.items()},
                "agent_call_counts": dict(self.agent_call_counts),
            }

    def reset(self) -> None:
        """Clear all collected metrics."""
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.llm_calls = 0
            self.llm_errors = 0
            self.tool_calls = 0
            self.retries = 0
            self._llm_start_times.clear()
            self.llm_latencies.clear()
            self._agent_start_times.clear()
            self.agent_latencies.clear()
            self.agent_call_counts.clear()
