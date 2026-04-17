"""Tests for OrchidMetricsHandler callback (#9)."""

from __future__ import annotations

import time
from uuid import uuid4

from langchain_core.outputs import LLMResult

from orchid_ai.observability.callbacks import OrchidMetricsHandler


class TestOrchidMetricsHandlerInit:
    """Fresh handler starts with zeroed counters."""

    def test_initial_state(self):
        h = OrchidMetricsHandler()
        metrics = h.get_metrics()
        assert metrics["total_tokens"] == 0
        assert metrics["prompt_tokens"] == 0
        assert metrics["completion_tokens"] == 0
        assert metrics["llm_calls"] == 0
        assert metrics["llm_errors"] == 0
        assert metrics["tool_calls"] == 0
        assert metrics["retries"] == 0
        assert metrics["avg_llm_latency_s"] == 0.0
        assert metrics["agent_latencies_s"] == {}
        assert metrics["agent_call_counts"] == {}


class TestLLMTracking:
    """LLM call tracking (tokens, latency, errors)."""

    def test_llm_call_counted(self):
        h = OrchidMetricsHandler()
        run_id = uuid4()

        h.on_llm_start({}, ["hello"], run_id=run_id)
        h.on_llm_end(
            LLMResult(
                generations=[],
                llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}},
            ),
            run_id=run_id,
        )

        metrics = h.get_metrics()
        assert metrics["llm_calls"] == 1
        assert metrics["prompt_tokens"] == 10
        assert metrics["completion_tokens"] == 20
        assert metrics["total_tokens"] == 30

    def test_multiple_llm_calls_accumulate(self):
        h = OrchidMetricsHandler()

        for _ in range(3):
            rid = uuid4()
            h.on_llm_start({}, ["q"], run_id=rid)
            h.on_llm_end(
                LLMResult(
                    generations=[],
                    llm_output={"token_usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}},
                ),
                run_id=rid,
            )

        metrics = h.get_metrics()
        assert metrics["llm_calls"] == 3
        assert metrics["total_tokens"] == 45

    def test_llm_latency_tracked(self):
        h = OrchidMetricsHandler()
        run_id = uuid4()

        h.on_llm_start({}, ["q"], run_id=run_id)
        time.sleep(0.01)  # tiny sleep to ensure measurable latency
        h.on_llm_end(
            LLMResult(generations=[], llm_output={}),
            run_id=run_id,
        )

        metrics = h.get_metrics()
        assert metrics["avg_llm_latency_s"] > 0

    def test_llm_error_counted(self):
        h = OrchidMetricsHandler()
        run_id = uuid4()

        h.on_llm_start({}, ["q"], run_id=run_id)
        h.on_llm_error(RuntimeError("boom"), run_id=run_id)

        metrics = h.get_metrics()
        assert metrics["llm_errors"] == 1
        assert metrics["llm_calls"] == 0  # error doesn't count as successful call

    def test_no_llm_output_no_crash(self):
        h = OrchidMetricsHandler()
        run_id = uuid4()

        h.on_llm_start({}, ["q"], run_id=run_id)
        h.on_llm_end(
            LLMResult(generations=[], llm_output=None),
            run_id=run_id,
        )

        metrics = h.get_metrics()
        assert metrics["llm_calls"] == 1
        assert metrics["total_tokens"] == 0


class TestToolTracking:
    """Tool call counting."""

    def test_tool_call_counted(self):
        h = OrchidMetricsHandler()
        h.on_tool_end("result", run_id=uuid4())
        h.on_tool_end("result2", run_id=uuid4())

        metrics = h.get_metrics()
        assert metrics["tool_calls"] == 2


class TestAgentTracking:
    """Agent latency and call count tracking."""

    def test_agent_latency_tracked(self):
        h = OrchidMetricsHandler()
        run_id = uuid4()

        h.on_chain_start({"name": "learning_agent"}, {}, run_id=run_id)
        time.sleep(0.01)
        h.on_chain_end({}, run_id=run_id)

        metrics = h.get_metrics()
        assert "learning" in metrics["agent_latencies_s"]
        assert metrics["agent_latencies_s"]["learning"] > 0
        assert metrics["agent_call_counts"]["learning"] == 1

    def test_non_agent_chain_ignored(self):
        h = OrchidMetricsHandler()
        run_id = uuid4()

        h.on_chain_start({"name": "supervisor"}, {}, run_id=run_id)
        h.on_chain_end({}, run_id=run_id)

        metrics = h.get_metrics()
        assert metrics["agent_latencies_s"] == {}
        assert metrics["agent_call_counts"] == {}

    def test_multiple_agents_tracked_separately(self):
        h = OrchidMetricsHandler()

        r1 = uuid4()
        h.on_chain_start({"name": "learning_agent"}, {}, run_id=r1)
        h.on_chain_end({}, run_id=r1)

        r2 = uuid4()
        h.on_chain_start({"name": "notifications_agent"}, {}, run_id=r2)
        h.on_chain_end({}, run_id=r2)

        metrics = h.get_metrics()
        assert "learning" in metrics["agent_call_counts"]
        assert "notifications" in metrics["agent_call_counts"]
        assert metrics["agent_call_counts"]["learning"] == 1
        assert metrics["agent_call_counts"]["notifications"] == 1


class TestRetryTracking:
    """Retry event counting."""

    def test_retry_counted(self):
        h = OrchidMetricsHandler()
        h.on_retry(None, run_id=uuid4())
        h.on_retry(None, run_id=uuid4())

        metrics = h.get_metrics()
        assert metrics["retries"] == 2


class TestReset:
    """Reset clears all metrics."""

    def test_reset_clears_everything(self):
        h = OrchidMetricsHandler()

        # Accumulate some state
        rid = uuid4()
        h.on_llm_start({}, ["q"], run_id=rid)
        h.on_llm_end(
            LLMResult(
                generations=[],
                llm_output={"token_usage": {"total_tokens": 100}},
            ),
            run_id=rid,
        )
        h.on_tool_end("x", run_id=uuid4())
        h.on_retry(None, run_id=uuid4())

        assert h.get_metrics()["total_tokens"] == 100
        assert h.get_metrics()["tool_calls"] == 1

        h.reset()
        metrics = h.get_metrics()
        assert metrics["total_tokens"] == 0
        assert metrics["tool_calls"] == 0
        assert metrics["retries"] == 0
        assert metrics["llm_calls"] == 0


class TestImportFromSDK:
    """OrchidMetricsHandler is importable from the SDK surface."""

    def test_import_from_orchid_ai(self):
        from orchid_ai import OrchidMetricsHandler as H

        assert H is OrchidMetricsHandler

    def test_import_from_observability(self):
        from orchid_ai.observability import OrchidMetricsHandler as H

        assert H is OrchidMetricsHandler
