"""RagPipeline — owns RAG retrieval, cache check, and dynamic injection.

M2 refactoring: extracted from ``GenericAgent`` (833 LOC).  Groups the
three RAG-related pipeline steps into a single SRP collaborator.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..core.decision import (
    OrchidDecisionError,
    OrchidDecisionPointBinding,
    OrchidDecisionQuestion,
    OrchidDecisionState,
    aggregate_shortlist,
)
from ..core.repository import OrchidVectorReader
from ..decision.observability import log_decision
from ..documents.strategies import build_ingestion_strategy
from ..rag.dynamic import inject_to_rag
from ..rag.scopes import OrchidRAGScope
from ..rag.strategies import RETRIEVAL_REGISTRY, get_retrieval_strategy
from ..rag.transformers import (
    TRANSFORMER_REGISTRY,
    get_query_transformer,
    resolve_transformer_kwargs,
)

logger = logging.getLogger(__name__)
perf_logger = logging.getLogger("orchid.perf")

_STRATEGY_QUESTION = "strategy"
_MAX_QUERY_CHARS = 4000
_MIN_STRATEGY_OPTIONS = 2


class RagPipeline:
    """Owns the RAG retrieval, cache check, and dynamic injection steps.

    Extracted from ``GenericAgent._step_rag_retrieval``,
    ``GenericAgent._step_cache_check``, and
    ``GenericAgent._step_dynamic_injection``.
    """

    def __init__(
        self,
        reader: OrchidVectorReader,
        chat_model: Any | None = None,
        graph_store: Any | None = None,
        *,
        decision_model: Any | None = None,
        confidence_threshold: float = 0.75,
        strategies: list[str] | None = None,
        retrieval_strategy_binding: OrchidDecisionPointBinding | None = None,
    ) -> None:
        self._reader = reader
        self._chat_model = chat_model
        self._graph_store = graph_store
        # Passage-gate provider (the legacy ``rag`` point). It is deliberately
        # separate from the strategy selector below: they answer different
        # questions and can be bound to different providers.
        self._decision_model = decision_model
        self._confidence_threshold = max(0.0, min(1.0, confidence_threshold))
        self._strategies = list(strategies) if strategies else None
        # Strategy selector (the additive ``retrieval_strategy`` point).
        self._retrieval_strategy_binding = retrieval_strategy_binding

    async def retrieve(
        self,
        query: str,
        scope: OrchidRAGScope,
        *,
        rag_namespace: str,
        k: int,
        enabled: bool,
        retrieval_strategy: str = "simple",
        retrieval_config: Any | None = None,
        exclude_dynamic: bool = False,
    ) -> list[dict[str, Any]]:
        """Step 1: RAG retrieval — strategy resolved by name from the registry."""
        if not enabled:
            return []

        import asyncio as _asyncio

        configured_strategy = retrieval_strategy or "simple"
        strategy_name = await self._select_retrieval_strategy(query, configured_strategy)
        strategy = get_retrieval_strategy(
            strategy_name,
            config=retrieval_config,
        )
        from ..core.retrieval import OrchidRAGExplorer

        if self._decision_model is not None and isinstance(strategy, OrchidRAGExplorer):
            strategy.bind_decision_model(
                self._decision_model,
                confidence_threshold=self._confidence_threshold,
            )

        prompts_cfg = retrieval_config.transformer_prompts if retrieval_config else None
        strategy_transformers = [
            get_query_transformer(name, **resolve_transformer_kwargs(name, prompts_cfg))
            for name in (retrieval_config.query_transformers or [] if retrieval_config else [])
            if not TRANSFORMER_REGISTRY[name].pre_strategy
        ]

        configured_filters = dict(retrieval_config.metadata_filters or {}) if retrieval_config else {}
        if exclude_dynamic:
            configured_filters.setdefault("dynamic", {"not": True})
        metadata_filters = configured_filters or None

        common_kwargs: dict[str, Any] = {
            "query": query,
            "scope": scope,
            "k": k,
            "reader": self._reader,
            "chat_model": self._chat_model,
            "transformers": strategy_transformers,
            "metadata_filters": metadata_filters,
            "graph_store": self._graph_store,
        }

        domain_results, upload_results = await _asyncio.gather(
            strategy.retrieve(namespace=rag_namespace, **common_kwargs),
            strategy.retrieve(namespace="uploads", **common_kwargs),
        )

        combined: list[dict[str, Any]] = []
        for r in [*domain_results, *upload_results]:
            content = r.document.metadata.get("parent_content", r.document.page_content)
            combined.append(
                {
                    "content": content,
                    "score": round(r.score, 3),
                    "metadata": {
                        mk: mv
                        for mk, mv in r.document.metadata.items()
                        if mk not in ("content", "embedding", "parent_content")
                    },
                }
            )

        combined.sort(key=lambda d: d.get("score", 0), reverse=True)
        # The selected strategy drives retrieval, but the gate keeps using the
        # *configured* strategy for its allowlist and log label so selection
        # cannot change the legacy chunk-gate behaviour.
        combined = await self._gate_passages(query, combined, configured_strategy)
        return combined[:k]

    async def check_cache(
        self,
        scope: OrchidRAGScope,
        *,
        rag_namespace: str,
        enabled: bool,
        tool_ttls: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Step 1.5: Check RAG for cached tool results within TTL."""
        if not (enabled and tool_ttls):
            return {}
        return await self._lookup_cached_tools(scope, rag_namespace, tool_ttls)

    async def inject(
        self,
        mcp_data: dict[str, Any],
        scope: OrchidRAGScope,
        *,
        rag_namespace: str,
        enabled: bool,
        injectable_tools: set[str] | None = None,
        effective_rag_resolver: Any = None,
    ) -> None:
        """Dynamic RAG injection — per-tool ``effective_rag``."""
        if not (enabled and injectable_tools):
            return

        for tool_name, tool_result in mcp_data.items():
            if tool_name not in injectable_tools and (f"builtin_{tool_name}" not in injectable_tools):
                continue

            effective = effective_rag_resolver(tool_name)
            target_namespace = effective.namespace or rag_namespace
            ingestion = build_ingestion_strategy(effective.ingestion)

            await inject_to_rag(
                self._reader,
                tool_name=tool_name,
                tool_result=tool_result,
                namespace=target_namespace,
                scope=scope,
                ingestion=ingestion,
            )

    async def _lookup_cached_tools(
        self,
        scope: OrchidRAGScope,
        namespace: str,
        tool_ttls: dict[str, int],
    ) -> dict[str, Any]:
        import asyncio as _asyncio

        async def _lookup(tool_name: str, ttl: int) -> tuple[str, Any]:
            min_time = time.time() - ttl
            result = await self._reader.lookup_cached_tool_results(
                namespace=namespace,
                scope=scope,
                tool_name=tool_name,
                min_injected_at=min_time,
            )
            if result is not None:
                logger.info("Cache hit for tool '%s' (TTL=%ds)", tool_name, ttl)
            return tool_name, result

        pairs = await _asyncio.gather(*(_lookup(name, ttl) for name, ttl in tool_ttls.items()))
        return {name: val for name, val in pairs if val is not None}

    async def _select_retrieval_strategy(self, query: str, configured: str) -> str:
        """Ask the ``retrieval_strategy`` point which strategy to use.

        The binding is optional: without it (point disabled, unconfigured, or
        its provider missing from the runtime) the configured strategy is used
        unchanged. A selected name is accepted only when it is in the offered
        candidate set — the retrieval registry, optionally narrowed by
        ``options.strategies``; every other outcome logs one structured
        fallback line and returns ``configured``.
        """
        binding = self._retrieval_strategy_binding
        if binding is None:
            return configured

        candidates = tuple(sorted(RETRIEVAL_REGISTRY))
        allowlist = binding.options.get("strategies")
        allowlist_active = isinstance(allowlist, list) and bool(allowlist)
        if allowlist_active:
            candidates = tuple(name for name in candidates if name in allowlist)
        if len(candidates) < _MIN_STRATEGY_OPTIONS:
            return self._strategy_fallback(binding, configured, "no_candidates")

        question = OrchidDecisionQuestion(
            name=_STRATEGY_QUESTION,
            kind="choice",
            options=candidates,
            instructions=(
                "Choose the retrieval strategy that best answers the query. "
                "Prefer the simplest strategy that fits; choose a graph or hybrid "
                "strategy only when the query needs multi-hop or mixed evidence."
            ),
        )
        started = time.perf_counter()
        try:
            result = await binding.decide(
                OrchidDecisionState(text=query[:_MAX_QUERY_CHARS]),
                [question],
            )
        except OrchidDecisionError as exc:
            logger.warning("[RAG] retrieval strategy provider failed code=%s", exc.code)
            return self._strategy_fallback(
                binding,
                configured,
                exc.code,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                provider=exc.provider,
            )
        except Exception:
            logger.warning("[RAG] retrieval strategy provider raised an unexpected error", exc_info=True)
            return self._strategy_fallback(
                binding,
                configured,
                "provider_error",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        provider = result.provider or binding.provider
        answer = result.answers.get(_STRATEGY_QUESTION)
        if answer is None:
            return self._strategy_fallback(
                binding,
                configured,
                "invalid_response",
                latency_ms=latency_ms,
                provider=provider,
            )
        if answer.refusal is not None:
            return self._strategy_fallback(
                binding,
                configured,
                "refusal",
                latency_ms=latency_ms,
                confidence=answer.confidence,
                provider=provider,
            )
        if answer.confidence < binding.confidence_threshold:
            return self._strategy_fallback(
                binding,
                configured,
                "low_confidence",
                latency_ms=latency_ms,
                confidence=answer.confidence,
                provider=provider,
            )

        selected = str(answer.value)
        if selected not in candidates:
            return self._strategy_fallback(
                binding,
                configured,
                "out_of_allowlist" if allowlist_active else "unknown_strategy",
                latency_ms=latency_ms,
                confidence=answer.confidence,
                provider=provider,
            )

        log_decision(
            point="retrieval_strategy",
            path="decision",
            provider=provider,
            latency_ms=latency_ms,
            confidence=answer.confidence,
            outcome="accepted",
        )
        logger.info("[RAG] retrieval strategy selected %r", selected)
        return selected

    def _strategy_fallback(
        self,
        binding: OrchidDecisionPointBinding,
        configured: str,
        reason: str,
        *,
        latency_ms: float | None = None,
        confidence: float | None = None,
        provider: str = "",
    ) -> str:
        """Log one structured fallback line and return the configured strategy."""
        logger.warning("[RAG] retrieval strategy decision fell back to %r (reason=%s)", configured, reason)
        log_decision(
            point="retrieval_strategy",
            path="llm_fallback",
            provider=provider or binding.provider,
            latency_ms=latency_ms,
            confidence=confidence,
            outcome="fallback",
            fallback_reason=reason,
        )
        return configured

    async def _gate_passages(
        self,
        query: str,
        passages: list[dict[str, Any]],
        strategy: str = "simple",
    ) -> list[dict[str, Any]]:
        """Keep only passages the decision provider judges useful.

        Fails open: no provider, a strategy excluded by config, too many
        candidates, a provider error, or an empty/low-confidence selection all
        return the original ranking.
        """
        if self._decision_model is None or not passages:
            return passages
        if self._strategies is not None and strategy not in self._strategies:
            return passages
        candidates = passages[:32]
        if len(passages) > 32:
            return passages

        payload_passages: list[dict[str, Any]] = []
        questions: list[OrchidDecisionQuestion] = []
        for index, passage in enumerate(candidates):
            text = str(passage.get("content") or passage.get("text") or "")
            metadata = passage.get("metadata") or {}
            source = metadata.get("source") if isinstance(metadata, dict) else None
            score = passage.get("score")
            payload_passages.append(
                {
                    "index": index,
                    "text": text[:500],
                    "score": score if isinstance(score, (int, float)) else None,
                    "source": str(source) if source is not None else None,
                }
            )
            detail = f"source={source}, score={score}" if source is not None else f"score={score}"
            questions.append(
                OrchidDecisionQuestion(
                    name=f"chunk:{index}",
                    kind="boolean",
                    instructions=f"Does passage {index} ({detail}) help answer the query? {text[:300]}",
                )
            )

        try:
            result = await self._decision_model.decide(
                OrchidDecisionState(text=query[:4000], payload={"passages": payload_passages}),
                questions,
            )
        except OrchidDecisionError as exc:
            logger.warning("[RAG] decision provider failed code=%s; keeping the original ranking", exc.code)
            return passages
        except Exception:
            logger.warning("[RAG] decision provider raised an unexpected error", exc_info=True)
            return passages

        selected = aggregate_shortlist(
            result,
            [f"chunk:{index}" for index in range(len(candidates))],
            threshold=self._confidence_threshold,
        )
        if not selected:
            return passages
        keep = {int(name.split(":", 1)[1]) for name in selected}
        logger.info("[RAG] gated %d/%d passages (strategy=%s)", len(keep), len(passages), strategy)
        return [passage for index, passage in enumerate(passages) if index in keep]
