"""RagPipeline — owns RAG retrieval, cache check, and dynamic injection.

M2 refactoring: extracted from ``GenericAgent`` (833 LOC).  Groups the
three RAG-related pipeline steps into a single SRP collaborator.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..core.repository import OrchidVectorReader
from ..documents.strategies import build_ingestion_strategy
from ..rag.dynamic import inject_to_rag
from ..rag.scopes import OrchidRAGScope
from ..rag.strategies import get_retrieval_strategy
from ..rag.transformers import (
    TRANSFORMER_REGISTRY,
    get_query_transformer,
    resolve_transformer_kwargs,
)

logger = logging.getLogger(__name__)
perf_logger = logging.getLogger("orchid.perf")


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
    ) -> None:
        self._reader = reader
        self._chat_model = chat_model
        self._graph_store = graph_store
        self._decision_model = decision_model
        self._confidence_threshold = max(0.0, min(1.0, confidence_threshold))

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

        strategy = get_retrieval_strategy(
            retrieval_strategy or "simple",
            config=retrieval_config,
        )

        prompts_cfg = retrieval_config.transformer_prompts if retrieval_config else None
        strategy_transformers = [
            get_query_transformer(name, **resolve_transformer_kwargs(name, prompts_cfg))
            for name in (retrieval_config.query_transformers or [])
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
        combined = await self._gate_passages(query, combined)
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

    async def _gate_passages(self, query: str, passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep only passages the decision provider judges useful.

        Fails open: no provider, too many candidates, a provider error, or an
        empty/low-confidence selection all return the original ranking.
        """
        if self._decision_model is None or not passages:
            return passages
        candidates = passages[:32]
        if len(passages) > 32:
            return passages

        from ..core.decision import (
            OrchidDecisionError,
            OrchidDecisionQuestion,
            OrchidDecisionState,
            aggregate_shortlist,
        )

        questions = [
            OrchidDecisionQuestion(
                name=f"chunk:{index}",
                kind="boolean",
                instructions=f"Does this passage help answer the query? {str(passage.get('text', ''))[:300]}",
            )
            for index, passage in enumerate(candidates)
        ]
        try:
            result = await self._decision_model.decide(
                OrchidDecisionState(
                    text=query[:4000],
                    payload={"passages": [str(p.get("text", ""))[:500] for p in candidates]},
                ),
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
        logger.info("[RAG] gated %d/%d passages", len(keep), len(passages))
        return [passage for index, passage in enumerate(passages) if index in keep]
