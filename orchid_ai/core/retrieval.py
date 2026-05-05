"""
Pluggable retrieval primitives (ADR-023).

Two ABCs:

  * :class:`OrchidRetrievalStrategy` — turn a query into a ranked list
    of :class:`OrchidSearchResult`.  Strategies own the choice of lane
    (dense / sparse / graph), fusion algorithm, and how to consume
    transformers.
  * :class:`OrchidQueryTransformer` — turn one query into one (or
    several) reformulated query strings.  The :attr:`pre_strategy`
    class flag splits transformers into two scopes:
      - ``True`` (e.g. reformulate): runs once at agent entry, returns
        exactly one replacement query that feeds RAG **and** the agentic
        loop.  Same query everywhere — no double work, no scope drift.
      - ``False`` (e.g. multi_query, HyDE, decompose): runs inside the
        retrieval strategy, may return N queries that are fanned out
        for retrieval only.  Strategy decides how to merge.

The split is intentional and load-bearing: the wide-impact rewrite is
**not** the same kind of transformation as a fan-out, so they get
different types and different call sites.

Both ABCs live in ``core/`` so strategies, transformers, and the agent
hot path can depend on them without risking a cycle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from .doc_store import OrchidDocStore
from .graph_store import OrchidGraphStore
from .repository import OrchidSearchResult, OrchidVectorReader
from .scopes import OrchidRAGScope


class OrchidQueryTransformer(ABC):
    """Convert one query into one-or-more queries.

    The :attr:`pre_strategy` class flag is the contract distinguishing
    agent-level rewrites from strategy-internal fan-outs:

    * ``pre_strategy=True`` — must return **exactly one** query, used as
      a drop-in replacement for the original.  The agent applies these
      transformers in order at the start of the turn; the rewritten
      query feeds RAG, the agentic loop, and any other downstream step.

    * ``pre_strategy=False`` — may return any number of queries.  The
      agent passes these transformers (resolved from the registry) into
      the retrieval strategy via the ``transformers`` kwarg.  Each
      strategy decides how to consume them (e.g. fan out + RRF merge,
      cascade with deduplication).

    A runtime check in :meth:`apply_pre_strategy` enforces the
    "exactly one" rule for ``pre_strategy=True`` transformers so a
    misbehaving implementation raises immediately instead of silently
    dropping or duplicating queries.
    """

    pre_strategy: ClassVar[bool] = False

    @abstractmethod
    async def transform(
        self,
        query: str,
        *,
        chat_model: Any,
        history: list[dict[str, str]] | None = None,
    ) -> list[str]:
        """Return the transformed queries.

        ``chat_model`` is the duck-typed LangChain chat model; ``history``
        is the standard list of ``{"role": ..., "content": ...}`` dicts
        produced by ``OrchidAgent.extract_conversation_history``.
        """
        ...


class OrchidRetrievalStrategy(ABC):
    """Turn a query into a ranked list of :class:`OrchidSearchResult`.

    The signature is wide on purpose: each strategy uses only the
    side-channel resources it needs.  ISP is preserved because the
    strategy depends on the **narrow** :class:`OrchidVectorReader`
    (not the full repository), and graph / doc stores are explicitly
    optional.

    Strategies must be **stateless** across calls — any per-call state
    lives inside ``retrieve()`` — so they can be registered once at
    process startup and shared across requests.

    Per-strategy YAML knobs (e.g. ``hyde.n_hypothetical``,
    ``hybrid.fusion``) flow into the strategy via :meth:`from_config`
    rather than widening :meth:`retrieve`'s signature.  The default
    impl simply calls ``cls()``; subclasses that need YAML knobs
    override it.
    """

    @classmethod
    def from_config(cls, config: Any) -> "OrchidRetrievalStrategy":
        """Build a strategy instance from an :class:`OrchidRetrievalConfig`.

        ``config`` is duck-typed (``Any``) so this ABC stays free of
        the Pydantic config import in ``core/``.  Default impl is
        zero-arg construction; override to read strategy-specific
        knobs (e.g. ``HyDERetrieval`` reads ``config.hyde.n_hypothetical``).
        """
        return cls()

    @abstractmethod
    async def retrieve(
        self,
        *,
        query: str,
        namespace: str,
        scope: OrchidRAGScope,
        k: int,
        reader: OrchidVectorReader,
        chat_model: Any | None = None,
        graph_store: OrchidGraphStore | None = None,
        doc_store: OrchidDocStore | None = None,
        transformers: list[OrchidQueryTransformer] | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[OrchidSearchResult]:
        """Return the top-``k`` results for ``query`` in ``namespace``.

        ``transformers`` carries only ``pre_strategy=False`` transformers
        — agent-entry transformers have already rewritten the query by
        the time the strategy is called.  ``metadata_filters`` follows
        the operator mini-language defined in ADR-027 (used from
        Stage 6 onward; ignored by strategies that haven't wired it up).
        """
        ...


async def apply_pre_strategy(
    transformers: list[OrchidQueryTransformer],
    query: str,
    *,
    chat_model: Any,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Run every ``pre_strategy=True`` transformer in order, return the rewrite.

    Each ``pre_strategy=True`` transformer **must** return a 1-element
    list — anything else is a contract violation and raises
    :class:`RuntimeError`.  ``pre_strategy=False`` transformers in the
    list are silently skipped here (they belong inside the retrieval
    strategy).
    """
    rewritten = query
    for transformer in transformers:
        if not transformer.pre_strategy:
            continue
        result = await transformer.transform(rewritten, chat_model=chat_model, history=history)
        if len(result) != 1:
            raise RuntimeError(
                f"Pre-strategy transformer {type(transformer).__name__} must return exactly 1 query, got {len(result)}."
            )
        rewritten = result[0]
    return rewritten
