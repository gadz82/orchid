"""
Groundedness guardrail — check that agent output is supported by context.

Uses a lightweight heuristic: if RAG context was provided, verify that
the agent's response references or aligns with the retrieved documents.

For LLM-based groundedness checking (more accurate but slower), consumers
can subclass this and override the ``check()`` method to call an LLM
evaluator.  The ``llm_provider`` config parameter is reserved for that
future extension.

This built-in version uses keyword overlap as a simple proxy for
groundedness — it checks what fraction of key response terms appear
in the provided context.
"""

from __future__ import annotations

import re

from ..core.guardrails import (
    Guardrail,
    GuardrailAction,
    GuardrailContext,
    GuardrailResult,
)

# Stop words to ignore when computing keyword overlap
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "because",
        "but",
        "and",
        "or",
        "if",
        "while",
        "about",
        "up",
        "it",
        "its",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "they",
        "them",
        "this",
        "that",
        "these",
        "those",
    }
)


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from text (lowercase, no stop words)."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return {w for w in words if w not in _STOP_WORDS}


class GroundednessGuardrail(Guardrail):
    """
    Check that output content is grounded in the provided RAG context.

    Expects ``context.metadata["rag_context"]`` to contain the RAG
    documents (as a string or list of dicts with "content" keys).

    Parameters
    ----------
    fail_action : str
        Action when ungrounded: "warn" (default), "block", or "log".
    min_overlap : float
        Minimum fraction of response keywords that must appear in the
        RAG context.  Default: 0.3 (30%).
    """

    def __init__(
        self,
        *,
        fail_action: str = "warn",
        min_overlap: float = 0.3,
    ) -> None:
        self._fail_action = GuardrailAction(fail_action)
        self._min_overlap = min_overlap

    @property
    def name(self) -> str:
        return "groundedness"

    async def check(self, content: str, context: GuardrailContext) -> GuardrailResult:
        # Extract RAG context from metadata
        rag_data = context.metadata.get("rag_context")
        if not rag_data:
            # No RAG context available — skip check (can't assess groundedness)
            return GuardrailResult.passed(self.name)

        # Normalize RAG context to a single string
        if isinstance(rag_data, list):
            rag_text = " ".join(
                item.get("content", str(item)) if isinstance(item, dict) else str(item) for item in rag_data
            )
        else:
            rag_text = str(rag_data)

        # Extract keywords
        response_keywords = _extract_keywords(content)
        context_keywords = _extract_keywords(rag_text)

        if not response_keywords:
            return GuardrailResult.passed(self.name)

        # Compute overlap
        overlap = response_keywords & context_keywords
        overlap_ratio = len(overlap) / len(response_keywords)

        if overlap_ratio >= self._min_overlap:
            return GuardrailResult.passed(self.name)

        return GuardrailResult(
            triggered=True,
            action=self._fail_action,
            guardrail_name=self.name,
            message=(
                "The response may contain information not fully supported by the available context. "
                "Please verify the accuracy of the information provided."
            ),
            details={
                "overlap_ratio": round(overlap_ratio, 3),
                "min_overlap": self._min_overlap,
                "response_keywords_count": len(response_keywords),
                "context_keywords_count": len(context_keywords),
                "overlapping_keywords_count": len(overlap),
            },
        )
