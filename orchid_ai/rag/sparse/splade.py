"""
Splade sparse encoder (ADR-025 §"Sparse vector encoder abstraction").

Wraps Splade-style learned sparse retrieval models (e.g.
``naver/splade-cocondenser-ensembledistil``) behind the optional
``splade`` extra: ``pip install orchid-ai[splade]``.

The constructor eagerly checks that ``torch`` and ``transformers``
import — surfacing the missing-extra error at registry lookup time
rather than burying it in the first ``encode_query`` call.  The
actual model is lazily loaded on first encode call so registry
lookups don't pay the multi-hundred-megabyte download cost when
nobody calls the encoder.
"""

from __future__ import annotations

import logging
from typing import Any

from ...core.sparse import OrchidSparseEncoder, OrchidSparseVector

logger = logging.getLogger(__name__)


_INSTALL_HINT = "SpladeEncoder requires the 'splade' extra. Install with: pip install orchid-ai[splade]"


class SpladeEncoder(OrchidSparseEncoder):
    """Splade learned-sparse encoder — opt-in via the ``splade`` extra.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.  Default is the popular
        ``naver/splade-cocondenser-ensembledistil`` checkpoint.
    device : str
        ``"cpu"`` (default), ``"cuda"``, ``"mps"`` — passed to
        ``torch.device``.  ``"cuda"`` requires a GPU-capable PyTorch
        build.
    max_length : int
        Token limit per input; longer inputs are truncated.

    Raises
    ------
    ImportError
        When the ``splade`` extra is not installed.
    """

    def __init__(
        self,
        *,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        # Eager import check — surfaces ``pip install orchid-ai[splade]``
        # at registry lookup time, not at first encode call.
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:  # pragma: no cover — exercised via test_rag_splade_encoder
            raise ImportError(_INSTALL_HINT) from exc

        self._model_name = model_name
        self._device = device
        self._max_length = max_length
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    # ── ABC contract ──────────────────────────────────────────

    async def encode_documents(
        self,
        texts: list[str],
        namespace: str | None = None,
    ) -> list[OrchidSparseVector]:
        self._ensure_loaded()
        return [self._encode_one(t) for t in texts]

    async def encode_query(
        self,
        text: str,
        namespace: str | None = None,
    ) -> OrchidSparseVector:
        self._ensure_loaded()
        return self._encode_one(text)

    # ── Internal helpers ──────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Imports are deferred to first use — they trigger the
        # ~hundreds-of-megabytes model download.
        import torch  # noqa: F401  — referenced via globals() below
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        logger.info("[SpladeEncoder] Loading model %s on %s", self._model_name, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        model = AutoModelForMaskedLM.from_pretrained(self._model_name)
        model.to(self._device)
        model.eval()
        self._model = model

    def _encode_one(self, text: str) -> OrchidSparseVector:
        import torch

        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None  # noqa: S101 — _ensure_loaded ran

        with torch.no_grad():
            tokens = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_length,
            ).to(self._device)
            output = model(**tokens)
            # Standard Splade aggregation: max-over-sequence of
            # log(1 + relu(logits)).
            relu_log = torch.log(1 + torch.relu(output.logits))
            attention_mask = tokens["attention_mask"].unsqueeze(-1)
            weighted = relu_log * attention_mask
            vec = torch.max(weighted, dim=1).values.squeeze()

        nonzero = torch.nonzero(vec, as_tuple=False).squeeze(-1)
        return OrchidSparseVector(
            indices=nonzero.cpu().tolist(),
            values=vec[nonzero].cpu().tolist(),
        )
