"""Cross-encoder reranker.

We support two backbones:

* ``cross-encoder/ms-marco-MiniLM-L-6-v2`` — the default, 90 MB, ~100 qps on
  CPU. Fast enough to rerank 500 → 200 candidates in the 5-min budget.
* ``BAAI/bge-reranker-base`` — Agent 4 upgrade, 278M params (~570 MB raw /
  ~140 MB int8). Much stronger out-of-the-box on retrieval-style pairs.

At runtime the model name comes from ``configs/build.yaml:cross_encoder.model_name``.
The ranker falls back to ms-marco-MiniLM-L-6-v2 if the configured model is
not on disk (so the build still works in offline sandboxes).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np

log = logging.getLogger("cross_encoder")

# Fallback when no fine-tuned / stronger model is on disk.
FALLBACK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
STRONG_MODEL = "BAAI/bge-reranker-base"


def _resolve_model_name(model_name: str | None) -> str:
    """Return the model name to load, falling back if the artifact is missing.

    A model name is treated as a local path if it exists on disk.
    Otherwise we fall back to the ms-marco-MiniLM default. The previous
    version warned but returned the original name, which made the
    ranker try to download a 570 MB model from HuggingFace at rank
    time (impossible in the offline sandbox).
    """
    if model_name and Path(model_name).exists():
        return model_name
    if model_name and model_name != FALLBACK_MODEL:
        log.warning(
            "Configured CE model %s not on disk; falling back to %s",
            model_name, FALLBACK_MODEL,
        )
        return FALLBACK_MODEL
    return model_name or FALLBACK_MODEL


def rerank(
    query: str,
    candidates: list[tuple[str, str]],
    model_name: str = FALLBACK_MODEL,
    top_k: int | None = 200,
    batch_size: int = 64,
    max_length: int = 256,
    device: str = "cpu",
    cache_dir: str | None = None,
) -> list[tuple[str, float]]:
    """Score (query, document) pairs. Return top-k by score, descending.

    Uses bge-reranker-base's preferred max_length of 512 if the model is
    the bge variant; otherwise defaults to 256 (ms-marco friendly).
    """
    from sentence_transformers import CrossEncoder

    if not candidates:
        return []
    model_name = _resolve_model_name(model_name)
    # bge-reranker models benefit from longer sequences.
    if "bge-reranker" in model_name and max_length == 256:
        max_length = 512
    model = CrossEncoder(model_name, max_length=max_length, device=device, cache_dir=cache_dir)
    pairs = [(query, doc) for _, doc in candidates]
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    if isinstance(scores, np.ndarray):
        scores = scores.tolist()
    scored = [(candidates[i][0], float(s)) for i, s in enumerate(scores)]
    scored.sort(key=lambda x: x[1], reverse=True)
    if top_k is not None:
        scored = scored[:top_k]
    return scored


def rerank_pairs(
    pairs: Iterable[tuple[str, str]],
    model_name: str = FALLBACK_MODEL,
    top_k: int | None = 200,
    batch_size: int = 64,
    max_length: int = 256,
    device: str = "cpu",
    cache_dir: str | None = None,
) -> list[tuple[float, str, str]]:
    """Score arbitrary (query, doc) pairs. Returns [(score, qid, doc_id)]."""
    from sentence_transformers import CrossEncoder

    pairs = list(pairs)
    if not pairs:
        return []
    model_name = _resolve_model_name(model_name)
    if "bge-reranker" in model_name and max_length == 256:
        max_length = 512
    model = CrossEncoder(model_name, max_length=max_length, device=device, cache_dir=cache_dir)
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    if isinstance(scores, np.ndarray):
        scores = scores.tolist()
    out = [(float(s), pairs[i][0], pairs[i][1]) for i, s in enumerate(scores)]
    out.sort(key=lambda x: x[0], reverse=True)
    if top_k is not None:
        out = out[:top_k]
    return out
