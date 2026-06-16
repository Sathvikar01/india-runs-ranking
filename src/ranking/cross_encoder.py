"""Cross-encoder reranker.

We use a small, fast cross-encoder (default `ms-marco-MiniLM-L-6-v2`, ~90 MB)
that runs at > 100 qps on CPU. 500 → 100 rerank in well under the 5-min
budget.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def rerank(
    query: str,
    candidates: list[tuple[str, str]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int | None = 200,
    batch_size: int = 64,
    max_length: int = 256,
    device: str = "cpu",
    cache_dir: str | None = None,
) -> list[tuple[str, float]]:
    """Score (query, document) pairs. Return top-k by score, descending."""
    from sentence_transformers import CrossEncoder

    if not candidates:
        return []
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
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int | None = 200,
    batch_size: int = 64,
    max_length: int = 256,
    device: str = "cpu",
    cache_dir: str | None = None,
) -> list[tuple[str, float, str]]:
    """Score arbitrary (query, doc) pairs. Returns [(score, qid, doc_id)]."""
    from sentence_transformers import CrossEncoder

    pairs = list(pairs)
    if not pairs:
        return []
    model = CrossEncoder(model_name, max_length=max_length, device=device, cache_dir=cache_dir)
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    if isinstance(scores, np.ndarray):
        scores = scores.tolist()
    out = [(float(s), pairs[i][0], pairs[i][1]) for i, s in enumerate(scores)]
    out.sort(key=lambda x: x[0], reverse=True)
    if top_k is not None:
        out = out[:top_k]
    return out
