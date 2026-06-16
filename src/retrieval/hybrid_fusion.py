"""Reciprocal Rank Fusion and weighted hybrid retrieval fusion.

Both rankers return lists of (doc_id, score). We support two fusion modes:
  * **rrf** — Reciprocal Rank Fusion (no score normalization needed).
  * **weighted** — min-max normalise each score list and combine with weights.
"""

from __future__ import annotations

from collections.abc import Iterable


def rrf(rankings: Iterable[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over multiple ranked lists.

    Each input list must already be sorted best-first. The k constant damps
    the contribution of low-ranked hits.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _minmax(d: dict[str, float]) -> dict[str, float]:
    if not d:
        return {}
    vs = list(d.values())
    lo, hi = min(vs), max(vs)
    if hi - lo < 1e-9:
        return {k: 0.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def weighted(
    rankings: Iterable[list[tuple[str, float]]],
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Min-max normalise each ranking, then combine with weights."""
    rankings = list(rankings)
    if not rankings:
        return []
    if weights is None:
        weights = [1.0 / len(rankings)] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match rankings length")
    final: dict[str, float] = {}
    for r, w in zip(rankings, weights, strict=True):
        d = _minmax({doc_id: score for doc_id, score in r})
        for doc_id, s in d.items():
            final[doc_id] = final.get(doc_id, 0.0) + w * s
    return sorted(final.items(), key=lambda x: x[1], reverse=True)


def union_top_k(*rankings: list[tuple[str, float]], k: int = 500) -> list[tuple[str, float]]:
    """Take the union of the top-k from each ranking, return ranked by max score."""
    out: dict[str, float] = {}
    for r in rankings:
        for doc_id, s in r:
            if doc_id not in out or s > out[doc_id]:
                out[doc_id] = s
    return sorted(out.items(), key=lambda x: x[1], reverse=True)[:k]
