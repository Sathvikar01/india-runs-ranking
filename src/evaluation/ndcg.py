"""Ranking-quality metrics.

The challenge composite is `0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`.
We implement each component so we can compute it locally against our proxy
ground truth.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    """Discounted Cumulative Gain at k. 0-indexed relevances list."""
    s = 0.0
    for i, rel in enumerate(relevances[:k]):
        # Use the canonical 2^rel - 1 gain formula.
        s += (2 ** rel - 1) / math.log2(i + 2)
    return s


def ndcg_at_k(relevances: Sequence[float], k: int) -> float:
    """Normalised DCG at k. Returns 0 if no positive relevances exist."""
    ideal = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(relevances, k) / idcg


def precision_at_k(relevances: Sequence[float], k: int, threshold: float = 0.0) -> float:
    """Fraction of top-k with relevance > threshold."""
    if k == 0:
        return 0.0
    return sum(1 for r in relevances[:k] if r > threshold) / k


def recall_at_k(relevances: Sequence[float], total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return sum(1 for r in relevances[:k] if r > 0) / total_relevant


def average_precision(relevances: Sequence[float], threshold: float = 0.0) -> float:
    """Mean Average Precision for a single query."""
    if not relevances:
        return 0.0
    num_relevant = sum(1 for r in relevances if r > threshold)
    if num_relevant == 0:
        return 0.0
    score = 0.0
    hits = 0
    for i, r in enumerate(relevances, 1):
        if r > threshold:
            hits += 1
            score += hits / i
    return score / num_relevant


def mean_reciprocal_rank(relevances: Sequence[float], threshold: float = 0.0) -> float:
    for i, r in enumerate(relevances, 1):
        if r > threshold:
            return 1.0 / i
    return 0.0


def hit_rate_at_k(relevances: Sequence[float], k: int, threshold: float = 0.0) -> float:
    return float(any(r > threshold for r in relevances[:k]))


def composite_score(ndcg_10: float, ndcg_50: float, map_: float, p_10: float) -> float:
    return 0.50 * ndcg_10 + 0.30 * ndcg_50 + 0.15 * map_ + 0.05 * p_10


def ranking_correlation(predicted: list[str], ground_truth: list[str]) -> float:
    """Spearman-like rank correlation between two orderings of the same items."""
    pred_pos = {x: i for i, x in enumerate(predicted)}
    gt_pos = {x: i for i, x in enumerate(ground_truth)}
    common = set(pred_pos) & set(gt_pos)
    if not common:
        return 0.0
    n = len(common)
    d2 = sum((pred_pos[x] - gt_pos[x]) ** 2 for x in common)
    return 1.0 - 6.0 * d2 / (n * (n * n - 1)) if n > 1 else 0.0


def evaluate_ranking(
    predicted_ids: list[str],
    relevance: dict[str, float],
    k_list: tuple[int, ...] = (5, 10, 50, 100),
) -> dict[str, float]:
    """Compute the standard ranking metrics for one query."""
    rels = [relevance.get(cid, 0.0) for cid in predicted_ids]
    out: dict[str, float] = {}
    for k in k_list:
        out[f"ndcg@{k}"] = ndcg_at_k(rels, k)
        out[f"p@{k}"] = precision_at_k(rels, k)
    out["map"] = average_precision(rels)
    out["mrr"] = mean_reciprocal_rank(rels)
    out["composite"] = composite_score(
        ndcg_at_k(rels, 10),
        ndcg_at_k(rels, 50),
        average_precision(rels),
        precision_at_k(rels, 10),
    )
    return out
