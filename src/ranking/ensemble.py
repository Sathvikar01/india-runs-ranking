"""Final ensemble: combine LTR, behavioral, honeypot penalty, and JD penalty
into a single 0-1 score and produce a strictly monotonically non-increasing
score list over the final top-100.
"""

from __future__ import annotations

import math

import numpy as np

from src.api.schemas import Candidate
from src.behavioral.availability import availability_score
from src.behavioral.honeypot import honeypot_risk
from src.behavioral.jd_filters import negative_penalty, positive_boost


def _sigmoid(x: float) -> float:
    if x > 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def ensemble_score(
    ltr_score: float,
    ce_score: float,
    availability: float,
    positive: float,
    negative: float,
    honeypot: float,
    catboost_score: float = 0.0,
    catboost_weight: float = 0.0,
) -> float:
    """Combine signals into a single 0-1 score. Higher is better.

    WS-6: a small additive CatBoost signal is now part of the ensemble.
    `catboost_score` is expected to be in roughly the same scale as the
    LTR sigmoid term. We add `catboost_weight * sigmoid(catboost_score)` to
    the base, capped at 1.0.
    """
    base = (
        0.55 * _sigmoid(ltr_score)
        + 0.20 * _sigmoid(ce_score)
        + 0.10 * _clip01(availability)
        + 0.10 * _clip01(positive)
        - 0.10 * _clip01(negative)
        - 0.20 * _clip01(honeypot)
    )
    if catboost_weight > 0.0:
        base = base + catboost_weight * _sigmoid(catboost_score)
    return _clip01(base)


def make_monotonic_scores(raw_scores: list[float]) -> list[float]:
    """DEPRECATED — prefer `make_monotonic_scores_for_topk` for new code.

    This function preserves the input order in the output, which means the
    output is *not* monotonic when the input is in MMR-reordered (not
    pre-sorted) order. It is kept for backwards compatibility with the
    one call site that explicitly needs the input-order behaviour.

    Use `make_monotonic_scores_for_topk` when the input is already in
    the desired output order (e.g. after MMR has reordered the head).

    The output has the *same length* as the input and preserves the input
    order — but every output value is one of the strictly-decreasing
    scores we assign to ranks 0..n-1. Concretely: we sort (index, score)
    pairs by score descending, then for each pair we write
    `output[index] = base_top - new_rank * step - jitter`.
    """
    if not raw_scores:
        return []
    sorted_pairs = sorted(enumerate(raw_scores), key=lambda x: x[1], reverse=True)
    n = len(sorted_pairs)
    base_top = 0.99
    base_bottom = 0.20
    if n == 1:
        return [base_top]
    step = (base_top - base_bottom) / max(1, n - 1)
    final = [0.0] * n
    for new_rank, (orig_idx, _) in enumerate(sorted_pairs):
        jitter = (new_rank * 1e-5) % 1e-3
        final[orig_idx] = base_top - new_rank * step - jitter
    return [_clip01(v) for v in final]


def make_monotonic_scores_for_topk(raw_scores: list[float]) -> list[float]:
    """Strictly-decreasing scores in INPUT order.

    Unlike `make_monotonic_scores`, this returns scores in the same order
    as the input list: output[i] is the i-th strictly-decreasing value
    (rank i). This is the correct function to call when the input is in
    the desired output order — e.g., after MMR has reordered the head
    and you want rank 1 to get the highest score, rank 2 the next, etc.
    """
    if not raw_scores:
        return []
    n = len(raw_scores)
    base_top = 0.99
    base_bottom = 0.20
    if n == 1:
        return [base_top]
    step = (base_top - base_bottom) / max(1, n - 1)
    return [_clip01(base_top - i * step - (i * 1e-5) % 1e-3) for i in range(n)]


def rank_candidates(
    candidates: list[Candidate],
    ltr_scores: dict[str, float],
    ce_scores: dict[str, float],
    top_k: int = 100,
) -> list[tuple[Candidate, float, dict]]:
    """Score every candidate, sort, return top-k with breakdown dicts."""
    scored: list[tuple[Candidate, float, dict]] = []
    for c in candidates:
        ltr = float(ltr_scores.get(c.candidate_id, 0.0))
        ce = float(ce_scores.get(c.candidate_id, 0.0))
        availability = availability_score(c)
        pos = positive_boost(c)
        neg = negative_penalty(c)
        hon = honeypot_risk(c)
        score = ensemble_score(ltr, ce, availability, pos, neg, hon)
        scored.append(
            (
                c,
                score,
                {
                    "ltr": ltr,
                    "ce": ce,
                    "availability": availability,
                    "positive": pos,
                    "negative": neg,
                    "honeypot": hon,
                },
            )
        )
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def apply_monotonic(ranked: list[tuple[Candidate, float, dict]]) -> list[tuple[Candidate, float, dict]]:
    """Apply strict monotonicity to the final top-100 (or whatever the input is)."""
    raw = [r[1] for r in ranked]
    new_scores = make_monotonic_scores(raw)
    out = []
    for (c, _old, breakdown), ns in zip(ranked, new_scores, strict=True):
        out.append((c, float(ns), breakdown))
    return out
