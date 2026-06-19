"""Final ensemble: combine LTR, behavioral, honeypot penalty, and JD penalty
into a single 0-1 score and produce a strictly monotonically non-increasing
score list over the final top-100.

Agent 7 update: the ensemble weights are now configurable via
``EnsembleWeights``. ``search_ensemble_weights.py`` runs a Bayesian /
coordinate-descent search over the dev split to find weights that
maximise min(proxy, eval_rubric). The best weights are saved to
``artifacts/best_ensemble_weights.json``.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from src.api.schemas import Candidate
from src.behavioral.availability import availability_score
from src.behavioral.honeypot import honeypot_risk
from src.behavioral.jd_filters import negative_penalty, positive_boost


@dataclass
class EnsembleWeights:
    """Configurable weights for the final ensemble.

    Defaults match the original hard-coded ensemble so existing call sites
    keep the same behaviour unless they explicitly load new weights.
    """

    w_ltr: float = 0.55
    w_ce: float = 0.20
    w_avail: float = 0.10
    w_positive: float = 0.10
    w_negative: float = 0.10       # subtracted
    w_honeypot: float = 0.20       # subtracted
    w_catboost: float = 0.0        # optional extra head
    w_multitask: float = 0.0       # optional multi-task LTR head (Agent 1)
    w_topk: float = 0.0            # optional top-K reranker head (Agent 3)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "EnsembleWeights":
        d = json.loads(s)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EnsembleWeights":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


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

    Backwards-compatible signature; new code should call
    ``ensemble_score_v2`` with an ``EnsembleWeights`` instance.
    """
    weights = EnsembleWeights(w_catboost=catboost_weight)
    return ensemble_score_v2(
        ltr_score=ltr_score, ce_score=ce_score,
        availability=availability, positive=positive, negative=negative,
        honeypot=honeypot,
        catboost_score=catboost_score,
        weights=weights,
    )


def ensemble_score_v2(
    ltr_score: float,
    ce_score: float,
    availability: float,
    positive: float,
    negative: float,
    honeypot: float,
    weights: EnsembleWeights,
    catboost_score: float = 0.0,
    multitask_score: float = 0.0,
    topk_score: float = 0.0,
) -> float:
    """Combine signals into a single 0-1 score using ``EnsembleWeights``.

    Each signal is normalised to [0, 1] before being weighted. Negative
    weights subtract (honeypot, negative JD penalty). The result is
    clipped to [0, 1].

    New heads (multitask, topk) start at weight 0; the grid-search
    (``scripts/search_ensemble_weights.py``) sets them based on dev
    performance.
    """
    base = (
        weights.w_ltr * _sigmoid(ltr_score)
        + weights.w_ce * _sigmoid(ce_score)
        + weights.w_avail * _clip01(availability)
        + weights.w_positive * _clip01(positive)
        - weights.w_negative * _clip01(negative)
        - weights.w_honeypot * _clip01(honeypot)
    )
    if weights.w_catboost > 0.0 and catboost_score != 0.0:
        base = base + weights.w_catboost * _sigmoid(catboost_score)
    if weights.w_multitask > 0.0 and multitask_score != 0.0:
        base = base + weights.w_multitask * _sigmoid(multitask_score)
    if weights.w_topk > 0.0 and topk_score != 0.0:
        base = base + weights.w_topk * topk_score  # already in 0-1
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
