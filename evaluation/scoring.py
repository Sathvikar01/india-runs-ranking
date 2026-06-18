"""Composite-score formula for the end-to-end evaluation.

Defines a single 0-100 score with a letter grade, broken down into
four sub-scores:

    composite = 0.40 * ranking_score      (NDCG@10/50, MAP, P@10 vs proxy + eval-rubric)
             + 0.30 * reasoning_score   (Stage 4 checks: specific facts, JD connection,
                                          honest concerns, no hallucination, variation,
                                          rank consistency)
             + 0.20 * system_score       (tests, lint, output spec, monotonicity, build)
             + 0.10 * audit_score        (architecture + docs)

Letter grade: A >= 90, B >= 80, C >= 70, D >= 60, F < 60.

Every sub-score is in [0.0, 1.0] before the final * 100.
"""
from __future__ import annotations

from collections.abc import Iterable

GRADE_THRESHOLDS: list[tuple[float, str]] = [
    (90.0, "A"),
    (80.0, "B"),
    (70.0, "C"),
    (60.0, "D"),
    (0.0, "F"),
]

# Component weights — sum to 1.0.
WEIGHTS: dict[str, float] = {
    "ranking_score": 0.40,
    "reasoning_score": 0.30,
    "system_score": 0.20,
    "audit_score": 0.10,
}


def letter_grade(score_0_100: float) -> str:
    """Map a 0-100 score to a letter grade A/B/C/D/F."""
    if score_0_100 is None or score_0_100 < 0:
        return "F"
    for threshold, letter in GRADE_THRESHOLDS:
        if score_0_100 >= threshold:
            return letter
    return "F"


def clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def composite_score(sub_scores: dict[str, float]) -> dict:
    """Compute the weighted composite + letter grade.

    `sub_scores` is a dict of {component_name: 0..1 score}. Missing
    components default to 0 *and* the weights are renormalised so the
    system isn't penalised for not measuring something. If all components
    are missing, the score is 0.0 (F).
    """
    total = 0.0
    used_weights = 0.0
    breakdown: dict[str, float] = {}
    for name, weight in WEIGHTS.items():
        v = sub_scores.get(name)
        if v is None:
            # Missing → not measured. Don't penalise; renormalise.
            continue
        v = clip01(float(v))
        breakdown[name] = v
        total += weight * v
        used_weights += weight
    if used_weights <= 0:
        return {
            "score_0_100": 0.0,
            "score_0_1": 0.0,
            "grade": "F",
            "weights": dict(WEIGHTS),
            "sub_scores": {},
            "sub_scores_weighted": {},
        }
    final_0_1 = total / used_weights
    final_0_100 = round(100.0 * final_0_1, 2)
    return {
        "score_0_100": final_0_100,
        "score_0_1": round(final_0_1, 4),
        "grade": letter_grade(final_0_100),
        "weights": dict(WEIGHTS),
        "sub_scores": {k: round(v, 4) for k, v in breakdown.items()},
        "sub_scores_weighted": {k: round(WEIGHTS[k] * breakdown[k], 4) for k in breakdown},
    }


def mean(values: Iterable[float]) -> float:
    vs = list(values)
    return sum(vs) / len(vs) if vs else 0.0
