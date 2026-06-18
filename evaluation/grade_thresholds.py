"""Per-sub-score grade thresholds and human-readable descriptions.

Each sub-score is in [0, 1]. The thresholds below turn a sub-score
into a letter grade and a one-line description. Used by the run_evaluation
script to produce the per-component breakdown in FINAL_GRADE.md.
"""
from __future__ import annotations

# Per-sub-score grade thresholds. Same A/B/C/D/F as the composite, but
# each sub-score can have slightly different cutoffs (e.g. it's
# realistic for "system" to be A with sub-score 0.85 but "ranking" to
# need 0.90 to be A).
SUB_SCORE_THRESHOLDS: dict[str, list[tuple[float, str, str]]] = {
    "ranking_score": [
        (0.90, "A", "ranker matches ground truth closely"),
        (0.80, "B", "ranker is competitive"),
        (0.65, "C", "ranker is acceptable, room to improve"),
        (0.50, "D", "ranker is below the proxy baseline"),
        (0.00, "F", "ranker is no better than random"),
    ],
    "reasoning_score": [
        (0.85, "A", "all 6 Stage 4 checks largely pass"),
        (0.75, "B", "Stage 4 mostly pass; some improvements needed"),
        (0.60, "C", "Stage 4 partially pass; template/audit needs work"),
        (0.45, "D", "Stage 4 mostly fail; major rewrite needed"),
        (0.00, "F", "Stage 4 all fail"),
    ],
    "system_score": [
        (0.95, "A", "all tests pass, lint clean, spec-compliant, reproducible"),
        (0.85, "B", "all tests pass, minor lint warnings"),
        (0.70, "C", "most tests pass, some gaps"),
        (0.55, "D", "tests failing or build not reproducible"),
        (0.00, "F", "system broken"),
    ],
    "audit_score": [
        (0.90, "A", "architecture and docs are accurate and complete"),
        (0.80, "B", "minor gaps in architecture / docs"),
        (0.65, "C", "some gaps; reviewer can follow the design"),
        (0.50, "D", "major gaps"),
        (0.00, "F", "architecture is incoherent"),
    ],
}


def grade_sub_score(name: str, value: float) -> tuple[str, str]:
    """Return (letter, description) for a sub-score.

    `value` is in [0, 1]. If `name` is unknown, returns ("F", "no rubric").
    """
    for threshold, letter, desc in SUB_SCORE_THRESHOLDS.get(name, []):
        if value >= threshold:
            return letter, desc
    return "F", "no rubric"
