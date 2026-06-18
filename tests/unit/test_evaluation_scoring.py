"""Tests for the evaluation scoring module."""
from __future__ import annotations

from evaluation.grade_thresholds import grade_sub_score
from evaluation.scoring import WEIGHTS, clip01, composite_score, letter_grade, mean


def test_letter_grade_a():
    assert letter_grade(95) == "A"
    assert letter_grade(90) == "A"


def test_letter_grade_b():
    assert letter_grade(85) == "B"
    assert letter_grade(80) == "B"


def test_letter_grade_c():
    assert letter_grade(75) == "C"
    assert letter_grade(70) == "C"


def test_letter_grade_d():
    assert letter_grade(65) == "D"
    assert letter_grade(60) == "D"


def test_letter_grade_f():
    assert letter_grade(59) == "F"
    assert letter_grade(0) == "F"
    assert letter_grade(-1) == "F"
    assert letter_grade(None) == "F"


def test_composite_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_composite_score_perfect():
    out = composite_score(
        {"ranking_score": 1.0, "reasoning_score": 1.0, "system_score": 1.0, "audit_score": 1.0}
    )
    assert out["score_0_100"] == 100.0
    assert out["grade"] == "A"


def test_composite_score_zero():
    out = composite_score(
        {"ranking_score": 0.0, "reasoning_score": 0.0, "system_score": 0.0, "audit_score": 0.0}
    )
    assert out["score_0_100"] == 0.0
    assert out["grade"] == "F"


def test_composite_score_weights_match():
    """Verify the weighted sub-scores add up correctly."""
    out = composite_score(
        {"ranking_score": 0.8, "reasoning_score": 0.6, "system_score": 1.0, "audit_score": 0.4}
    )
    expected = (0.40 * 0.8 + 0.30 * 0.6 + 0.20 * 1.0 + 0.10 * 0.4) * 100
    assert abs(out["score_0_100"] - round(expected, 2)) < 0.01


def test_composite_score_handles_missing_keys():
    """Missing keys default to 0 — the system is penalised for not measuring that axis."""
    out = composite_score({"ranking_score": 0.8})
    expected = 0.8 * 100  # only ranking_score; weight is renormalised to 1.0
    assert abs(out["score_0_100"] - expected) < 0.01


def test_composite_score_clamps_to_unit_interval():
    """Inputs > 1 or < 0 are clipped to [0, 1]."""
    out = composite_score(
        {"ranking_score": 2.0, "reasoning_score": -0.5, "system_score": 1.0, "audit_score": 0.0}
    )
    assert out["sub_scores"]["ranking_score"] == 1.0
    assert out["sub_scores"]["reasoning_score"] == 0.0


def test_clip01():
    assert clip01(0.5) == 0.5
    assert clip01(-0.1) == 0.0
    assert clip01(1.5) == 1.0
    assert clip01(0.0) == 0.0
    assert clip01(1.0) == 1.0


def test_mean():
    assert mean([1, 2, 3]) == 2.0
    assert mean([]) == 0.0
    assert mean([0.5]) == 0.5


def test_grade_sub_score_rubric_known():
    letter, desc = grade_sub_score("ranking_score", 0.92)
    assert letter == "A"
    assert "ranker" in desc.lower()


def test_grade_sub_score_unknown_returns_f():
    letter, desc = grade_sub_score("nonexistent_score", 0.99)
    assert letter == "F"
    assert "no rubric" in desc.lower()
