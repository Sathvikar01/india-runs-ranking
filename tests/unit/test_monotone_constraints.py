"""Tests for the monotone-constraint helper."""
from __future__ import annotations

from src.training.train_ltr import _build_monotone_constraints


def test_monotone_constraints_default_zero():
    """Columns not in the YAML get 0 (no constraint)."""
    cols = ["n_named_jd_skills_continuous", "education_prestige", "unknown_col"]
    out = _build_monotone_constraints(cols)
    assert out == [1, 1, 0]


def test_monotone_constraints_respects_yaml():
    """The helper reads configs/ranking.yaml's monotone_constraints block."""
    cols = ["education_prestige", "consulting_share", "honeypot"]
    out = _build_monotone_constraints(cols)
    assert out == [1, -1, -1]


def test_monotone_constraints_length_matches_input():
    cols = ["a", "b", "c", "d", "e"]
    out = _build_monotone_constraints(cols)
    assert len(out) == len(cols)
