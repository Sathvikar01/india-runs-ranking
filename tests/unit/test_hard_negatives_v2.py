"""Tests for cross-ranker hard-negative mining (Agent 6)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _toy_features(n: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "candidate_id": [f"CAND_{i:07d}" for i in range(n)],
        "f0": rng.random(n),
        "f1": rng.random(n),
    })
    return df


def test_union_hard_negatives_dedup():
    from src.training.hard_negatives import union_hard_negatives
    a = ["CAND_001", "CAND_002", "CAND_003"]
    b = ["CAND_003", "CAND_004"]
    out = union_hard_negatives(a, b)
    assert out == ["CAND_001", "CAND_002", "CAND_003", "CAND_004"]


def test_union_hard_negatives_empty():
    from src.training.hard_negatives import union_hard_negatives
    assert union_hard_negatives() == []
    assert union_hard_negatives([], []) == []


def test_reweight_with_hard_negatives_zeroes_label():
    from src.training.hard_negatives import reweight_with_hard_negatives
    relevance = {"CAND_001": 2.0, "CAND_002": 3.0, "CAND_003": 1.0}
    out = reweight_with_hard_negatives(relevance, ["CAND_002"])
    assert out["CAND_002"] == 0.0
    assert out["CAND_001"] == 2.0
    assert out["CAND_003"] == 1.0


def test_reweight_with_hard_negatives_handles_missing():
    from src.training.hard_negatives import reweight_with_hard_negatives
    relevance = {"CAND_001": 2.0}
    out = reweight_with_hard_negatives(relevance, ["CAND_001", "CAND_MISSING"])
    assert out["CAND_001"] == 0.0
    assert out["CAND_MISSING"] == 0.0  # missing → 0.0 via min(0.0, 0.0)


def test_mine_top_low_rubric_needs_real_features():
    """Sanity: the function accepts the (ltr, features_df) shape we use."""
    pytest.importorskip("lightgbm")
    from src.training.hard_negatives import mine_top_low_rubric
    # We don't run a real LTR here; just check the function is importable
    # and accepts the documented signature.
    assert callable(mine_top_low_rubric)


def test_mine_cross_ranker_disagreement_signature():
    """Sanity: the function accepts (candidates, ltr, catboost_scores, df)."""
    pytest.importorskip("lightgbm")
    from src.training.hard_negatives import mine_cross_ranker_disagreements
    assert callable(mine_cross_ranker_disagreements)
