from __future__ import annotations

import math

from src.evaluation.ndcg import (
    average_precision,
    composite_score,
    dcg_at_k,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    ranking_correlation,
    recall_at_k,
)


def test_dcg_perfect_ranking():
    rels = [3, 2, 1, 0]
    dcg = dcg_at_k(rels, 4)
    assert dcg > 0
    assert math.isclose(ndcg_at_k(rels, 4), 1.0)


def test_ndcg_zero_ideal():
    assert ndcg_at_k([0, 0, 0], 3) == 0.0


def test_precision_at_k_threshold():
    rels = [1, 0, 1, 1]
    assert precision_at_k(rels, 4) == 0.75
    assert precision_at_k(rels, 2) == 0.5


def test_recall_at_k():
    assert recall_at_k([1, 0, 1, 0], total_relevant=2, k=4) == 1.0
    assert recall_at_k([1, 0, 0, 0], total_relevant=2, k=4) == 0.5


def test_map_perfect():
    assert math.isclose(average_precision([1, 1, 1]), 1.0)


def test_map_zero():
    assert average_precision([0, 0, 0]) == 0.0


def test_mrr_first_hit():
    assert mean_reciprocal_rank([0, 1, 0]) == 0.5
    assert mean_reciprocal_rank([1, 0, 0]) == 1.0


def test_hit_rate():
    assert hit_rate_at_k([0, 0, 1, 0], 4) == 1.0
    assert hit_rate_at_k([0, 0, 0, 0], 4) == 0.0


def test_composite_in_range():
    s = composite_score(0.5, 0.5, 0.5, 0.5)
    assert 0.0 <= s <= 1.0


def test_ranking_correlation_perfect():
    a = ["x", "y", "z"]
    b = ["x", "y", "z"]
    assert math.isclose(ranking_correlation(a, b), 1.0)


def test_ranking_correlation_reverse():
    a = ["x", "y", "z"]
    b = ["z", "y", "x"]
    assert math.isclose(ranking_correlation(a, b), -1.0)
