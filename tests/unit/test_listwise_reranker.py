"""Tests for the listwise top-K reranker (Agent 3)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _toy_dataset(n: int = 600, seed: int = 0):
    """Toy data with integer labels and groups of size 200."""
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(8)]
    df = pd.DataFrame(rng.random((n, len(cols))), columns=cols).astype(float)
    raw = 0.7 * df["f0"] + 0.3 * df["f1"]
    y = np.minimum(4, np.maximum(0, np.floor(raw * 5).astype(int)))
    return df, y


def test_listwise_topk_train_and_predict():
    pytest.importorskip("lightgbm")
    from src.ranking.listwise_reranker import ListwiseTopKReranker

    X, y = _toy_dataset()
    rk = ListwiseTopKReranker.train(X, y, num_boost_round=20, cat_columns=[])
    preds = rk.predict(X)
    assert preds.shape == (len(X),)
    assert np.isfinite(preds).all()


def test_listwise_topk_save_and_load(tmp_path: Path):
    pytest.importorskip("lightgbm")
    from src.ranking.listwise_reranker import ListwiseTopKReranker

    X, y = _toy_dataset()
    rk = ListwiseTopKReranker.train(X, y, num_boost_round=10, cat_columns=[])
    rk.save(tmp_path / "ltr_topk.cbm")
    assert (tmp_path / "ltr_topk.cbm").exists()
    rk2 = ListwiseTopKReranker.load(tmp_path / "ltr_topk.cbm", cat_columns=[])
    p1 = rk.predict(X)
    p2 = rk2.predict(X)
    np.testing.assert_allclose(p1, p2, atol=1e-5)


def test_rank_top_k_reorders_top_window():
    pytest.importorskip("lightgbm")
    from src.ranking.listwise_reranker import ListwiseTopKReranker, rank_top_k

    X, y = _toy_dataset(n=400)
    rk = ListwiseTopKReranker.train(X, y, num_boost_round=20, cat_columns=[])
    cands = X.to_dict("records")
    upstream = y.astype(float)
    out = rank_top_k(cands, upstream, rk, top_k_input=200, top_k_output=100, blend_weight=0.7)
    assert len(out) == 400
    # The top-200 after rerank must be non-increasing in score.
    top200_scores = [s for _, s in out[:200]]
    for i in range(len(top200_scores) - 1):
        assert top200_scores[i] >= top200_scores[i + 1] - 1e-9, (i, top200_scores[i], top200_scores[i + 1])
    # The output preserves all 400 candidates.
    assert len({id(c) for c, _ in out}) == 400


def test_rank_top_k_returns_same_set():
    pytest.importorskip("lightgbm")
    from src.ranking.listwise_reranker import ListwiseTopKReranker, rank_top_k

    X, y = _toy_dataset(n=300)
    rk = ListwiseTopKReranker.train(X, y, num_boost_round=10, cat_columns=[])
    cands = X.to_dict("records")
    upstream = y.astype(float)
    out = rank_top_k(cands, upstream, rk, top_k_input=200, top_k_output=100, blend_weight=0.5)
    # Same set of records, just reordered.
    out_ids = sorted(id(c) for c, _ in out)
    in_ids = sorted(id(c) for c in cands)
    assert out_ids == in_ids
