"""Tests for the multi-task LTR (Agent 1)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _toy_dataset(n: int = 200, seed: int = 0):
    """Build a small synthetic feature matrix + multi-task labels.

    Uses ONLY numeric columns so the test is small and deterministic.
    """
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(8)]
    df = pd.DataFrame(rng.random((n, len(cols))), columns=cols).astype(float)
    # Labels: simple linear combinations, then quantised to 0-4 (lambdarank
    # requires integer labels).
    raw_a = 0.7 * df["f0"] + 0.3 * df["f1"]
    raw_b = 0.5 * df["f2"] + 0.5 * df["f3"]
    y_a = np.minimum(4, np.maximum(0, np.floor(raw_a * 5).astype(int)))
    y_b = np.minimum(4, np.maximum(0, np.floor(raw_b * 5).astype(int)))
    n_groups = 4
    group = np.full(n_groups, n // n_groups, dtype=int)
    return df, y_a, y_b, group


def test_multitask_ltr_train_and_predict():
    pytest.importorskip("lightgbm")
    from src.ranking.ltr_multitask import MultiTaskLTR

    X, y_a, y_b, group = _toy_dataset()
    mt = MultiTaskLTR.train(
        X, y_a, y_b, group=group, num_boost_round=10, cat_columns=[],
    )
    preds = mt.predict(X)
    assert preds.shape == (len(X),)
    assert np.isfinite(preds).all()


def test_multitask_ltr_save_and_load(tmp_path: Path):
    pytest.importorskip("lightgbm")
    from src.ranking.ltr_multitask import MultiTaskLTR

    X, y_a, y_b, group = _toy_dataset()
    mt = MultiTaskLTR.train(
        X, y_a, y_b, group=group, num_boost_round=10,
        weight_a=0.7, weight_b=0.3, cat_columns=[],
    )
    mt.save(tmp_path / "mt")
    assert (tmp_path / "mt" / "ltr_multitask_a.cbm").exists()
    assert (tmp_path / "mt" / "ltr_multitask_b.cbm").exists()
    assert (tmp_path / "mt" / "ltr_multitask_meta.json").exists()

    meta = json.loads((tmp_path / "mt" / "ltr_multitask_meta.json").read_text())
    assert meta["weight_a"] == 0.7
    assert meta["weight_b"] == 0.3

    mt2 = MultiTaskLTR.load(tmp_path / "mt", cat_columns=[])
    assert mt2.weight_a == 0.7
    assert mt2.weight_b == 0.3
    p1 = mt.predict(X)
    p2 = mt2.predict(X)
    np.testing.assert_allclose(p1, p2, atol=1e-5)


def test_multitask_ltr_predict_per_head():
    pytest.importorskip("lightgbm")
    from src.ranking.ltr_multitask import MultiTaskLTR

    X, y_a, y_b, group = _toy_dataset()
    mt = MultiTaskLTR.train(
        X, y_a, y_b, group=group, num_boost_round=5, cat_columns=[],
    )
    a, b, c = mt.predict_per_head(X)
    assert a.shape == (len(X),)
    assert b.shape == (len(X),)
    assert c is None  # 2-head model has no c head


def test_multitask_ltr_weighted_average():
    pytest.importorskip("lightgbm")
    from src.ranking.ltr_multitask import MultiTaskLTR

    X, y_a, y_b, group = _toy_dataset()
    mt = MultiTaskLTR.train(
        X, y_a, y_b, group=group, num_boost_round=5, cat_columns=[],
    )
    a, b, _ = mt.predict_per_head(X)
    pred_default = mt.predict(X)
    np.testing.assert_allclose(pred_default, 0.5 * a + 0.5 * b, atol=1e-5)
    mt.weight_a, mt.weight_b = 0.7, 0.3
    pred_70 = mt.predict(X)
    np.testing.assert_allclose(pred_70, 0.7 * a + 0.3 * b, atol=1e-5)
