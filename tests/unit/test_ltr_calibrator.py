"""Tests for the LTR isotonic calibrator (WS-11)."""
from __future__ import annotations

import numpy as np

from src.ranking.ltr_calibrator import LTRCalibrator


def test_calibrator_fit_transform():
    """A perfect-positive linear relationship: x → y."""
    np.random.seed(0)
    x = np.linspace(-3, 3, 200)
    y = 1.0 / (1.0 + np.exp(-x))  # sigmoid
    cal = LTRCalibrator().fit(x, y * 4)  # y in [0,4] then divided by 4
    out = cal.transform(x)
    assert out.shape == x.shape
    assert (out >= 0.0).all() and (out <= 1.0).all()
    # Monotonic: x1 < x2 → out1 <= out2.
    assert all(out[i] <= out[i + 1] for i in range(len(out) - 1))


def test_calibrator_clip_at_zero_and_one():
    """Out-of-range x should be clipped, not extrapolated."""
    cal = LTRCalibrator().fit(np.array([-1.0, 0.0, 1.0]), np.array([0.0, 2.0, 4.0]))
    out = cal.transform(np.array([-100.0, 0.0, 100.0]))
    assert out[0] == 0.0
    assert out[2] == 1.0


def test_calibrator_save_load(tmp_path):
    """Round-trip via pickle."""
    cal = LTRCalibrator().fit(np.array([-1.0, 0.0, 1.0]), np.array([0.0, 2.0, 4.0]))
    p = tmp_path / "cal.pkl"
    cal.save(p)
    cal2 = LTRCalibrator.load(p)
    assert cal2.n_samples == cal.n_samples
    out1 = cal.transform(np.array([0.0]))
    out2 = cal2.transform(np.array([0.0]))
    np.testing.assert_allclose(out1, out2)


def test_calibrator_unfitted_raises():
    cal = LTRCalibrator()
    try:
        cal.transform(np.array([0.0]))
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError on unfitted transform")
