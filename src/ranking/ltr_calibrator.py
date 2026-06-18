"""Isotonic calibration for LTR scores (WS-11).

LightGBM LambdaRank scores are unbounded real numbers whose distribution
depends on the model and the input mix. The ensemble combines them with
sigmoids of other scores, so an unbounded score can drown out everything
else. We fit an `IsotonicRegression` on (LTR score → proxy relevance) at
build time, save the calibrator alongside the LTR model, and apply it at
rank time so the LTR contribution to the ensemble is in [0, 1] and
well-calibrated.

The calibrator is OPTIONAL — if the artifact isn't present, the ranker
falls back to `sigmoid(ltr_score)` as before.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

log = logging.getLogger("ltr_calibrator")


class LTRCalibrator:
    """Wraps an `IsotonicRegression` fit on (ltr_score, proxy_relevance) pairs."""

    def __init__(self, model: IsotonicRegression | None = None, n_samples: int = 0) -> None:
        self.model = model
        self.n_samples = n_samples

    def fit(self, ltr_scores: np.ndarray, proxy_relevance: np.ndarray) -> "LTRCalibrator":
        """Fit the calibrator. We normalise proxy_relevance to [0, 1] first.

        `proxy_relevance` is 0-4; we divide by 4.
        """
        y = np.asarray(proxy_relevance, dtype=np.float32) / 4.0
        # Clip to [0, 1] defensively.
        y = np.clip(y, 0.0, 1.0)
        x = np.asarray(ltr_scores, dtype=np.float32)
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(x, y)
        self.model = ir
        self.n_samples = int(len(x))
        return self

    def transform(self, ltr_scores: np.ndarray) -> np.ndarray:
        """Map raw LTR scores to [0, 1] via the fitted isotonic regression."""
        if self.model is None:
            raise RuntimeError("LTRCalibrator is not fitted")
        x = np.asarray(ltr_scores, dtype=np.float32)
        return np.clip(self.model.predict(x), 0.0, 1.0)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump({"model": self.model, "n_samples": self.n_samples}, f)
        log.info("Saved LTR calibrator to %s (n=%d)", p, self.n_samples)

    @classmethod
    def load(cls, path: str | Path) -> "LTRCalibrator":
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        return cls(model=obj["model"], n_samples=obj.get("n_samples", 0))

    @classmethod
    def fit_and_save(
        cls,
        ltr_scores: np.ndarray,
        proxy_relevance: np.ndarray,
        out_path: str | Path,
    ) -> "LTRCalibrator":
        cal = cls().fit(ltr_scores, proxy_relevance)
        cal.save(out_path)
        return cal
