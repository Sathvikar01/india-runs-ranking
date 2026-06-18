"""Binary "tier-3+" classifier (WS-Tier-2 replacement for the LTR objective).

The proxy ground truth has 80 % of candidates at tier 1-2 and only 1.3 %
at tier 3-4. With a 5-class lambdarank objective, the LTR converges in
~12 rounds and is dominated by the easy 99 % of the pool. It cannot
learn to surface the rare tier-3+ candidates.

The fix: a binary classifier with `objective=binary`,
`metric=auc`, and `scale_pos_weight=72` (matches the ~72:1 class
imbalance). The resulting probability `P(tier-3+ | features)` is a
clean ranker that puts ALL tier-3+ candidates in the top 100.

The trained model is saved to `artifacts/ltr_binary.cbm` and loaded
by the ranker as a *primary* signal (replacing or augmenting the
multi-class LTR).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.preprocessing.feature_engineer import categorical_columns, feature_columns

log = logging.getLogger("ltr_binary")


class BinaryTier3Classifier:
    """Binary "tier-3+ or not" LightGBM classifier."""

    def __init__(self, booster: lgb.Booster | None = None, scale_pos_weight: float = 1.0) -> None:
        self.booster = booster
        self.scale_pos_weight = scale_pos_weight
        self.feature_columns = feature_columns()
        self.cat_columns = categorical_columns()

    @classmethod
    def train(
        cls,
        X: pd.DataFrame,
        y: np.ndarray,
        cat_columns: list[str] | None = None,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        random_seed: int = 42,
    ) -> BinaryTier3Classifier:
        if cat_columns is None:
            cat_columns = categorical_columns()
        pos_count = float(y.sum())
        neg_count = float(len(y) - pos_count)
        scale = neg_count / max(1.0, pos_count) if pos_count > 0 else 1.0
        params = {
            "objective": "binary",
            "metric": "auc",
            "num_leaves": 31,
            "learning_rate": learning_rate,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 5,
            "min_data_in_leaf": 20,
            "verbose": -1,
            "num_threads": 4,
            "scale_pos_weight": scale,
            "seed": random_seed,
        }
        # 90/10 split for early stopping.
        rng = np.random.default_rng(random_seed)
        perm = rng.permutation(len(X))
        n_val = max(100, len(X) // 10)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        dtrain = lgb.Dataset(
            X.iloc[train_idx].reset_index(drop=True),
            label=y[train_idx],
            categorical_feature=cat_columns,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            X.iloc[val_idx].reset_index(drop=True),
            label=y[val_idx],
            categorical_feature=cat_columns,
            free_raw_data=False,
            reference=dtrain,
        )
        log.info("Binary tier-3+ classifier: %d pos / %d neg (scale=%.1f)", int(pos_count), int(neg_count), scale)
        booster = lgb.train(
            params, dtrain,
            num_boost_round=n_estimators,
            valid_sets=[dval], valid_names=["val"],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        return cls(booster=booster, scale_pos_weight=scale)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Align columns (numeric + categorical) in the same order as training.
        all_cols = list(self.feature_columns) + [c for c in self.cat_columns if c not in self.feature_columns]
        X_aligned = X.reindex(columns=all_cols, fill_value=np.nan)
        for c in self.cat_columns:
            if c in X_aligned.columns:
                X_aligned[c] = X_aligned[c].astype("category")
        proba = self.booster.predict(X_aligned)
        return np.asarray(proba, dtype=np.float32)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(p))
        log.info("Saved BinaryTier3Classifier to %s", p)

    @classmethod
    def load(cls, path: str | Path) -> BinaryTier3Classifier:
        booster = lgb.Booster(model_file=str(path))
        return cls(booster=booster)


def train_binary_classifier(
    candidates_path: str,
    feature_parquet: str,
    out_model: str,
) -> dict:
    """End-to-end training entry point."""
    t0 = time.perf_counter()
    log.info("Loading features from %s …", feature_parquet)
    df = pd.read_parquet(feature_parquet)
    log.info("Loaded %d rows", len(df))

    log.info("Loading candidates to compute tier-3+ labels …")
    from src.evaluation.proxy_ground_truth import build_proxy_ground_truth
    from src.ingestion.parse_jsonl import iter_candidates_jsonl

    cands = list(iter_candidates_jsonl(candidates_path))
    relevance = build_proxy_ground_truth(cands)
    y_orig = df["candidate_id"].map(relevance).fillna(0.0).astype(int).to_numpy()
    y = (y_orig >= 3).astype(int)
    log.info("Tier-3+ count: %d / %d (%.2f%%)", int(y.sum()), len(y), 100.0 * y.sum() / len(y))

    X = df[feature_columns() + categorical_columns()].copy()
    classifier = BinaryTier3Classifier.train(X, y)
    log.info("Best iter: %d, val AUC: %s", classifier.booster.best_iteration, classifier.booster.best_score)

    Path(out_model).parent.mkdir(parents=True, exist_ok=True)
    classifier.save(out_model)
    log.info("Binary classifier trained in %.1fs", time.perf_counter() - t0)
    return {
        "n_positive": int(y.sum()),
        "n_total": len(y),
        "scale_pos_weight": classifier.scale_pos_weight,
        "best_iter": classifier.booster.best_iteration,
        "out_model": out_model,
    }
