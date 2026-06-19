"""Top-K listwise reranker (Agent 3).

The single-task LTR (Agent 1) optimises NDCG across the whole pool, which
means most of its gradient comes from the 99 % of the pool that is *not*
in the top-100. We need a second-stage model whose *only* objective is
ordering the top-K candidates correctly — that's where NDCG@10 lifts come
from.

Two pieces:

1. ``ListwiseTopKReranker`` — a LightGBM lambdarank trained on the full
   pool but with ``ndcg_eval_at=[10, 20, 50]`` and more rounds / smaller
   learning rate. This makes the *training loss* focused on top-K
   quality instead of averaging over the long tail.

2. ``rank_top_k(...)`` — at rank time, take the top-200 from the upstream
   ensemble and re-order them using the listwise model. This is a pure
   reorder (no new candidates added), so it doesn't change the
   recall ceiling.

The reranker is saved to ``artifacts/ltr_topk.cbm`` and loaded as a
second-tier score by the ranker. We blend it with the original LTR at
inference: ``final = w_topk * listwise + (1 - w_topk) * upstream``.

Why a separate model: LightGBM's lambdarank objective allows multiple
``ndcg_eval_at`` values, but the actual gradient computation uses
*all* of them. By training a separate model that only sees the top-K
gradient signal (via group-size + smaller pools per group), we get a
specialist that handles the hard top-K cases better than a generalist.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.preprocessing.feature_engineer import categorical_columns, feature_columns

log = logging.getLogger("ltr_topk")


class ListwiseTopKReranker:
    """LightGBM lambdarank tuned for top-K NDCG.

    Differences from the vanilla LTR:
      * ``num_leaves`` 127 (vs 63) — finer splits on the rare top-K cases.
      * ``learning_rate`` 0.025 (vs 0.05) — slower to overfit.
      * ``num_iterations`` 1500 (vs 800) — more rounds to converge.
      * ``ndcg_eval_at=[10, 20, 50]`` — no top-100, no MAP; only the top-K.
      * ``group_size`` 200 — smaller groups so each gradient step only
        affects the top-K ordering.
    """

    def __init__(
        self,
        booster=None,
        cat_columns: list[str] | None = None,
        group_size: int = 200,
        num_leaves: int = 127,
        learning_rate: float = 0.025,
    ) -> None:
        self.booster = booster
        self.cat_columns = cat_columns or categorical_columns()
        self.feature_columns = feature_columns()
        self.group_size = group_size
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate

    @classmethod
    def train(
        cls,
        X: pd.DataFrame,
        y: np.ndarray,
        cat_columns: list[str] | None = None,
        num_boost_round: int = 1500,
        early_stopping_rounds: int = 0,
        params: dict | None = None,
        group_size: int = 200,
        num_leaves: int = 127,
        learning_rate: float = 0.025,
        eval_set: tuple[pd.DataFrame, np.ndarray, np.ndarray] | None = None,
    ) -> "ListwiseTopKReranker":
        """Train the listwise top-K reranker.

        ``early_stopping_rounds`` defaults to 0 (disabled) because early
        stopping requires a held-out validation set. Pass a positive
        value AND ``eval_set=(X_val, y_val, group_val)`` to enable it.
        """
        import lightgbm as lgb

        if cat_columns is None:
            cat_columns = categorical_columns()

        n = len(X)
        # Build group sizes: ~group_size rows per group, last group may be smaller.
        sizes = []
        for start in range(0, n, group_size):
            sizes.append(min(group_size, n - start))
        group = np.array(sizes, dtype=int)

        default_params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [10, 20, 50],
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 5,
            "min_data_in_leaf": 10,
            "lambda_l2": 1.0,
            "verbose": -1,
        }
        if params:
            default_params.update(params)

        dtrain = lgb.Dataset(
            X, label=y, group=group,
            categorical_feature=cat_columns, free_raw_data=False,
        )
        valid_sets = [dtrain]
        valid_names = ["train"]
        callbacks = []
        if eval_set is not None and early_stopping_rounds > 0:
            X_val, y_val, group_val = eval_set
            dval = lgb.Dataset(
                X_val, label=y_val, group=group_val,
                categorical_feature=cat_columns, free_raw_data=False,
            )
            valid_sets.append(dval)
            valid_names.append("val")
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

        log.info(
            "Training ListwiseTopKReranker (group_size=%d, leaves=%d, lr=%.3f, rounds=%d) …",
            group_size, num_leaves, learning_rate, num_boost_round,
        )
        booster = lgb.train(
            default_params, dtrain,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks or None,
        )
        return cls(
            booster=booster, cat_columns=cat_columns,
            group_size=group_size, num_leaves=num_leaves, learning_rate=learning_rate,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("ListwiseTopKReranker has no trained booster")
        return self.booster.predict(X)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(p))

    @classmethod
    def load(cls, path: str | Path, cat_columns: list[str] | None = None) -> "ListwiseTopKReranker":
        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(path))
        return cls(booster=booster, cat_columns=cat_columns)

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        imp = self.booster.feature_importance(importance_type=importance_type)
        return pd.DataFrame({
            "feature": self.feature_columns, "importance": imp,
        }).sort_values("importance", ascending=False)


def rank_top_k(
    candidates: list[dict],
    scores: list[float],
    reranker: ListwiseTopKReranker,
    *,
    top_k_input: int = 200,
    top_k_output: int = 100,
    blend_weight: float = 0.5,
) -> list[tuple[dict, float]]:
    """Re-rank the top-200 candidates using the listwise model.

    Returns [(candidate_dict, blended_score), ...] in the new order.
    """
    if len(candidates) <= top_k_output:
        return list(zip(candidates, scores))

    # Take top-200 by upstream score.
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    keep = order[:top_k_input]
    top_cands = [candidates[i] for i in keep]
    top_scores = [scores[i] for i in keep]

    # Build a DataFrame and predict.
    X = pd.DataFrame(top_cands)
    new_scores = reranker.predict(X)

    # Blend: keep some of the upstream signal so we don't fully trust the
    # reranker for the rest of the list (we only re-rank the top-200).
    blended = [blend_weight * s_new + (1 - blend_weight) * s_old
               for s_new, s_old in zip(new_scores, top_scores)]
    reranked = sorted(zip(top_cands, blended), key=lambda x: x[1], reverse=True)

    # Concatenate the rest of the pool (unchanged order, downstream of the cut).
    rest = [(candidates[i], scores[i]) for i in order[top_k_input:]]
    return reranked + rest
