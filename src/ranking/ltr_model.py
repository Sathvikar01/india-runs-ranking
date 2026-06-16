"""LightGBM LambdaRank LTR model.

We wrap a LightGBM Booster with feature column metadata so we can save and
reload it cleanly. Synthetic-relevance labels are produced by the proxy
ground-truth builder; hard-negative mining is done by `src.training`.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.preprocessing.feature_engineer import categorical_columns, feature_columns


class LTRModel:
    """Thin LightGBM-Booster wrapper with stable feature ordering."""

    def __init__(self, booster: lgb.Booster | None = None, cat_columns: list[str] | None = None) -> None:
        self.booster = booster
        self.cat_columns = cat_columns or categorical_columns()
        self.feature_columns = feature_columns()

    @classmethod
    def train(
        cls,
        X: pd.DataFrame,
        y: np.ndarray,
        group: np.ndarray,
        cat_columns: list[str] | None = None,
        params: dict | None = None,
        num_boost_round: int = 600,
    ) -> "LTRModel":
        if cat_columns is None:
            cat_columns = categorical_columns()
        dtrain = lgb.Dataset(
            X,
            label=y,
            group=group,
            categorical_feature=cat_columns,
            free_raw_data=False,
        )
        default_params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [10, 50, 100],
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 5,
            "min_data_in_leaf": 20,
            "verbose": -1,
        }
        if params:
            default_params.update(params)
        booster = lgb.train(
            default_params,
            dtrain,
            num_boost_round=num_boost_round,
        )
        return cls(booster, cat_columns)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("LTRModel is not trained yet")
        return self.booster.predict(X)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(p))

    @classmethod
    def load(cls, path: str | Path) -> "LTRModel":
        booster = lgb.Booster(model_file=str(path))
        return cls(booster=booster)

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        if self.booster is None:
            raise RuntimeError("LTRModel is not trained yet")
        imp = self.booster.feature_importance(importance_type=importance_type)
        return pd.DataFrame({"feature": self.feature_columns, "importance": imp}).sort_values(
            "importance", ascending=False
        )
