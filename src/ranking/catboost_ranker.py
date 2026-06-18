"""CatBoost YetiRank second ranker (WS-6).

A second gradient-boosted tree model trained with CatBoost's YetiRank
objective. Used as a *diversity* ensemble member with the LightGBM
LambdaRank model: the ranker that disagrees more with LightGBM on hard
cases is more useful as a tiebreaker.

The class follows the same `LTRModel`-style API: a stable `predict` and a
`save`/`load` pair. Optional — if CatBoost is not installed, the loader
raises a clear error and the ranker falls back to LightGBM-only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.feature_engineer import categorical_columns, feature_columns


class CatBoostRanker:
    """Thin CatBoost YetiRank wrapper with stable feature ordering."""

    def __init__(self, model, cat_columns: list[str] | None = None) -> None:
        self.model = model
        self.cat_columns = cat_columns or categorical_columns()
        self.feature_columns = feature_columns()

    @classmethod
    def train(
        cls,
        X: pd.DataFrame,
        y: np.ndarray,
        group: np.ndarray,
        cat_columns: list[str] | None = None,
        iterations: int = 500,
        learning_rate: float = 0.05,
        depth: int = 6,
        random_seed: int = 42,
    ) -> CatBoostRanker:
        try:
            from catboost import CatBoostRanker, Pool  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "CatBoost is not installed. Run `pip install catboost` "
                "(or `pip install -e \".[catboost]\"`)."
            ) from e
        if cat_columns is None:
            cat_columns = categorical_columns()
        # Order by group for CatBoost's group-wise objective.
        order = np.argsort(group, kind="stable")
        X_ord = X.iloc[order].reset_index(drop=True)
        y_ord = y[order]
        group_ord = group[order]
        pool = Pool(
            data=X_ord,
            label=y_ord,
            group_id=group_ord,
            cat_features=cat_columns,
        )
        model = CatBoostRanker(
            loss_function="YetiRank",
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=random_seed,
            verbose=False,
        )
        model.fit(pool)
        return cls(model, cat_columns=cat_columns)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # CatBoost expects the same column set AND order as during training,
        # which includes the categorical columns. The class stores only the
        # numeric feature_columns; the categorical ones are added on top.
        all_cols = list(self.feature_columns) + [c for c in self.cat_columns if c not in self.feature_columns]
        X_aligned = X.reindex(columns=all_cols, fill_value=np.nan)
        # Cast categorical columns to str (CatBoost dislikes mixed types).
        for c in self.cat_columns:
            if c in X_aligned.columns:
                X_aligned[c] = X_aligned[c].astype(str)
        scores = self.model.predict(X_aligned)
        return np.asarray(scores, dtype=np.float32)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(p))

    @classmethod
    def load(cls, path: str | Path, cat_columns: list[str] | None = None) -> CatBoostRanker:
        try:
            from catboost import CatBoostRanker  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "CatBoost is not installed. Run `pip install catboost`."
            ) from e
        m = CatBoostRanker()
        m.load_model(str(path))
        return cls(m, cat_columns=cat_columns)
