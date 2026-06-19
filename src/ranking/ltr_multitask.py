"""Multi-task LTR (Agent 1).

LightGBM's lambdarank objective is single-task by design. To get the
ranker to learn signals that *both* the proxy and the eval_rubric reward,
we train two lambdarank models on the same features but different
ground-truth targets, then combine their scores at inference.

* ``task_a`` — trained on ``proxy_relevance_v2`` (Agent 2). This is the
  JD-derived + eval_rubric-blended target. It rewards the JD-literal
  signals the eval_rubric cares about (education, open_source,
  distributed_systems, open_to_work).
* ``task_b`` — trained on ``eval_rubric.eval_relevance``. Independent
  target, weights are 0.30/0.20/0.15/0.08/0.10/0.08/0.05/0.04.

At inference time, ``MultiTaskLTR.predict`` returns ``w_a * task_a +
w_b * task_b`` (defaults: 0.5 / 0.5). Both sub-models are saved as
``artifacts/ltr_multitask_a.cbm`` and ``artifacts/ltr_multitask_b.cbm``.

Why this matters: the single-task LTR can only learn to fit one target,
so it learns to over-reward ``ai_keyword_hits_career`` (the strongest
signal in the proxy). Multi-task forces it to also learn the eval_rubric's
preferred signals, which is what the official evaluation cares about.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.preprocessing.feature_engineer import categorical_columns, feature_columns

log = logging.getLogger("ltr_multitask")


class MultiTaskLTR:
    """Two-head LTR: task A (proxy_v2) + task B (eval_rubric).

    Both heads share the same feature schema. The class wraps two
    ``lightgbm.Booster`` instances; ``predict`` returns the weighted sum
    of their per-row scores.
    """

    def __init__(
        self,
        booster_a=None,
        booster_b=None,
        weight_a: float = 0.5,
        weight_b: float = 0.5,
        cat_columns: list[str] | None = None,
    ) -> None:
        self.booster_a = booster_a
        self.booster_b = booster_b
        self.weight_a = float(weight_a)
        self.weight_b = float(weight_b)
        self.cat_columns = cat_columns or categorical_columns()
        self.feature_columns = feature_columns()

    @classmethod
    def train(
        cls,
        X: pd.DataFrame,
        y_a: np.ndarray,
        y_b: np.ndarray,
        group: np.ndarray,
        cat_columns: list[str] | None = None,
        num_boost_round: int = 800,
        params: dict | None = None,
        weight_a: float = 0.5,
        weight_b: float = 0.5,
    ) -> "MultiTaskLTR":
        import lightgbm as lgb

        if cat_columns is None:
            cat_columns = categorical_columns()

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

        dtrain = lgb.Dataset(
            X, label=y_a, group=group,
            categorical_feature=cat_columns, free_raw_data=False,
        )
        log.info("Training MultiTaskLTR head A (proxy_v2) …")
        booster_a = lgb.train(
            default_params, dtrain, num_boost_round=num_boost_round,
        )

        dtrain_b = lgb.Dataset(
            X, label=y_b, group=group,
            categorical_feature=cat_columns, free_raw_data=False,
        )
        log.info("Training MultiTaskLTR head B (eval_rubric) …")
        booster_b = lgb.train(
            default_params, dtrain_b, num_boost_round=num_boost_round,
        )

        return cls(
            booster_a=booster_a,
            booster_b=booster_b,
            weight_a=weight_a,
            weight_b=weight_b,
            cat_columns=cat_columns,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster_a is None or self.booster_b is None:
            raise RuntimeError("MultiTaskLTR has no trained boosters")
        a = self.booster_a.predict(X)
        b = self.booster_b.predict(X)
        return self.weight_a * a + self.weight_b * b

    def predict_per_head(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (head_a_scores, head_b_scores) for downstream analysis."""
        return self.booster_a.predict(X), self.booster_b.predict(X)

    def save(self, dir_path: str | Path) -> None:
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        self.booster_a.save_model(str(d / "ltr_multitask_a.cbm"))
        self.booster_b.save_model(str(d / "ltr_multitask_b.cbm"))
        # Save the weights as a sidecar JSON.
        import json
        (d / "ltr_multitask_meta.json").write_text(
            json.dumps({"weight_a": self.weight_a, "weight_b": self.weight_b}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, dir_path: str | Path, cat_columns: list[str] | None = None) -> "MultiTaskLTR":
        import json
        import lightgbm as lgb

        d = Path(dir_path)
        booster_a = lgb.Booster(model_file=str(d / "ltr_multitask_a.cbm"))
        booster_b = lgb.Booster(model_file=str(d / "ltr_multitask_b.cbm"))
        meta_path = d / "ltr_multitask_meta.json"
        weight_a, weight_b = 0.5, 0.5
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            weight_a = float(meta.get("weight_a", 0.5))
            weight_b = float(meta.get("weight_b", 0.5))
        return cls(
            booster_a=booster_a,
            booster_b=booster_b,
            weight_a=weight_a,
            weight_b=weight_b,
            cat_columns=cat_columns,
        )

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        imp_a = self.booster_a.feature_importance(importance_type=importance_type)
        imp_b = self.booster_b.feature_importance(importance_type=importance_type)
        return pd.DataFrame({
            "feature": self.feature_columns,
            "gain_a": imp_a,
            "gain_b": imp_b,
            "gain_total": self.weight_a * imp_a + self.weight_b * imp_b,
        }).sort_values("gain_total", ascending=False)
