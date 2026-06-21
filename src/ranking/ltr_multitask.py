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
    """Two- or three-head LTR: task A (proxy_v2) + task B (eval_rubric) [+ task C (jd_literal)].

    All heads share the same feature schema. The class wraps up to three
    ``lightgbm.Booster`` instances; ``predict`` returns the weighted sum
    of their per-row scores.

    Iteration 3: a third head on the jd_literal target is opt-in. With
    it, the ranker learns signals that all three ground-truth rubrics
    reward (the "robust to ground-truth choice" promise).
    """

    def __init__(
        self,
        booster_a=None,
        booster_b=None,
        booster_c=None,
        weight_a: float = 0.5,
        weight_b: float = 0.5,
        weight_c: float = 0.0,
        cat_columns: list[str] | None = None,
    ) -> None:
        self.booster_a = booster_a
        self.booster_b = booster_b
        self.booster_c = booster_c
        self.weight_a = float(weight_a)
        self.weight_b = float(weight_b)
        self.weight_c = float(weight_c)
        self.cat_columns = cat_columns or categorical_columns()
        self.feature_columns = feature_columns()

    @classmethod
    def train(
        cls,
        X: pd.DataFrame,
        y_a: np.ndarray,
        y_b: np.ndarray,
        group: np.ndarray,
        y_c: np.ndarray | None = None,
        cat_columns: list[str] | None = None,
        num_boost_round: int = 800,
        params: dict | None = None,
        weight_a: float = 0.5,
        weight_b: float = 0.5,
        weight_c: float = 0.0,
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

        booster_c = None
        if y_c is not None and weight_c > 0.0:
            dtrain_c = lgb.Dataset(
                X, label=y_c, group=group,
                categorical_feature=cat_columns, free_raw_data=False,
            )
            log.info("Training MultiTaskLTR head C (jd_literal) …")
            booster_c = lgb.train(
                default_params, dtrain_c, num_boost_round=num_boost_round,
            )

        return cls(
            booster_a=booster_a,
            booster_b=booster_b,
            booster_c=booster_c,
            weight_a=weight_a,
            weight_b=weight_b,
            weight_c=weight_c,
            cat_columns=cat_columns,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster_a is None or self.booster_b is None:
            raise RuntimeError("MultiTaskLTR has no trained boosters")
        a = self.booster_a.predict(X)
        b = self.booster_b.predict(X)
        if self.booster_c is not None and self.weight_c > 0.0:
            c = self.booster_c.predict(X)
            return self.weight_a * a + self.weight_b * b + self.weight_c * c
        return self.weight_a * a + self.weight_b * b

    def predict_per_head(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Return (head_a, head_b, head_c_or_None) for downstream analysis."""
        a = self.booster_a.predict(X)
        b = self.booster_b.predict(X)
        c = self.booster_c.predict(X) if self.booster_c is not None else None
        return a, b, c

    def save(self, dir_path: str | Path) -> None:
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        self.booster_a.save_model(str(d / "ltr_multitask_a.cbm"))
        self.booster_b.save_model(str(d / "ltr_multitask_b.cbm"))
        if self.booster_c is not None:
            self.booster_c.save_model(str(d / "ltr_multitask_c.cbm"))
        # Save the weights as a sidecar JSON.
        import json
        meta = {"weight_a": self.weight_a, "weight_b": self.weight_b}
        if self.weight_c > 0.0:
            meta["weight_c"] = self.weight_c
        (d / "ltr_multitask_meta.json").write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, dir_path: str | Path, cat_columns: list[str] | None = None) -> "MultiTaskLTR":
        import json
        import lightgbm as lgb

        d = Path(dir_path)
        booster_a = lgb.Booster(model_file=str(d / "ltr_multitask_a.cbm"))
        booster_b = lgb.Booster(model_file=str(d / "ltr_multitask_b.cbm"))
        booster_c = None
        c_path = d / "ltr_multitask_c.cbm"
        if c_path.exists():
            booster_c = lgb.Booster(model_file=str(c_path))
        meta_path = d / "ltr_multitask_meta.json"
        weight_a, weight_b, weight_c = 0.5, 0.5, 0.0
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            weight_a = float(meta.get("weight_a", 0.5))
            weight_b = float(meta.get("weight_b", 0.5))
            weight_c = float(meta.get("weight_c", 0.0))
        return cls(
            booster_a=booster_a,
            booster_b=booster_b,
            booster_c=booster_c,
            weight_a=weight_a,
            weight_b=weight_b,
            weight_c=weight_c,
            cat_columns=cat_columns,
        )

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        imp_a = self.booster_a.feature_importance(importance_type=importance_type)
        imp_b = self.booster_b.feature_importance(importance_type=importance_type)
        out = {
            "feature": self.feature_columns,
            "gain_a": imp_a,
            "gain_b": imp_b,
            "gain_total": self.weight_a * imp_a + self.weight_b * imp_b,
        }
        if self.booster_c is not None and self.weight_c > 0.0:
            imp_c = self.booster_c.feature_importance(importance_type=importance_type)
            out["gain_c"] = imp_c
            out["gain_total"] = out["gain_total"] + self.weight_c * imp_c
        return pd.DataFrame(out).sort_values("gain_total", ascending=False)
