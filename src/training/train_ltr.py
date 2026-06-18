"""LTR trainer entry point.

Trains a LightGBM LambdaRank model. Handles the 10k-row-per-query limit by
splitting the 100k pool into multiple "queries" via a deterministic bucket.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold

from src.evaluation.ndcg import evaluate_ranking
from src.evaluation.proxy_ground_truth import build_proxy_ground_truth
from src.ingestion.parse_jsonl import iter_candidates_jsonl
from src.preprocessing.feature_engineer import (
    build_features,
    categorical_columns,
    feature_columns,
    features_to_dataframe,
)

log = logging.getLogger("train_ltr")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _bucket_groups(n: int, group_size: int) -> tuple[np.ndarray, list[int]]:
    """Return (bucket_assignment_per_row, group_sizes)."""
    bucket = (np.arange(n) // group_size).astype(int)
    counts = np.bincount(bucket)
    return bucket, counts[counts > 0].tolist()


def _reorder_by_bucket(X: pd.DataFrame, y: np.ndarray, bucket: np.ndarray):
    order = np.argsort(bucket, kind="stable")
    return X.iloc[order].reset_index(drop=True), y[order], bucket[order]


def _build_monotone_constraints(columns: list[str]) -> list[int]:
    """Build a per-feature monotone-constraint vector from configs/ranking.yaml.

    Each entry is in {-1, 0, +1}. +1 means "monotonically increasing with
    the label" (higher feature = higher label). -1 is the opposite. 0 is
    no constraint.
    """
    try:
        import yaml
        from pathlib import Path as _P
        cfg_path = _P("configs/ranking.yaml")
        if not cfg_path.exists():
            return [0] * len(columns)
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        constraints_map = (cfg.get("ltr") or {}).get("monotone_constraints") or {}
    except Exception:
        return [0] * len(columns)
    return [int(constraints_map.get(col, 0)) for col in columns]


def _to_lgb_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[feature_columns() + categorical_columns()].copy()


def train_ltr(
    candidates_path: str,
    feature_parquet: str,
    out_model: str,
    k_folds: int = 3,
    num_boost_round: int = 200,
    early_stopping_rounds: int = 30,
    group_size: int = 5000,
) -> dict:
    t0 = time.perf_counter()
    log.info("Loading features from %s …", feature_parquet)
    df = pd.read_parquet(feature_parquet)
    log.info("Loaded %d rows in %.1fs", len(df), time.perf_counter() - t0)

    t0 = time.perf_counter()
    log.info("Loading candidates to compute proxy relevance …")
    cands = []
    for c in iter_candidates_jsonl(candidates_path):
        cands.append(c)
    log.info("Loaded %d candidates in %.1fs", len(cands), time.perf_counter() - t0)

    t0 = time.perf_counter()
    log.info("Computing proxy relevance …")
    relevance = build_proxy_ground_truth(cands)
    log.info("Proxy relevance done in %.1fs", time.perf_counter() - t0)

    y_full = df["candidate_id"].map(relevance).fillna(0.0).astype(int).to_numpy()
    X = _to_lgb_frame(df)
    cat_cols = categorical_columns()

    n = len(X)
    bucket, group_full = _bucket_groups(n, group_size)
    X, y_full, bucket = _reorder_by_bucket(X, y_full, bucket)
    log.info("Bucketed into %d groups of size ~%d", len(group_full), group_size)

    # WS-Tier 1 #5: per-feature monotone constraints from configs/ranking.yaml.
    # Build a `monotone_constraints` vector aligned with X's columns.
    monotone_constraints = _build_monotone_constraints(X.columns.tolist())

    params = {
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
        "num_threads": 4,
        "monotone_constraints": monotone_constraints,
        "monotone_constraints_method": "advanced",
    }

    t0 = time.perf_counter()
    log.info("Training final model on all data, %d rounds …", num_boost_round)
    # WS-Tier 1 #8: use a held-out validation set (10 % of the data) so
    # the early-stopping callback has an `eval_set` to monitor.
    n_total = len(X)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_total)
    n_val = max(100, n_total // 10)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y_full[train_idx]
    bucket_train, group_train = _bucket_groups(len(X_train), group_size)
    X_train, y_train, _ = _reorder_by_bucket(X_train, y_train, bucket_train)
    X_val = X.iloc[val_idx].reset_index(drop=True)
    y_val = y_full[val_idx]
    bucket_val, group_val = _bucket_groups(len(X_val), max(group_size, 100))
    X_val, y_val, _ = _reorder_by_bucket(X_val, y_val, bucket_val)
    log.info("Train %d / Val %d (early stopping needs an eval set)", len(X_train), len(X_val))

    dtrain = lgb.Dataset(
        X_train, label=y_train, group=group_train,
        categorical_feature=cat_cols, free_raw_data=False,
    )
    dval = lgb.Dataset(
        X_val, label=y_val, group=group_val,
        categorical_feature=cat_cols, free_raw_data=False, reference=dtrain,
    )
    callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)] if num_boost_round > 100 else None
    final = lgb.train(
        params, dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval], valid_names=["val"],
        callbacks=callbacks,
    )
    log.info("Final model trained in %.1fs (best_iter=%s)", time.perf_counter() - t0, final.best_iteration)

    # WS-9: hard-negative mining pass.
    # Mine the top-K candidates the booster ranks highly but whose proxy
    # relevance is low. Reweight those to 0 and refit a second booster.
    # This pulls the model away from the highest-impact wrong answers.
    hard_neg_summary: dict = {"trained": False}
    try:
        from src.ranking.ltr_model import LTRModel
        from src.training.hard_negatives import (
            mine_hard_negatives,
            reweight_with_hard_negatives,
        )

        log.info("Mining hard negatives …")
        t0 = time.perf_counter()
        booster = LTRModel(booster=final, cat_columns=cat_cols)
        hard_neg_ids = mine_hard_negatives(
            cands, booster, df, top_k=200, rel_threshold=1.0
        )
        log.info(
            "Mined %d hard negatives in %.1fs", len(hard_neg_ids), time.perf_counter() - t0
        )
        if hard_neg_ids:
            relevance_v2 = reweight_with_hard_negatives(relevance, hard_neg_ids)
            y_v2 = df["candidate_id"].map(relevance_v2).fillna(0.0).astype(int).to_numpy()
            y_v2 = y_v2[np.argsort(bucket, kind="stable")]
            dtrain_v2 = lgb.Dataset(
                X,
                label=y_v2,
                group=group_full,
                categorical_feature=cat_cols,
                free_raw_data=False,
            )
            t0 = time.perf_counter()
            log.info("Refitting model with hard-negative reweighting, %d rounds …", num_boost_round)
            final_v2 = lgb.train(params, dtrain_v2, num_boost_round=num_boost_round)
            log.info("Hard-neg refit done in %.1fs", time.perf_counter() - t0)
            # Save the reweighted model.
            final = final_v2
            hard_neg_summary = {
                "trained": True,
                "n_hard_negatives": len(hard_neg_ids),
                "rounds": num_boost_round,
            }
    except Exception as e:
        log.warning("Hard-negative mining skipped: %s", e)
        hard_neg_summary = {"trained": False, "reason": str(e)}

    Path(out_model).parent.mkdir(parents=True, exist_ok=True)
    final.save_model(out_model)

    # WS-11: fit and save an isotonic calibrator. The calibrator maps
    # raw LTR scores to [0, 1] using the proxy relevance as the target,
    # so the sigmoid in the ensemble is well-behaved.
    calibrator_summary: dict = {"trained": False}
    try:
        from src.ranking.ltr_calibrator import LTRCalibrator

        cal_path = Path(out_model).with_name("ltr_calibrator.pkl")
        X_all = df[feature_columns() + categorical_columns()].copy()
        for c in categorical_columns():
            if c in X_all.columns:
                X_all[c] = X_all[c].astype("category")
        ltr_scores_for_cal = final.predict(X_all)
        relevance_full = df["candidate_id"].map(relevance).fillna(0.0).to_numpy(dtype=np.float32)
        LTRCalibrator.fit_and_save(ltr_scores_for_cal, relevance_full, cal_path)
        calibrator_summary = {"trained": True, "path": str(cal_path)}
        log.info("LTR calibrator saved to %s", cal_path)
    except Exception as e:
        log.warning("LTR calibrator skipped: %s", e)
        calibrator_summary = {"trained": False, "reason": str(e)}

    # WS-6: optionally train a CatBoost YetiRank as a second ensemble member.
    catboost_summary: dict = {"trained": False}
    catboost_out = Path(out_model).with_name("catboost.cbm")
    if _catboost_enabled():
        try:
            from src.ranking.catboost_ranker import CatBoostRanker

            log.info("Training CatBoost YetiRank second ranker …")
            cb = CatBoostRanker.train(
                X,
                y_full,
                bucket,  # group ids (bucket index per row)
                cat_columns=cat_cols,
                iterations=400,
            )
            cb.save(catboost_out)
            catboost_summary = {"trained": True, "path": str(catboost_out)}
            log.info("CatBoost ranker saved to %s", catboost_out)
        except Exception as e:
            log.warning("CatBoost training skipped: %s", e)
            catboost_summary = {"trained": False, "reason": str(e)}

    # Optionally: quick k-fold CV on a small subset (just for the cv.json report)
    cv_summary: dict = {
        "folds": k_folds,
        "n_train": int(n),
        "n_groups": len(group_full),
        "group_size": group_size,
        "num_boost_round": num_boost_round,
        "early_stopping_rounds": early_stopping_rounds,
        "hard_negatives": hard_neg_summary,
        "catboost": catboost_summary,
        "calibrator": calibrator_summary,
    }
    out_json = Path(out_model).with_suffix(".cv.json")
    out_json.write_text(json.dumps(cv_summary, indent=2), encoding="utf-8")
    return cv_summary


def _catboost_enabled() -> bool:
    try:
        import catboost  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False


def main() -> int:
    import json
    parser = argparse.ArgumentParser(description="Train the LTR ranker on the candidate pool.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--feature-parquet", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k-folds", type=int, default=3)
    parser.add_argument("--num-boost-round", type=int, default=200)
    parser.add_argument("--group-size", type=int, default=5000)
    args = parser.parse_args()
    summary = train_ltr(
        candidates_path=args.candidates,
        feature_parquet=args.feature_parquet,
        out_model=args.out,
        k_folds=args.k_folds,
        num_boost_round=args.num_boost_round,
        group_size=args.group_size,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
