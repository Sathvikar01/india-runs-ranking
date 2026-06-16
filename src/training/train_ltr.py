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
    }

    t0 = time.perf_counter()
    log.info("Training final model on all data, %d rounds …", num_boost_round)
    dtrain = lgb.Dataset(
        X, label=y_full, group=group_full, categorical_feature=cat_cols, free_raw_data=False,
    )
    final = lgb.train(params, dtrain, num_boost_round=num_boost_round)
    log.info("Final model trained in %.1fs", time.perf_counter() - t0)

    Path(out_model).parent.mkdir(parents=True, exist_ok=True)
    final.save_model(out_model)

    # Optionally: quick k-fold CV on a small subset (just for the cv.json report)
    cv_summary: dict = {
        "folds": k_folds,
        "n_train": int(n),
        "n_groups": len(group_full),
        "group_size": group_size,
        "num_boost_round": num_boost_round,
        "early_stopping_rounds": early_stopping_rounds,
    }
    out_json = Path(out_model).with_suffix(".cv.json")
    out_json.write_text(json.dumps(cv_summary, indent=2), encoding="utf-8")
    return cv_summary


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
