"""Quick-build features + multi-task LTR + top-K reranker for the 5k dev split.

This is the dev-time equivalent of the full build pipeline. It runs in
~5 min on the dev CPU and produces the three LTR artifacts that the
ranker needs.

Usage:
    python scripts/dev_build_ltrs.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.evaluation.proxy_ground_truth import proxy_relevance
from src.evaluation.eval_rubric import eval_relevance
from src.ingestion.parse_jsonl import iter_candidates_jsonl
from src.preprocessing.feature_engineer import (
    build_features,
    categorical_columns,
    feature_columns,
)
from src.ranking.ltr_multitask import MultiTaskLTR
from src.ranking.listwise_reranker import ListwiseTopKReranker

log = logging.getLogger("dev_build_ltrs")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    candidates_path = "data/raw/candidates_5k.jsonl"
    out_dir = REPO_ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    log.info("Loading %s …", candidates_path)
    candidates = list(iter_candidates_jsonl(candidates_path))
    log.info("  %d candidates", len(candidates))

    log.info("Building features (113 columns) …")
    t_feat = time.perf_counter()
    feats = [build_features(c) for c in candidates]
    df = pd.DataFrame(feats)
    log.info("  %d rows × %d cols in %.1fs",
             len(df), len(df.columns), time.perf_counter() - t_feat)

    feat_cols = feature_columns()
    cat_cols = categorical_columns()
    X = df[feat_cols + cat_cols].copy()
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")

    log.info("Computing proxy_v2 + eval_rubric labels …")
    t_lbl = time.perf_counter()
    y_proxy = np.array([int(proxy_relevance(c)) for c in candidates], dtype=int)
    y_eval = np.array([int(eval_relevance(c)) for c in candidates], dtype=int)
    log.info("  labels in %.1fs (proxy tier-3+: %d, eval tier-3+: %d)",
             time.perf_counter() - t_lbl,
             int((y_proxy >= 3).sum()),
             int((y_eval >= 3).sum()))

    n = len(X)
    group_size = 200
    sizes = []
    for start in range(0, n, group_size):
        sizes.append(min(group_size, n - start))
    group = np.array(sizes, dtype=int)

    # Multi-task LTR
    log.info("Training multi-task LTR (proxy_v2 + eval_rubric) …")
    t_mt = time.perf_counter()
    mt = MultiTaskLTR.train(
        X, y_proxy, y_eval, group=group,
        num_boost_round=400, cat_columns=cat_cols,
        weight_a=0.5, weight_b=0.5,
    )
    mt.save(out_dir / "ltr_multitask")
    log.info("  multi-task LTR in %.1fs", time.perf_counter() - t_mt)

    # Top-K listwise reranker (proxy_v2 as target)
    log.info("Training top-K listwise reranker …")
    t_topk = time.perf_counter()
    rk = ListwiseTopKReranker.train(
        X, y_proxy,
        cat_columns=cat_cols,
        num_boost_round=600,
        group_size=200,
        num_leaves=127,
        learning_rate=0.025,
    )
    rk.save(out_dir / "ltr_topk.cbm")
    log.info("  top-K reranker in %.1fs", time.perf_counter() - t_topk)

    # Also train a fresh ltr.cbm with the new 113-feature schema so the
    # existing ranker can load it directly. This overwrites the old
    # 75-feature booster.
    log.info("Training single-task LTR (ltr.cbm) on the new schema …")
    t_single = time.perf_counter()
    import lightgbm as lgb
    dtrain = lgb.Dataset(
        X, label=y_proxy, group=group,
        categorical_feature=cat_cols, free_raw_data=False,
    )
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
    }
    booster = lgb.train(params, dtrain, num_boost_round=600)
    booster.save_model(str(out_dir / "ltr.cbm"))
    log.info("  ltr.cbm in %.1fs", time.perf_counter() - t_single)

    # Save the feature parquet too (so the ranker can load it).
    df.to_parquet(out_dir / "feature_store.parquet", index=False)
    log.info("Saved %d features to %s", len(df), out_dir / "feature_store.parquet")

    log.info("Total wall clock: %.1fs", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
