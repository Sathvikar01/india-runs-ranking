"""Train the top-K listwise reranker (Agent 3).

Trains a LightGBM lambdarank focused on top-K NDCG. Saves to
``artifacts/ltr_topk.cbm``.

Usage:
    python scripts/train_listwise_reranker.py \\
        --candidates data/raw/candidates_5k.jsonl \\
        --feature-parquet artifacts/feature_store.parquet \\
        --out artifacts/ltr_topk.cbm
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.proxy_ground_truth import proxy_relevance
from src.ingestion.parse_jsonl import iter_candidates_jsonl
from src.preprocessing.feature_engineer import (
    categorical_columns,
    feature_columns,
)
from src.ranking.listwise_reranker import ListwiseTopKReranker

log = logging.getLogger("train_listwise")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", default="data/raw/candidates_5k.jsonl")
    p.add_argument("--feature-parquet", default="artifacts/feature_store.parquet")
    p.add_argument("--out", default="artifacts/ltr_topk.cbm")
    p.add_argument("--num-boost-round", type=int, default=1500)
    p.add_argument("--group-size", type=int, default=200)
    p.add_argument("--num-leaves", type=int, default=127)
    p.add_argument("--learning-rate", type=float, default=0.025)
    args = p.parse_args()

    t0 = time.perf_counter()

    log.info("Loading feature parquet …")
    df = pd.read_parquet(args.feature_parquet)
    log.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    if "candidate_id" not in df.columns:
        log.warning("No candidate_id; rebuilding from JSONL …")
        from src.preprocessing.feature_engineer import build_features
        rows = []
        for c in iter_candidates_jsonl(args.candidates):
            rows.append(build_features(c))
        df = pd.DataFrame(rows)
        df.to_parquet(args.feature_parquet, index=False)

    if "candidate_id" not in df.columns:
        log.error("No candidate_id column. Aborting.")
        return 1

    log.info("Loading candidates for ground truth …")
    cand_map = {c.candidate_id: c for c in iter_candidates_jsonl(args.candidates)}

    log.info("Computing proxy_relevance labels (proxy_v2 blend) …")
    y = np.zeros(len(df), dtype=int)
    for i, cid in enumerate(df["candidate_id"].tolist()):
        c = cand_map.get(cid)
        if c is None:
            continue
        y[i] = int(proxy_relevance(c))

    feat_cols = feature_columns()
    cat_cols = categorical_columns()
    X = df[feat_cols + cat_cols].copy()
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")

    log.info("Training listwise reranker …")
    rk = ListwiseTopKReranker.train(
        X, y,
        cat_columns=cat_cols,
        num_boost_round=args.num_boost_round,
        group_size=args.group_size,
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
    )
    rk.save(args.out)

    summary = {
        "trained": True,
        "n_samples": int(len(df)),
        "out": args.out,
        "num_boost_round": args.num_boost_round,
        "group_size": args.group_size,
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "wall_clock_s": time.perf_counter() - t0,
    }
    Path(args.out).with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
