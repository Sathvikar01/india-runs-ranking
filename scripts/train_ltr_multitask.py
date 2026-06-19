"""Train the multi-task LTR (Agent 1).

Trains two LightGBM lambdarank models on the same feature schema but
different ground-truth targets:

  task_a: proxy_relevance_v2 (Agent 2 — JD + eval_rubric blend)
  task_b: eval_rubric.eval_relevance

Both are saved into ``artifacts/ltr_multitask/`` and the ranker picks them
up at inference time.
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
from src.evaluation.eval_rubric import eval_relevance
from src.ingestion.parse_jsonl import iter_candidates_jsonl
from src.preprocessing.feature_engineer import (
    categorical_columns,
    feature_columns,
)
from src.ranking.ltr_multitask import MultiTaskLTR

log = logging.getLogger("train_ltr_multitask")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _bucket_groups(n: int, group_size: int) -> tuple[np.ndarray, list[int]]:
    """Return (group_sizes_array, list_of_sizes). LightGBM expects sizes."""
    sizes = []
    for start in range(0, n, group_size):
        end = min(start + group_size, n)
        sizes.append(end - start)
    return np.array(sizes, dtype=int), sizes


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", default="data/raw/candidates_5k.jsonl")
    p.add_argument("--feature-parquet", default="artifacts/feature_store.parquet")
    p.add_argument("--out-dir", default="artifacts/ltr_multitask")
    p.add_argument("--num-boost-round", type=int, default=600)
    p.add_argument("--group-size", type=int, default=2000)
    p.add_argument("--weight-a", type=float, default=0.5)
    p.add_argument("--weight-b", type=float, default=0.5)
    args = p.parse_args()

    t0 = time.perf_counter()

    log.info("Loading feature parquet …")
    df = pd.read_parquet(args.feature_parquet)
    log.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # The parquet was built with the old schema; this command is intended
    # for a fresh build. We rebuild features here from the JSONL when the
    # parquet is missing or stale.
    if "candidate_id" not in df.columns:
        log.warning("No candidate_id in feature parquet; rebuilding from JSONL …")
        from src.preprocessing.feature_engineer import build_features
        rows = []
        id_to_cand = {}
        for c in iter_candidates_jsonl(args.candidates):
            id_to_cand[c.candidate_id] = c
            rows.append(build_features(c))
        df = pd.DataFrame(rows)
        if "candidate_id" not in df.columns and len(df) > 0:
            log.warning("No candidate_id column produced; assuming order matches JSONL")
        df.to_parquet(args.feature_parquet, index=False)

    if "candidate_id" not in df.columns:
        log.error("No candidate_id column. Aborting.")
        return 1

    log.info("Loading candidates for ground truth …")
    cand_map = {c.candidate_id: c for c in iter_candidates_jsonl(args.candidates)}

    log.info("Computing ground truth labels …")
    y_a = np.zeros(len(df), dtype=int)
    y_b = np.zeros(len(df), dtype=int)
    for i, cid in enumerate(df["candidate_id"].tolist()):
        c = cand_map.get(cid)
        if c is None:
            continue
        y_a[i] = int(proxy_relevance(c))
        y_b[i] = int(eval_relevance(c))

    # Build the LTR frame.
    feat_cols = feature_columns()
    cat_cols = categorical_columns()
    X = df[feat_cols + cat_cols].copy()
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")

    group, sizes = _bucket_groups(len(X), args.group_size)
    log.info("Groups: %d (sizes min=%d max=%d)", len(sizes), min(sizes), max(sizes))

    log.info("Training multi-task LTR …")
    mt = MultiTaskLTR.train(
        X, y_a, y_b, group=group,
        num_boost_round=args.num_boost_round,
        cat_columns=cat_cols,
        weight_a=args.weight_a,
        weight_b=args.weight_b,
    )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    mt.save(args.out_dir)
    log.info("Saved multi-task LTR to %s", args.out_dir)

    summary = {
        "trained": True,
        "n_samples": int(len(df)),
        "n_groups": len(sizes),
        "weight_a": mt.weight_a,
        "weight_b": mt.weight_b,
        "out_dir": args.out_dir,
        "wall_clock_s": time.perf_counter() - t0,
    }
    Path(args.out_dir, "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
