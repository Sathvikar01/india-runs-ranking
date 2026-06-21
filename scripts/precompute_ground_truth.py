"""Precompute and cache the 3 ground truth rubrics for a candidate pool.

This is a one-time cost per pool. Once cached, the eval and bench scripts
load the cached ground truth instead of re-computing it (which is the
bottleneck for the 5k+ pools since proxy_v2 calls both the JD rubric
and the eval_rubric internals).

Usage:
    python scripts/precompute_ground_truth.py --candidates data/raw/candidates_5k.jsonl --out artifacts/ground_truth_5k.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.proxy_ground_truth import build_proxy_ground_truth
from src.evaluation.eval_rubric import build_eval_ground_truth
from src.evaluation.jd_literal_rubric import build_jd_literal_ground_truth
from src.ingestion.parse_jsonl import iter_candidates_jsonl

log = logging.getLogger("precompute_gt")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    t0 = time.perf_counter()
    log.info("Loading candidates from %s …", args.candidates)
    candidates = list(iter_candidates_jsonl(args.candidates))
    log.info("  %d candidates", len(candidates))

    log.info("Computing proxy_v2 ground truth …")
    t = time.perf_counter()
    proxy = build_proxy_ground_truth(candidates)
    log.info("  proxy done in %.1fs", time.perf_counter() - t)

    log.info("Computing eval_rubric ground truth …")
    t = time.perf_counter()
    eval_ = build_eval_ground_truth(candidates)
    log.info("  eval done in %.1fs", time.perf_counter() - t)

    log.info("Computing jd_literal ground truth …")
    t = time.perf_counter()
    jd = build_jd_literal_ground_truth(candidates)
    log.info("  jd_literal done in %.1fs", time.perf_counter() - t)

    out = {
        "candidates_path": args.candidates,
        "n": len(candidates),
        "proxy": proxy,
        "eval_rubric": eval_,
        "jd_literal": jd,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out), encoding="utf-8")
    log.info("Saved %d ground truths to %s in %.1fs total",
             len(candidates), out_path, time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
