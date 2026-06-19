"""Build a BM25 index from a JSONL candidate pool.

Standalone script (does not require the full build_artifacts.py
pipeline). Writes ``artifacts/bm25.pkl`` from the candidate's
``deep_profile`` text.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ingestion.parse_jsonl import iter_candidates_jsonl
from src.preprocessing.deep_profile import build_deep_profile
from src.retrieval.bm25 import build_bm25

log = logging.getLogger("build_bm25")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True)
    p.add_argument("--out", default="artifacts/bm25.pkl")
    p.add_argument("--k1", type=float, default=1.5)
    p.add_argument("--b", type=float, default=0.75)
    args = p.parse_args()

    t0 = time.perf_counter()
    log.info("Building BM25 from %s …", args.candidates)
    candidates = list(iter_candidates_jsonl(args.candidates))
    log.info("  %d candidates", len(candidates))

    log.info("Building deep profiles …")
    pairs = [(c.candidate_id, build_deep_profile(c) or "") for c in candidates]

    log.info("Fitting BM25Okapi …")
    idx = build_bm25(pairs, k1=args.k1, b=args.b)
    idx.save(args.out)
    log.info("Saved %d docs to %s in %.1fs",
             len(candidates), args.out, time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
