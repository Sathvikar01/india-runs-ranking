"""Sample N candidates from the full pool to a smaller JSONL for dev builds.

Usage:
    python scripts/sample_pool.py --n 20000 --out data/raw/candidates_20k.jsonl --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ingestion.parse_jsonl import iter_candidates_jsonl


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/raw/candidates.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Reservoir sampling for a deterministic, evenly-spread subset.
    # We don't read all 100k into memory; we stream and apply a
    # skip-with-replacement scheme.
    import random
    rng = random.Random(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    selected: list[str] = []
    count = 0
    t0 = __import__("time").perf_counter()
    for c in iter_candidates_jsonl(args.src):
        count += 1
        if len(selected) < args.n:
            selected.append(c.candidate_id)
        else:
            # Skip each (count - args.n)/count of the time.
            j = rng.randint(0, count - 1)
            if j < args.n:
                selected[j] = c.candidate_id
        if count % 20000 == 0:
            print(f"  scanned {count} candidates, kept {len(selected)}", flush=True)

    print(f"Scanned {count} in {__import__('time').perf_counter() - t0:.1f}s. Selected {len(selected)} ids.")
    selected_set = set(selected)

    print(f"Streaming again to write {len(selected)} candidates to {out_path} …")
    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for c in iter_candidates_jsonl(args.src):
            if c.candidate_id in selected_set:
                f.write(c.model_dump_json() + "\n")
                n_written += 1
                if n_written % 5000 == 0:
                    print(f"  wrote {n_written}", flush=True)
                if n_written == len(selected):
                    break
    print(f"Wrote {n_written} candidates to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
