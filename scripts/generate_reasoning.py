"""Standalone reasoning-generation runner. Useful for resume / restart."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_artifacts import _generate_reasoning
from src.ingestion.parse_jsonl import iter_candidates_jsonl

log = logging.getLogger("reasoning")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--job-description", required=True)
    parser.add_argument("--out", default="artifacts/portraits.jsonl")
    parser.add_argument("--max", type=int, default=0)
    args = parser.parse_args()
    import yaml

    llm_cfg = yaml.safe_load(Path("configs/llm.yaml").read_text(encoding="utf-8"))["llm"]
    jd_text = Path(args.job_description).read_text(encoding="utf-8")
    cands = []
    for c in iter_candidates_jsonl(args.candidates):
        cands.append(c)
        if args.max and len(cands) >= args.max:
            break
    log.info("Loaded %d candidates.", len(cands))
    n = _generate_reasoning(cands, jd_text, llm_cfg, Path(args.out), resume=True)
    log.info("Generated %d new portraits.", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
