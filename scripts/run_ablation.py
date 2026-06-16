"""Run the ablation suite on a dev split (default 5 000 candidates).

Generates `outputs/ablations/summary.csv` and `reports/benchmark.md`.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api.schemas import Candidate
from src.evaluation.ablation_runner import run_ablations, write_markdown_report
from src.ingestion.parse_jsonl import iter_candidates_jsonl

log = logging.getLogger("ablation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _dev_split(path: str, size: int, seed: int = 42) -> list[Candidate]:
    rng = random.Random(seed)
    cands: list[Candidate] = []
    for c in iter_candidates_jsonl(path):
        cands.append(c)
    if size and size < len(cands):
        cands = rng.sample(cands, size)
    return cands


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--size", type=int, default=5000)
    parser.add_argument("--out-dir", default="outputs/ablations")
    parser.add_argument("--report", default="reports/benchmark.md")
    args = parser.parse_args()

    log.info("Loading dev split (%d candidates) …", args.size)
    candidates = _dev_split(args.candidates, args.size)

    # Build a minimal ablation set: each fn just returns a deterministic ordering
    # that the proxy ground truth can score against.
    from src.evaluation.proxy_ground_truth import build_proxy_ground_truth
    relevance = build_proxy_ground_truth(candidates)

    # Ablation 1: random order
    def fn_random(cs: list[Candidate]) -> list[str]:
        ids = [c.candidate_id for c in cs]
        random.Random(0).shuffle(ids)
        return ids

    # Ablation 2: by YOE desc
    def fn_yoe(cs: list[Candidate]) -> list[str]:
        return [c.candidate_id for c in sorted(cs, key=lambda c: c.profile.years_of_experience, reverse=True)]

    # Ablation 3: by current_industry == AI/ML
    def fn_industry(cs: list[Candidate]) -> list[str]:
        from src.preprocessing.normalize import normalize_industry
        return [c.candidate_id for c in sorted(cs, key=lambda c: 0 if normalize_industry(c.profile.current_industry) == "ai_ml" else 1)]

    # Ablation 4: by proxy relevance
    def fn_proxy(cs: list[Candidate]) -> list[str]:
        return [cid for cid, _ in sorted(relevance.items(), key=lambda x: x[1], reverse=True)]

    # Ablation 5: by AI skill count desc + YOE filter
    def fn_skills(cs: list[Candidate]) -> list[str]:
        from src.evaluation.proxy_ground_truth import _ai_evidence_score
        return [c.candidate_id for c in sorted(cs, key=lambda c: (_ai_evidence_score(c), c.profile.years_of_experience), reverse=True)]

    ablations = {
        "01_random": fn_random,
        "02_yoe_only": fn_yoe,
        "03_industry_ai_ml": fn_industry,
        "04_proxy_relevance": fn_proxy,
        "05_skills_ai_count": fn_skills,
    }

    log.info("Running %d ablations …", len(ablations))
    t0 = time.perf_counter()
    results = run_ablations(candidates, ablations, out_dir=args.out_dir)
    write_markdown_report(results, args.report)
    log.info("Ablations done in %.1fs. Report: %s", time.perf_counter() - t0, args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
