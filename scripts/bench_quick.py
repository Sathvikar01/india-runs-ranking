"""Quick benchmark (Agent 10) - run from CI on every PR.

Skips the ranker; uses an existing ``outputs/team_xxx.csv`` (or the most
recent dry-run) and scores it against all 3 rubrics. Reports the 4
sub-scores so a regression in any one axis is visible. Output is a
single Markdown table written to ``reports/bench.md``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", default="data/raw/candidates_5k.jsonl",
                   help="JSONL to use as the candidate pool (ground-truth source). "
                        "If --csv contains ids from a larger pool, the bench "
                        "automatically scopes the ground truth to the CSV's ids.")
    p.add_argument("--csv", default=None,
                   help="CSV to score; default = newest outputs/dry_run/*.csv")
    p.add_argument("--out", default="reports/bench.md")
    p.add_argument("--random-baseline", action="store_true",
                   help="If no CSV is available, score a random-order baseline.")
    args = p.parse_args()

    t0 = time.perf_counter()

    csv_path = Path(args.csv) if args.csv else _find_newest_dry_run()
    if not csv_path.exists():
        if args.random_baseline:
            csv_path = _write_random_baseline(args.candidates)
        else:
            print(f"ERROR: no CSV found at {csv_path}. Use --random-baseline to generate one.")
            return 1

    from evaluation.ranking_metrics import ranking_score
    from evaluation.scoring import composite_score
    from evaluation.ranking_metrics import load_candidates
    from src.evaluation.proxy_ground_truth import build_proxy_ground_truth
    from src.evaluation.eval_rubric import build_eval_ground_truth
    from src.evaluation.jd_literal_rubric import build_jd_literal_ground_truth

    cands = load_candidates(args.candidates)
    csv_ids = _read_csv_ids(csv_path)
    cand_ids = {c.candidate_id for c in cands}
    if csv_ids and not csv_ids.issubset(cand_ids):
        missing = len(csv_ids - cand_ids)
        log_msg = (
            f"Warning: {missing}/{len(csv_ids)} ids in the CSV are not in the "
            f"candidate pool ({args.candidates}). The bench will report a lower "
            f"ranking_score than the ranker actually achieves. Use --candidates "
            f"with the matching pool (e.g. data/raw/candidates.jsonl)."
        )
        print(log_msg)

    proxy = build_proxy_ground_truth(cands)
    eval_ = build_eval_ground_truth(cands)
    jd_literal = build_jd_literal_ground_truth(cands)

    r = ranking_score(csv_path, proxy, eval_, jd_literal)
    # reasoning_score / system_score / audit_score are not part of the
    # quick bench (they require LLM/audit calls); we report zeros.
    composite = composite_score({
        "ranking_score": r["ranking_score_0_1"],
        "reasoning_score": 0.0,
        "system_score": 0.0,
        "audit_score": 0.0,
    })

    md_lines = [
        "# Quick Bench (Agent 10)",
        f"CSV: `{csv_path}`",
        f"Candidates: {len(cands)}",
        f"Wall clock: {time.perf_counter() - t0:.1f}s",
        "",
        "## Ranking",
        "",
        f"| Rubric        | Composite | NDCG@10 | NDCG@50 | MAP | P@10 |",
        f"|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("proxy", "eval_rubric", "jd_literal"):
        m = r[name]
        comp = r.get(f"{name}_composite", 0.0)
        md_lines.append(
            f"| {name:13s} | {comp:.4f}    | "
            f"{m.get('ndcg@10', 0):.4f} | {m.get('ndcg@50', 0):.4f} | "
            f"{m.get('map', 0):.4f} | {m.get('p@10', 0):.4f} |"
        )
    md_lines += [
        f"| **ranking_score (min)** | **{r['ranking_score_0_1']:.4f}** | | | | |",
        "",
        "## Composite (without reasoning/system/audit)",
        f"Score 0-1: {composite['score_0_1']:.4f}",
        f"Score 0-100: {composite['score_0_100']:.2f}",
        f"Grade: {composite['grade']}",
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Also print a single JSON for CI consumption.
    summary = {
        "ranking_score": r["ranking_score_0_1"],
        "ranking_components": {
            "proxy": r["proxy_composite"],
            "eval_rubric": r["eval_rubric_composite"],
            "jd_literal": r.get("jd_literal_composite"),
        },
        "composite_partial": composite["score_0_1"],
        "wall_clock_s": time.perf_counter() - t0,
    }
    print(json.dumps(summary, indent=2))
    return 0


def _find_newest_dry_run() -> Path:
    dry = REPO_ROOT / "outputs" / "dry_run"
    candidates = sorted(dry.glob("team_xxx_dryrun_*.csv"))
    if not candidates:
        return REPO_ROOT / "outputs" / "team_xxx.csv"
    return candidates[-1]


def _read_csv_ids(path: Path) -> set[str]:
    """Return the set of candidate_ids in a ranker-output CSV."""
    import csv as _csv
    out: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in _csv.DictReader(f):
            cid = row.get("candidate_id")
            if cid:
                out.add(cid)
    return out


def _write_random_baseline(candidates_path: str | Path) -> Path:
    """Write a 100-row CSV with random candidates and monotonically decreasing scores.

    Used by the CI bench when no ranker output exists yet.
    """
    import csv as _csv
    import random as _r

    from evaluation.ranking_metrics import load_candidates
    cands = load_candidates(candidates_path)
    _r.seed(0)
    sample = _r.sample(cands, k=min(100, len(cands)))
    out = REPO_ROOT / "outputs" / "bench_random_baseline.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        w.writeheader()
        for i, c in enumerate(sample, 1):
            w.writerow({
                "candidate_id": c.candidate_id,
                "rank": i,
                "score": round(1.0 - i / 100.0, 4),
                "reasoning": "(random baseline)",
            })
    print(f"Wrote random baseline to {out}")
    return out


if __name__ == "__main__":
    sys.exit(main())
