"""End-to-end evaluation harness.

Runs the ranker, scores the output on 4 axes, and produces a single
composite score (0-100) with letter grade. Writes to `evaluation/results/`.

Usage:
    python evaluation/run_evaluation.py [--full-pool] [--skip-ranker]
    make eval
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.feature_importance import (
    cohort_feature_diff,
    ltr_feature_importance,
    write_feature_importance_md,
)
from evaluation.grade_thresholds import grade_sub_score
from evaluation.ranking_metrics import (
    ranking_score as compute_ranking_score,
)
from evaluation.ranking_metrics import (
    write_ranking_metrics_csv,
)
from evaluation.reasoning_audit import reasoning_score as compute_reasoning_score
from evaluation.reasoning_audit import run_audit_for_eval
from evaluation.scoring import composite_score
from evaluation.system_quality import system_quality

log = logging.getLogger("eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_CANDIDATES = REPO_ROOT / "data" / "raw" / "candidates_5k.jsonl"
DEFAULT_JD = REPO_ROOT / "data" / "raw" / "job_description.md"
DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts"
RESULTS_DIR = REPO_ROOT / "evaluation" / "results"
VALIDATOR = (
    REPO_ROOT.parent
    / "[PUB] India_runs_data_and_ai_challenge"
    / "India_runs_data_and_ai_challenge"
    / "validate_submission.py"
)


def _run_ranker(candidates: Path, jd: Path, artifacts: Path, out: Path) -> dict:
    """Run the ranker in --dry-run mode. Returns metadata (timing, output path)."""
    cmd = [
        sys.executable, "-m", "src.serving.rank",
        "--dry-run",
        "--candidates", str(candidates),
        "--job-description", str(jd),
        "--artifacts", str(artifacts),
        "--out", str(out),
    ]
    log.info("Running ranker: %s", " ".join(cmd))
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, timeout=900)
    dt = time.perf_counter() - t0
    log.info("Ranker done in %.1fs (rc=%d)", dt, r.returncode)
    if r.returncode != 0:
        log.error("ranker failed: %s\nstderr: %s", r.stdout, r.stderr)
        raise RuntimeError(f"ranker failed with rc={r.returncode}")
    # Find the actual output file (under outputs/dry_run/ with a timestamp).
    candidates_paths = sorted((REPO_ROOT / "outputs" / "dry_run").glob("team_xxx_dryrun_*.csv"))
    if not candidates_paths:
        raise RuntimeError("no dry-run output found under outputs/dry_run/")
    actual_out = candidates_paths[-1]
    # Copy to a stable name in evaluation/results/.
    stable = RESULTS_DIR / "submission.csv"
    shutil.copy(actual_out, stable)
    return {
        "output_path": str(stable),
        "wall_clock_s": round(dt, 1),
        "log_tail": r.stdout[-400:],
    }


def _build_ground_truths(candidates: Path) -> tuple[dict, dict]:
    """Build proxy + eval-rubric ground truths over the candidate pool."""
    from src.evaluation.eval_rubric import build_eval_ground_truth
    from src.evaluation.proxy_ground_truth import build_proxy_ground_truth

    from evaluation.ranking_metrics import load_candidates

    cands = load_candidates(candidates)
    proxy = build_proxy_ground_truth(cands)
    eval_ = build_eval_ground_truth(cands)
    log.info("Built ground truths: %d proxy, %d eval", len(proxy), len(eval_))
    return proxy, eval_


def _audit_score(csv_path: Path, candidates: Path) -> dict:
    """Run the audit and compute the reasoning_score."""
    out_md = RESULTS_DIR / "reasoning_quality.md"
    summary = run_audit_for_eval(csv_path, candidates, out_md)
    if not summary:
        return {"reasoning_score_0_1": 0.0, "audit_summary": {}, "per_check_pass_rate": {}}
    rs = compute_reasoning_score(summary)
    return {
        "reasoning_score_0_1": rs["reasoning_score_0_1"],
        "audit_summary": summary,
        "per_check_pass_rate": rs["per_check_pass_rate"],
        "weights": rs.get("weights", {}),
    }


def _ranking_score_block(csv_path: Path, proxy: dict, eval_: dict) -> dict:
    rs = compute_ranking_score(csv_path, proxy, eval_)
    # Write the per-metric CSV.
    write_ranking_metrics_csv(csv_path, proxy, eval_, RESULTS_DIR / "ranking_metrics.csv")
    return rs


def _audit_score_block(features_df, csv_path, ltr_path) -> dict:
    """Architecture + docs audit: a sub-score for the audit_score component.

    - Architecture: does the ranker use the documented components?
    - Docs: do they match the code?
    """
    arch = 1.0
    doc = 1.0
    notes: list[str] = []
    # LTR model present
    if not Path(ltr_path).exists():
        arch -= 0.2
        notes.append("LTR model not at expected path")
    # Calibrator
    if not (Path(ltr_path).parent / "ltr_calibrator.pkl").exists():
        arch -= 0.1
        notes.append("LTR calibrator not at expected path")
    # CatBoost
    if not (Path(ltr_path).parent / "catboost.cbm").exists():
        arch -= 0.1
        notes.append("CatBoost model not at expected path")
    # Feature store
    if not (Path(ltr_path).parent / "feature_store.parquet").exists():
        arch -= 0.2
        notes.append("Feature store not at expected path")
    # Docs
    docs_root = REPO_ROOT / "docs"
    if not (docs_root / "methodology.md").exists():
        doc -= 0.3
        notes.append("methodology.md missing")
    if not (docs_root / "future_work.md").exists():
        doc -= 0.2
        notes.append("future_work.md missing")
    if not (REPO_ROOT / "reports" / "evaluation_report.md").exists():
        doc -= 0.2
        notes.append("evaluation_report.md missing")
    arch = max(0.0, arch)
    doc = max(0.0, doc)
    audit_score_0_1 = 0.5 * arch + 0.5 * doc
    return {
        "audit_score_0_1": round(audit_score_0_1, 4),
        "architecture": round(arch, 4),
        "docs": round(doc, 4),
        "notes": notes,
    }


def _feature_importance_block(csv_path: Path, ltr_path: Path) -> dict:
    import pandas as pd
    feats = pd.read_parquet(REPO_ROOT / "artifacts" / "feature_store.parquet")
    importance = ltr_feature_importance(ltr_path, top_k=20)
    cohort = cohort_feature_diff(feats, csv_path, top_n=10, bottom_n=10)
    write_feature_importance_md(importance, cohort, RESULTS_DIR / "feature_importance.md")
    return {
        "top_20_by_gain": importance,
        "top10_vs_bottom10_diff": cohort[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run end-to-end evaluation.")
    parser.add_argument("--full-pool", action="store_true",
                        help="Use the full 100K candidates pool (overnight build).")
    parser.add_argument("--skip-ranker", action="store_true",
                        help="Skip running the ranker; evaluate an existing CSV (set --csv).")
    parser.add_argument("--csv", default=None,
                        help="Path to a submission CSV to evaluate (used with --skip-ranker).")
    parser.add_argument("--no-system-quality", action="store_true",
                        help="Skip the system-quality checks (tests, lint, etc.).")
    args = parser.parse_args()

    candidates = REPO_ROOT / "data" / "raw" / "candidates.jsonl" if args.full_pool else DEFAULT_CANDIDATES
    jd = DEFAULT_JD
    artifacts = DEFAULT_ARTIFACTS
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Run the ranker (unless --skip-ranker).
    if args.skip_ranker:
        if not args.csv:
            log.error("--csv is required with --skip-ranker")
            return 1
        out_path = Path(args.csv)
        if not out_path.exists():
            log.error("CSV not found: %s", out_path)
            return 1
        ranker_meta = {"output_path": str(out_path), "wall_clock_s": None, "skipped": True}
    else:
        ranker_meta = _run_ranker(candidates, jd, artifacts, RESULTS_DIR / "submission.csv")
        out_path = Path(ranker_meta["output_path"])

    # 2. Build ground truths.
    proxy, eval_ = _build_ground_truths(candidates)

    # 3. Ranking score.
    ranking = _ranking_score_block(out_path, proxy, eval_)

    # 4. Reasoning score (audit).
    audit = _audit_score(out_path, candidates)
    rs = compute_reasoning_score(audit["audit_summary"])
    reasoning_block = {
        "reasoning_score_0_1": rs["reasoning_score_0_1"],
        "per_check_pass_rate": rs["per_check_pass_rate"],
        "audit_summary": audit["audit_summary"],
    }

    # 5. System quality (tests, lint, etc.).
    if args.no_system_quality:
        system = {"system_score_0_1": None, "per_check": {}, "skipped": True}
    else:
        system = system_quality(
            out_path,
            validator=str(VALIDATOR) if VALIDATOR.exists() else None,
            skip_reproducible=True,
        )

    # 6. Audit score (architecture + docs).
    audit_block = _audit_score_block(out_path, artifacts / "feature_store.parquet", artifacts / "ltr.cbm")

    # 7. Feature importance.
    try:
        importance = _feature_importance_block(out_path, artifacts / "ltr.cbm")
    except Exception as e:
        log.warning("feature importance failed: %s", e)
        importance = {"error": str(e)}

    # 8. Composite.
    sub_scores = {
        "ranking_score": ranking["ranking_score_0_1"],
        "reasoning_score": reasoning_block["reasoning_score_0_1"],
        "system_score": system.get("system_score_0_1") or 0.0,
        "audit_score": audit_block["audit_score_0_1"],
    }
    composite = composite_score(sub_scores)

    # 9. Write EVAL.json.
    final = {
        "ranker": ranker_meta,
        "candidates_path": str(candidates),
        "ranking": ranking,
        "reasoning": reasoning_block,
        "system": system,
        "audit": audit_block,
        "feature_importance_summary": {
            "top_20_by_gain": importance.get("top_20_by_gain", []),
            "top_10_vs_bottom_10_diff_top20": importance.get("top10_vs_bottom10_diff", [])[:20],
        },
        "composite": composite,
    }
    (RESULTS_DIR / "EVAL.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    log.info("Wrote %s", RESULTS_DIR / "EVAL.json")

    # 10. Write FINAL_GRADE.md.
    _write_final_grade_md(final, out_path)

    print()
    print("=" * 60)
    print(f"  COMPOSITE SCORE: {composite['score_0_100']}  GRADE: {composite['grade']}")
    print("=" * 60)
    print()
    for k, v in composite["sub_scores"].items():
        letter, desc = grade_sub_score(k, v)
        print(f"  {k:18s}  {v:.3f}  ({letter})  {desc}")
    print()
    print(f"  Full report:  {RESULTS_DIR / 'FINAL_GRADE.md'}")
    print(f"  JSON:         {RESULTS_DIR / 'EVAL.json'}")
    print(f"  CSV:          {out_path}")
    return 0


def _write_final_grade_md(final: dict, csv_path: Path) -> None:
    """Write a one-page human-readable summary."""
    composite = final["composite"]
    out_lines: list[str] = []
    out_lines.append("# Final Grade\n")
    out_lines.append(f"_Generated by `evaluation/run_evaluation.py` on {csv_path.name}._\n")
    out_lines.append("## Composite\n")
    out_lines.append(f"- **Score: {composite['score_0_100']} / 100**")
    out_lines.append(f"- **Grade: {composite['grade']}**\n")
    out_lines.append("## Sub-scores\n")
    out_lines.append("| Component | Weight | Score | Grade | Description |")
    out_lines.append("|---|---:|---:|---|---|")
    for k, w in composite["weights"].items():
        v = composite["sub_scores"][k]
        letter, desc = grade_sub_score(k, v)
        weighted = composite["sub_scores_weighted"][k]
        out_lines.append(f"| `{k}` | {w:.2f} | {v:.3f} | {letter} | {desc} ({weighted:.3f} weighted) |")
    out_lines.append("\n## Ranking\n")
    rs = final["ranking"]
    out_lines.append(f"- proxy composite: **{rs['proxy_composite']:.4f}**")
    out_lines.append(f"- eval-rubric composite: **{rs['eval_rubric_composite']:.4f}**")
    out_lines.append(f"- ranking_score (min of the two): **{rs['ranking_score_0_1']:.4f}**\n")
    out_lines.append("### proxy")
    for k, v in rs["proxy"].items():
        out_lines.append(f"- {k}: {v:.4f}")
    out_lines.append("\n### eval rubric")
    for k, v in rs["eval_rubric"].items():
        out_lines.append(f"- {k}: {v:.4f}")
    out_lines.append("\n## Reasoning\n")
    rs2 = final["reasoning"]
    out_lines.append(f"- reasoning_score: **{rs2['reasoning_score_0_1']:.4f}**\n")
    out_lines.append("| Check | Pass rate |")
    out_lines.append("|---|---:|")
    for k, v in rs2.get("per_check_pass_rate", {}).items():
        out_lines.append(f"| `{k}` | {v:.1%} |")
    out_lines.append("\n## System\n")
    sysq = final["system"]
    out_lines.append(f"- system_score: **{sysq.get('system_score_0_1', 'n/a')}**\n")
    out_lines.append("| Check | Passed |")
    out_lines.append("|---|:-:|")
    for k, v in sysq.get("per_check", {}).items():
        passed = "✓" if v.get("passed") else "✗"
        out_lines.append(f"| `{k}` | {passed} |")
    out_lines.append("\n## Audit (architecture + docs)\n")
    a = final["audit"]
    out_lines.append(f"- audit_score: **{a['audit_score_0_1']:.4f}**")
    out_lines.append(f"- architecture: **{a['architecture']:.4f}**")
    out_lines.append(f"- docs: **{a['docs']:.4f}**")
    if a.get("notes"):
        out_lines.append("\n### Notes")
        for n in a["notes"]:
            out_lines.append(f"- {n}")
    out_lines.append("\n## Feature importance (top 10 by LTR gain)\n")
    fi = final.get("feature_importance_summary", {}).get("top_20_by_gain", [])[:10]
    out_lines.append("| Feature | Gain | Gain % |")
    out_lines.append("|---|---:|---:|")
    for r in fi:
        out_lines.append(f"| `{r['feature']}` | {r['gain']:.1f} | {r['gain_pct']:.1f}% |")
    out_lines.append("\n## Top 10 vs bottom 10 — biggest deltas\n")
    out_lines.append("Positive delta → feature is higher in the top-10 than the bottom-10.\n")
    out_lines.append("| Feature | Top-10 mean | Bottom-10 mean | Delta |")
    out_lines.append("|---|---:|---:|---:|")
    for r in final.get("feature_importance_summary", {}).get("top_10_vs_bottom_10_diff_top20", [])[:10]:
        out_lines.append(
            f"| `{r['feature']}` | {r['top_mean']} | {r['bottom_mean']} | {r['delta']:+.3f} |"
        )
    out_lines.append("\n## Interpretation\n")
    out_lines.append(_interpretation(final))
    (RESULTS_DIR / "FINAL_GRADE.md").write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    log.info("Wrote %s", RESULTS_DIR / "FINAL_GRADE.md")


def _interpretation(final: dict) -> str:
    """One-paragraph human interpretation of the result."""
    score = final["composite"]["score_0_100"]
    grade = final["composite"]["grade"]
    rs = final["ranking"]["ranking_score_0_1"]
    rea = final["reasoning"]["reasoning_score_0_1"]
    final["audit"]["audit_score_0_1"]
    # Identify the strongest and weakest sub-scores.
    sub = final["composite"]["sub_scores"]
    if sub:
        strongest = max(sub, key=sub.get)
        weakest = min(sub, key=sub.get)
    else:
        strongest = weakest = "n/a"
    parts: list[str] = []
    if rs < 0.5:
        parts.append(f"the ranker scores {rs:.2f} on the eval rubric, no better than a heuristic on this pool")
    elif rs < 0.7:
        parts.append(f"the ranker scores {rs:.2f} on the eval rubric — competitive but leaves NDCG@10 headroom")
    else:
        parts.append(f"the ranker scores {rs:.2f} on the eval rubric — matches it closely")
    if rea < 0.5:
        parts.append("reasoning quality is below the Stage 4 bar")
    elif rea < 0.8:
        parts.append("reasoning is mostly Stage-4-compliant with some gaps")
    else:
        parts.append("reasoning is largely Stage-4-compliant")
    return (
        f"With a composite of {score} (grade {grade}), the system's strongest axis is "
        f"`{strongest}` ({sub.get(strongest, 0):.2f}) and the weakest is `{weakest}` "
        f"({sub.get(weakest, 0):.2f}). On the substance: " + "; ".join(parts) + "."
    )


if __name__ == "__main__":
    sys.exit(main())
