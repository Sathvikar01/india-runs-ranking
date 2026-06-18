# Evaluation

> End-to-end quality evaluation for the Candidate Intelligence Platform.
> Produces a single composite score (0-100) with letter grade and
> per-component breakdowns.

This folder is self-contained: a single command runs the full
evaluation pipeline and writes its output to `evaluation/results/`.

## Quick start

```bash
# From the repo root
python evaluation/run_evaluation.py
# Or, equivalently:
make eval
```

The script will:
1. Run the ranker on the available candidate pool (5K subset
   by default; full 100K if `--full-pool`).
2. Score the output CSV against the **proxy ground truth** and the
   **independent eval rubric** for ranking quality (NDCG@10/50, MAP, P@10).
3. Score the reasoning column on the **6 Stage 4 checks** from
   `submission_spec.md:75-95`.
4. Score the **system quality**: tests, lint, output format, build
   pipeline integrity.
5. Compute a **composite score (0-100)** with a letter grade
   (A/B/C/D/F).
6. Write:
   - `evaluation/results/EVAL.json` — machine-readable
   - `evaluation/results/FINAL_GRADE.md` — human-readable
   - `evaluation/results/ranking_metrics.csv` — per-cutoff metrics
   - `evaluation/results/feature_importance.md` — top features
   - `evaluation/results/system_quality.md` — non-ML checks

## Score formula

The composite is a weighted average of 4 sub-scores, each in [0, 1]:

```
composite = 0.40 * ranking_score
         + 0.30 * reasoning_score
         + 0.20 * system_score
         + 0.10 * audit_score
```

Where:

- `ranking_score` — how well the top-K matches the eval rubric
  (proxy and eval-rubric separately; the *lower* of the two is
  reported to defend against proxy-overfit).
- `reasoning_score` — weighted average of the 6 Stage 4 audit checks
  (specific facts, JD connection, honest concerns, no hallucination,
  variation, rank consistency).
- `system_score` — tests passing, lint clean, output spec-compliant,
  build pipeline reproducible, monotonic scores.
- `audit_score` — composite of (a) "is the ranker doing what we said
  it would do?" (architecture audit) and (b) "is the documentation
  accurate?" (docs audit).

Letter grade: A ≥ 90, B ≥ 80, C ≥ 70, D ≥ 60, F < 60.

## Files

| File | Purpose |
|---|---|
| `run_evaluation.py` | Main entry point |
| `ranking_metrics.py` | NDCG, MAP, P@k, MRR against proxy + eval rubric |
| `reasoning_audit.py` | Stage 4 reasoning-quality scoring (wraps `scripts/audit_reasoning_quality.py`) |
| `system_quality.py` | Tests, lint, output spec, monotonicity, reproducibility |
| `feature_importance.py` | Top features by LTR gain, top reasons for "rank-N is AI" |
| `grade_thresholds.py` | A/B/C/D/F thresholds and per-component weights |
| `scoring.py` | The composite-score formula |
| `results/` | Output directory (gitignored) |
| `FINAL_GRADE.md` | One-page summary written by the latest run |

## Re-running

The output goes to `evaluation/results/EVAL.json` and overwrites prior
results. Pass `--full-pool` to use the full 100K (overnight build).
