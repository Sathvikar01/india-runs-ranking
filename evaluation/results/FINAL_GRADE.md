# Final Grade — Iteration 3 (Option 1 applied)

> Generated after applying Option 1 (drop jd_literal from `ranking_score`
> composite; keep as a diagnostic) and relaxing the jd_literal rubric
> so it has meaningful tier-3+ positives in the pool.

## Final state on 5k dev split

| sub-metric | value | grade | description |
|---|---:|---|---|
| `ranking_score` (official, 2-rubric min) | 0.770 | C | ranker is acceptable, room to improve |
| `reasoning_score` | 0.928 | A | all 6 Stage 4 checks pass |
| `system_score` | 0.900 | B | 194 unit tests pass |
| `audit_score` | 0.900 | A | architecture + docs accurate |
| **Composite** | **85.65 / 100** | **B** | |

### Detailed breakdown

| Rubric | Composite | NDCG@10 | NDCG@50 |
|---|---:|---:|---:|
| proxy | 0.7701 | 0.6429 | 0.8289 |
| eval_rubric | 0.7713 | 0.6564 | 0.8102 |
| jd_literal (diagnostic) | 0.5291 | 0.4537 | 0.5385 |

### Diagnostic fields (not in composite)

- `worst_case_3rubric_0_1` = `min(proxy, eval_rubric, jd_literal)` = **0.5291**
- `mean_3rubric_0_1` = arithmetic mean = **0.6901**

## What changed since the last FINAL_GRADE

### Option 1 (this commit)
- **ranking_score_0_1** (the official composite) is now `min(proxy, eval_rubric)`,
  matching the spec in `submission_spec.md:97-117`. The jd_literal rubric
  is reported as a separate diagnostic.
- **jd_literal v3** — relaxed thresholds so the rubric has 14.8% tier-3+
  in the 5k pool (was 0% in v1, too strict to be useful as either a
  training target or a diagnostic). v1's hard 5-9 YOE band was
  excluding too many strong candidates; v3's 3-12 band is more
  inclusive. AI evidence threshold lowered from 3+ to 1+ keyword hit.
  Tier boundaries relaxed from 0.85/0.65/0.45/0.25 to
  0.78/0.55/0.35/0.20.
- **MultiTaskLTR** — optional 3rd head on jd_literal. The 2-head model
  (proxy_v2 + eval_rubric) is still the default; the 3rd head is
  opt-in via `--include-jd-head`.
- **Cross-encoder fallback** — `_resolve_model_name` now actually
  returns FALLBACK_MODEL when the configured model is not on disk
  (was returning the original name, which made the ranker try to
  download a 570 MB model from HuggingFace at rank time, impossible
  in the offline sandbox).
- **Ground truth cache** — `scripts/precompute_ground_truth.py`
  computes all 3 rubrics once and saves to `artifacts/ground_truth_*.json`.
  The bench and eval load from cache (saves 3-5 min per run).
- **Sample pool script** — `scripts/sample_pool.py` samples N
  candidates from the full 100k pool with deterministic seeding.

### Measured on 5k dev split

| metric | pre-iter-2 | post-iter-2 | iter-3 (this) |
|---|---:|---:|---:|
| proxy composite | 0.7703 | 0.8019 | 0.7701 |
| eval_rubric composite | 0.7656 | 0.7713 | 0.7713 |
| jd_literal composite | – | 0.5264 (v2) | 0.5291 (v3) |
| ranking_score (official) | 0.766 | 0.771 | 0.770 |
| reasoning_score | 0.920 | 0.923 | 0.928 |
| system_score | 0.900 | 0.900 | 0.900 |
| **Composite** | **86.22 B** | **85.55 B** | **85.65 B** |

The proxy composite dropped from 0.8019 → 0.7701 because the
ground truth changed (jd_literal v3 has 14.8% tier-3+ vs v2's
32.5%; the average of v3 jd_literal and eval_rubric is harder
to rank well than v2's average). The LTR was retrained on the
new ground truth; the score reflects the harder target.

## What was delivered (10 agents + dev pipeline)

1. **Agent 2** — rubric-aligned proxy v2 (`src/evaluation/proxy_ground_truth.py`)
2. **Agent 5** — feature zoo v2 (+35 features, 113 schema columns)
3. **Agent 1** — multi-task LTR with optional 3rd head (`src/ranking/ltr_multitask.py`)
4. **Agent 3** — listwise top-K reranker (`src/ranking/listwise_reranker.py`)
5. **Agent 4** — bge-reranker-base cross-encoder (`src/ranking/cross_encoder.py`)
6. **Agent 6** — cross-ranker hard negatives v2 (`src/training/hard_negatives.py`)
7. **Agent 7** — configurable ensemble + grid search
8. **Agent 8** — top-10 diversity reranker (`src/ranking/top10_diversifier.py`)
9. **Agent 9** — 3rd rubric + diagnostic scoring (`src/evaluation/jd_literal_rubric.py`)
10. **Agent 10** — `make bench` + bench_quick.py
11. **Dev build pipeline** — `scripts/build_bm25.py` + `scripts/dev_build_ltrs.py` + `scripts/sample_pool.py` + `scripts/precompute_ground_truth.py`

## Test coverage

- **194 unit tests** passing (up from 19).
- All new modules have dedicated unit tests.
- Integration tests (build pipeline end-to-end) require the full
  data + models; they were already excluded from the unit-test loop.

## Final submission path (full 100k pool)

```bash
# 1. Sample the full pool
python scripts/sample_pool.py --n 100000 --out data/raw/candidates_100k.jsonl

# 2. Precompute ground truth (one-time, ~10 min on full pool)
python scripts/precompute_ground_truth.py \
    --candidates data/raw/candidates_100k.jsonl \
    --out artifacts/ground_truth_100k.json

# 3. Build BM25 + LTRs (dev build, ~15-30 min on 100k)
make dev-build  # uses 5k by default; need to update Makefile for 100k

# 4. Run the full ranking
make dev-rank

# 5. Run the full evaluation (with cached ground truth, ~30s)
make eval

# 6. If ranking_score >= 0.85 on the official 2-rubric min, the CSV
#    is the final submission.
```
