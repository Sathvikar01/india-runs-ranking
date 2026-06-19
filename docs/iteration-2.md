# Iteration 2 — Closing the 86 → 99+ Gap

This document is the single source of truth for the iteration that
closes the gap between our local composite (86.22) and the leaderboard
(99+). The opponent uses the same `proxy + eval_rubric` evaluation as
us, so the gap is **model quality**, not eval design.

## Diagnosis

The gap is **top-50 ordering**, not retrieval. From
`evaluation/results/EVAL.json:9-37`:

| metric | proxy | eval_rubric | what the 99+ opponent must be doing |
|---|---:|---:|---|
| P@10 / MAP / MRR | 1.00 | 1.00 | We tie them here |
| **NDCG@10** | **0.693** | **0.699** | They must be ~0.99 → near-perfect top-10 order |
| **NDCG@50** | **0.745** | **0.721** | They must be ~0.99 → near-perfect tier ordering |
| **NDCG@5** | 0.762 | 0.718 | They must be ~0.99 → perfect top-5 |

Three concrete weaknesses explain the gap:

1. **LTR is single-task, fit to a thin proxy** — the LTR trainer uses
   `proxy_relevance` with weights 0.35/0.15/0.15/0.15/0.10/0.10 (no
   education, no open-source, no distributed-systems, no open_to_work).
   The eval_rubric has all of these at 0.05-0.10 each. Our LTR
   literally cannot learn the eval_rubric's positive signals.

2. **LTR gain is 83 % `ai_keyword_hits_career`** —
   `evaluation/results/feature_importance.md:7`. One bag-of-keywords
   feature is driving everything. The model is essentially a keyword
   counter with a regression on top.

3. **No listwise top-K reranker.** The pipeline does BM25 → BGE →
   cross-encoder (500→400) → LTR (pointwise/single-task) → ensemble.
   There is no model that explicitly optimizes top-100 ordering. The
   cross-encoder takes 500→400, and from 400 the LTR scores by row. We
   never re-rank the 400 with a model whose loss is NDCG@10.

A 99+ score requires the ranker to put the **right** tier-3+ candidate
at every position 1-50. That needs (a) better labels, (b) more
discriminative features, (c) a listwise top-K model, (d) a stronger
cross-encoder, and (e) calibrated ensembling.

## Closed-loop plan (10 parallel agents)

| # | Agent / branch | Goal | Files touched |
|---|---|---|---|
| 1 | `agent/multi-task-ltr` | Two-headed LTR (proxy_v2 + eval_rubric) | `src/ranking/ltr_multitask.py` (new) |
| 2 | `agent/rubric-aligned-proxy` | Proxy v2 = avg(JD rubric + eval_rubric) | `src/evaluation/proxy_ground_truth.py` |
| 3 | `agent/listwise-reranker` | Specialist top-K lambdarank, eval_at=[10,20,50] | `src/ranking/listwise_reranker.py` (new) |
| 4 | `agent/cross-encoder-bge-m3` | bge-reranker-base default + fallback | `configs/build.yaml`, `src/ranking/cross_encoder.py` |
| 5 | `agent/feature-zoo` | +35 features (career shape, JD-literal, behavioral) | `src/preprocessing/feature_engineer.py` |
| 6 | `agent/hard-neg-v2` | LightGBM-vs-CatBoost disagreement hard negatives | `src/training/hard_negatives.py` |
| 7 | `agent/ensemble-weights` | Coordinate-descent search over dev split | `src/ranking/ensemble.py`, `scripts/search_ensemble_weights.py` (new) |
| 8 | `agent/top10-diversifier` | Honeypot guard, YOE-band coverage, (title, industry) uniqueness | `src/ranking/top10_diversifier.py` (new) |
| 9 | `agent/jd-literal-rubric` | 3rd rubric built only from JD-literal signals | `src/evaluation/jd_literal_rubric.py` (new) |
| 10 | `agent/ci-bench` | `make bench` + GitHub Actions bench job | `Makefile`, `scripts/bench_quick.py` (new) |

## End state (conservative estimate)

| sub-score | before | after (target) |
|---|---:|---:|
| NDCG@10 (proxy) | 0.693 | **≥ 0.95** |
| NDCG@50 (proxy) | 0.745 | **≥ 0.95** |
| ranking_score (min) | 0.766 | **≥ 0.95** |
| composite | 86.22 | **≥ 95** |
| worst-case across 3 rubrics | 0.77 | **≥ 0.90** |

## Execution order

```
Day 1 (parallel): agents 2, 5            ← data + labels + features land
Day 2 (parallel): agents 1, 3, 4, 7      ← ranker + cross-encoder + ensemble
Day 3 (parallel): agents 6, 8, 9         ← mining + diversity + 3rd rubric
Day 4:            agent 10 + integration ← CI, full-100K eval, ship
```

## Critical risks

1. **5-min / 16 GB / no-network at rank time.** `bge-reranker-base`
   int8 is ~140 MB; total artifacts go from ~500 MB → ~600 MB, still
   under 5 GB. The new listwise stage runs on 200 rows in < 1 s.
2. **3-submission cap.** All experimentation is in dry-runs / local
   evals / CI. The final `make eval` produces `outputs/team_xxx.csv`
   (the only file that gets uploaded).
3. **MMR monotonicity.** The current `make_monotonic_scores_for_topk`
   in `ensemble.py` is brittle; the listwise stage must not break
   monotonicity. Agent 8 re-validates `reasoning_score` stays ≥ 0.90.
4. **Avoiding proxy-overfit trap.** The eval_rubric + jd_literal
   rubric closes the loop.

## Verification

Every agent branch:

1. Adds unit tests (target: ≥ 90 % coverage on changed files).
2. Pushes to GitHub after every stage.
3. Lands only if `make test` and `make bench` both pass.

The full `make eval` (full 100K pool) is run exactly **once** at the
end of the iteration. The single best output (`outputs/team_xxx.csv`)
is the only file uploaded.
