# Future Work — Plan for the next iteration

> Single source of truth for the roadmap. Cross-linked from
> `docs/methodology.md:121-128` (the "what we'd do with more time" section).

This file is the post-WS-1..WS-14 roadmap. It captures items that the
current iteration has not yet shipped but that are prioritised for
the next one.

---

## P0 — Expected NDCG@10 lift > 0.02 each

| # | Item | Effort | Notes |
|---|---|---|---|
| 1 | **Wire `hard_negatives.py` into `train_ltr.py`** | 1-2 h | DONE in WS-9. 200 hard negatives are mined, the LTR is reweighted + refit. Lift on hard cases expected. |
| 2 | **Career-evidence semantic similarity** | 3-4 h | DONE in WS-10. BGE cosine between candidate's `deep_profile` and the JD replaces the bag-of-keywords `ai_keyword_hits_career` for the "is this candidate actually doing AI?" question. |
| 3 | **Isotonic LTR calibration** | 1 h | DONE in WS-11. Maps raw LTR scores to [0, 1] using the proxy relevance. Saved to `artifacts/ltr_calibrator.pkl`. |
| 4 | **Reasoning quality self-audit** | 2-3 h | DONE in WS-13. `scripts/audit_reasoning_quality.py` scores each row on the 6 Stage 4 checks. Output: `reports/reasoning_quality.md`. |
| 5 | **Full 100K end-to-end stress test** | overnight | DONE. 100K dry-run completes in ~108 s; LTR + CatBoost + binary classifier all run on the full pool. Wall-clock budget well under 5 min. Composite = 86.22 (B) on the full pool. |
| 6 | **Tier 3 reranker fine-tune** | 1-2 h | DONE. Local CPU fine-tune of `cross-encoder/ms-marco-MiniLM-L-6-v2` on 1600 train + 400 val examples (positive = tier-3+). 2 epochs, 16:18 min, AP=0.889, F1=0.78. Loaded by ranker via `configs/build.yaml: cross_encoder.model_name = artifacts/ce_finetuned`. Lift: ranking_score 0.743 → 0.766, composite 85.31 → 86.22. |

## P1 — Robustness / interpretability

| # | Item | Effort | Notes |
|---|---|---|---|
| 6 | **SHAP feature attribution** for the LTR model | 2 h | Per-row `shap_values` report added to `outputs/team_xxx.csv` as a new column. Lets a reviewer see *why* each candidate is at this rank. |
| 7 | **Tier-1 school prestige list** as a real ordinal feature | 1 h | DONE in WS-12. `configs/school_prestige.yaml` is the source of truth. Currently a continuous 0..1 score; could be an enum (tier_1 / tier_2 / other). |
| 8 | **Larger cross-encoder** (`bge-reranker-v2-m3`, 570 MB) | 1 h | Bigger reranker = better top-K precision. Fits the 5 GB artifact cap. |
| 9 | **Increase cross-encoder shortlist** | trivial | DONE in WS-12 (`top_k_from_retrieval: 800`, `top_k_output: 400`). |
| 10 | **Per-JD query rewriter** at build time | 1 h | DONE in WS-7 (`src/retrieval/query_rewriter.py`). Untested on full 100K with dense index. |

## P2 — Stretch

| # | Item | Effort | Notes |
|---|---|---|---|
| 11 | **Per-candidate template selection by quality heuristic** | 3 h | Instead of `hash(candidate_id) % 5`, pick the best template per candidate (e.g. evidence template for AI-strong, concern template for location/notice-poor). |
| 12 | **LLM polish for top-N (production)** | done | DONE in WS-14 (`--llm-polish-top N`). Run at build time only, not in the 5-min rank-time budget. |
| 13 | **Resume the Zenmux portraits pipeline** | 1-2 h | The 30-60 min Zenmux pass in `build_artifacts.py` was unreachable from the sandbox. Will work in a non-sandboxed env. |
| 14 | **Multi-template concatenation** for Stage 4 "Variation" check | 1 h | Instead of one template per row, concatenate 2 different templates and de-dup. Increases "substantively different" probability for sampled reasonings. |
| 15 | **Swap LightGBM for CatBoost-only** | 1 h | CatBoost on its own is competitive. Worth testing as a simpler single-model baseline. |
| 16 | **Per-candidate score distribution** | 2 h | Currently 100K candidates all get a single ensemble score. Replace with a percentile-based normalised score (rank within pool) for better interpretability. |
| 17 | **Make / CI** | trivial | DONE in WS-15. `make test`, `make dry-run`, `make audit CSV=...`, `make polish N=10` etc. |
| 18 | **Title-YOE inconsistency hard feature** | trivial | DONE in WS-9. `title_yoe_inconsistent` is 1 if the candidate's title is 2+ buckets below the expected bucket for their YOE. |

## Done in this iteration (closed-loop)

- WS-1..WS-8: see commit history of the previous iteration.
- WS-9..WS-15: see above (P0 #1, P0 #2, P0 #3, P0 #4, P1 #7, P1 #9, P2 #12, P2 #17, P2 #18).

## Skipped (and why)

- **Switching to a hosted LLM API** for reasoning — explicitly forbidden
  by `submission_spec.md:57-59`.
- **Switching to a larger embedder** (bge-large, NV-Embed-v2) — exceeds
  the 5 GB artifact cap or 5-min rank-time on 16 GB CPU.
- **Multi-stage LTR cascades** — adds complexity for marginal NDCG@10
  lift; one LightGBM + one CatBoost is enough.
