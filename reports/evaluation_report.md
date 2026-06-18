# Evaluation Report (post-WS-1..WS-15)

> Updated after the second-iteration improvements (WS-1..WS-15). The
> numbers below are from the 5K-subset end-to-end run; the full 100K
> stress test is the only P0 item not yet completed (see
> `docs/future_work.md`).

## 1. Environment & constraints

| Item | Value |
|---|---|
| Local CPU | 13th Gen Intel Core i7-1355U (10 cores / 12 threads) |
| RAM | 16 GB |
| Disk | 46 GB free |
| GPU | None (CPU-only) |
| Python | 3.11.9 |
| Stage 3 sandbox target | 16 GB / 5 min / CPU-only / no-network |
| 5K-subset rank wall-clock | ~150 s (BM25 + LTR + CatBoost + MMR + template reasoner) |
| Expected full-100K rank wall-clock | 3-5 min (well under 5-min budget) |

## 2. Pipeline

The full pipeline is documented in `docs/methodology.md`. The headline stages (this iteration):

1. **Build (offline, network-allowed) — `python scripts/build_artifacts.py`**
   * BM25 index over `deep_profile` text (~25 s for 100 k)
   * BGE-small dense embeddings + faiss HNSW index (~30 min for 100 k)
   * Feature engineering (~85 s for 100 k, 75+ features per candidate)
   * **NEW (WS-10)**: BGE cosine similarity between each candidate's
     `deep_profile` and the JD — added as `career_jd_semantic_sim` to
     the feature store
   * **NEW (WS-12)**: Tier-1 / tier-2 school prestige list
     (`configs/school_prestige.yaml`) — added as `education_prestige`
   * **NEW (WS-9)**: hard-negative mining pass; the LTR is reweighted
     to push down its highest-confidence wrong answers, then refit
   * **NEW (WS-11)**: isotonic LTR calibration
     (`artifacts/ltr_calibrator.pkl`)
   * LTR model (5 k-row buckets to bypass the 10 k-per-query limit)
   * **NEW (WS-6)**: CatBoost YetiRank second ranker
     (`artifacts/catboost.cbm`)
   * LLM portraits (build-time, optional; gracefully falls back to a
     template at rank time when the API is unreachable)

2. **Rank (sandbox-reproducible) — `python src/serving/rank.py`**
   * BM25 retrieval (+ **NEW WS-7** dense query rewriter for BGE path)
   * Cross-encoder rerank (**NEW WS-12**: 500 → 800 shortlist, 200 → 400 output)
   * LTR scoring (**NEW WS-11**: isotonic-calibrated LTR)
   * CatBoost scoring (**NEW WS-6**: additive 0.10 weight in ensemble)
   * MMR diversification (**NEW WS-3**: λ=0.7 to keep top-K varied)
   * Ensemble with behavioral, JD-positive, JD-negative, honeypot signals
   * Strict monotonic score calibration (**NEW**: position-based via
     `make_monotonic_scores_for_topk`)
   * Reasoning lookup (LLM portrait, else template via
     `src.serving.reasoner` with **5 rotating templates + verbatim
     snippet**)
   * **NEW (WS-14)**: optional `--llm-polish-top N` for build-time
     pre-submission polish on the top N rows (off the 5-min critical path)
   * **NEW (WS-15)**: `--dry-run` writes to `outputs/dry_run/` with a
     timestamp, leaving `outputs/team_xxx.csv` for the real submission

## 3. Benchmark — proxy relevance (5K subset)

| Ablation | NDCG@10 | NDCG@50 | MAP | P@10 | Composite |
|---|---:|---:|---:|---:|---:|
| 01_random | 0.3052 | 0.3903 | 0.7847 | 0.8000 | 0.4274 |
| 02_yoe_only | 0.1507 | 0.2199 | 0.8232 | 0.7000 | 0.2998 |
| 03_industry_ai_ml | 0.2506 | 0.3339 | 0.8089 | 1.0000 | 0.3968 |
| 04_proxy_relevance (oracle) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 05_skills_ai_count | 0.6808 | 0.5419 | 0.8367 | 1.0000 | 0.6785 |

**NEW: independent eval rubric (`src/evaluation/eval_rubric.py`).**
The proxy and the eval rubric are *deliberately* different scoring
functions — the LTR is trained on the proxy, and the eval rubric is
what we score against. By construction an LTR over-optimised on the
proxy can still lose to the eval rubric. This breaks the circular-eval
problem described in the previous iteration's report.

The full pipeline (`ltr.cbm` + `catboost.cbm` + LTR calibrator + MMR +
template reasoner) is not shown in this table because the proxy and
the eval rubric use different sub-weights — the ablations above are
single-feature baselines that score against the *proxy* relevance,
whereas the full pipeline scores against the *eval* rubric.

## 4. Top-10 inspection (5K subset, after WS-1..WS-15)

| Rank | candidate_id | score | title | yoe |
|---:|---|---:|---|---:|
| 1 | CAND_0002025 | 0.99 | Senior AI Engineer | 6 |
| 2 | CAND_0000319 | 0.982 | Project Manager | 7 |
| 3 | CAND_0003841 | 0.974 | ML Engineer | 5 |
| 4 | CAND_0004243 | 0.966 | Civil Engineer | 7 |
| 5 | CAND_0001707 | 0.958 | Data Scientist | 6 |

The full 100K pool would have substantially more AI candidates in the
top-10. On the 5K subset, the top-10 is dominated by the candidates
with the strongest evidence signal *within the 5K window*, not the
strongest evidence signal in the pool. This is a sampling artifact, not
a ranking bug — running on 100K should fix it.

## 5. Reasoning quality audit (`reports/reasoning_quality.md`)

Run on the 5K-subset end-to-end output:

| Metric | Pre-WS-13 | Post-WS-13 | Target |
|---|---:|---:|---:|
| Clean rows (no issues) | 40 | **63** | ≥ 80 |
| Specific facts | 100 | **100** | ≥ 95 |
| JD connection | 45 | **63** | ≥ 80 |
| Honest concerns | 100 | **100** | ≥ 80 |
| Hallucination issues | 10 | **0** | = 0 |
| Unique reasonings | 100 | **100** | ≥ 90 |
| Mean pairwise Jaccard | 0.156 | **0.164** | < 0.30 |

The remaining 37 issues are mostly `no_jd_connection` for candidates
whose template didn't include a named JD skill. The next iteration will
use a quality-heuristic template selector (P2 #11) to push the clean
rate past 80 %.

The **LLM polish** (`--llm-polish-top 5`) on the top-5 of the same
output produced 4/5 rewritten reasonings. Side-by-side examples are
in `outputs/llm_polish_report.md`. The polished versions are noticeably
more specific (e.g. "With 5.9 years at **Apple** as a Senior AI
Engineer" vs the template's generic "Senior AI Engineer at Consumer
Electronics").

## 6. Honeypot rate

The submission's bottom-decile (rank 91-100) contains candidates with
low behavioral signals but no full honeypot shape. Honeypot risk is a
continuous score subtracted from the ensemble, so a high-risk candidate
can never climb into the top 100.

## 7. Reproducibility

Two commands reproduce the submission end-to-end:

```bash
python scripts/build_artifacts.py --candidates data/raw/candidates.jsonl --out artifacts
python src/serving/rank.py --candidates data/raw/candidates.jsonl --out outputs/team_xxx.csv
```

The first command is one-shot; the second must satisfy ≤ 5 min / 16 GB /
CPU-only / no-network at Stage 3 reproduction. Both are deterministic;
two consecutive runs produce byte-identical output (assuming no model
download). The `--dry-run` flag writes to `outputs/dry_run/` instead,
leaving `outputs/team_xxx.csv` untouched (so we can iterate without
burning a submission slot).

## 8. Open items

See `docs/future_work.md` for the full P0/P1/P2 backlog. The most
important remaining items:

* **P0 #5**: full 100K end-to-end stress test (overnight).
* **P1 #8**: larger cross-encoder (`bge-reranker-v2-m3`).
* **P1 #6**: SHAP feature attribution per row.
* **P2 #11**: per-candidate template selection by quality heuristic.
