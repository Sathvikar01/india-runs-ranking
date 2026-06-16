# Final Report — Candidate Intelligence Platform

> The complete, competition-ready submission for the Redrob Hackathon v4 — Intelligent Candidate Discovery & Ranking Challenge.

## What is in this repo

```
india-runs-ranking/
├── README.md                         # entry point
├── LICENSE
├── CONTRIBUTING.md
├── submission_metadata.yaml          # portal metadata mirror
├── pyproject.toml                    # pip-installable
├── requirements.txt                  # runtime deps
├── Dockerfile
├──
├── docs/
│   ├── system_report.md              # machine audit (16 GB CPU, no GPU)
│   ├── research_findings.md          # surveyed literature, sources cited
│   ├── methodology.md                # design narrative
│   ├── reproducibility.md             # two-command reproduction
│   ├── final_presentation.pptx       # 15-slide deck
│   └── final_presentation.pdf        # exported via PowerPoint COM
├──
├── configs/                          # all tunables in YAML
├──
├── src/                              # the production-grade code
│   ├── ingestion/        parse_jsonl, schema_validator
│   ├── preprocessing/    normalize, feature_engineer, deep_profile
│   ├── retrieval/        bm25, dense_index, hybrid_fusion
│   ├── ranking/          cross_encoder, ltr_model, ensemble
│   ├── behavioral/       availability, honeypot, jd_filters  (vectorized)
│   ├── feature_store/    parquet_store
│   ├── evaluation/       ndcg, ablation_runner, proxy_ground_truth
│   ├── serving/          rank (sandbox-reproducible), sandbox_app (Streamlit)
│   ├── training/         hard_negatives, train_ltr (bucketed LambdaRank)
│   └── api/              Pydantic schemas
├──
├── tests/                            # unit + integration + pipeline + evaluation
├── scripts/                          # build_artifacts, rank, ablation, presentation
├── artifacts/                        # produced by build_artifacts (~500 MB)
├── reports/                          # data_profile, evaluation_report, benchmark
└── outputs/team_xxx.csv              # the submission (validates against official rules)
```

## Headline numbers

| Metric | Value |
|---|---|
| Pool size | 100,000 candidates |
| Top-K ranked | 100 |
| Embedding model | `BAAI/bge-small-en-v1.5` (384-d, 90 MB) — optional |
| Sparse retriever | BM25 (rank_bm25) over `deep_profile` text |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90 MB) |
| LTR model | LightGBM LambdaRank, 67 features, bucketed groups |
| Wall-clock on 16 GB CPU | **~ 120 s** for full ranking (well within 5-min budget) |
| Artifact footprint | ~ 500 MB shipped, well within 5 GB cap |
| Honeypot detection | 7-rule ensemble, subtracted from final ensemble |
| Tests | 63 unit + integration + pipeline, all passing |

## How to reproduce

Two commands, no GPU, no manual steps. See `docs/reproducibility.md`.

## Methodology

The full design narrative is in `docs/methodology.md`. The 30-second version:

1. **Trap-aware framing.** The JD is explicit that keyword matching is the trap. We weight career-history evidence above the skill list, build a `deep_profile` text accordingly, and subtract honeypot risk from the final score.
2. **Hybrid retrieval.** BM25 + optional dense + RRF; cross-encoder reranks the top 500.
3. **LightGBM LambdaRank** on 67 engineered features, trained with 5 k-row buckets to satisfy the 10 k-per-query limit.
4. **Ensemble** blends LTR, cross-encoder, behavioral availability, JD-positive boosters, JD-negative filters, and honeypot risk.
5. **Strict monotonic score calibration** in [0.20, 0.99] with per-rank jitter.
6. **Reasoning** is pre-stored (LLM) or feature-driven template at rank time, with a hard fallback when the LLM API is unreachable.

## Why this can win

* **Stage 1 (format validation)** — submission validates against the official rules.
* **Stage 2 (scoring)** — the LTR + ensemble closes ~ 80 % of the gap between the random baseline and the proxy oracle on the dev split. The top-10 surface area is all AI/ML roles, in or near the 5-9 yrs band, with low honeypot risk.
* **Stage 3 (reproduction)** — the pipeline runs in ~ 120 s on a 16 GB CPU laptop, well within the 5-min budget.
* **Stage 4 (manual review)** — reasoning is template-based but specific, JD-connected, and honest about gaps.
* **Stage 5 (defend-your-work)** — clean architecture, 63 tests, full documentation, methodology coherent with the JD.

## Why a single best-effort submission

The 3-attempt cap, no live leaderboard, and the JD's explicit warnings against experiment-driven submissions all favor one strong run. We use one slot; two are kept in reserve in case the leaderboard shows a surprising ranking pattern after the reveal.

## Limitations and what we would do with more time

* Replace `bge-small-en` with `bge-m3` or `NV-Embed-v2` in a one-time cloud build (5 GB disk cap limits the artifact set we ship).
* Add a ColBERTv2 late-interaction rerank over the top 200.
* Bootstrap a much larger LTR training set with active learning on the top-100 disagreements.
* Add a per-JD query rewriter (LLM, build-time) to expand the BM25 query with synonyms and related terms.
* Use the Zenmux API for LLM portraits at build time when the network is reachable; in this run, the API was unreachable from the sandbox so the template fallback was used.

## Acknowledgements

* Redrob / India Runs organizers for the well-designed challenge
* BAAI for the BGE embedding family
* HuggingFace for `sentence-transformers` and `cross-encoder`
* Microsoft for LightGBM
* Xiaomi for MiMo v2.5 via Zenmux
* Open-source maintainers of every dependency in `requirements.txt`
