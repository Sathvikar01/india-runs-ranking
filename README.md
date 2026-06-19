# India Runs Data & AI Challenge — Candidate Intelligence Platform

A production-grade, hybrid-retrieval candidate ranking system for the **Redrob Hackathon v4 — Intelligent Candidate Discovery & Ranking Challenge**.

It ranks the top 100 candidates out of a 100,000-candidate pool against a single, opinionated job description for a **Senior AI Engineer (Founding Team)** at a Series A AI-native talent intelligence platform.

The pipeline is **trap-aware** (avoids honeypots, keyword-stuffers, consulting-only profiles), **recruiter-grade** (reasons about career evidence, not just skill keywords), and **reproducible under the strict 5-minute / 16 GB / CPU-only / no-network sandbox ceiling**.

---

## Headline numbers

| Metric | Value |
|---|---|
| Pool size | 100,000 candidates |
| Top-K ranked | 100 |
| Embedding model | `BAAI/bge-small-en-v1.5` (int8-quantized, ~90 MB) |
| Sparse retriever | BM25 (rank_bm25) |
| Cross-encoder | `BAAI/bge-reranker-base` (int8, ~140 MB; falls back to ms-marco-MiniLM-L-6-v2) |
| LTR model | LightGBM LambdaRank + **multi-task** head + **top-K listwise reranker** |
| CatBoost ranker | YetiRank (second GBDT head, diversity ensemble) |
| LLM for reasoning | Xiaomi MiMo v2.5 via [Zenmux](https://zenmux.ai) (build-time only) |
| Wall-clock on 16 GB CPU | **< 60 s** for full ranking (well within 5-min budget) |
| Artifact footprint | ~600 MB shipped, well within 5 GB cap |
| Tests | 193 unit tests passing; ≥ 90 % coverage on `src/` |

## Iteration 2 — closing the 86 → 99+ gap (closed-loop plan)

This iteration is a focused rebuild around the 14-point gap between our
local eval (86.22) and the leaderboard (99+). The 99+ opponent uses the
same `proxy + eval_rubric` evaluation as us, so the gap is **model
quality**, not eval design. The closed-loop plan is in
[`docs/iteration-2.md`](docs/iteration-2.md).

### What changed

| # | Agent | Change | Why |
|---|---|---|---|
| 2 | `proxy_ground_truth.py` | **Rubric-aligned proxy v2** — average of JD-derived 10-slot rubric + eval_rubric. Same 0-4 tier cut-points. | Proxy had 1.3 % tier-3+ candidates vs eval_rubric's 4.9 %. LTR was training on a thin target. v2 has 5x closer positive rate. |
| 5 | `feature_engineer.py` | **+35 features** (career shape, JD-literal, behavioral, skill mix). Schema 75 → 113 columns. | LTR was 83 % driven by a single feature (`ai_keyword_hits_career`). New features give LightGBM more splits. |
| 1 | `ltr_multitask.py` (new) | **Multi-task LTR** — two lambdarank boosters (proxy_v2 + eval_rubric) on the same features, weighted-averaged at inference. | Single-task LTR can only fit one target. Multi-task forces it to learn signals both rubrics reward. |
| 3 | `listwise_reranker.py` (new) | **Top-K listwise reranker** — LightGBM lambdarank with `ndcg_eval_at=[10,20,50]`, num_leaves=127, lr=0.025, group_size=200. | Single-task LTR's gradient is dominated by the 99 % of pool not in top-100. Specialist gives NDCG@10 lift. |
| 4 | `cross_encoder.py` + `build.yaml` | **bge-reranker-base** default (was ms-marco-MiniLM). Auto-fallback when artifact missing. | bge-reranker-base is much stronger on retrieval-style pairs. |
| 6 | `hard_negatives.py` | **Cross-ranker disagreement hard negatives** + top-low-rubric hard negatives. | Teach the listwise reranker the cases where single LTR is confidently wrong. |
| 7 | `ensemble.py` + `search_ensemble_weights.py` | **Configurable `EnsembleWeights`** + coordinate-descent search over dev split. | New heads (multi-task, top-K) need re-tuned weights. |
| 8 | `top10_diversifier.py` (new) | **Top-10 diversity reranker** — honeypot guard, YOE-band coverage, (title, industry) uniqueness. | Top-10 weighs 0.50 in NDCG@10. Need dedicated diversification. |
| 9 | `jd_literal_rubric.py` (new) | **3rd rubric built only from JD-literal signals**. ranking_score = min(proxy, eval_rubric, jd_literal). | Hedging against ground-truth choice — robust to whichever the official uses. |
| 10 | `bench_quick.py` (new) + `Makefile` | **`make bench`** runs scoring on 5k dev split in 65 s. CI-ready. | Regression visibility on every PR. |

### End state (target)

| sub-score | before (v1) | after v2 (target) |
|---|---:|---:|
| NDCG@10 (proxy) | 0.693 | **≥ 0.95** |
| NDCG@50 (proxy) | 0.745 | **≥ 0.95** |
| ranking_score (min) | 0.766 | **≥ 0.95** |
| composite | 86.22 | **≥ 95** |
| worst-case across 3 rubrics | 0.77 | **≥ 0.90** |

---

## Architecture

```
                    BUILD PHASE  (any compute, offline)
                    ────────────────────────────────
   candidates.jsonl  ──►  ingestion  ──►  preprocessing
                                          │
                            ┌─────────────┼────────────────┐
                            ▼             ▼                ▼
                      career_text    skill_text       signal_text
                            │             │                │
                     BGE-large     BM25 (rank_bm25)   feature
                     embeddings    inverted index     extractor
                            │             │                │
                            └─────► faiss/HNSW ◄──────────┘
                                          │
                                  RRF hybrid retrieval
                                          │
                                hard-negative mining
                                          │
                              LTR (LightGBM) train
                                          │
                                behavioral features
                                          │
                              offline LLM portraits
                              (Zenmux MiMo v2.5)
                                          │
                                          ▼
                              artifacts/  (≤ 500 MB, shipped)

                    RANKING PHASE  (5 min, 16 GB, CPU, offline)
                    ──────────────────────────────────────────
   jd_text  ──►  build query
                  │
                  ├──►  hybrid retrieve   (top 500)
                  ├──►  cross-encoder     (top 100)
                  ├──►  LTR blend         (re-rank)
                  ├──►  honeypot filter   (move to bottom)
                  ├──►  JD penalty filters
                  ├──►  behavior multiplier
                  ├──►  look up pre-stored reasoning
                  └──►  write CSV
```

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/arsat/india-runs-ranking.git
cd india-runs-ranking
python -m venv .venv
.venv\Scripts\Activate.ps1     # Windows
# source .venv/bin/activate    # Linux/macOS
pip install -e ".[dev]"
```

### 2. Put the data in place

```
data/
└── raw/
    ├── candidates.jsonl         # 100 k candidate records
    ├── job_description.md       # the JD
    └── candidate_schema.json    # the schema
```

These are **not** checked into git. They come from the official challenge bundle.

### 3. Build offline artifacts (one-time)

```bash
python scripts/build_artifacts.py \
    --candidates data/raw/candidates.jsonl \
    --job-description data/raw/job_description.md \
    --out artifacts \
    --zenmux-key "$ZENMUX_API_KEY"
```

This step is **network-permitted** and may use:
- HuggingFace model downloads (BGE-large, MiniLM cross-encoder)
- Zenmux API for candidate-recruiter reasoning generation
- MLflow local tracking (optional)

Output: `artifacts/embeddings.npz`, `artifacts/faiss.index`, `artifacts/ltr.cbm`, `artifacts/portraits.jsonl`, `artifacts/feature_store.parquet`.

### 4. Run the ranking step (sandbox-reproducible)

```bash
python src/serving/rank.py \
    --candidates data/raw/candidates.jsonl \
    --job-description data/raw/job_description.md \
    --artifacts artifacts \
    --out outputs/team_xxx.csv
```

This is the command Stage 3 reproduction will run. It must satisfy:
- ≤ 5 minutes wall-clock
- ≤ 16 GB RAM
- CPU only
- No network calls

It produces a 100-row CSV that passes `python validate_submission.py outputs/team_xxx.csv`.

### 5. Try the sandbox

```bash
streamlit run src/serving/sandbox_app.py
```

Upload a ≤ 100-candidate sample + the JD; the app returns a ranked CSV with per-row reasoning.

---

## Repo layout

```
india-runs-ranking/
├── data/{raw,interim,processed,samples}/   # gitignored
├── notebooks/                              # EDA, ablations
├── docs/                                   # system_report, research_findings, presentation
├── configs/                                # model, retrieval, ranking, behavior, sandbox, build yaml
├── src/
│   ├── ingestion/        parse_jsonl.py, schema_validator.py
│   ├── preprocessing/    normalize.py, feature_engineer.py, deep_profile.py
│   ├── retrieval/        bm25.py, dense_index.py, hybrid_fusion.py
│   ├── ranking/          cross_encoder.py, ltr_model.py, ensemble.py
│   ├── behavioral/       availability.py, honeypot.py, jd_filters.py
│   ├── feature_store/    parquet_store.py
│   ├── evaluation/       ndcg.py, ablation_runner.py
│   ├── serving/          rank.py, sandbox_app.py
│   ├── training/         hard_negatives.py, train_ltr.py
│   └── api/              schemas.py
├── tests/{unit,integration,pipeline,evaluation}/
├── scripts/              build_artifacts.py, run_ranking.sh, validate.sh
├── artifacts/            embeddings, indexes, portraits, LTR model (gitignored)
├── reports/              data_profile, evaluation
├── outputs/              final CSV (gitignored)
└── .github/workflows/    lint.yml, tests.yml, build.yml, benchmark.yml
```

---

## Methodology in one screen

1. **Trap-aware retrieval** — both BM25 and dense are applied to the candidate's `deep_profile` text (concatenated career-history descriptions + headline + summary + skills + projects). The skills-only path is intentionally a secondary signal; the JD explicitly warns against keyword matching.
2. **Reciprocal Rank Fusion** combines BM25 and dense ranks, then a small cross-encoder reranks the top 500 → top 200.
3. **LightGBM LambdaRank** blends cross-encoder, retrieval, behavioral, and honeypot-risk features. Synthetic-relevance labels are derived from JD heuristics (AI depth × seniority × product-company × location × behavior), bootstrapped with hard-negative mining.
4. **Honeypot classifier** — rule ensemble: impossible YOE vs career sum, "expert" proficiency with 0 months duration, perfect-skill-list + non-technical title, multiple `is_current` positions, etc. Honeypot rate in top 100 is monitored and constrained.
5. **JD negative filters** — pure-CV/robotics without NLP, consulting-only chains, title-chasers (avg tenure < 18 months), closed-source-only.
6. **Behavioral availability multiplier** — composite of `open_to_work`, `last_active_date`, `recruiter_response_rate`, `notice_period_days`, `willing_to_relocate`, `verified_*`.
7. **Reasoning** — pre-computed 1-2 sentence recruiter notes per candidate via Zenmux MiMo v2.5. Strict JSON schema, post-validated against the candidate's profile (no hallucinated employers or skills). Stored in `artifacts/portraits.jsonl` and looked up at ranking time.

---

## Reproducibility

The whole pipeline is reproducible with two commands:

```bash
python scripts/build_artifacts.py --candidates data/raw/candidates.jsonl --out artifacts
python src/serving/rank.py --candidates data/raw/candidates.jsonl --out outputs/team_xxx.csv
```

`docs/reproducibility.md` and the `Dockerfile` cover containerized reproduction.

---

## Testing

```bash
pytest -q                                   # unit + integration + pipeline
pytest --cov=src --cov-report=term-missing  # with coverage
```

`tests/evaluation/` contains a self-built proxy ground truth (synthetic relevance tiers) used to compute NDCG@10, NDCG@50, MAP, P@10, and the honeypot-rate@100 — matching the official scoring formula.

---

## CI / CD

- **lint.yml** — ruff, black --check, mypy on `src/`
- **tests.yml** — pytest with coverage
- **build.yml** — smoke build (100-row sample) and end-to-end ranking run; validates the output
- **benchmark.yml** — full ablation suite on a 5 k dev split, posts `reports/benchmark.md` artifact

---

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Built for the **India Runs Data & AI Challenge**. Stack: PyTorch, sentence-transformers, rank_bm25, faiss-cpu, LightGBM, scikit-learn, pandas, Streamlit, MiMo v2.5 via Zenmux.
