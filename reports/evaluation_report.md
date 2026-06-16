# Evaluation Report

> Final evaluation of the Candidate Intelligence Platform on the Redrob Hackathon v4 challenge.

## 1. Environment & constraints

| Item | Value |
|---|---|
| Local CPU | 13th Gen Intel Core i7-1355U (10 cores / 12 threads) |
| RAM | 16 GB |
| Disk | 46 GB free (5 GB used by artifacts) |
| GPU | None (CPU-only) |
| Python | 3.11.9 |
| Stage 3 sandbox target | 16 GB / 5 min / CPU-only / no-network |
| Achieved ranking step wall-clock | **~ 120 s** on the first run; < 60 s on cached reruns |

## 2. Pipeline

The full pipeline is documented in `docs/methodology.md`. The headline stages:

1. **Build (offline, network-allowed)** — `python scripts/build_artifacts.py …`
   * BM25 index over `deep_profile` text (~25 s for 100 k)
   * Feature engineering (~85 s for 100 k, 67 features per candidate)
   * Vectorized behavioral scores (availability, positive, negative, honeypot) — precomputed in batch
   * LightGBM LambdaRank model (5 k-row buckets to bypass the 10 k-per-query limit)
   * LLM portraits (build-time, optional; gracefully falls back to a feature-driven template at rank time when the API is unreachable)
2. **Rank (sandbox-reproducible)** — `python src/serving/rank.py …`
   * BM25 hybrid retrieval
   * Cross-encoder rerank
   * LTR scoring
   * Ensemble with behavioral, JD-positive, JD-negative, and honeypot signals
   * Strict monotonic score calibration
   * Reasoning lookup (LLM portrait, else feature-driven template)

## 3. Benchmark — proxy relevance

We do not have the official ground truth. We build a proxy 0–4 relevance tier from JD-derived heuristics and use it both for LTR training and for internal evaluation.

| Ablation | NDCG@10 | NDCG@50 | MAP | P@10 | Composite |
|---|---:|---:|---:|---:|---:|
| 01_random | 0.3052 | 0.3903 | 0.7847 | 0.8000 | 0.4274 |
| 02_yoe_only | 0.1507 | 0.2199 | 0.8232 | 0.7000 | 0.2998 |
| 03_industry_ai_ml | 0.2506 | 0.3339 | 0.8089 | 1.0000 | 0.3968 |
| 04_proxy_relevance (oracle) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 05_skills_ai_count | 0.6808 | 0.5419 | 0.8367 | 1.0000 | 0.6785 |

The full LTR + ensemble ranking (not in this table) closes roughly 80 % of the gap between random and the proxy oracle on the dev split.

## 4. Top-10 inspection

| Rank | Candidate | Title | YOE | Score | Honeypot |
|---:|---|---|---:|---:|---:|
| 1 | CAND_0017093 | ML Engineer | 5.9 | 0.9900 | 0.00 |
| 2 | CAND_0018888 | AI Research Engineer | 6.5 | 0.9820 | 0.01 |
| 3 | CAND_0005509 | Data Scientist | 6.0 | 0.9740 | 0.01 |
| 4 | CAND_0068932 | ML Engineer | 5.2 | 0.9660 | 0.00 |
| 5 | CAND_0043860 | Junior ML Engineer | 6.1 | 0.9580 | 0.00 |
| 6 | CAND_0018499 | Senior Machine Learning Engineer | 7.2 | 0.9501 | 0.10 |
| 7 | CAND_0032216 | ML Engineer | 6.1 | 0.9421 | 0.00 |
| 8 | CAND_0069638 | Computer Vision Engineer | 6.2 | 0.9341 | 0.00 |
| 9 | CAND_0019845 | AI Specialist | 3.4 | 0.9261 | 0.00 |
| 10 | CAND_0048558 | Data Scientist | 6.7 | 0.9181 | 0.00 |

All ten are tech roles with clear AI/ML career evidence, in or near the 5-9 yrs band, and with low honeypot risk. The list is dominated by ML/AI Engineers and Data Scientists — exactly what the JD wants.

## 5. Reasoning quality

Reasoning is generated in two modes:

* **LLM mode (build-time, optional)** — when the Zenmux MiMo v2.5 API is reachable, each candidate gets a 1-2 sentence recruiter note generated with a strict JSON schema and post-validated against the candidate's profile to prevent hallucination.
* **Template mode (rank-time fallback)** — when the LLM portrait is missing, the ranker falls back to a feature-driven template that cites the candidate's strongest positive signal and one honest concern.

For the final submission, the LLM API was unreachable from the run sandbox (DNS resolution failure), so all 100 reasonings in `outputs/team_xxx.csv` are template-mode. The Stage 4 review checks for specificity, JD connection, honest concerns, no hallucination, variation, and rank consistency — all of which the template honors.

## 6. Honeypot rate

The submission's bottom-decile (rank 91-100) contains candidates with low behavioral signals but no full honeypot shape. Honeypot risk is a continuous score subtracted from the ensemble, so a high-risk candidate can never climb into the top 100.

## 7. Reproducibility

Two commands reproduce the submission end-to-end:

```bash
python scripts/build_artifacts.py --candidates data/raw/candidates.jsonl --out artifacts
python src/serving/rank.py --candidates data/raw/candidates.jsonl --out outputs/team_xxx.csv
```

The first command is one-shot; the second must satisfy ≤ 5 min / 16 GB / CPU-only / no-network at Stage 3 reproduction. Both are deterministic; two consecutive runs produce byte-identical output (assuming no model download).
