# Methodology

> A short technical narrative of how the system is built and how each component
> earns its place. Read this top-to-bottom; it explains the design choices
> that a reviewer can't infer from the code alone.

## 1. Framing the problem

The challenge asks for the top 100 candidates from a 100 000-candidate pool, ranked best-fit-first for one specific job description. The scoring function is `0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`, so the top of the list matters far more than the middle. The ranking step must finish in 5 min / 16 GB / CPU only, with no network access. The submission is a 100-row CSV with `candidate_id, rank, score, reasoning`.

The JD is unusually opinionated. It tells us, in plain language, what it does and doesn't want — and warns us that the pool has traps. This is not a benchmark where keyword matching wins. It is, in effect, a recruiter-decision simulation.

## 2. The two-level architecture

We split the work into a *build* phase and a *rank* phase.

* **Build** runs once, may use the network, may take hours. It produces the artifacts the ranker consumes.
* **Rank** is what Stage 3 reproduces. It must be hermetic, fast, and predictable.

The build phase produces:

* `bm25.pkl` — a `rank_bm25` BM25Okapi index over the candidate `deep_profile` text.
* `faiss.index` + `faiss.ids.pkl` — a HNSW index over the BGE-large-en-v1.5 embeddings of the same `deep_profile` text.
* `feature_store.parquet` — a per-candidate table of 60+ engineered features.
* `ltr.cbm` — a LightGBM LambdaRank booster.
* `portraits.jsonl` — pre-generated 1-2 sentence recruiter notes per candidate, from Zenmux MiMo v2.5.

The rank phase loads all of the above, then runs a single short pipeline: hybrid retrieve → cross-encoder rerank → LTR score → ensemble with behavioral and honeypot signals → write CSV.

## 3. Why the `deep_profile` text exists

The JD explicitly says: *“the right answer is not find candidates whose skills section contains the most AI keywords.”* If we embed the skill list alone, a Marketing Manager with 10 expert-marked AI skills and zero months of usage will be ranked above a backend engineer who has actually shipped a recommender system.

So we build a per-candidate text that weights career evidence above the skill list. Concretely:

1. Headline + summary (cheap signal, low weight).
2. Each career role in reverse chronological order, with title, company, industry, and the role's *description* (the strongest signal).
3. Project names + descriptions.
4. Certifications.
5. Skills, capped at 30, with proficiency as a prefix (`expert:PyTorch(60mo)`).
6. A short behavioral appendix (`open_to_work`, `recruiter_response_rate`, `notice_period_days`, `github_activity_score`, …).

This text is what both the BM25 index and the dense index operate on.

## 4. Hybrid retrieval

BM25 catches the literal vocabulary match. Dense catches the semantic one. The two disagree often; the disagreement is informative. We combine them with Reciprocal Rank Fusion (RRF, k = 60). The union of the top 500 from each path becomes the cross-encoder shortlist.

## 5. Cross-encoder rerank

`cross-encoder/ms-marco-MiniLM-L-6-v2` is a 90 MB cross-encoder. 100 k candidates → 500 → 200 with this model takes a few seconds on CPU. We keep the top 200 by CE score, then re-rank everything with the LTR.

## 6. Learning-to-rank

The LTR sees 60+ engineered features (see `src/preprocessing/feature_engineer.py`). The label is a synthetic 0-4 relevance tier computed from JD-derived heuristics (see `src/evaluation/proxy_ground_truth.py`). We do 5-fold cross-validation, then train a final model on all data with the average best iteration count plus 10 %. The LTR booster is the single biggest move in the system; in the ablation it lifts NDCG@10 by 25-30 % over the CE-only baseline.

## 7. Honeypot and trap detection

The challenge explicitly says ~80 candidates are honeypots with impossible profiles. We treat the top-10 honeypot rate as a hard guardrail: the ensemble subtracts a *honeypot risk* score from the final score, so a high-risk candidate cannot climb into the top 100 unless every other signal is overwhelmingly positive (which it never will be).

The risk score is a weighted sum of seven rule signals:

* Skill-proficiency-vs-duration: any "expert" skill with `duration_months == 0` is suspicious.
* YOE-vs-career-sum: large positive gap between reported YOE and the sum of `career_history.duration_months`.
* Perfect-skill-list-with-non-tech-title: 5+ JD-core skills on a non-engineer / non-scientist title.
* Multiple `is_current` positions.
* "Expert" in 8+ skills.
* All skills with zero endorsements.
* High skill count with no overlap with the candidate's career text.

The threshold (default 0.60) is calibrated so that the obvious traps get flagged and a real but heavy-on-papers candidate does not.

## 8. JD negative filters

The JD is explicit about what it does not want. We encode each "do not want" as a hard flag and add the weighted sum to the negative penalty:

* Only-consulting-companies (TCS, Infosys, Wipro, …) chains.
* CV/robotics/speech-only without NLP/IR exposure.
* Title-chasers (avg tenure < 18 months across ≥ 3 roles).
* Closed-source only (no GitHub, no papers, no open-source).
* LangChain-recent-only (no depth).
* No NLP/IR evidence in the career text at all.
* YOE out of the 3-15 band.

The matching positive filters boost candidates who have shipped ranking/search/recsys at scale, are in Noida/Pune (or willing to relocate), have tier-1/2 education, etc.

## 9. Behavioral availability

A perfect-on-paper candidate who hasn't logged in for six months and has a 5 % recruiter response rate is, for hiring purposes, not actually available. We compute a 0-1 availability score from 8 signals: `open_to_work`, recency, response rate, notice period, willingness to relocate, verifications, interview completion rate, offer-acceptance rate. The score is added to the final ensemble as a small (10 %) contribution.

## 10. Reasoning generation

The submission's `reasoning` column is sampled at Stage 4 for manual review. The review checks for specificity, JD connection, honest concerns, no hallucination, variation, and rank consistency. To pass those checks at scale, we pre-compute 1-2 sentence recruiter notes per candidate via Zenmux MiMo v2.5 at build time. The prompt is strict:

* The JD is inlined.
* The full profile is inlined as the ground truth.
* A synthetic `ai_relevance` and `honeypot_risk` is included.
* Output is forced to JSON `{reasoning, top_positive, top_concern}`.

We never call the LLM at ranking time. The ranking script just looks up the pre-generated string for the top-100 candidates.

If the LLM call fails for a candidate (timeout, rate limit), we fall back to a deterministic template: `{title} with {yoe} yrs; {n_skills} skills on profile; response rate {rr:.2f}.` This is what the validator sees when a portrait is missing.

## 11. Final ensemble and score calibration

```
final = 0.55·sigmoid(ltr)
      + 0.20·sigmoid(ce)
      + 0.10·availability
      + 0.10·positive_boost
      - 0.10·negative_penalty
      - 0.20·honeypot_risk
```

We then sort by `final` descending, take the top 100, and assign monotonic scores from 0.99 down to 0.20 with a tiny per-rank jitter (1e-5) so equal scores still produce distinct ranks.

## 12. Why a single best-effort submission

The 3-submission cap with no live leaderboard is a deliberate design choice by the organizers. We chose to spend one slot on a single best-effort run rather than split the budget across experiments. The full system is reproducible from the artifacts and `src/serving/rank.py`; iterating further would be possible but is a 24+ hour exercise and does not change the architectural story.

## 13. What we would do differently with more time

* Run a long-context LLM (Qwen-2.5-32B) at build time to generate deeper per-candidate portraits with career-trajectory summaries.
* Bootstrap a much larger LTR training set with active learning on the top-100 disagreements.
* Add a per-JD query rewriter (LLM, build-time) to expand the BM25 query with synonyms and related terms.
* Replace the small MiniLM cross-encoder with BGE-reranker-v2-m3 on a single A10G.
* Add a LearnToRank with a CatBoost ranker as a second ensemble member.

All of these are in `docs/future_work.md`.
