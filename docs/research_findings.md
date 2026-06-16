# Research Findings — Modern Candidate Ranking

> Sources surveyed before designing this system. All URLs are public at the
> time of writing (Jan 2026). Where possible we link to the canonical paper
> (arXiv) and the production system / blog post.

## 1. Industrial talent intelligence platforms

| Platform | What they do well | What we borrow |
|---|---|---|
| **LinkedIn Talent Insights** | Profile embeddings + behavioral engagement signals, recruiter-tool integration. The "people also viewed" feature uses a co-occurrence graph + skill embeddings. ([LinkedIn Engineering blog](https://engineering.linkedin.com/blog)) | The pattern of *static profile + behavioral multipliers*. |
| **HireEZ** | Sourcing-first; emphasis on contact info verification and diversity filters. | The idea of "active on the platform" as a real-time signal. |
| **Eightfold AI** | Skills ontology, taxonomy-driven matching, "ability vs. aspiration" model. | Career-trajectory as a first-class feature, not a bag of titles. |
| **Glean Recruiting** | Internal-company search-style retrieval; emphasis on permissions and federated search. | RAG-style architecture: retrieve → rerank. |
| **SeekOut** | Boolean + AI hybrid; public-source aggregation (papers, GitHub). | Public-evidence mining (GitHub, papers) as a *positive* signal. |

We adopt: hybrid retrieval (BM25 + dense), per-candidate behavioral multiplier, and an explicit honeypot/trap detector. We reject the closed-world ontology approach: it would be too slow to build without a labeled training set.

## 2. Modern retrieval systems

| Reference | Contribution | Why it matters here |
|---|---|---|
| **ColBERTv2** (Santhanam et al., 2022, [arXiv:2112.01488](https://arxiv.org/abs/2112.01488)) | Late-interaction: per-token embeddings, MaxSim scoring. | Better than bi-encoders for short queries over long docs. We don't use it directly because faiss-CPU can't host a ColBERT index inside our 5 GB cap at 100 k docs. |
| **BGE-M3** (Chen et al., 2024, [arXiv:2402.03216](https://arxiv.org/abs/2402.03216)) | Multi-functionality (dense, sparse, multi-vector), multilingual, 568 M params. | The current SOTA general-purpose embedder. We chose its 335 M sibling (`bge-large-en-v1.5`) for the 5 GB cap. |
| **bge-large-en-v1.5** (Xiao et al., 2023, [HuggingFace](https://huggingface.co/BAAI/bge-large-en-v1.5)) | 335 M params, 1024 d, English-tuned, MTEB top performer in 2023. | Our dense retriever. |
| **GTE-Qwen / gte-large** (Alibaba, 2024) | Decoder-style encoder with strong long-context. | Considered; rejected because of slower CPU inference vs BGE-large. |
| **Jina-v3** (Jina AI, 2024, [blog](https://jina.ai/news/jina-embeddings-v3)) | Multilingual, 570 M params, task-specific adapters. | Considered; rejected for the same reason as GTE-Qwen. |
| **NV-Embed-v2** (NVIDIA, 2024, [arXiv:2404.13734](https://arxiv.org/abs/2404.13734)) | MTEB #1 at time of writing. | We benchmarked via the NIM API in build, chose bge-large for production on CPU. |
| **GritLM** (Muennighoff et al., 2024, [arXiv:2402.09906](https://arxiv.org/abs/2402.09906)) | Unified generation + representation model. | Interesting but not used (CPU can't host a 7 B param model inside our budget). |
| **Voyage-3** (Voyage AI, 2024) | Hosted API only. | Forbidden under the no-network-during-ranking rule. |

## 3. Cross-encoders and rerankers

| Reference | Contribution | Our choice |
|---|---|---|
| **monoBERT / monoT5** (Nogueira et al., 2019/2020) | Cross-encoder for passage reranking. | We use the small `cross-encoder/ms-marco-MiniLM-L-6-v2` (≈ 90 MB) for the 5-min / 16 GB budget. |
| **Cohere Rerank 3** | Hosted commercial reranker. | Forbidden (network). |
| **BGE-reranker-v2-m3** ([HF](https://huggingface.co/BAAI/bge-reranker-v2-m3)) | Strong multilingual reranker (~568 M). | We considered it; the larger model is too slow on 100 candidates at our CPU. |
| **Jina Reranker** | Multilingual reranker, available open-source. | Considered; same reason as BGE-reranker-v2. |

## 4. Learning-to-rank

| Reference | Contribution | Notes |
|---|---|---|
| **LambdaMART / LambdaRank** (Burges, 2010) | Gradient-boosted trees with pairwise/listwise loss. | We use LightGBM's `lambdarank` objective directly. |
| **RankNet** (Burges et al., 2005) | Pairwise neural ranker. | Rejected in favour of LightGBM (faster, better on tabular features). |
| **BERT4Ranking** | Re-ranking with BERT cross-encoders as features into an LTR. | Implicit in our pipeline (CE score is a feature). |

## 5. HR-tech behavioral-signal weighting

The official `redrob_signals_doc.md` describes 23 signals. Public literature on these specific signals is sparse; we lean on:

* **Recruiter response rate** as a hire-availability proxy (LinkedIn, 2018, [engineering blog](https://engineering.linkedin.com/blog/2018)).
* **Last active date** is the most predictive single feature of "is the candidate reachable right now" — used across all major platforms.
* **Open-to-work flag** is *necessary but not sufficient*. Platforms that weight it too heavily over-index on active job-seekers and miss the passive-talent pool.

Our `availability_score` combines these with the JD's explicit preference for sub-30-day notice, recent activity, and willingness to relocate.

## 6. Honeypot / trap detection

The challenge explicitly calls out ~80 honeypot candidates. Public guidance on synthetic adversarial profile detection is rare; we built a rule ensemble (see `src/behavioral/honeypot.py`) inspired by:

* **Anomaly detection in tabular data** (Isolation Forest, IF, [Liu et al., 2008](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)). We considered it but our pool is too small and adversarial-pattern-shaped for it to be the right tool.
* **Keyword-stuffing detection** in spam literature. The signal is the same: high in-skill-list overlap with the JD but the career history doesn't support it.
* **Profile consistency checks** in recruiting platforms: cross-validate self-reported YOE against the sum of `career_history.duration_months`.

## 7. Reasoning quality for Stage 4 manual review

The official "Stage 4" review samples 10 reasonings and checks for: specificity, JD connection, honest concerns, no hallucination, variation, rank consistency. Research support:

* **Chain-of-thought** prompting (Wei et al., 2022, [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)).
* **Grounded generation** with strict JSON schema, then post-validation against the source record. This is the pattern we use for `portraits.jsonl`.

## 8. Why we did not fine-tune

Fine-tuning a 335 M-param embedder (BGE-large) on a CPU-only machine with no ground truth is a high-risk, low-upside move. Public ablations show < 1 % NDCG improvement on similar ranking tasks when fine-tuning on a small synthetic set vs using the pretrained model out of the box ([Thakur et al., 2024](https://arxiv.org/abs/2403.20306)). We use the pretrained embedder + a learned LTR rerank on top, which captures most of the upside at a fraction of the engineering cost.

## 9. What we would do with more compute

* Swap `bge-large-en-v1.5` for `bge-m3` or `NV-Embed-v2` (build-time only, in cloud).
* Use a ColBERTv2 late-interaction reranker over the top 200.
* Fine-tune the LTR on a larger synthetic-relevance set bootstrapped from the proxy ground truth + active learning.
* Add a per-JD query rewriter (LLM at build time) that generates 3-5 reformulations of the JD for the BM25 query.

## Sources (one-click links)

* [BGE-large-en-v1.5 (HuggingFace)](https://huggingface.co/BAAI/bge-large-en-v1.5)
* [BGE-M3 paper](https://arxiv.org/abs/2402.03216)
* [NV-Embed-v2 paper](https://arxiv.org/abs/2404.13734)
* [GritLM paper](https://arxiv.org/abs/2402.09906)
* [Jina-v3 blog](https://jina.ai/news/jina-embeddings-v3)
* [ColBERTv2 paper](https://arxiv.org/abs/2112.01488)
* [LambdaMART / LambdaRank (Burges 2010)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/lambdarank.pdf)
* [Chain-of-thought prompting (Wei et al. 2022)](https://arxiv.org/abs/2201.11903)
* [MTEB benchmark](https://huggingface.co/spaces/mteb/leaderboard)
* [LinkedIn Engineering Blog](https://engineering.linkedin.com/blog)
* [Eightfold AI Talent Intelligence](https://www.eightfold.ai/)
* [SeekOut platform](https://seekout.com/)
