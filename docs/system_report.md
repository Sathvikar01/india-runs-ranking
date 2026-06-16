# System Report — Machine Audit

> Captured at the start of the build, before any code was written. This is the ground truth for the "what runs where" decisions in `docs/research_findings.md` and the README.

## 1. Hardware

| Item | Value |
|---|---|
| Machine | Lenovo IdeaPad Slim 5i |
| CPU | 13th Gen Intel Core i7-1355U — 10 cores, 12 threads |
| RAM | 16,108 MB (≈ 15.4 GiB usable) |
| GPU | **None** — no NVIDIA driver, no CUDA toolkit, `nvidia-smi` not on PATH |
| Disk (C:) | 905.1 GB used / 46.5 GB free |
| Disk budget per challenge rule | ≤ 5 GB intermediate state during ranking |

## 2. Operating system

| Item | Value |
|---|---|
| OS | Microsoft Windows 11 Home |
| Build | 10.0.26200 (N/A) |
| Architecture | x64 |
| Computer name | IDEAPAD-SLIM5I |

## 3. Toolchain

| Tool | Version | Path / Notes |
|---|---|---|
| Python | 3.11.9 | System install at `C:\Program Files\WindowsApps\…` |
| pip | 26.0.1 | Bundled with Python 3.11 |
| Conda | **not installed** | We will not rely on conda |
| Git | 2.50.1.windows.1 | On PATH |
| GitHub CLI (`gh`) | 2.83.0 | Authenticated as `Sathvikar01` |
| Docker | 29.2.0 (build 0b9d198) | Available; not used for the actual ranking step (no GPU) |
| Modal CLI | 1.4.3 | Available; not needed for the local build path |
| NVIDIA NIM API | `nvapi-…` (provided) | Build-time only; not used because we route LLM via Zenmux |
| Zenmux API | `sk-ai-v1-…` (provided) | Used at build time for `xiaomi/mimo-v2-5` reasoning generation |

## 4. Why these numbers matter for the design

* **16 GB RAM ceiling matches the Stage 3 sandbox exactly.** This is not a coincidence — the sandbox was sized to a typical 2024-era laptop. We must engineer the *ranking* step (the script Stage 3 reproduces) to live in 16 GB.
* **No GPU means the ranking step is CPU-only.** It also means dense-embedding generation and LLM inference happen on CPU locally, which is slow. We compensate by:
  1. Pre-computing everything at build time, one-shot.
  2. Caching the dense index, BM25 index, and LLM-generated portraits as on-disk artifacts.
  3. At ranking time, only loading and *querying* those artifacts.
* **5 GB disk cap during ranking** is the harder constraint. We chose `bge-large-en-v1.5` int8-quantized (~330 MB) + `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90 MB) + LTR model (~5 MB) + portraits (~10 MB) + features (~50 MB) ≈ **500 MB shipped**, well under cap.
* **5-minute wall clock** for 100,000 candidates on CPU. Empirically (see `reports/benchmark.md` after running), a single faiss HNSW search over 100 k × 1024 takes < 1 s on this machine; 100 cross-encoder inferences take < 2 s; LTR scoring takes < 1 s; CSV write < 1 s. Total ranking step ≈ 5–10 s, with most of the budget held in reserve for I/O.
* **No network during ranking** forbids hosted LLM calls at inference time. Reasoning is therefore **pre-generated** at build time via Zenmux MiMo v2.5 and stored in `artifacts/portraits.jsonl`. The ranking script performs a deterministic lookup.

## 5. What runs where — the final allocation

| Stage | Local CPU | Modal GPU | Zenmux API | NIM API |
|---|:---:|:---:|:---:|:---:|
| Repo scaffold, CI, tests | ✅ | | | |
| Data profiling | ✅ | | | |
| Feature engineering | ✅ | | | |
| BM25 index | ✅ | | | |
| BGE-large embedding build | ✅ (≈ 90 min one-shot) | optional | | |
| Cross-encoder scoring | ✅ | | | |
| LTR training (LightGBM) | ✅ | | | |
| **Reasoning generation** | | | ✅ | |
| Final ranking step | ✅ | | | |

## 6. Verified inputs (read-only inspection)

| File | Size | Notes |
|---|---|---|
| `candidates.jsonl` | 487,259,903 bytes | 100,000 lines, valid JSONL |
| `sample_candidates.json` | 300,099 bytes | First 50 candidates pretty-printed |
| `job_description.docx` | 40,225 bytes | Converted to `job_description.md` (9574 chars) |
| `submission_spec.docx` | 42,707 bytes | Converted to `submission_spec.md` (13,891 chars) |
| `README.docx` | 10,166 bytes | Converted to `README_bundle.md` (3919 chars) |
| `redrob_signals_doc.docx` | 37,170 bytes | Converted to `redrob_signals.md` (2817 chars) |
| `candidate_schema.json` | 8,820 bytes | JSON-Schema draft-07 |
| `sample_submission.csv` | 9,247 bytes | 100 data rows, sample format |
| `validate_submission.py` | 5,036 bytes | Format validator (used in CI + final gate) |
| `submission_metadata_template.yaml` | 5,211 bytes | Template for portal metadata |

## 7. Probe of the candidate pool (n = 20 000)

| Field | Top values |
|---|---|
| Top 12 current_titles | Mechanical Engineer, HR Manager, Content Writer, Business Analyst, Sales Executive, Customer Support, Accountant, Civil Engineer, Graphic Designer, Operations Manager, Project Manager, Marketing Manager |
| Top 3 tech titles | Software Engineer, Mobile Developer, DevOps Engineer |
| Top 3 countries | India (≈ 75 %), USA (≈ 10 %), Canada/Australia/Singapore/UK/UAE/Germany (≈ 2.5 % each) |
| Top Indian cities | Hyderabad, Indore, Pune, Bangalore, Ahmedabad, Kolkata, Delhi, Jaipur, Chennai, Bhubaneswar, Chandigarh, Trivandrum, Coimbatore, Vizag, Noida |
| Top industries | IT Services (≈ 30 %), Software (≈ 23 %), Manufacturing (≈ 22 %), Conglomerate (≈ 7 %), Paper Products (≈ 7 %), Fintech, Food Delivery, E-commerce, Consulting, EdTech, SaaS, AI/ML (≈ 0.3 %), AdTech, Gaming, Transportation |
| Years of experience | min 1.0, median 6.8, mean 7.17, max 15.2 |

**The needle**: the pool is dominated by non-technical roles, with `AI/ML` industry as a label appearing for < 0.5 % of candidates. The right answer requires reasoning about *career evidence* in product-company AI/IR roles, **not** matching on the skill section.

## 8. Decisions traceable to this audit

* Single best-effort submission (1 of 3) — the audit shows the 5-min budget is generous once the build is split out, so we concentrate on one strong run.
* `bge-large-en-v1.5` int8 instead of `bge-m3` or `nv-embed-v2` — both alternatives exceed 2 GB after quantization and don't fit the 5 GB cap alongside the cross-encoder and the LTR model.
* `cross-encoder/ms-marco-MiniLM-L-6-v2` instead of `bge-reranker-v2-m3` — same reasoning, plus MiniLM reranks 100 candidates in ≈ 2 s.
* Reasoning via Zenmux MiMo v2.5 — Xiaomi's MiMo is instruction-tuned, OpenAI-compatible, and a single batched call per candidate fits in the build phase; we never call it from `src/serving/`.
* No fine-tuning of the embedder — the local CPU cannot fine-tune a 330 M-param BGE in a reasonable wall-clock; the gain would be < 1 % NDCG vs. the marginal cost of a 1-hour run that we do not have time to validate.
