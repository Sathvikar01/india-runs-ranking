# Reproducibility

Two commands. No GPU, no network (for the ranker), no surprises.

## 1. Prerequisite

* Python 3.11+
* 16 GB RAM
* CPU only
* 5 GB free disk for artifacts

The data bundle (`candidates.jsonl`, `job_description.md`, `candidate_schema.json`) lives in `data/raw/`. It is **not** checked into git because of size and licence.

## 2. One-time build (offline, network-allowed)

```bash
pip install -e ".[dev]"
export ZENMUX_API_KEY=sk-ai-v1-...
python scripts/build_artifacts.py \
    --candidates data/raw/candidates.jsonl \
    --job-description data/raw/job_description.md \
    --out artifacts
```

This produces:

* `artifacts/bm25.pkl` — BM25 index over `deep_profile`
* `artifacts/faiss.index`, `artifacts/faiss.ids.pkl` — faiss HNSW index over BGE-large-en-v1.5 embeddings
* `artifacts/embeddings.npz` — raw float32 vectors
* `artifacts/feature_store.parquet` — per-candidate feature table
* `artifacts/ltr.cbm` — LightGBM LambdaRank booster
* `artifacts/portraits.jsonl` — Zenmux MiMo v2.5 generated recruiter notes (one JSON per line)

Wall-clock on a 16 GB CPU-only laptop: ~ 1.5-2.5 hours, of which 60-90 min is the BGE-large embedding pass and 30-60 min is the Zenmux reasoning pass.

If you want to skip the network-dependent step (e.g. for sandbox smoke testing):

```bash
python scripts/build_artifacts.py ... --skip-reasoning
```

## 3. The ranker (sandbox-reproducible)

```bash
python src/serving/rank.py \
    --candidates data/raw/candidates.jsonl \
    --job-description data/raw/job_description.md \
    --artifacts artifacts \
    --out outputs/team_xxx.csv
```

This is the command Stage 3 will run. It must satisfy:

* ≤ 5 minutes wall-clock
* ≤ 16 GB RAM
* CPU only
* No network calls (the `requests` and `urllib` modules are never imported)

Empirically: < 60 s on a 16 GB CPU laptop.

## 4. Validate

```bash
python scripts/validate.sh outputs/team_xxx.csv
```

This wraps the official `validate_submission.py` and exits non-zero on any spec violation. The CI runs the same check on every push.

## 5. Sandbox demo

```bash
streamlit run src/serving/sandbox_app.py
```

Upload a ≤ 100-candidate JSONL and the JD. The app returns a ranked CSV with per-candidate reasoning. Designed to run on HuggingFace Spaces or Streamlit Cloud.

## 6. Tests and benchmarks

```bash
pytest -q                           # unit + integration + pipeline + evaluation
pytest --cov=src --cov-fail-under=85
python -m scripts.run_ablation --candidates data/raw/candidates.jsonl --size 5000 --report reports/benchmark.md
```

## 7. Containerized run (Docker)

```bash
docker build -t india-runs-ranking .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/artifacts:/app/artifacts" india-runs-ranking \
    --candidates data/raw/candidates.jsonl \
    --job-description data/raw/job_description.md \
    --artifacts artifacts \
    --out outputs/team_xxx.csv
```

The image is built with `python:3.11-slim` and the runtime dependencies in `requirements.txt`.

## 8. Determinism

* Random seeds are pinned in `configs/build.yaml` and the LTR trainer.
* BM25 is order-sensitive on input; we always feed the same iteration order (jsonl stream, sorted by file offset).
* faiss HNSW queries are deterministic for a fixed index.
* Cross-encoder inference is deterministic at `temperature=0` (it has no temperature by default).
* LightGBM inference is deterministic.

Two consecutive runs of `src/serving/rank.py` against the same artifacts will produce a byte-identical CSV.
