# Makefile for the India Runs Data & AI Challenge project.
#
# Standard targets for the iteration loop. Anything destructive (rebuild,
# retrain) is gated behind an explicit `confirm` variable.
#
# Usage examples:
#   make test
#   make dry-run
#   make audit CSV=outputs/team_xxx.csv
#   make build confirm=1

.PHONY: help test unit integration dry-run build retrain audit polish clean eval bench search-weights train-multitask train-topk dev-build dev-bm25

PYTHON ?= python
ARTIFACTS ?= artifacts
DATA := data/raw/candidates.jsonl
JD := data/raw/job_description.md

help:
	@echo "make targets:"
	@echo "  test         Run unit tests (no coverage threshold)"
	@echo "  unit         Run unit tests only"
	@echo "  dry-run      Run the ranker with --dry-run (writes to outputs/dry_run/)"
	@echo "  build        Build all artifacts from scratch (slow, ~2h on 16 GB CPU)"
	@echo "  retrain      Retrain the LTR (uses existing feature_store.parquet)"
	@echo "  audit        Run the reasoning quality audit on a CSV (set CSV=...)"
	@echo "  polish       Run LLM polish on top N (set N=10) of a CSV"
	@echo "  eval         Run the full evaluation harness (writes to evaluation/results/)"
	@echo "  clean        Remove local artifacts (does not delete the model GGUF)"

test:
	$(PYTHON) -m pytest tests/unit --no-cov

unit:
	$(PYTHON) -m pytest tests/unit --no-cov -q

integration:
	$(PYTHON) -m pytest tests/integration --no-cov -q

dry-run:
	$(PYTHON) -m src.serving.rank --dry-run \
	    --candidates $(DATA) \
	    --job-description $(JD) \
	    --artifacts $(ARTIFACTS) \
	    --out outputs/team_xxx.csv

build:
	@if [ -z "$(confirm)" ]; then \
	    echo "Refusing to rebuild without confirm=1 (this overwrites artifacts)"; \
	    exit 1; \
	fi
	$(PYTHON) -m scripts.build_artifacts \
	    --candidates $(DATA) \
	    --job-description $(JD) \
	    --out $(ARTIFACTS)

retrain:
	$(PYTHON) -m scripts.train_ltr \
	    --candidates $(DATA) \
	    --feature-parquet $(ARTIFACTS)/feature_store.parquet \
	    --out $(ARTIFACTS)/ltr.cbm \
	    --k-folds 5 \
	    --num-boost-round 600

audit:
	@if [ -z "$(CSV)" ]; then \
	    echo "Usage: make audit CSV=outputs/team_xxx.csv"; \
	    exit 1; \
	fi
	$(PYTHON) scripts/audit_reasoning_quality.py $(CSV) \
	    --candidates $(DATA) \
	    --out reports/reasoning_quality.md

polish:
	@if [ -z "$(CSV)" ]; then echo "set CSV=..."; exit 1; fi
	@if [ -z "$(N)" ]; then N=10; fi
	$(PYTHON) -m src.serving.rank \
	    --candidates $(DATA) \
	    --job-description $(JD) \
	    --artifacts $(ARTIFACTS) \
	    --out $(CSV) \
	    --llm-polish-top $(N)

clean:
	rm -rf $(ARTIFACTS)/*.parquet $(ARTIFACTS)/*.cbm $(ARTIFACTS)/*.pkl $(ARTIFACTS)/*.npz
	rm -rf outputs/dry_run
	rm -f reports/reasoning_quality.md
	rm -f outputs/llm_polish_report.md

eval:
	@if [ -z "$(CSV)" ]; then \
	    $(PYTHON) evaluation/run_evaluation.py; \
	else \
	    $(PYTHON) evaluation/run_evaluation.py --skip-ranker --csv $(CSV); \
	fi

# Agent 10 — quick local benchmarks (5k dev split). Use these to validate
# every change in <2 min without rebuilding artifacts.
bench:
	$(PYTHON) scripts/bench_quick.py

search-weights:
	$(PYTHON) scripts/search_ensemble_weights.py \
	    --candidates data/raw/candidates_5k.jsonl \
	    --out artifacts/best_ensemble_weights.json \
	    --rounds 2

train-multitask:
	$(PYTHON) scripts/train_ltr_multitask.py \
	    --candidates data/raw/candidates_5k.jsonl \
	    --out-dir artifacts/ltr_multitask \
	    --num-boost-round 600

train-topk:
	$(PYTHON) scripts/train_listwise_reranker.py \
	    --candidates data/raw/candidates_5k.jsonl \
	    --out artifacts/ltr_topk.cbm \
	    --num-boost-round 1500

# Agent dev — full dev pipeline on the 5k split (~3 min on dev CPU).
# Trains: BM25 + features + multi-task LTR + top-K reranker + ltr.cbm.
# Output: a fresh dev-build of artifacts/ that the ranker can consume
# directly. The new artifacts use the 113-feature schema from
# Agent 5 (vs the legacy 75-feature schema).
dev-build:
	$(PYTHON) scripts/build_bm25.py --candidates data/raw/candidates_5k.jsonl
	$(PYTHON) scripts/dev_build_ltrs.py

dev-bm25:
	$(PYTHON) scripts/build_bm25.py --candidates data/raw/candidates_5k.jsonl

# Run the ranker on the 5k dev split with the dev-built artifacts.
dev-rank:
	$(PYTHON) -m src.serving.rank \
	    --candidates data/raw/candidates_5k.jsonl \
	    --job-description data/raw/job_description.md \
	    --artifacts artifacts \
	    --out outputs/team_xxx_v2.csv \
	    --dry-run
