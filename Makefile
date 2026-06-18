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

.PHONY: help test unit integration dry-run build retrain audit polish clean eval

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
