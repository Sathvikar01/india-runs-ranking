#!/usr/bin/env bash
# Run the ranking step (this is what Stage 3 reproduces).
# No network, CPU-only, must finish in <= 5 min on 16 GB RAM.

set -euo pipefail
cd "$(dirname "$0")/.."

TEAM_ID="${TEAM_ID:-team_xxx}"
python src/serving/rank.py \
  --candidates data/raw/candidates.jsonl \
  --job-description data/raw/job_description.md \
  --artifacts artifacts \
  --out "outputs/${TEAM_ID}.csv"
