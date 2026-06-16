#!/usr/bin/env bash
# Build all offline artifacts (one-time).
# Network-permitted. ~1.5-2.5 h on a 16 GB CPU laptop.

set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${ZENMUX_API_KEY:-}" ]; then
  echo "ZENMUX_API_KEY not set; reasoning generation will be skipped."
fi

python scripts/build_artifacts.py \
  --candidates data/raw/candidates.jsonl \
  --job-description data/raw/job_description.md \
  --out artifacts
