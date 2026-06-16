#!/usr/bin/env bash
# Validate a submission CSV against the official rules.

set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -lt 1 ]; then
  echo "Usage: scripts/validate.sh outputs/team_xxx.csv"
  exit 1
fi
python "[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/validate_submission.py" "$1"
