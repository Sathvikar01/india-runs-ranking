from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.integration
def test_validate_submission_passes(tmp_path: Path):
    """Build a synthetic 100-row CSV that must pass the official validator."""
    out = tmp_path / "team_xxx.csv"
    rows = []
    for rank in range(1, 101):
        cid = f"CAND_{rank:07d}"
        score = round(0.99 - (rank - 1) * 0.008, 4)
        rows.append({"candidate_id": cid, "rank": rank, "score": score, "reasoning": f"Test row {rank}."})
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        w.writeheader()
        w.writerows(rows)

    cmd = [
        sys.executable,
        "[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/validate_submission.py",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"validator rejected: {r.stdout}\n{r.stderr}"


@pytest.mark.integration
def test_validate_submission_rejects_bad_rows(tmp_path: Path):
    out = tmp_path / "team_xxx.csv"
    # 99 rows instead of 100
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank in range(1, 100):
            w.writerow([f"CAND_{rank:07d}", rank, 0.5, ""])
    cmd = [
        sys.executable,
        "[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/validate_submission.py",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode != 0
