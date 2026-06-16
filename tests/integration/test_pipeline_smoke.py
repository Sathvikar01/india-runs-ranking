from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.integration
def test_build_artifacts_smoke(tmp_path: Path):
    """Smoke build on the first 200 candidates. ~2-3 min on a 16 GB CPU."""
    src = Path("[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl")
    jd = Path("data/raw/job_description.md")
    if not jd.exists():
        # Convert from docx
        import zipfile
        import re
        import html
        with zipfile.ZipFile("[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/job_description.docx") as z:
            x = z.read("word/document.xml").decode("utf-8", "ignore")
        x = re.sub(r"</w:p>", "\n", x)
        x = re.sub(r"<[^>]+>", "", x)
        jd.parent.mkdir(parents=True, exist_ok=True)
        jd.write_text(html.unescape(x), encoding="utf-8")

    out = tmp_path / "artifacts"
    cmd = [
        sys.executable,
        "-m",
        "scripts.build_artifacts",
        "--candidates",
        str(src),
        "--job-description",
        str(jd),
        "--out",
        str(out),
        "--max-candidates",
        "200",
        "--skip-reasoning",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        pytest.fail(f"build_artifacts failed: {r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    assert (out / "bm25.pkl").exists()
    assert (out / "faiss.index").exists()
    assert (out / "feature_store.parquet").exists()
    assert (out / "ltr.cbm").exists()
    # artifacts directory must be < 500 MB
    size = sum(p.stat().st_size for p in out.rglob("*"))
    assert size < 500 * 1024 * 1024, f"artifacts too large: {size}"


@pytest.mark.integration
def test_rank_sandbox_runs(tmp_path: Path):
    """Run rank.py on the pre-built 200-candidate artifacts and validate output."""
    src = Path("[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl")
    jd = Path("data/raw/job_description.md")
    if not jd.exists():
        # Reuse logic above
        import zipfile, re, html
        with zipfile.ZipFile("[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/job_description.docx") as z:
            x = z.read("word/document.xml").decode("utf-8", "ignore")
        x = re.sub(r"</w:p>", "\n", x)
        x = re.sub(r"<[^>]+>", "", x)
        jd.parent.mkdir(parents=True, exist_ok=True)
        jd.write_text(html.unescape(x), encoding="utf-8")

    # First build a tiny artifact set
    art = tmp_path / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "scripts.build_artifacts",
        "--candidates", str(src),
        "--job-description", str(jd),
        "--out", str(art),
        "--max-candidates", "200",
        "--skip-reasoning",
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=900)

    out_csv = tmp_path / "team_xxx.csv"
    cmd = [
        sys.executable, "-m", "src.serving.rank",
        "--candidates", str(src),
        "--job-description", str(jd),
        "--artifacts", str(art),
        "--out", str(out_csv),
        "--top-k", "100",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        pytest.fail(f"rank failed: {r.stdout[-2000:]}\n{r.stderr[-2000:]}")

    df = pd.read_csv(out_csv)
    assert len(df) == 100
    assert list(df.columns) == ["candidate_id", "rank", "score", "reasoning"]
    assert df["rank"].tolist() == list(range(1, 101))
    # monotonic non-increasing
    assert all(df["score"].iloc[i] >= df["score"].iloc[i + 1] for i in range(len(df) - 1))
    # CAND_9999999 (honeypot) should NOT be in top 100
    assert "CAND_9999999" not in df["candidate_id"].values
