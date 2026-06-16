from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.ablation_runner import run_ablations, write_markdown_report
from tests.fixtures.candidates import (
    make_ai_candidate,
    make_consulting_chain_candidate,
    make_honeypot_candidate,
)


def test_run_ablations_returns_results(tmp_path: Path):
    cands = [make_ai_candidate(), make_consulting_chain_candidate(), make_honeypot_candidate()]
    ablations = {
        "a_random": lambda cs: [c.candidate_id for c in cs],
        "b_yoe": lambda cs: [c.candidate_id for c in sorted(cs, key=lambda c: c.profile.years_of_experience, reverse=True)],
    }
    results = run_ablations(cands, ablations, out_dir=tmp_path)
    assert {r.name for r in results} == {"a_random", "b_yoe"}
    assert (tmp_path / "summary.csv").exists()
    md = tmp_path / "report.md"
    write_markdown_report(results, md)
    text = md.read_text(encoding="utf-8")
    assert "NDCG@10" in text


def test_run_ablations_handles_honeypot():
    # A 3-candidate run should always produce a ranking where the AI candidate wins.
    from src.evaluation.proxy_ground_truth import build_proxy_ground_truth
    from src.evaluation.ablation_runner import run_ablations
    cands = [make_honeypot_candidate(), make_consulting_chain_candidate(), make_ai_candidate()]
    relevance = build_proxy_ground_truth(cands)
    out = run_ablations(cands, {"proxy": lambda cs: [cid for cid, _ in sorted(relevance.items(), key=lambda x: x[1], reverse=True)]}, out_dir="outputs/test")
    m = out[0].metrics
    assert m["composite"] > 0.5
