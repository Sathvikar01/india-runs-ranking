"""Tests for the reasoning quality audit (WS-13)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_reasoning_quality import (  # type: ignore
    _has_honest_concerns,
    _has_jd_connection,
    _has_specific_facts,
    _load_candidates,
    _no_hallucination,
    _rank_consistency,
    _variation_score,
    audit,
)


def test_specific_facts_positive():
    assert _has_specific_facts("ML Engineer with 7 yrs at Acme AI; response 60%")
    assert _has_specific_facts("Notice period 30 days, recency 80%")


def test_specific_facts_negative():
    assert not _has_specific_facts("Great fit for the role.")


def test_jd_connection_positive():
    assert _has_jd_connection("Hands-on RAG and vector search experience.")
    assert _has_jd_connection("Built and shipped a hybrid retrieval + cross-encoder ranking system.")
    assert _has_jd_connection("LLM fine-tuning with LoRA and PEFT.")


def test_jd_connection_negative():
    assert not _has_jd_connection("Manager with 8 yrs of experience")


def test_honest_concerns_positive():
    assert _has_honest_concerns("Concern: location is not Noida/Pune.")
    assert _has_honest_concerns("Limited AI/ML evidence in the career history.")


def test_honest_concerns_negative():
    assert not _has_honest_concerns("Excellent fit.")


def test_no_hallucination_clean():
    profile = {"all_profile_text": "acme ai machine learning pytorch"}
    issues = _no_hallucination("Worked at Acme AI on PyTorch.", profile)
    assert issues == []


def test_no_hallucination_flags_unknown_employer():
    profile = {"all_profile_text": "acme ai machine learning pytorch"}
    # "Acme AI" appears in the profile, "Quantum Dynamics" does not.
    issues = _no_hallucination("Worked at Acme AI and Quantum Dynamics on PyTorch.", profile)
    # "Quantum Dynamics" is a capitalized bigram not in the profile.
    assert any("Quantum Dynamics" in i for i in issues)


def test_no_hallucination_flags_superlative():
    profile = {"all_profile_text": "machine learning pytorch"}
    issues = _no_hallucination("An impressive and stellar engineer.", profile)
    assert "superlative" in issues


def test_rank_consistency_top_concern_mismatch():
    """A rank-1 with 'Concern:' is a tone mismatch."""
    assert _rank_consistency(1, "Concern: limited AI/ML evidence.") == "tone_mismatch_top_concern"


def test_rank_consistency_bottom_no_concern_mismatch():
    """A rank-95 with no concerns is also a tone mismatch."""
    assert _rank_consistency(95, "Strong production signal; open to work.") == "tone_mismatch_bottom_no_concern"


def test_rank_consistency_ok():
    assert _rank_consistency(1, "Strong production signal; open to work.") == "ok"
    assert _rank_consistency(95, "Concern: location is not Noida/Pune.") == "ok"


def test_variation_score_low_for_distinct():
    """5 different reasonings → low mean Jaccard."""
    rs = [
        "Senior ML Engineer at Acme AI with 7 yrs of RAG and ranking experience.",
        "Backend Engineer with 5 yrs of Java and Spring Boot.",
        "Data Scientist specializing in NLP and LLM fine-tuning with PyTorch.",
        "Civil Engineer with no AI experience but solid project management.",
        "Marketing Manager with strong stakeholder engagement.",
    ]
    out = _variation_score(rs)
    assert out["mean_jaccard"] < 0.4
    assert out["n_unique"] == 5


def test_variation_score_high_for_identical():
    """All-identical reasonings → Jaccard 1.0 and 1 unique."""
    rs = ["Same reasoning."] * 5
    out = _variation_score(rs)
    assert out["mean_jaccard"] == 1.0
    assert out["n_unique"] == 1


def test_audit_writes_markdown(tmp_path):
    """End-to-end audit on a synthetic 5-row CSV."""
    out = tmp_path / "submission.csv"
    md = tmp_path / "report.md"
    rows = []
    for rank in range(1, 6):
        rows.append({
            "candidate_id": f"CAND_{rank:07d}",
            "rank": rank,
            "score": 0.9 - rank * 0.01,
            "reasoning": (
                f"Candidate {rank} with 6 yrs of RAG and retrieval experience, "
                f"response 60%, recency 80%. Concern: location is not Noida."
            ),
        })
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        w.writeheader()
        w.writerows(rows)
    summary = audit(str(out), None, str(md))
    assert summary["n_rows"] == 5
    assert summary["n_with_facts"] == 5
    assert summary["n_with_jd_connection"] == 5
    assert summary["n_hallucination_issues"] == 0
    assert md.exists()
    # The report should mention our summary.
    text = md.read_text(encoding="utf-8")
    assert "Reasoning Quality Audit" in text
    assert "Rows:" in text
