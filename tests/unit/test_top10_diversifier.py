"""Tests for the top-10 diversity reranker (Agent 8)."""
from __future__ import annotations

from src.ranking.top10_diversifier import (
    Top10Candidate,
    diversify_records,
    diversify_top10,
)


def _cands(n: int = 30, yoe_in_band: bool = True):
    """Build n candidates with score = 1.0 - i/100, varying titles/industries."""
    out = []
    titles = ["Senior ML Engineer", "ML Engineer", "Data Scientist",
              "AI Specialist", "Junior ML Engineer", "Software Engineer",
              "Senior AI Engineer", "ML Researcher", "Backend Engineer",
              "Full Stack Developer"]
    industries = ["AI/ML", "SaaS", "Fintech", "Edtech", "Consulting",
                  "Gaming", "Ecommerce", "Healthcare", "Manufacturing",
                  "Logistics"]
    for i in range(n):
        out.append(Top10Candidate(
            candidate_id=f"CAND_{i:07d}",
            score=1.0 - i / 100.0,
            title=titles[i % len(titles)],
            industry=industries[i % len(industries)],
            company=f"Co{i}",
            yoe=7.0 if yoe_in_band else 11.0 + (i % 5),
            honeypot=0.0,
        ))
    return out


def test_diversify_top10_preserves_top_when_no_constraints_trigger():
    """When no constraints trigger, the top-10 ordering is unchanged."""
    cands = _cands(n=30, yoe_in_band=True)
    out = diversify_top10(cands, top_k=10)
    # The top-10 should be the highest-scoring 10 (no honeypot, all in band).
    top10_ids = [c.candidate_id for c in out[:10]]
    expected = [c.candidate_id for c in cands[:10]]
    assert top10_ids == expected


def test_diversify_top10_pushes_honeypot_candidates():
    """Honeypot candidates in the top-10 must be pushed below rank 10."""
    cands = _cands(n=30)
    # Mark candidate 2 (3rd highest) as honeypot.
    cands[2].honeypot = 0.95
    out = diversify_top10(cands, top_k=10)
    top10 = out[:10]
    assert all(c.honeypot < 0.7 for c in top10)


def test_diversify_top10_yoe_band_coverage():
    """When the top-30 has no candidate in 5-9 YOE, swap one in from the tail."""
    cands = _cands(n=30, yoe_in_band=False)  # all > 9 yrs
    # Make candidate 25 a 5-9 YOE candidate.
    cands[25].yoe = 7.0
    out = diversify_top10(cands, top_k=10)
    # The result should contain cands[25] somewhere in the top window.
    head_ids = {c.candidate_id for c in out[:10]}
    assert "CAND_0000025" in head_ids


def test_diversify_top10_dedup_title_industry():
    """Two candidates with the same (title, industry) in top-10 → one drops."""
    cands = _cands(n=30)
    # Force candidate 1 to share (title, industry) with candidate 0.
    cands[1].title = cands[0].title
    cands[1].industry = cands[0].industry
    out = diversify_top10(cands, top_k=10)
    head = out[:10]
    pairs = {(c.title.lower(), c.industry.lower()) for c in head}
    assert len(pairs) == len(head), (pairs, head)


def test_diversify_top10_handles_empty_input():
    out = diversify_top10([])
    assert out == []


def test_diversify_top10_handles_short_input():
    cands = _cands(n=5)
    out = diversify_top10(cands, top_k=10)
    assert len(out) == 5


def test_diversify_records_wrapper():
    records = [
        {"candidate_id": f"CAND_{i:07d}", "score": 1.0 - i / 10,
         "title": "ML Engineer", "industry": "AI/ML", "company": "Co",
         "yoe": 7.0, "honeypot": 0.0}
        for i in range(15)
    ]
    out = diversify_records(records, top_k=10)
    assert len(out) == 15
    assert {r["candidate_id"] for r in out} == {r["candidate_id"] for r in records}


def test_diversify_top10_returns_same_set():
    """Diversification must be a permutation of the input (no drops)."""
    cands = _cands(n=30)
    out = diversify_top10(cands, top_k=10)
    in_ids = {c.candidate_id for c in cands}
    out_ids = {c.candidate_id for c in out}
    assert in_ids == out_ids
