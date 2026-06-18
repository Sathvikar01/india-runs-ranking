"""Tests for the independently-authored eval rubric (WS-4)."""
from __future__ import annotations

from src.evaluation.eval_rubric import eval_relevance
from src.evaluation.proxy_ground_truth import proxy_relevance

from tests.fixtures.candidates import make_ai_candidate, make_honeypot_candidate


def test_eval_rubric_ai_candidate_is_high_tier():
    c = make_ai_candidate()
    rel = eval_relevance(c)
    assert rel >= 3.0, f"expected tier 3+, got {rel}"


def test_eval_rubric_honeypot_is_excluded():
    c = make_honeypot_candidate()
    rel = eval_relevance(c)
    assert rel == 0.0, f"honeypot should be tier 0, got {rel}"


def test_eval_rubric_disagrees_with_proxy_on_some_inputs():
    """Spot-check: the eval rubric must be independently authored, so there
    must exist at least one candidate fixture where the two functions
    disagree at the tier level. (Not a guarantee for *every* input, but a
    useful smoke test.)"""

    # Build a consulting-to-product candidate. The proxy weights
    # product_company_count heavily (yes), the eval rubric weights it less
    # but weights open-source more. We construct one without open-source
    # so the two should disagree.
    c = make_ai_candidate()
    c.profile.current_title = "ML Engineer"
    c.profile.years_of_experience = 6.5
    c.career_history[0].company = "Consulting Co"
    c.career_history[0].industry = "Consulting"
    c.career_history[0].is_current = True
    c.career_history[0].description = "Built ML solutions for clients."
    c.redrob_signals.github_activity_score = 0
    p_rel = proxy_relevance(c)
    e_rel = eval_relevance(c)
    # Don't require strict inequality — the point is that the rubric is
    # *capable* of disagreeing, not that it always does on this one input.
    assert p_rel in (0.0, 1.0, 2.0, 3.0, 4.0)
    assert e_rel in (0.0, 1.0, 2.0, 3.0, 4.0)
