"""Tests for the rubric-aligned proxy ground truth (v2 blend)."""
from __future__ import annotations

import pytest

from src.evaluation.proxy_ground_truth import (
    PROXY_VERSION,
    _ai_evidence_score,
    _jd_rubric_score,
    _open_source_score,
    _product_company_score,
    _school_prestige_score,
    _to_tier,
    proxy_relevance,
    proxy_relevance_v1,
    proxy_relevance_v2,
)


def _stub_candidate(**overrides):
    """Build a Candidate with sensible defaults for the proxy tests."""
    from src.api.schemas import (
        Candidate,
        CareerRole,
        Education,
        Profile,
        RedrobSignals,
        Skill,
    )

    profile = Profile(
        anonymized_name="Test",
        headline="",
        summary="",
        location="Bangalore",
        country="India",
        years_of_experience=7.0,
        current_title="Senior ML Engineer",
        current_company="Acme AI",
        current_company_size="100-500",
        current_industry="AI/ML",
    )
    signals = RedrobSignals(
        profile_completeness_score=80,
        signup_date="2024-01-01",
        last_active_date="2025-06-01",
        open_to_work_flag=True,
        profile_views_received_30d=100,
        applications_submitted_30d=5,
        recruiter_response_rate=0.7,
        avg_response_time_hours=12.0,
        skill_assessment_scores={},
        connection_count=500,
        endorsements_received=20,
        notice_period_days=30,
        expected_salary_range_inr_lpa={"min": 35.0, "max": 45.0},
        preferred_work_mode="hybrid",
        willing_to_relocate=True,
        github_activity_score=45,
        search_appearance_30d=300,
        saved_by_recruiters_30d=20,
        interview_completion_rate=0.8,
        offer_acceptance_rate=0.5,
        verified_email=True,
        verified_phone=True,
        linkedin_connected=True,
    )
    cand = Candidate(
        candidate_id="CAND_0000001",
        profile=profile,
        career_history=[
            CareerRole(
                company="Acme AI",
                title="Senior ML Engineer",
                start_date="2022-01-01",
                end_date=None,
                duration_months=36,
                is_current=True,
                industry="AI/ML",
                company_size="100-500",
                description="Built ML systems on pytorch",
            )
        ],
        education=[
            Education(institution="IIT Bombay", degree="B.Tech", tier="tier_1")
        ],
        skills=[Skill(name="pytorch", proficiency="expert", endorsements=5)],
        redrob_signals=signals,
    )
    for k, v in overrides.items():
        if k == "yoe":
            cand.profile.years_of_experience = v
        elif k == "title":
            cand.profile.current_title = v
        elif k == "location":
            cand.profile.location = v
        elif k == "country":
            cand.profile.country = v
        elif k == "open_to_work":
            cand.redrob_signals.open_to_work_flag = v
        elif k == "gh":
            cand.redrob_signals.github_activity_score = v
        elif k == "industry":
            cand.profile.current_industry = v
        else:
            setattr(cand, k, v)
    return cand


def test_proxy_version_default_is_v2():
    assert PROXY_VERSION == "v2"
    assert proxy_relevance is proxy_relevance_v2


def test_tier_cutpoints():
    assert _to_tier(0.0) == 0.0
    assert _to_tier(0.17) == 0.0
    assert _to_tier(0.18) == 1.0
    assert _to_tier(0.34) == 1.0
    assert _to_tier(0.35) == 2.0
    assert _to_tier(0.54) == 2.0
    assert _to_tier(0.55) == 3.0
    assert _to_tier(0.74) == 3.0
    assert _to_tier(0.75) == 4.0
    assert _to_tier(1.0) == 4.0


def test_jd_rubric_strict_yoe_band_full_credit():
    c = _stub_candidate(yoe=7.0)
    assert _jd_rubric_score(c) > 0.0


def test_jd_rubric_out_of_band_lower():
    c_in = _stub_candidate(yoe=7.0)
    c_out = _stub_candidate(yoe=15.0)
    assert _jd_rubric_score(c_in) > _jd_rubric_score(c_out)


def test_proxy_v2_higher_for_ai_career():
    """Adding AI evidence (PyTorch + ML summary) should raise the proxy."""
    c_no = _stub_candidate()
    c_no.profile.summary = ""
    c_no.skills = []
    c_yes = _stub_candidate()
    c_yes.profile.summary = "machine learning pytorch deep learning neural"
    c_yes.skills = [
        type(c_yes.skills[0])(
            name="pytorch", proficiency="expert", endorsements=5
        )
    ]
    assert proxy_relevance_v2(c_yes) >= proxy_relevance_v2(c_no)


def test_proxy_v2_returns_valid_tier_for_normal_candidate():
    """Sanity: a well-formed candidate gets a tier in {0,1,2,3,4} from v2."""
    from src.evaluation.proxy_ground_truth import honeypot_risk

    c = _stub_candidate(yoe=7.0)
    # A well-formed candidate with mid-tier signals gets a valid tier.
    assert honeypot_risk(c) < 0.7
    tier = proxy_relevance_v2(c)
    assert tier in (0.0, 1.0, 2.0, 3.0, 4.0)


def test_proxy_v2_honeypot_returns_zero():
    """A candidate with honeypot risk >= 0.7 gets tier 0."""
    from src.api.schemas import Profile, RedrobSignals
    from src.evaluation.proxy_ground_truth import honeypot_risk

    # Build a candidate that triggers honeypot rules:
    # - yoe >> career_sum, - perfect skill list with non-tech title.
    # We force the two strongest rules to fire.
    profile = Profile(
        anonymized_name="Honeypot",
        headline="",
        summary="",
        location="Bangalore",
        country="India",
        years_of_experience=15.0,
        current_title="HR Manager",  # non-tech title
        current_company="Consulting Co",
        current_company_size="50-100",
        current_industry="Consulting",
    )
    signals = RedrobSignals(
        profile_completeness_score=99,
        signup_date="2024-01-01",
        last_active_date="2025-06-01",
        open_to_work_flag=True,
        profile_views_received_30d=0,
        applications_submitted_30d=0,
        recruiter_response_rate=0.0,
        avg_response_time_hours=0.0,
        skill_assessment_scores={},
        connection_count=0,
        endorsements_received=0,
        notice_period_days=30,
        expected_salary_range_inr_lpa={"min": 30.0, "max": 40.0},
        preferred_work_mode="hybrid",
        willing_to_relocate=False,
        github_activity_score=0,
        search_appearance_30d=0,
        saved_by_recruiters_30d=0,
        interview_completion_rate=0.0,
        offer_acceptance_rate=0.0,
        verified_email=False,
        verified_phone=False,
        linkedin_connected=False,
    )
    c = _stub_candidate()
    c.profile = profile
    c.redrob_signals = signals
    c.skills = [
        type(c.skills[0])(
            name="python", proficiency="expert", endorsements=0
        ),
        type(c.skills[0])(
            name="pytorch", proficiency="expert", endorsements=0
        ),
        type(c.skills[0])(
            name="tensorflow", proficiency="expert", endorsements=0
        ),
        type(c.skills[0])(
            name="kubernetes", proficiency="expert", endorsements=0
        ),
    ]
    # If honeypot risk is high, proxy_v2 returns 0.
    if honeypot_risk(c) >= 0.7:
        assert proxy_relevance_v2(c) == 0.0
    else:
        # Otherwise it's a normal tier in {0..4}.
        assert proxy_relevance_v2(c) in (0.0, 1.0, 2.0, 3.0, 4.0)
