"""Tests for the JD-literal 3rd rubric (Agent 9)."""
from __future__ import annotations

import pytest

from src.api.schemas import (
    Candidate,
    CareerRole,
    Education,
    Profile,
    RedrobSignals,
    Skill,
)
from src.evaluation.jd_literal_rubric import (
    _has_ai_career,
    _has_distributed_systems,
    _has_open_source,
    _has_school_prestige,
    _is_in_yoe_band,
    _is_product_company,
    _is_title_chaser,
    jd_literal_relevance,
    jd_literal_score,
)


def _stub_candidate(**overrides):
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
                description="Built ML systems on pytorch, retrieval and rag for distributed microservices",
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
            # Also update career_history so title-based checks work.
            if cand.career_history:
                cand.career_history[0].title = v
        elif k == "industry":
            cand.profile.current_industry = v
            # Also update career_history so product-company check works.
            if cand.career_history:
                cand.career_history[0].industry = v
        elif k == "company":
            cand.profile.current_company = v
            if cand.career_history:
                cand.career_history[0].company = v
        elif k == "desc":
            if cand.career_history:
                cand.career_history[0].description = v
        elif k == "gh":
            cand.redrob_signals.github_activity_score = v
        elif k == "relocate":
            cand.redrob_signals.willing_to_relocate = v
        elif k == "location":
            cand.profile.location = v
    return cand


def test_jd_literal_score_strong_candidate():
    """All checks should pass for a well-formed AI candidate."""
    c = _stub_candidate()
    s = jd_literal_score(c)
    assert s >= 0.80, s


def test_jd_literal_score_low_yoe_band():
    c = _stub_candidate(yoe=3.0)
    assert _is_in_yoe_band(c) is False


def test_jd_literal_score_in_yoe_band():
    c = _stub_candidate(yoe=7.0)
    assert _is_in_yoe_band(c) is True


def test_jd_literal_score_title_chaser():
    """Junior with 7 yrs → title chaser penalty."""
    c = _stub_candidate(title="Junior ML Engineer", yoe=7.0)
    assert _is_title_chaser(c) is True
    s = jd_literal_score(c)
    assert s < 0.5  # severely penalized


def test_jd_literal_score_consulting_company():
    c = _stub_candidate(company="Big Consulting Co", industry="Consulting")
    assert _is_product_company(c) is False


def test_jd_literal_score_ai_career():
    c = _stub_candidate()
    assert _has_ai_career(c) is True


def test_jd_literal_score_open_source_github():
    """High GitHub activity → has_open_source."""
    c = _stub_candidate(gh=50)
    assert _has_open_source(c) is True


def test_jd_literal_score_distributed_systems():
    c = _stub_candidate(desc="Built distributed microservices on kubernetes with kafka")
    assert _has_distributed_systems(c) is True


def test_jd_literal_relevance_strong():
    c = _stub_candidate()
    tier = jd_literal_relevance(c)
    assert tier in (3.0, 4.0)


def test_jd_literal_relevance_weak():
    """A consultant with no AI evidence → low tier."""
    c = _stub_candidate(
        title="HR Manager",
        yoe=10.0,
        company="Big Consulting Co",
        industry="Consulting",
        desc="Project management",
        gh=0,
    )
    tier = jd_literal_relevance(c)
    assert tier <= 1.0
