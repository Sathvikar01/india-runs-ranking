"""Unit tests for JD negative / positive filters."""

from __future__ import annotations

from src.behavioral.jd_filters import (
    has_ai_career_evidence,
    is_closed_source_only,
    is_cv_robotics_speech,
    is_langchain_recent_only,
    is_only_consulting_companies,
    is_title_chaser,
    negative_flags,
    negative_penalty,
    positive_boost,
    positive_flags,
    shipped_ranking_or_search_at_scale,
)
from tests.fixtures.candidates import (
    make_ai_candidate,
    make_consulting_chain_candidate,
    make_honeypot_candidate,
)


def test_ai_candidate_positive():
    flags = positive_flags(make_ai_candidate())
    assert flags["has_ai_career_evidence"]
    assert flags["location_noida_pune_or_relocate"]
    assert flags["github_open_source_evidence"]
    assert positive_boost(make_ai_candidate()) > 0.3


def test_consulting_chain_negative():
    flags = negative_flags(make_consulting_chain_candidate())
    assert flags["only_consulting_companies"]
    assert flags["no_nlp_ir_in_career"]
    assert negative_penalty(make_consulting_chain_candidate()) > 0.1


def test_is_only_consulting_companies():
    assert is_only_consulting_companies(make_consulting_chain_candidate())
    assert not is_only_consulting_companies(make_ai_candidate())


def test_helper_functions():
    assert has_ai_career_evidence(make_ai_candidate())
    assert shipped_ranking_or_search_at_scale(make_ai_candidate())
    assert not is_only_consulting_companies(make_ai_candidate())
    assert is_only_consulting_companies(make_consulting_chain_candidate())


def test_title_chaser():
    # Build a clearly title-chasing candidate: 3 short stints.
    from src.api.schemas import Candidate, CareerRole
    base = make_ai_candidate()
    short = base.model_copy(deep=True)
    short.career_history = [
        CareerRole(
            company=f"Co{i}", title="Senior Engineer",
            start_date=f"202{i}-01-01", end_date=f"202{i}-09-01",
            duration_months=8, is_current=False, industry="Software", company_size="11-50",
            description="Short stint.",
        )
        for i in range(1, 4)
    ]
    # 3 roles × 8 months = 24 / 3 = 8 months avg. Definitely title chaser.
    assert is_title_chaser(short)
    assert not is_title_chaser(base)


def test_cv_robotics_negative():
    # Build a CV-only candidate
    from src.api.schemas import Candidate, CareerRole
    base = make_ai_candidate().model_copy(deep=True)
    base.career_history[0].description = (
        "Built computer vision pipelines for autonomous driving. Object detection "
        "and segmentation with PyTorch. Worked on ROS and embedded deployment."
    )
    base.career_history[1].description = (
        "Image classification with TensorFlow. OpenCV pipelines."
    )
    assert is_cv_robotics_speech(base)


def test_langchain_recent_only():
    from src.api.schemas import Candidate
    base = make_ai_candidate().model_copy(deep=True)
    base.career_history[0].description = "Built a LangChain agent using OpenAI for Q&A."
    base.career_history[1].description = "Worked on LangChain prototypes."
    assert is_langchain_recent_only(base)


def test_closed_source_only():
    # AI candidate has github activity, so should not be flagged
    assert not is_closed_source_only(make_ai_candidate())
