"""Unit tests for the feature engineer."""

from __future__ import annotations

from datetime import date

from src.preprocessing.feature_engineer import build_features, feature_columns
from tests.fixtures.candidates import make_ai_candidate, make_consulting_chain_candidate, make_honeypot_candidate


def test_build_features_ai_candidate_keys_present():
    f = build_features(make_ai_candidate(), today=date(2026, 6, 17))
    for k in feature_columns():
        assert k in f, k
    assert f["has_ai_career_evidence"] == 1
    assert f["location_is_noida_or_pune"] == 1
    assert f["yoe_in_5_9_band"] == 1
    # WS-5 new features
    assert "title_yoe_consistency" in f
    assert "title_yoe_inconsistent" in f
    assert "career_progression" in f
    assert "n_distinct_industries" in f
    assert "n_named_jd_skills" in f
    assert "consulting_to_product_transition" in f
    assert "endorsement_entropy" in f
    # The AI fixture has 2 roles, so n_distinct_industries is in {1, 2}.
    assert 1 <= f["n_distinct_industries"] <= 2
    # n_named_jd_skills should be > 0 because the AI fixture mentions RAG etc.
    assert f["n_named_jd_skills"] >= 1
    # The fixture's title is "Senior ML Engineer" with 7 yrs YOE → not
    # title_YOE inconsistent (delta = 0).
    assert f["title_yoe_inconsistent"] == 0


def test_build_features_honeypot_flags():
    f = build_features(make_honeypot_candidate(), today=date(2026, 6, 17))
    assert f["skill_expert_zero_months"] >= 5
    assert f["perfect_skill_list_with_non_tech_title"] == 1
    assert f["expert_skill_count"] >= 5


def test_school_prestige_known_iit_returns_one():
    """IIT Bombay (tier-1) → education_prestige == 1.0."""
    from src.preprocessing.feature_engineer import _school_prestige_score
    from src.api.schemas import Education

    from tests.fixtures.candidates import make_ai_candidate

    c = make_ai_candidate()
    c.education = [Education(institution="IIT Bombay", degree="B.Tech", field_of_study="CS")]
    assert _school_prestige_score(c) == 1.0


def test_school_prestige_known_tier2_returns_half():
    from src.preprocessing.feature_engineer import _school_prestige_score
    from src.api.schemas import Education

    from tests.fixtures.candidates import make_ai_candidate

    c = make_ai_candidate()
    c.education = [Education(institution="BITS Pilani", degree="B.E.", field_of_study="CS")]
    assert _school_prestige_score(c) == 0.5


def test_n_named_jd_skills_continuous_in_unit_interval():
    """The continuous version is in [0, 1]."""
    from src.preprocessing.feature_engineer import _n_named_jd_skills_continuous
    from tests.fixtures.candidates import make_ai_candidate

    c = make_ai_candidate()
    v = _n_named_jd_skills_continuous(c, "RAG with vector search and LoRA", "PyTorch, Transformers")
    assert 0.0 <= v <= 1.0
    assert v > 0.0  # the AI candidate should have several


def test_distributed_systems_count_increments_per_role():
    """A candidate with 2 distributed-systems roles → count >= 2."""
    from src.api.schemas import CareerRole
    from src.preprocessing.feature_engineer import _distributed_systems_count
    from tests.fixtures.candidates import make_ai_candidate

    c = make_ai_candidate()
    c.career_history = [
        CareerRole(company="A", title="X", start_date="2020-01-01", end_date="2022-01-01",
                   duration_months=24, is_current=False, industry="AI/ML",
                   description="Built a distributed training system with Spark and Kafka."),
        CareerRole(company="B", title="Y", start_date="2022-01-01", end_date=None,
                   duration_months=24, is_current=True, industry="AI/ML",
                   description="Deployed GPU inference on Kubernetes."),
    ]
    assert _distributed_systems_count(c, "") >= 2


def test_open_source_count_for_github_mentions():
    from src.api.schemas import CareerRole
    from src.preprocessing.feature_engineer import _open_source_count
    from tests.fixtures.candidates import make_ai_candidate

    c = make_ai_candidate()
    c.career_history = [
        CareerRole(company="A", title="OSS", start_date="2020-01-01", end_date="2022-01-01",
                   duration_months=24, is_current=False, industry="AI/ML",
                   description="Open source contributor to github.com/foo/bar, arxiv paper."),
    ]
    assert _open_source_count(c, "") >= 1


def test_school_prestige_unknown_returns_zero():
    from src.preprocessing.feature_engineer import _school_prestige_score
    from tests.fixtures.candidates import make_ai_candidate

    c = make_ai_candidate()
    c.education = []
    assert _school_prestige_score(c) == 0.0


def test_title_yoe_inconsistent_flags_junior_with_high_yoe():
    """A 'Junior ML Engineer' with 6+ yrs should be flagged as inconsistent."""
    from src.preprocessing.feature_engineer import _title_yoe_inconsistent
    from tests.fixtures.candidates import make_ai_candidate

    c = make_ai_candidate()
    c.profile.current_title = "Junior ML Engineer"
    c.profile.years_of_experience = 7.0
    val = _title_yoe_inconsistent(c, yoe=7.0)
    assert val == 1


def test_title_yoe_inconsistent_clean_for_normal_title():
    """A 'Senior Engineer' with 7 yrs should NOT be flagged."""
    from src.preprocessing.feature_engineer import _title_yoe_inconsistent
    from tests.fixtures.candidates import make_ai_candidate

    c = make_ai_candidate()
    c.profile.current_title = "Senior ML Engineer"
    c.profile.years_of_experience = 7.0
    val = _title_yoe_inconsistent(c, yoe=7.0)
    assert val == 0


def test_build_features_consulting_chain():
    f = build_features(make_consulting_chain_candidate(), today=date(2026, 6, 17))
    assert f["consulting_share"] >= 0.9
    assert f["product_company_count"] == 0
    # We don't surface has_nlp_ir_in_career as a feature; it's a JD-filter helper.


def test_ai_candidate_high_ai_score():
    f = build_features(make_ai_candidate(), today=date(2026, 6, 17))
    assert f["ai_keyword_hits_career"] >= 5
    # n_ai_skill_advanced is proficiency >= advanced; we have PyTorch + Transformers + Python(advanced, generic)
    assert f["n_ai_skill_advanced"] >= 2
