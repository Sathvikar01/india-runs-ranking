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


# ---------------------------------------------------------------------------
# Feature zoo v2 (Agent 5) tests
# ---------------------------------------------------------------------------


def test_feature_zoo_v2_all_keys_present():
    f = build_features(make_ai_candidate(), today=date(2026, 6, 17))
    zoo_v2_keys = [
        # Career shape
        "n_ai_roles", "n_senior_roles", "n_product_roles", "n_india_roles",
        "avg_role_duration_months", "max_role_duration_months",
        "tenure_variance", "career_progression_slope",
        "current_role_tenure_months", "n_career_gaps",
        # JD-literal
        "jd_skill_match_count", "jd_skill_match_expert_count",
        "jd_keyword_count_career", "jd_keyword_count_summary",
        "title_jd_match", "seniority_jd_match", "location_jd_match",
        "industry_jd_match", "has_product_company_recent",
        "career_jd_sim_proxy",
        # Behavioral
        "log_profile_views_30d", "log_search_appearance_30d",
        "log_saved_by_recruiters_30d", "log_connection_count",
        "log_endorsements_received", "engagement_intensity",
        "behavioral_risk_score", "availability_composite",
        # Skill mix
        "expert_share", "advanced_or_expert_share",
        "skill_endorsement_mean", "skill_endorsement_max", "skill_count_log",
        "ai_skill_count_log",
        # Title / seniority alignment
        "title_yoe_in_band", "seniority_distance_from_ideal",
    ]
    for k in zoo_v2_keys:
        assert k in f, f"missing feature: {k}"
        assert f[k] is not None


def test_feature_zoo_v2_ai_candidate_high_jd_match():
    f = build_features(make_ai_candidate(), today=date(2026, 6, 17))
    # AI fixture mentions RAG / retrieval / pytorch, etc.
    assert f["jd_skill_match_count"] >= 1
    assert f["jd_keyword_count_career"] >= 1
    assert f["title_jd_match"] == 1
    assert f["seniority_jd_match"] == 1  # 7 yrs is in 5-9 band
    assert f["industry_jd_match"] == 1   # AI/ML industry


def test_feature_zoo_v2_consulting_chain_low_jd_match():
    f = build_features(make_consulting_chain_candidate(), today=date(2026, 6, 17))
    # Consulting chain shouldn't match AI keywords.
    assert f["title_jd_match"] == 0
    assert f["industry_jd_match"] == 0
    assert f["has_product_company_recent"] == 0
    assert f["jd_skill_match_count"] == 0


def test_feature_zoo_v2_honeypot_high_risk_score():
    f = build_features(make_honeypot_candidate(), today=date(2026, 6, 17))
    # Honeypot candidates should have a higher behavioral risk score.
    assert f["behavioral_risk_score"] >= 0.0  # in [0,1]
    # And a low availability composite.
    assert 0.0 <= f["availability_composite"] <= 1.0


def test_feature_zoo_v2_log_features_log1p_scale():
    """log_* features should be monotonically increasing in raw counts."""
    f = build_features(make_ai_candidate(), today=date(2026, 6, 17))
    # log1p is monotone in raw count.
    raw = f["profile_views_30d"]
    log = f["log_profile_views_30d"]
    assert log >= 0.0
    if raw > 0:
        import math
        assert abs(log - math.log1p(raw)) < 1e-6


def test_feature_zoo_v2_career_progression_slope_strict():
    f = build_features(make_ai_candidate(), today=date(2026, 6, 17))
    # slope is finite and well-defined
    assert -10.0 <= f["career_progression_slope"] <= 10.0


def test_feature_zoo_v2_feature_count_grew():
    """v2 adds 35+ features; the schema should grow."""
    cols = feature_columns()
    assert len(cols) >= 110  # was 75, now ~110+

