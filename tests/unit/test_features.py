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


def test_build_features_honeypot_flags():
    f = build_features(make_honeypot_candidate(), today=date(2026, 6, 17))
    assert f["skill_expert_zero_months"] >= 5
    assert f["perfect_skill_list_with_non_tech_title"] == 1
    assert f["expert_skill_count"] >= 5


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
