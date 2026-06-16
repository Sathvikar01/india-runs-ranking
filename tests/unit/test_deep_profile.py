"""Unit tests for the deep profile builder."""

from __future__ import annotations

from src.preprocessing.deep_profile import build_career_text, build_deep_profile, build_signals_text, build_skills_text
from tests.fixtures.candidates import make_ai_candidate, make_honeypot_candidate


def test_deep_profile_contains_career():
    text = build_deep_profile(make_ai_candidate())
    assert "Senior ML Engineer" in text
    assert "Acme AI" in text
    assert "RAG" in text or "retrieval" in text.lower()
    assert "Skills:" in text


def test_career_text_only_career():
    text = build_career_text(make_ai_candidate())
    assert "Senior ML Engineer" in text
    assert "Skills:" not in text


def test_skills_text_canonicalized():
    text = build_skills_text(make_ai_candidate())
    assert "pytorch" in text
    assert "advanced" in text


def test_signals_text_contains_github():
    text = build_signals_text(make_ai_candidate())
    assert "github_activity_score=78.0" in text


def test_honeypot_deep_profile_does_not_oversell():
    text = build_deep_profile(make_honeypot_candidate())
    # The career description is short and is what the embedder should weight
    assert "Marketing Manager" in text
    assert "email campaigns" in text
