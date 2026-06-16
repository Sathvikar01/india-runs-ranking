"""Unit tests for honeypot detection."""

from __future__ import annotations

from src.behavioral.honeypot import honeypot_risk, honeypot_signals, is_honeypot
from tests.fixtures.candidates import (
    make_ai_candidate,
    make_consulting_chain_candidate,
    make_honeypot_candidate,
)


def test_honeypot_signals_ai_candidate_low():
    sub = honeypot_signals(make_ai_candidate())
    assert sub["perfect_skill_list_with_non_tech_title"] == 0.0
    assert sub["expert_in_too_many_skills"] <= 0.5


def test_honeypot_candidate_high_risk():
    assert honeypot_risk(make_honeypot_candidate()) >= 0.4
    assert is_honeypot(make_honeypot_candidate())


def test_consulting_chain_not_honeypot():
    # consulting chain is a JD-negative filter, not a honeypot
    assert honeypot_risk(make_consulting_chain_candidate()) < 0.5
