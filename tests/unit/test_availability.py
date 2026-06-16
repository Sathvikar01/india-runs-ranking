"""Unit tests for behavioral availability scoring."""

from __future__ import annotations

from datetime import date

from src.behavioral.availability import availability_score, is_stale
from tests.fixtures.candidates import make_ai_candidate, make_consulting_chain_candidate, make_honeypot_candidate


def test_ai_candidate_high_availability():
    a = availability_score(make_ai_candidate(), today=date(2026, 6, 17))
    assert a > 0.7


def test_consulting_chain_medium():
    a = availability_score(make_consulting_chain_candidate(), today=date(2026, 6, 17))
    assert 0.3 < a < 0.9


def test_honeypot_low_availability():
    a = availability_score(make_honeypot_candidate(), today=date(2026, 6, 17))
    assert a < 0.5


def test_is_stale():
    c = make_ai_candidate()
    assert not is_stale(c, today=date(2026, 6, 17))
    c.redrob_signals.last_active_date = "2024-01-01"
    assert is_stale(c, today=date(2026, 6, 17))
