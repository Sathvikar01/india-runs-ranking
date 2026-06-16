from __future__ import annotations

from src.ranking.ensemble import apply_monotonic, ensemble_score, make_monotonic_scores, rank_candidates
from tests.fixtures.candidates import make_ai_candidate, make_consulting_chain_candidate, make_honeypot_candidate


def test_ensemble_score_in_unit_range():
    s = ensemble_score(ltr_score=0.0, ce_score=0.0, availability=0.5, positive=0.5, negative=0.0, honeypot=0.0)
    assert 0.0 <= s <= 1.0


def test_ensemble_honeypot_demoted():
    clean = ensemble_score(ltr_score=1.0, ce_score=1.0, availability=0.8, positive=0.8, negative=0.0, honeypot=0.0)
    risky = ensemble_score(ltr_score=1.0, ce_score=1.0, availability=0.8, positive=0.8, negative=0.0, honeypot=0.9)
    assert clean > risky


def test_make_monotonic_scores_strictly_decreasing():
    raw = [0.9, 0.5, 0.5, 0.3, 0.1]
    out = make_monotonic_scores(raw)
    assert len(out) == len(raw)
    assert all(0.0 <= v <= 1.0 for v in out)
    # Strictly decreasing with tiny jitter
    for a, b in zip(out, out[1:]):
        assert a > b


def test_rank_candidates_orders_ai_first():
    cs = [make_consulting_chain_candidate(), make_ai_candidate(), make_honeypot_candidate()]
    out = rank_candidates(cs, ltr_scores={c.candidate_id: 0.0 for c in cs}, ce_scores={c.candidate_id: 0.0 for c in cs}, top_k=3)
    assert out[0][0].candidate_id == "CAND_0000001"
    assert out[-1][0].candidate_id == "CAND_9999999"


def test_apply_monotonic_preserves_order():
    cs = [make_consulting_chain_candidate(), make_ai_candidate()]
    ranked = [(cs[1], 0.9, {}), (cs[0], 0.3, {})]
    out = apply_monotonic(ranked)
    assert out[0][0].candidate_id == "CAND_0000001"
    assert out[0][1] > out[1][1]
