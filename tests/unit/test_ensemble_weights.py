"""Tests for the configurable ensemble + grid search (Agent 7)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ranking.ensemble import (
    EnsembleWeights,
    ensemble_score,
    ensemble_score_v2,
)


def test_ensemble_weights_defaults_match_v1():
    """Default weights reproduce the v1 hard-coded ensemble."""
    w = EnsembleWeights()
    s_v2 = ensemble_score_v2(
        ltr_score=1.0, ce_score=1.0, availability=1.0,
        positive=1.0, negative=0.0, honeypot=0.0, weights=w,
    )
    s_v1 = ensemble_score(
        ltr_score=1.0, ce_score=1.0, availability=1.0,
        positive=1.0, negative=0.0, honeypot=0.0,
    )
    assert abs(s_v2 - s_v1) < 1e-6


def test_ensemble_weights_clipping():
    """Out-of-range signals are clipped to [0, 1]."""
    w = EnsembleWeights(w_ltr=1.0, w_ce=1.0, w_avail=1.0, w_positive=1.0,
                        w_negative=0.0, w_honeypot=0.0)
    # All positive signals at max → ensemble = sum(weights).
    s = ensemble_score_v2(
        ltr_score=10.0, ce_score=10.0, availability=1.0,
        positive=1.0, negative=0.0, honeypot=0.0, weights=w,
    )
    assert 0.0 <= s <= 1.0
    # All negative → 0.
    s_neg = ensemble_score_v2(
        ltr_score=-10.0, ce_score=-10.0, availability=0.0,
        positive=0.0, negative=0.0, honeypot=1.0, weights=w,
    )
    assert 0.0 <= s_neg <= 1.0


def test_ensemble_weights_roundtrip_json():
    w = EnsembleWeights(w_ltr=0.7, w_ce=0.1, w_topk=0.2)
    s = w.to_json()
    w2 = EnsembleWeights.from_json(s)
    assert w2.w_ltr == 0.7
    assert w2.w_ce == 0.1
    assert w2.w_topk == 0.2


def test_ensemble_weights_save_load(tmp_path: Path):
    w = EnsembleWeights(w_multitask=0.3, w_topk=0.15)
    path = tmp_path / "weights.json"
    w.save(path)
    w2 = EnsembleWeights.load(path)
    assert w2.w_multitask == 0.3
    assert w2.w_topk == 0.15


def test_ensemble_score_uses_multitask_when_weight_positive():
    w = EnsembleWeights(w_ltr=0.5, w_multitask=0.3)
    s_with = ensemble_score_v2(
        ltr_score=1.0, ce_score=0.0, availability=0.0,
        positive=0.0, negative=0.0, honeypot=0.0,
        weights=w, multitask_score=1.0,
    )
    s_without = ensemble_score_v2(
        ltr_score=1.0, ce_score=0.0, availability=0.0,
        positive=0.0, negative=0.0, honeypot=0.0,
        weights=w,
    )
    assert s_with > s_without


def test_ensemble_score_uses_topk_when_weight_positive():
    w = EnsembleWeights(w_ltr=0.5, w_topk=0.4)
    s_high_topk = ensemble_score_v2(
        ltr_score=1.0, ce_score=0.0, availability=0.0,
        positive=0.0, negative=0.0, honeypot=0.0,
        weights=w, topk_score=1.0,
    )
    s_low_topk = ensemble_score_v2(
        ltr_score=1.0, ce_score=0.0, availability=0.0,
        positive=0.0, negative=0.0, honeypot=0.0,
        weights=w, topk_score=0.0,
    )
    assert s_high_topk > s_low_topk


def test_ensemble_weights_unknown_field_ignored_on_load():
    """Forward-compat: extra fields in the JSON don't crash."""
    raw = json.dumps({**EnsembleWeights().__dict__, "future_field": 0.99})
    w = EnsembleWeights.from_json(raw)
    assert w.w_ltr == 0.55
