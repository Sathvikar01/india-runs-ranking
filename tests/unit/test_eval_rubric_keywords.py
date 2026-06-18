"""Tests for the Tier-1 eval_rubric keyword expansion."""
from __future__ import annotations

from src.evaluation.eval_rubric import (
    _distributed_systems_count,
    _has_distributed_systems,
    _has_open_source_evidence,
    _open_source_count,
)
from src.api.schemas import Candidate, CareerRole, Education, Profile, RedrobSignals, Skill
from tests.fixtures.candidates import make_ai_candidate


def _mk_candidate_with_desc(desc: str) -> Candidate:
    c = make_ai_candidate()
    c.career_history = [
        CareerRole(company="X", title="Eng", start_date="2020-01-01", end_date=None,
                   duration_months=24, is_current=True, industry="AI/ML",
                   description=desc),
    ]
    return c


def test_distributed_systems_recognises_gpu_tpu():
    assert _has_distributed_systems(_mk_candidate_with_desc("trained on GPU clusters")) == 1
    assert _has_distributed_systems(_mk_candidate_with_desc("TPU-based inference")) == 1
    assert _has_distributed_systems(_mk_candidate_with_desc("distributed training with DeepSpeed")) == 1


def test_distributed_systems_count_is_continuous():
    c = make_ai_candidate()
    c.career_history = [
        CareerRole(company="A", title="X", start_date="2020-01-01", end_date="2022-01-01",
                   duration_months=24, is_current=False, industry="AI/ML",
                   description="built a Spark pipeline"),
        CareerRole(company="B", title="Y", start_date="2022-01-01", end_date=None,
                   duration_months=24, is_current=True, industry="AI/ML",
                   description="deployed GPU inference"),
    ]
    n = _distributed_systems_count(c)
    assert n >= 2


def test_open_source_recognises_huggingface_kaggle_medium():
    c = _mk_candidate_with_desc("published a model on huggingface.co and wrote on medium")
    assert _has_open_source_evidence(c) == 1


def test_open_source_count_continuous():
    c = make_ai_candidate()
    c.career_history = [
        CareerRole(company="A", title="OSS", start_date="2020-01-01", end_date="2022-01-01",
                   duration_months=24, is_current=False, industry="AI/ML",
                   description="open source contributor to github.com/foo/bar"),
        CareerRole(company="B", title="Y", start_date="2022-01-01", end_date=None,
                   duration_months=24, is_current=True, industry="AI/ML",
                   description="arxiv paper"),
    ]
    n = _open_source_count(c)
    assert n >= 2
