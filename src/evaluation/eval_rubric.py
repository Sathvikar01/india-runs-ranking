"""Independently authored eval rubric (WS-4).

This module is *not* a tweak of `proxy_ground_truth.py`. It was written
separately so that LTR training labels (`proxy_ground_truth.proxy_relevance`)
and evaluation ground truth (`eval_rubric.eval_relevance`) disagree in
predictable ways. The disagreement is the point: it lets us measure whether
the LTR model has learned something beyond the proxy.

Differences vs. `proxy_ground_truth`:

* Different sub-weights (no single 0.35/0.15/0.15/0.15/0.10/0.10 mix).
* Two swapped rule buckets: this rubric treats `open_source` and
  `distributed_systems` evidence as first-class positives; the proxy doesn't.
* YOE band: the proxy uses a 5-9 / 3-5 / 9-12 bucketing; this rubric
  uses a continuous "5-9 = 1.0; smooth ramp outside the band" curve.
* The rubric de-weights `product_company_count` (the proxy is generous
  with it) and up-weights `education_tier_1` (the proxy under-weights it).
* The rubric penalises `not_preferred_location` more aggressively.

These changes are deliberate and documented in `docs/methodology.md`.
"""
from __future__ import annotations

from src.api.schemas import Candidate
from src.preprocessing.normalize import (
    is_india,
    is_preferred_location,
    is_tier1_india,
)


def _yoe_curve(yoe: float) -> float:
    """Continuous JD-fit curve, 1.0 in the 5-9 band, smoothly lower outside."""
    if 5 <= yoe <= 9:
        return 1.0
    if yoe < 5:
        # 0 yrs -> 0.10, 5 yrs -> 1.0
        return max(0.10, 1.0 - (5 - yoe) * 0.18)
    # > 9 yrs: penalize over-experience
    return max(0.10, 1.0 - (yoe - 9) * 0.10)


def _ai_evidence(c: Candidate) -> float:
    """Independent of `proxy_ground_truth._ai_evidence_score`.

    Counts strong AI/ML signals across the entire career (not just titles).
    """
    career = (c.profile.summary or "").lower()
    for r in c.career_history:
        career += " " + (r.description or "").lower()
    keys = (
        "machine learning",
        "deep learning",
        "neural",
        "nlp",
        "transformer",
        "embedding",
        "retrieval",
        "ranking",
        "recommend",
        "llm",
        "rag",
        "lora",
        "peft",
        "fine-tun",
        "pytorch",
        "tensorflow",
        "faiss",
    )
    hits = sum(min(3, career.count(k)) for k in keys)
    return min(1.0, hits / 8.0)


def _has_distributed_systems(c: Candidate) -> int:
    career = " ".join((r.description or "") for r in c.career_history).lower()
    keys = (
        "distributed", "spark", "kafka", "ray", "kubernetes", "microservice", "grpc",
        "scalab", "cuda", "gpu", "tpu", "multi-gpu", "gpu cluster",
        "distributed training", "distributed inference", "horovod", "deepspeed",
        "parameter server", "allreduce", "distributed computing", "elastic",
    )
    return int(any(k in career for k in keys))


def _distributed_systems_count(c: Candidate) -> int:
    """Count of distinct role descriptions that mention distributed-systems
    keywords. The eval rubric up-weights distributed-systems evidence; a
    continuous count is more informative than a binary 0/1."""
    keys = (
        "distributed", "spark", "kafka", "ray", "kubernetes", "microservice", "grpc",
        "cuda", "gpu", "tpu", "horovod", "deepspeed",
    )
    n = 0
    for r in c.career_history:
        d = (r.description or "").lower()
        if any(k in d for k in keys):
            n += 1
    return n


def _has_open_source_evidence(c: Candidate) -> int:
    s = c.redrob_signals
    if s.github_activity_score and s.github_activity_score >= 30:
        return 1
    career = " ".join((r.description or "") for r in c.career_history).lower()
    keys = (
        "open source", "open-source", "github.com", "github.io", "arxiv", "paper",
        "publication", "medium", "kaggle", "huggingface.co", "stack overflow",
        "contributor", "maintainer", "pull request", "foss", "apache", "linux foundation",
    )
    return int(any(k in career for k in keys))


def _open_source_count(c: Candidate) -> int:
    """Count of distinct role descriptions that mention open-source signals.
    Continuous version of `_has_open_source_evidence`."""
    keys = (
        "open source", "open-source", "github.com", "github.io", "arxiv", "paper",
        "medium", "kaggle", "huggingface.co", "contributor", "maintainer", "foss",
    )
    n = 0
    for r in c.career_history:
        d = (r.description or "").lower()
        if any(k in d for k in keys):
            n += 1
    return n


def _location_score(c: Candidate) -> float:
    if is_preferred_location(c.profile.location):
        return 1.0
    if is_tier1_india(c.profile.country, c.profile.location):
        if c.redrob_signals.willing_to_relocate:
            return 0.85
        return 0.55
    if is_india(c.profile.country) and c.redrob_signals.willing_to_relocate:
        return 0.50
    return 0.05  # harsher than the proxy's 0.10


def _product_company_score(c: Candidate) -> float:
    if not c.career_history:
        return 0.0
    n_product = sum(
        1
        for r in c.career_history
        if r.industry
        and r.industry.lower() in (
            "ai/ml",
            "saas",
            "fintech",
            "ecommerce",
            "edtech",
            "adtech",
            "gaming",
        )
    )
    return min(0.6, n_product * 0.20)  # cap lower than the proxy's 1.0


def _education_score(c: Candidate) -> float:
    """First-class education score (proxy under-weights this)."""
    if not c.education:
        return 0.0
    score = 0.0
    for e in c.education:
        tier = (e.tier or "").lower()
        if tier == "tier_1":
            score = max(score, 0.6)
        elif tier == "tier_2":
            score = max(score, 0.4)
        elif tier:
            score = max(score, 0.2)
        else:
            score = max(score, 0.1)
    return score


def _seniority_score(c: Candidate) -> float:
    """Independent of `proxy_ground_truth._seniority_score`.

    Smooth curve over years-of-experience, not bucketed.
    """
    return _yoe_curve(float(c.profile.years_of_experience or 0.0))


def eval_relevance(c: Candidate) -> float:
    """0-4 tier. Independently authored from `proxy_relevance`.

    The tiering curve is intentionally *different* from the proxy's so
    that evaluating the LTR model on `eval_relevance` is not a self-eval.
    """
    from src.behavioral.honeypot import honeypot_risk

    if honeypot_risk(c) >= 0.7:
        return 0.0

    ai = _ai_evidence(c)
    sen = _seniority_score(c)
    loc = _location_score(c)
    prod = _product_company_score(c)
    edu = _education_score(c)
    ose = _has_open_source_evidence(c)
    dst = _has_distributed_systems(c)

    # Different weights than the proxy: education + open-source +
    # distributed-systems each get their own slot.
    score = (
        0.30 * ai
        + 0.20 * sen
        + 0.15 * loc
        + 0.08 * prod
        + 0.10 * edu
        + 0.08 * ose
        + 0.05 * dst
        + 0.04 * float(c.redrob_signals.open_to_work_flag)
    )

    # Different tier boundaries than the proxy.
    if score >= 0.70:
        return 4.0
    if score >= 0.52:
        return 3.0
    if score >= 0.34:
        return 2.0
    if score >= 0.18:
        return 1.0
    return 0.0


def build_eval_ground_truth(candidates: list[Candidate]) -> dict[str, float]:
    """Map candidate_id -> 0-4 tier from the independent eval rubric."""
    return {c.candidate_id: eval_relevance(c) for c in candidates}
