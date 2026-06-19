"""Proxy ground-truth construction.

We don't have the official ground truth, so we build a *proxy* relevance
score from JD-derived heuristics. This is what we evaluate against locally —
and it is also what the LTR trainer uses as labels.

Two versions live here:

* ``proxy_relevance_v1`` — the original. Weights are 0.35/0.15/0.15/0.15/0.10/0.10
  (ai / sen / loc / prod / pos / avail). This is what the LTR has been trained
  on historically.
* ``proxy_relevance_v2`` — rubric-aligned blend. We average two rubrics:
  (1) a JD-derived rubric that uses 10 slots (the same six as v1 plus
  education, open_source, distributed_systems, open_to_work), and
  (2) the independent ``eval_rubric.eval_relevance``. The average
  resolves the proxy-vs-eval disagreement: a candidate that is tier-3 in
  only one rubric becomes tier-2 in the average, while a candidate that
  is tier-3 in both becomes tier-3 in the average. This is the default
  starting this iteration; the LTR is retrained on it.

The tier boundary set is unchanged (0.75 / 0.55 / 0.35 / 0.18) so the eval
keeps the same 0–4 cut-points. Changing the boundaries would inflate
NDCG@10 without improving ranking quality.
"""
from __future__ import annotations

from src.api.schemas import Candidate
from src.behavioral.availability import availability_score
from src.behavioral.honeypot import honeypot_risk
from src.behavioral.jd_filters import positive_boost
from src.preprocessing.deep_profile import build_career_text
from src.preprocessing.feature_engineer import (
    _consulting_career_share,
    _product_company_count,
    _school_prestige_score,
)
from src.preprocessing.normalize import (
    AI_KEYWORDS,
    is_india,
    is_preferred_location,
    is_tier1_india,
)

PROXY_VERSION = "v2"


# ---------------------------------------------------------------------------
# Sub-scores (shared between v1 and v2; v2 just adds more slots)
# ---------------------------------------------------------------------------


def _ai_evidence_score(c: Candidate) -> float:
    """AI evidence in [0, 1] based on career text keyword matches.

    The formula is ``min(1.0, hits / 4.0)`` where hits is the number of
    AI keywords present in the career text. 4 hits is full credit; 99.5 %
    of candidates were < 0.05 under the old divisor-of-54 formula.
    """
    career = (build_career_text(c) or "").lower()
    hits = sum(min(1.0, career.count(k) / 2.0) for k in AI_KEYWORDS if k in career)
    return min(1.0, hits / 4.0)


def _seniority_score(c: Candidate) -> float:
    yoe = float(c.profile.years_of_experience or 0.0)
    if 5 <= yoe <= 9:
        return 1.0
    if 3 <= yoe < 5 or 9 < yoe <= 12:
        return 0.5
    return 0.1


def _location_score(c: Candidate) -> float:
    if is_preferred_location(c.profile.location):
        return 1.0
    if is_tier1_india(c.profile.country, c.profile.location):
        return 0.7
    if is_india(c.profile.country) and c.redrob_signals.willing_to_relocate:
        return 0.6
    return 0.1


def _product_company_score(c: Candidate) -> float:
    if not c.career_history:
        return 0.0
    return min(1.0, _product_company_count(c) / 2.0) * (1.0 - _consulting_career_share(c))


def _open_source_score(c: Candidate) -> float:
    """0-1. Continuous count of open-source signals in the career.

    Mirrors the eval_rubric's open-source sub-score: counts distinct role
    descriptions that mention open-source + 1 if GitHub activity is high.
    """
    keys = (
        "open source", "open-source", "github.com", "github.io", "arxiv",
        "paper", "medium", "kaggle", "huggingface.co", "contributor", "maintainer", "foss",
    )
    n = 0
    blob_parts: list[str] = []
    for r in c.career_history:
        d = (r.description or "").lower()
        if any(k in d for k in keys):
            n += 1
        blob_parts.append(d)
    blob = " ".join(blob_parts)
    if any(k in blob for k in keys):
        n = max(n, 1)
    if c.redrob_signals.github_activity_score and c.redrob_signals.github_activity_score >= 30:
        n = max(n, 1)
    return min(1.0, n / 2.0)


def _distributed_systems_score(c: Candidate) -> float:
    """0-1. Continuous count of distributed-systems signals.

    Mirrors the eval_rubric's distributed-systems sub-score.
    """
    keys = (
        "distributed", "spark", "kafka", "ray", "kubernetes", "microservice",
        "grpc", "cuda", "gpu", "tpu", "horovod", "deepspeed",
    )
    n = 0
    for r in c.career_history:
        d = (r.description or "").lower()
        if any(k in d for k in keys):
            n += 1
    return min(1.0, n / 2.0)


# ---------------------------------------------------------------------------
# V1 — original (kept for backwards compat + ablation)
# ---------------------------------------------------------------------------


def proxy_relevance_v1(c: Candidate) -> float:
    """0-4 tier. Original proxy formula (pre-iteration-2)."""
    honeypot = honeypot_risk(c)
    if honeypot >= 0.7:
        return 0.0

    ai = _ai_evidence_score(c)
    sen = _seniority_score(c)
    loc = _location_score(c)
    prod = _product_company_score(c)
    pos = positive_boost(c)
    avail = availability_score(c)

    score = 0.35 * ai + 0.15 * sen + 0.15 * loc + 0.15 * prod + 0.10 * pos + 0.10 * avail
    return _to_tier(score)


# ---------------------------------------------------------------------------
# V2 — rubric-aligned (default)
# ---------------------------------------------------------------------------


def proxy_relevance_v2(c: Candidate) -> float:
    """0-4 tier. Rubric-aligned blend (iteration 2 default).

    The label is the average of two independently-authored rubrics:

    1. ``_jd_rubric_score`` (this file, private) — uses 10 slots: ai, sen,
       loc, prod, pos, avail, education, open_source, distributed_systems,
       open_to_work. Weights are 0.30/0.20/0.12/0.10/0.06/0.06/0.08/0.05/0.03/0.04.
    2. ``eval_rubric.eval_relevance`` (independent file) — uses 8 slots:
       ai, sen, loc, prod, edu, ose, dst, open_to_work. Weights are
       0.30/0.20/0.15/0.08/0.10/0.08/0.05/0.04.

    Both are in [0, 1] before the tier cut. The average is also in [0, 1]
    and is fed to the same tier cut-points as v1.
    """
    honeypot = honeypot_risk(c)
    if honeypot >= 0.7:
        return 0.0

    jd = _jd_rubric_score(c)
    er = _eval_rubric_score(c)
    avg = 0.5 * jd + 0.5 * er
    return _to_tier(avg)


def _jd_rubric_score(c: Candidate) -> float:
    """0-1. JD-derived rubric, 10 slots. Used as one half of proxy_v2."""
    ai = _ai_evidence_score(c)
    sen = _seniority_score(c)
    loc = _location_score(c)
    prod = _product_company_score(c)
    pos = positive_boost(c)
    avail = availability_score(c)
    edu = _school_prestige_score(c)
    ose = _open_source_score(c)
    dst = _distributed_systems_score(c)
    otw = float(c.redrob_signals.open_to_work_flag)
    return (
        0.30 * ai
        + 0.20 * sen
        + 0.12 * loc
        + 0.10 * prod
        + 0.06 * pos
        + 0.06 * avail
        + 0.08 * edu
        + 0.05 * ose
        + 0.03 * dst
        + 0.04 * otw
    )


def _eval_rubric_score(c: Candidate) -> float:
    """0-1. Wrapper around the eval_rubric that returns the raw [0, 1] score.

    The eval_rubric returns a 0-4 tier; here we reconstruct the [0, 1] score
    so we can average it with the JD rubric. The reconstruction is
    monotone with the tier cut-points used by the eval_rubric.
    """
    from src.evaluation.eval_rubric import _yoe_curve, _ai_evidence, _location_score, _product_company_score, _education_score, _has_open_source_evidence, _has_distributed_systems

    ai = _ai_evidence(c)
    sen = _yoe_curve(float(c.profile.years_of_experience or 0.0))
    loc = _location_score(c)
    prod = _product_company_score(c)
    edu = _education_score(c)
    ose = min(1.0, _has_open_source_evidence(c) / 1.0)  # already 0/1+ count, normalize
    dst = _has_distributed_systems(c)  # 0/1
    otw = float(c.redrob_signals.open_to_work_flag)

    return (
        0.30 * ai
        + 0.20 * sen
        + 0.15 * loc
        + 0.08 * prod
        + 0.10 * edu
        + 0.08 * ose
        + 0.05 * dst
        + 0.04 * otw
    )


# Default entry point used by the LTR trainer and the local evaluator.
proxy_relevance = proxy_relevance_v2


def _to_tier(score: float) -> float:
    if score >= 0.75:
        return 4.0
    if score >= 0.55:
        return 3.0
    if score >= 0.35:
        return 2.0
    if score >= 0.18:
        return 1.0
    return 0.0


def build_proxy_ground_truth(candidates: list[Candidate]) -> dict[str, float]:
    """Map candidate_id -> 0-4 relevance tier (proxy v2)."""
    return {c.candidate_id: proxy_relevance(c) for c in candidates}


def build_proxy_ground_truth_v1(candidates: list[Candidate]) -> dict[str, float]:
    """Map candidate_id -> 0-4 relevance tier (proxy v1, for ablation)."""
    return {c.candidate_id: proxy_relevance_v1(c) for c in candidates}
