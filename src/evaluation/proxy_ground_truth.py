"""Proxy ground-truth construction.

We don't have the official ground truth, so we build a *proxy* relevance
score from JD-derived heuristics. This is what we evaluate against locally —
and it is also what the LTR trainer uses as labels.

The proxy is intentionally simple: each candidate gets a 0-4 tier based on a
weighted combination of:
  * AI career evidence (career-history text)
  * Seniority in 5-9 band
  * Product-company experience
  * Location fit
  * Behavioral availability

This matches the JD's stated ideal candidate profile.
"""

from __future__ import annotations

from src.api.schemas import Candidate
from src.behavioral.availability import availability_score
from src.behavioral.honeypot import honeypot_risk
from src.behavioral.jd_filters import positive_boost
from src.preprocessing.deep_profile import build_career_text
from src.preprocessing.feature_engineer import _consulting_career_share, _product_company_count
from src.preprocessing.normalize import (
    AI_KEYWORDS,
    is_india,
    is_preferred_location,
    is_tier1_india,
)


def _ai_evidence_score(c: Candidate) -> float:
    """AI evidence in [0, 1] based on career text keyword matches.

    WS-Tier-1 follow-up: the original formula divided the total matches
    by `len(AI_KEYWORDS)` (54), so even a strong AI engineer with 5
    keyword hits got 5/54 = 0.093. That's why the proxy ended up with
    60 % tier-1 and 0.2 % tier-3 — the AI signal was drowned out by
    the divisor. The new formula scales by `min(1.0, hits / 4.0)` — 4
    hits is full credit. 99.5 % of candidates were < 0.05 before; the
    new formula gives 0.5-1.0 to actual AI engineers.
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


def proxy_relevance(c: Candidate) -> float:
    """0-4 tier. 0 = exclude / honeypot. 4 = strong tier-1 candidate."""
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

    # Map the 0-1 continuous score to 0-4 discrete tiers for NDCG.
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
    """Map candidate_id -> 0-4 relevance tier."""
    return {c.candidate_id: proxy_relevance(c) for c in candidates}
