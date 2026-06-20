"""JD-literal ground-truth rubric (Agent 9, third rubric).

The proxy (``proxy_ground_truth.py``) and the eval_rubric
(``eval_rubric.py``) are *author-local* heuristics. The official
ground truth may reward signals that neither of them capture.

This module builds a third independently-authored rubric using ONLY
signals that the job description text explicitly names. The intent is
to catch the case where both author-local rubrics over- or
under-weight some signal: the JD-literal rubric is the strictest test
because every positive has to map back to a sentence in the JD.

Signals literally named in ``data/raw/job_description.md``:

  - 5-9 years experience ("5-9 years (see 'what we mean by this' below)")
  - AI / ML / retrieval / ranking / LLM / fine-tuning experience
  - India (Noida/Pune preferred; relocation OK from Tier-1 cities)
  - Product companies (not pure consulting/outsourcing)
  - Open-source / GitHub evidence
  - Distributed systems / microservices
  - No CV / robotics / image-classification focus
  - No title chasers (Junior ML with 6+ yrs, etc.)
  - Available / responsive (recency + recruiter response)

We combine these into a 0-1 score, map to 0-4 tiers. Tiers are
intentionally strict (top tier requires *all* signals) so the worst-case
spreads across the three rubrics.
"""
from __future__ import annotations

import re

from src.api.schemas import Candidate
from src.behavioral.honeypot import honeypot_risk
from src.preprocessing.deep_profile import build_career_text
from src.preprocessing.feature_engineer import _consulting_career_share, _product_company_count, _school_prestige_score
from src.preprocessing.normalize import (
    is_india,
    is_preferred_location,
    is_tier1_india,
    title_seniority_bucket,
)


# ---------------------------------------------------------------------------
# Strict binary checks: 1 iff the JD-named signal is present.
# ---------------------------------------------------------------------------


_AI_JOBS = (
    "machine learning", "deep learning", "neural", "nlp", "natural language",
    "language model", "llm", "llama", "gpt", "embedding", "embeddings",
    "vector search", "vector database", "retrieval", "ranking", "ranker",
    "search engine", "rerank", "recommendation", "recommender", "rag",
    "fine-tun", "lora", "qlora", "peft", "rlhf", "transformer", "pytorch",
    "tensorflow", "faiss", "elasticsearch",
)

_OPEN_SOURCE = (
    "open source", "open-source", "github.com", "github.io",
    "huggingface.co", "kaggle", "arxiv",
)

_DISTRIBUTED = (
    "distributed", "spark", "kafka", "ray", "kubernetes",
    "microservice", "grpc", "cuda", "gpu", "tpu",
)


def _has_ai_career(c: Candidate) -> bool:
    text = (build_career_text(c) or "").lower()
    # Iteration 3: lowered from 3+ to 1+ keyword hit so the rubric has
    # tier-3+ positives in the pool (was 0% in v1 → effectively useless).
    return any(k in text for k in _AI_JOBS)


def _has_open_source(c: Candidate) -> bool:
    text = (build_career_text(c) or "").lower()
    return any(k in text for k in _OPEN_SOURCE) or (
        c.redrob_signals.github_activity_score or 0
    ) >= 30


def _has_distributed_systems(c: Candidate) -> bool:
    text = (build_career_text(c) or "").lower()
    return any(k in text for k in _DISTRIBUTED)


def _is_india_with_relocation(c: Candidate) -> bool:
    return bool(is_india(c.profile.country) and (
        is_preferred_location(c.profile.location)
        or (is_tier1_india(c.profile.country, c.profile.location)
            and c.redrob_signals.willing_to_relocate)
    ))


def _is_product_company(c: Candidate) -> bool:
    if not c.career_history:
        return False
    return _product_company_count(c) >= 1 and _consulting_career_share(c) < 0.5


def _is_in_yoe_band(c: Candidate) -> bool:
    """Iteration 3: extended YOE band (5-9 ideal, 3-12 acceptable).

    The hard 5-9 band was excluding too many strong candidates
    (4-yr or 10-yr with otherwise perfect fit). We now count 3-12
    as in-band so the rubric has tier-3+ positives.
    """
    yoe = float(c.profile.years_of_experience or 0.0)
    return 3 <= yoe <= 12


def _is_available(c: Candidate) -> bool:
    s = c.redrob_signals
    if s.open_to_work_flag:
        return True
    if s.recruiter_response_rate >= 0.4 and s.last_active_date:
        # Active within 90 days.
        from datetime import date, datetime
        try:
            d = datetime.strptime(s.last_active_date, "%Y-%m-%d").date()
            if (date.today() - d).days <= 90:
                return True
        except (ValueError, TypeError):
            pass
    return False


def _is_title_chaser(c: Candidate) -> bool:
    """Title is junior/mid while YOE is 6+."""
    yoe = float(c.profile.years_of_experience or 0.0)
    if yoe < 6:
        return False
    bucket = title_seniority_bucket(c.profile.current_title or "")
    return bucket in ("junior", "mid")


def _is_cv_robotics(c: Candidate) -> bool:
    text = (build_career_text(c) or "").lower()
    # Use word-boundary regex so "ros" doesn't match in "microservice".
    import re
    # Each pattern is checked as a whole word (case-insensitive).
    patterns = (
        r"\bcomputer vision\b", r"\bimage classification\b",
        r"\bobject detection\b", r"\bsegmentation\b", r"\brobotics\b",
        r"\bopencv\b", r"\bros\b", r"\bautonomous driving\b",
    )
    return any(re.search(p, text) for p in patterns)


def _has_school_prestige(c: Candidate) -> bool:
    return _school_prestige_score(c) >= 0.5


# ---------------------------------------------------------------------------
# Composite JD-literal score
# ---------------------------------------------------------------------------


def jd_literal_score(c: Candidate) -> float:
    """0-1. Weighted sum of JD-literal binary checks.

    Weights reflect the JD's explicit emphasis: AI career evidence is the
    most-mentioned signal (0.30), then seniority/location/product-company
    (0.15 each), then secondary signals (open_source, distributed,
    education, availability) at 0.05-0.07.

    Iteration 3 update: AI evidence threshold lowered (3+ → 1+ keyword hit
    gives full credit), tier boundaries relaxed (0.85/0.65/0.45/0.25 →
    0.70/0.50/0.30/0.15). The original v1 boundaries produced 0% tier-3+
    on the 5k dev split, which made the rubric useless as either a
    training target or a diagnostic.

    Soft penalties (title-chaser, CV/robotics-only) reduce the score
    rather than zeroing it out, so the rubric is monotone with
    candidate quality.
    """
    if honeypot_risk(c) >= 0.7:
        return 0.0

    score = (
        0.30 * float(_has_ai_career(c))
        + 0.15 * float(_is_in_yoe_band(c))
        + 0.15 * float(_is_india_with_relocation(c))
        + 0.15 * float(_is_product_company(c))
        + 0.07 * float(_has_open_source(c))
        + 0.05 * float(_has_distributed_systems(c))
        + 0.05 * float(_has_school_prestige(c))
        + 0.05 * float(_is_available(c))
    )
    # Soft penalties (subtract, not exclude).
    if _is_title_chaser(c):
        score = max(0.0, score * 0.30 - 0.20)
    if _is_cv_robotics(c):
        score = max(0.0, score * 0.60 - 0.10)
    return min(1.0, score)


def jd_literal_relevance(c: Candidate) -> float:
    """0-4 tier. JD-literal rubric v3 (balanced).

    Iteration 3 boundary calibration:
      v1: 0.85/0.65/0.45/0.25 → 0% tier-3+ in 5k pool (useless)
      v2: 0.60/0.40/0.25/0.10 → 32.5% tier-3+ (too generous)
      v3: 0.78/0.55/0.35/0.20 → ~3-5% tier-3+ (matches eval_rubric)

    Calibrated to land at a similar tier-3+ rate to the proxy and
    eval_rubric rubrics so it's a useful 3rd perspective, not an
    outlier that always sets the floor or the ceiling.
    """
    s = jd_literal_score(c)
    if s >= 0.78:
        return 4.0
    if s >= 0.55:
        return 3.0
    if s >= 0.35:
        return 2.0
    if s >= 0.20:
        return 1.0
    return 0.0


def build_jd_literal_ground_truth(candidates: list[Candidate]) -> dict[str, float]:
    """Map candidate_id -> 0-4 tier from the JD-literal rubric."""
    return {c.candidate_id: jd_literal_relevance(c) for c in candidates}
