"""Per-candidate feature engineering.

The output is a flat dict of numeric + categorical features keyed by
candidate_id. The artifact lives in `artifacts/feature_store.parquet` and is
loaded at ranking time.

Features are deliberately *additive* and *interpretable*. Every feature is
defined here with a docstring, so a reviewer can read this file and predict
exactly what the model sees.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import date, datetime
from typing import Any

from src.api.schemas import Candidate
from src.preprocessing.deep_profile import build_career_text, build_skills_text
from src.preprocessing.normalize import (
    AI_KEYWORDS,
    is_consulting_company,
    is_india,
    is_preferred_location,
    is_tier1_india,
    location_tokens,
    normalize_industry,
    normalize_skill,
    seniority_at_least,
    title_seniority_bucket,
)

SENIORITY_FLOOR = 3  # 0=intern, 1=junior, 2=mid, 3=senior, 4=staff, 5=manager

# JD-specific named skills (referenced in the job description, used for the
# `has_named_jd_skill_*` features and the LLM anti-hallucination validator).
JD_NAMED_SKILLS: tuple[str, ...] = (
    "retrieval",
    "ranking",
    "rerank",
    "embeddings",
    "vector search",
    "rag",
    "fine-tuning",
    "lora",
    "peft",
    "rlhf",
    "eval",
    "pytorch",
    "transformers",
    "sentence-transformers",
    "faiss",
    "elasticsearch",
    "pinecone",
    "weaviate",
    "learning to rank",
    "lambdarank",
)


# WS-12: Tier-1/Tier-2 school prestige list (loaded from configs/school_prestige.yaml).
# Falls back to a small built-in list if the config isn't found.
_DEFAULT_TIER_1: tuple[str, ...] = (
    "IIT Bombay", "IIT Delhi", "IIT Madras", "IIT Kanpur", "IIT Kharagpur",
    "IIM Ahmedabad", "IIM Bangalore", "IIM Calcutta",
    "MIT", "Stanford", "Carnegie Mellon", "Berkeley", "ETH Zurich",
    "National University of Singapore", "NUS",
)
_DEFAULT_TIER_2: tuple[str, ...] = (
    "IIT Hyderabad", "IIT Indore", "IIT BHU", "IIT (BHU) Varanasi",
    "NIT Trichy", "NIT Warangal", "BITS Pilani",
    "University of Michigan", "Georgia Tech", "University of Toronto",
    "University of Waterloo", "University of Melbourne", "HKUST",
)


def _load_school_prestige() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Load the school prestige list from configs/school_prestige.yaml."""
    try:
        import yaml
        from pathlib import Path

        # Search in (a) CWD/configs, (b) the package-relative configs dir.
        candidates = [
            Path("configs/school_prestige.yaml"),
            Path(__file__).resolve().parents[2] / "configs" / "school_prestige.yaml",
        ]
        for cfg_path in candidates:
            if cfg_path.exists():
                data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                t1 = tuple(s.lower() for s in data.get("tier_1", ()))
                t2 = tuple(s.lower() for s in data.get("tier_2", ()))
                return t1, t2
    except Exception:
        pass
    return tuple(s.lower() for s in _DEFAULT_TIER_1), tuple(s.lower() for s in _DEFAULT_TIER_2)


def _school_prestige_score(c: Candidate) -> float:
    """1.0 if any education matches a tier-1 school, 0.5 for tier-2, else 0.

    Matching is case-insensitive substring on `institution`.
    """
    if not c.education:
        return 0.0
    t1, t2 = _load_school_prestige()
    best = 0.0
    for e in c.education:
        inst = (e.institution or "").lower()
        if not inst:
            continue
        for needle in t1:
            if needle in inst:
                return 1.0
        for needle in t2:
            if needle in inst:
                best = max(best, 0.5)
    return best


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _days_ago(s: str | None, today: date) -> int | None:
    d = _parse_date(s)
    if d is None:
        return None
    return (today - d).days


def _career_total_months(c: Candidate) -> int:
    return sum(max(0, role.duration_months) for role in c.career_history)


def _current_role(c: Candidate):
    for r in c.career_history:
        if r.is_current:
            return r
    return c.career_history[0] if c.career_history else None


def _avg_tenure_months(c: Candidate) -> float:
    completed = [r.duration_months for r in c.career_history if not r.is_current and r.duration_months > 0]
    if not completed:
        current = _current_role(c)
        return float(current.duration_months) if current else 0.0
    return sum(completed) / len(completed)


def _is_current_count(c: Candidate) -> int:
    return sum(1 for r in c.career_history if r.is_current)


def _ai_keyword_hits(text: str) -> int:
    if not text:
        return 0
    t = text.lower()
    return sum(1 for k in AI_KEYWORDS if k in t)


def _has_opensource_github(c: Candidate) -> bool:
    s = c.redrob_signals
    return s.github_activity_score >= 50 and s.github_activity_score > 0


def _consulting_career_count(c: Candidate) -> int:
    return sum(1 for r in c.career_history if is_consulting_company(r.company))


def _consulting_career_share(c: Candidate) -> float:
    if not c.career_history:
        return 0.0
    return _consulting_career_count(c) / len(c.career_history)


def _product_company_count(c: Candidate) -> int:
    return sum(
        1
        for r in c.career_history
        if normalize_industry(r.industry) in {"saas", "ai_ml", "fintech", "ecommerce", "edtech", "adtech", "gaming", "transportation", "food", "healthcare", "telecom"}
        and not is_consulting_company(r.company)
    )


def _skills_with_zero_months_expert(c: Candidate) -> int:
    return sum(1 for s in c.skills if s.proficiency == "expert" and s.duration_months == 0)


def _skills_with_zero_months_total(c: Candidate) -> int:
    return sum(1 for s in c.skills if s.duration_months == 0)


def _all_skills_zero_endorsements(c: Candidate) -> bool:
    return bool(c.skills) and all(s.endorsements == 0 for s in c.skills)


def _perfect_skill_match_with_non_tech_title(c: Candidate) -> bool:
    """If the candidate's skill list overlaps heavily with the JD core skills
    but their current_title is non-technical, treat as keyword stuffing.
    """
    title = (c.profile.current_title or "").lower()
    if any(t in title for t in ("engineer", "scientist", "developer", "analyst", "architect", "researcher")):
        return False
    skill_names = {normalize_skill(s.name) for s in c.skills}
    jd_core = {
        "machine learning", "nlp", "llm", "embedding", "vector search",
        "retrieval augmented generation", "pytorch", "tensorflow", "transformers",
        "ranker", "lora", "qlora", "peft", "xgboost", "lightgbm",
    }
    overlap = skill_names & jd_core
    n_expert = sum(1 for s in c.skills if s.proficiency == "expert")
    return len(overlap) >= 5 and n_expert >= 4


def _cv_robotics_only(c: Candidate) -> bool:
    """Primary expertise is CV/robotics/speech without NLP/IR exposure."""
    career_text = (build_career_text(c) or "").lower()
    has_cv = any(k in career_text for k in ("computer vision", "image classification", "object detection", "segmentation", "robotics", "opencv", "ros", "embedded", "autonomous"))
    has_nlp = any(k in career_text for k in (" nlp", "natural language", "language model", " llm", "rag", "retrieval", "search", "ranker", "rank"))
    return has_cv and not has_nlp


def _langchain_recent_only(c: Candidate) -> bool:
    """Has 'AI experience' but it's basically LangChain + recent only."""
    career_text = (build_career_text(c) or "").lower()
    has_langchain = "langchain" in career_text
    has_deep = any(k in career_text for k in ("pytorch", "tensorflow", "transformer", "embedding", "retrieval", "ranker", "xgboost", "lightgbm"))
    if has_langchain and not has_deep:
        return True
    return False


def _closed_source_only(c: Candidate) -> bool:
    s = c.redrob_signals
    return s.github_activity_score <= 0 and not any(
        k in (build_career_text(c) or "").lower()
        for k in ("open source", "open-source", "github.com", "arxiv", "publication", "paper")
    )


def _title_chaser(c: Candidate) -> bool:
    return _avg_tenure_months(c) < 18 and len(c.career_history) >= 3


def _is_location_good(c: Candidate) -> bool:
    if is_preferred_location(c.profile.location):
        return True
    if is_tier1_india(c.profile.country, c.profile.location):
        return True
    if c.redrob_signals.willing_to_relocate and is_india(c.profile.country):
        return True
    return False


def _notice_period_score(notice_days: int) -> float:
    if notice_days <= 0:
        return 1.0
    if notice_days <= 30:
        return 1.0
    if notice_days <= 60:
        return 0.7
    if notice_days <= 90:
        return 0.4
    if notice_days <= 120:
        return 0.2
    return 0.0


# ---------------------------------------------------------------------------
# New features (WS-5)
# ---------------------------------------------------------------------------

# Numeric ordering for title-seniority consistency. Higher bucket = more senior.
_SENIORITY_BUCKET_INDEX: dict[str, int] = {
    "intern": 0,
    "junior": 1,
    "mid": 2,
    "senior": 3,
    "staff": 4,
    "manager": 5,
    "unknown": 2,  # default: treat as mid-band
}


def _expected_seniority_for_yoe(yoe: float) -> int:
    """Map years-of-experience to an expected title-bucket index.

    Mirrors typical industry ladders: <2 yrs → junior, 2-5 → mid,
    5-9 → senior, 9+ → staff/manager.
    """
    if yoe < 2:
        return _SENIORITY_BUCKET_INDEX["junior"]
    if yoe < 5:
        return _SENIORITY_BUCKET_INDEX["mid"]
    if yoe <= 9:
        return _SENIORITY_BUCKET_INDEX["senior"]
    return _SENIORITY_BUCKET_INDEX["staff"]


def _title_yoe_consistency(c: Candidate, yoe: float) -> float:
    """Signed delta: expected-bucket index minus actual title-bucket index.

    Negative → title is junior for the experience; positive → title is more
    senior than typical. The JD explicitly warns about title/evidence
    mismatches (e.g. "Junior" with 6+ yrs).
    """
    title = c.profile.current_title or ""
    actual = _SENIORITY_BUCKET_INDEX.get(title_seniority_bucket(title), 2)
    expected = _expected_seniority_for_yoe(yoe)
    return float(expected - actual)


def _title_yoe_inconsistent(c: Candidate, yoe: float) -> int:
    """Hard 0/1: 1 if the title is at least 2 buckets below the expected
    bucket for the candidate's YOE. e.g. "Junior" with 6+ yrs, or "Mid"
    with 12+ yrs. This is the "title chaser" anti-pattern the JD warns
    about. Strong negative signal for the LTR.
    """
    if yoe < 2:
        return 0
    title = c.profile.current_title or ""
    actual = _SENIORITY_BUCKET_INDEX.get(title_seniority_bucket(title), 2)
    expected = _expected_seniority_for_yoe(yoe)
    return int((expected - actual) >= 2)


def _career_progression(c: Candidate) -> float:
    """Signed slope of title-seniority across roles (chronological order).

    Positive → titles became more senior over time (a JD-positive signal);
    negative → titles flattened or dropped.
    """
    roles = c.career_history
    if len(roles) < 2:
        return 0.0
    series = [
        _SENIORITY_BUCKET_INDEX.get(title_seniority_bucket(r.title), 2)
        for r in roles
    ]
    n = len(series)
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(series) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, series, strict=True))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den < 1e-9:
        return 0.0
    return float(num / den)


def _n_distinct_industries(c: Candidate) -> int:
    return len({(r.industry or "").strip().lower() for r in c.career_history if (r.industry or "").strip()})


def _has_named_jd_skill_count(c: Candidate, career: str, skills_text: str) -> int:
    """How many of the JD's named skills appear in the candidate's career
    text or skill list. Used as a single integer and as a count of buckets.
    """
    blob = (career or "").lower() + " " + (skills_text or "").lower()
    n = 0
    for needle in JD_NAMED_SKILLS:
        if needle in blob:
            n += 1
    return n


def _n_named_jd_skills_continuous(c: Candidate, career: str, skills_text: str) -> float:
    """WS-12 follow-up: continuous version of `n_named_jd_skills`.

    Returns the count normalised by the total number of JD-named skills, so
    the value lives in [0, 1] and LightGBM can give it a continuous weight
    instead of treating it as a sparse integer. Tier 1 #4.
    """
    n = _has_named_jd_skill_count(c, career, skills_text)
    return min(1.0, n / max(1, len(JD_NAMED_SKILLS)))


def _distributed_systems_count(c: Candidate, career: str) -> int:
    """WS-12 follow-up: count of role descriptions that mention
    distributed-systems keywords. Used as a continuous feature
    (vs the binary `is_distributed_systems_evidence` in the proxy).
    """
    keys = (
        "distributed", "spark", "kafka", "ray", "kubernetes", "microservice",
        "grpc", "cuda", "gpu", "tpu", "horovod", "deepspeed",
    )
    n = 0
    blob = (career or "").lower()
    for r in c.career_history:
        d = (r.description or "").lower()
        if any(k in d for k in keys):
            n += 1
    # Bonus: if the whole blob mentions these, +1
    if any(k in blob for k in keys):
        n = max(n, 1)
    return n


def _open_source_count(c: Candidate, career: str) -> int:
    """WS-12 follow-up: count of role descriptions that mention
    open-source signals. Used as a continuous feature.
    Matches the binary `_has_open_source_evidence` logic: counts any
    role with a description match + 1 if `github_activity_score >= 30`.
    """
    keys = (
        "open source", "open-source", "github.com", "github.io", "arxiv",
        "paper", "medium", "kaggle", "huggingface.co", "contributor", "foss",
    )
    n = 0
    blob = (career or "").lower()
    for r in c.career_history:
        d = (r.description or "").lower()
        if any(k in d for k in keys):
            n += 1
    if any(k in blob for k in keys):
        n = max(n, 1)
    if c.redrob_signals.github_activity_score and c.redrob_signals.github_activity_score >= 30:
        n = max(n, 1)
    return n


def _consulting_to_product_transition(c: Candidate) -> int:
    """1 iff the candidate's career went from consulting → product."""
    if len(c.career_history) < 2:
        return 0
    earliest = c.career_history[-1]
    latest = c.career_history[0]
    return int(is_consulting_company(earliest.company) and not is_consulting_company(latest.company))


def _endorsement_entropy(c: Candidate) -> float:
    """Shannon entropy of endorsement counts across skills.

    A profile where all endorsements concentrate in 1-2 skills scores low
    (low diversity, possible keyword stuffing); a balanced profile scores
    high. Returns 0 for empty skills.
    """
    if not c.skills:
        return 0.0
    counts = [max(1, int(s.endorsements)) for s in c.skills]
    total = sum(counts)
    if total <= 0:
        return 0.0
    import math as _math

    return -sum((c_n / total) * _math.log2(c_n / total) for c_n in counts if c_n > 0)


def _recency_score(last_active: str | None, today: date) -> float:
    days = _days_ago(last_active, today)
    if days is None:
        return 0.0
    if days <= 7:
        return 1.0
    if days <= 30:
        return 0.9
    if days <= 90:
        return 0.7
    if days <= 180:
        return 0.4
    if days <= 365:
        return 0.15
    return 0.0


def _behavioral_availability(c: Candidate) -> float:
    """Backwards-compat: the original availability_score column.

    Computed at build time so the ranker can read it directly from the
    feature store (avoids re-loading candidates at rank time).
    """
    from src.behavioral.availability import availability_score
    return float(availability_score(c))


def _behavioral_positive(c: Candidate) -> float:
    """Backwards-compat: the original positive_boost column."""
    from src.behavioral.jd_filters import positive_boost
    return float(positive_boost(c))


def _behavioral_negative(c: Candidate) -> float:
    """Backwards-compat: the original negative_penalty column."""
    from src.behavioral.jd_filters import negative_penalty
    return float(negative_penalty(c))


def _behavioral_honeypot(c: Candidate) -> float:
    """Backwards-compat: the original honeypot_risk column."""
    from src.behavioral.honeypot import honeypot_risk
    return float(honeypot_risk(c))


# ---------------------------------------------------------------------------
# Feature zoo v2 (Agent 5) — additional discriminative features added to
# the build_features dict and to feature_columns(). These split the LTR's
# 83% single-feature dependency on `ai_keyword_hits_career` into 30+ smaller,
# mutually-correlated signals so the LightGBM tree has more splits to work
# with when ordering the top-K.
# ---------------------------------------------------------------------------


def _n_ai_roles(c: Candidate) -> int:
    """Count of roles at AI/ML companies (industry == ai/ml)."""
    n = 0
    for r in c.career_history:
        if (r.industry or "").lower() == "ai/ml":
            n += 1
    return n


def _n_senior_roles(c: Candidate) -> int:
    """Count of senior / staff / manager roles."""
    n = 0
    for r in c.career_history:
        t = (r.title or "").lower()
        if any(k in t for k in ("senior", "staff", "principal", "manager", "lead", "head")):
            n += 1
    return n


def _n_product_roles(c: Candidate) -> int:
    """Count of roles at product companies (not consulting / outsourcing)."""
    return _product_company_count(c)


def _n_india_roles(c: Candidate) -> int:
    """Count of roles based in India (location or country indicator)."""
    n = 0
    for r in c.career_history:
        loc = (r.title or "") + " " + (r.description or "")
        if is_india(loc) or any(
            city in loc.lower()
            for city in ("bangalore", "bengaluru", "hyderabad", "pune", "noida", "delhi", "mumbai", "chennai")
        ):
            n += 1
    return n


def _avg_role_duration_months(c: Candidate) -> float:
    """Average role duration in months across all roles."""
    if not c.career_history:
        return 0.0
    durs = [r.duration_months for r in c.career_history if r.duration_months and r.duration_months > 0]
    return float(sum(durs) / len(durs)) if durs else 0.0


def _max_role_duration_months(c: Candidate) -> float:
    """Longest role duration in months (loyalty signal)."""
    if not c.career_history:
        return 0.0
    durs = [r.duration_months for r in c.career_history if r.duration_months and r.duration_months > 0]
    return float(max(durs)) if durs else 0.0


def _tenure_variance(c: Candidate) -> float:
    """Variance of role durations (high = job hopper, low = stable)."""
    if not c.career_history:
        return 0.0
    durs = [r.duration_months for r in c.career_history if r.duration_months and r.duration_months > 0]
    if len(durs) < 2:
        return 0.0
    mean = sum(durs) / len(durs)
    return float(sum((d - mean) ** 2 for d in durs) / len(durs))


def _career_progression_slope(c: Candidate) -> float:
    """Linear slope of role-seniority over time. Positive = growth pattern.

    Roles are ordered from oldest to newest; we map each role's title to
    a seniority bucket (0..5) and compute the slope of that line vs
    chronological order. Stable senior roles give a flat ~0; a Junior →
    Senior pattern over 7 years gives a strong positive slope.
    """
    if not c.career_history or len(c.career_history) < 2:
        return 0.0
    # Use chronological order (career_history is newest-first; reverse).
    roles = list(reversed(c.career_history))
    xs: list[int] = []
    ys: list[float] = []
    for i, r in enumerate(roles):
        bucket = _SENIORITY_BUCKET_INDEX.get(title_seniority_bucket(r.title or ""), 2)
        xs.append(i)
        ys.append(float(bucket))
    n = len(xs)
    if n < 2:
        return 0.0
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys, strict=True))
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0
    return float((n * sxy - sx * sy) / denom)


def _current_role_tenure_months(c: Candidate) -> float:
    """Months in the current (most-recent) role. Long tenure = stable."""
    if not c.career_history:
        return 0.0
    cur = c.career_history[0]
    return float(cur.duration_months or 0)


def _n_career_gaps(c: Candidate) -> int:
    """Number of gaps > 6 months between consecutive roles."""
    if not c.career_history or len(c.career_history) < 2:
        return 0
    # career_history is newest-first; walk oldest-first.
    roles = list(reversed(c.career_history))
    gaps = 0
    for i in range(1, len(roles)):
        prev = roles[i - 1]
        cur = roles[i]
        # Compare end_date of prev role with start_date of cur role.
        if prev.end_date and cur.start_date:
            try:
                p_end = datetime.strptime(prev.end_date, "%Y-%m-%d").date()
                c_start = datetime.strptime(cur.start_date, "%Y-%m-%d").date()
                gap_days = (c_start - p_end).days
                if gap_days > 180:
                    gaps += 1
            except (ValueError, TypeError):
                continue
    return gaps


def _jd_skill_match_count(c: Candidate, career: str, skills_text: str) -> int:
    """Count of JD-named skills found in the candidate's career + skills text."""
    blob = (career + " " + skills_text).lower()
    return sum(1 for s in JD_NAMED_SKILLS if s in blob)


def _jd_skill_match_expert_count(c: Candidate) -> int:
    """Count of JD-named skills at 'expert' proficiency on the profile."""
    n = 0
    blob = " ".join((sk.name or "").lower() for sk in c.skills)
    for s in JD_NAMED_SKILLS:
        if s in blob:
            # Check if any of the matching skills is at expert level.
            for sk in c.skills:
                if s in (sk.name or "").lower() and sk.proficiency == "expert":
                    n += 1
                    break
    return n


def _jd_keyword_count_career(career: str) -> int:
    """Count of JD-aligned keywords in the candidate's career text."""
    keys = (
        "embedding", "retrieval", "ranking", "ranker", "rerank", "rag",
        "fine-tun", "lora", "peft", "rlhf", "transformer", "pytorch",
        "vector search", "semantic search", "learning to rank", "lambdarank",
        "sentence-transformer", "faiss", "elasticsearch", "pinecone", "weaviate",
    )
    blob = (career or "").lower()
    return sum(min(3, blob.count(k)) for k in keys)


def _jd_keyword_count_summary(c: Candidate) -> int:
    """Count of JD-aligned keywords in the candidate's profile summary."""
    keys = (
        "embedding", "retrieval", "ranking", "rag", "fine-tun", "lora",
        "transformer", "pytorch", "vector search", "semantic search",
    )
    blob = (c.profile.summary or "").lower()
    return sum(min(3, blob.count(k)) for k in keys)


def _title_jd_match(c: Candidate) -> int:
    """1 if the current title matches JD-aligned keywords."""
    t = (c.profile.current_title or "").lower()
    keys = ("ml", "ai", "machine learning", "data scientist", "nlp", "llm")
    return int(any(k in t for k in keys))


def _seniority_jd_match(c: Candidate) -> int:
    """1 if YOE is in the JD's 5-9 band (the ideal band)."""
    yoe = float(c.profile.years_of_experience or 0.0)
    return int(5 <= yoe <= 9)


def _location_jd_match(c: Candidate) -> int:
    """1 if location is preferred OR willing to relocate from a tier-1 city."""
    if is_preferred_location(c.profile.location):
        return 1
    if is_tier1_india(c.profile.country, c.profile.location) and c.redrob_signals.willing_to_relocate:
        return 1
    return 0


def _industry_jd_match(c: Candidate) -> int:
    """1 if current industry is AI/ML."""
    return int(normalize_industry(c.profile.current_industry) == "ai_ml")


def _engagement_intensity(c: Candidate) -> float:
    """Composite engagement: profile_views + search + saved (log-scaled)."""
    s = c.redrob_signals
    return float(
        safe_log1p(s.profile_views_received_30d)
        + safe_log1p(s.search_appearance_30d)
        + safe_log1p(s.saved_by_recruiters_30d)
    )


def _log_profile_views_30d(c: Candidate) -> float:
    return safe_log1p(c.redrob_signals.profile_views_received_30d)


def _log_search_appearance_30d(c: Candidate) -> float:
    return safe_log1p(c.redrob_signals.search_appearance_30d)


def _log_saved_by_recruiters_30d(c: Candidate) -> float:
    return safe_log1p(c.redrob_signals.saved_by_recruiters_30d)


def _log_connection_count(c: Candidate) -> float:
    return safe_log1p(c.redrob_signals.connection_count)


def _log_endorsements_received(c: Candidate) -> float:
    return safe_log1p(c.redrob_signals.endorsements_received)


def _behavioral_risk_score(c: Candidate) -> float:
    """Composite behavioral risk. Higher = riskier to hire.

    Combines: very long notice period, no offer history, low recruiter
    response, profile_views = 0, no endorsements, no verifications.
    Each component is in [0, 1]; the sum is clipped to [0, 1].
    """
    s = c.redrob_signals
    risk = 0.0
    # Long notice period (>90d) is a strong no-hire signal in India.
    if s.notice_period_days > 90:
        risk += 0.30
    elif s.notice_period_days > 60:
        risk += 0.15
    # No offer history → unknown past behavior.
    if s.offer_acceptance_rate < 0:
        risk += 0.15
    # Very low recruiter response.
    if s.recruiter_response_rate < 0.10:
        risk += 0.15
    # No profile views in 30d → invisible.
    if s.profile_views_received_30d <= 0:
        risk += 0.10
    # No verifications.
    if not s.verified_email:
        risk += 0.10
    if not s.verified_phone:
        risk += 0.10
    if not s.linkedin_connected:
        risk += 0.05
    return min(1.0, risk)


def _availability_composite(c: Candidate) -> float:
    """Composite availability: positive signals only. In [0, 1]."""
    s = c.redrob_signals
    pos = 0.0
    if s.open_to_work_flag:
        pos += 0.25
    if s.willing_to_relocate:
        pos += 0.10
    if s.notice_period_days <= 30:
        pos += 0.20
    elif s.notice_period_days <= 60:
        pos += 0.10
    if s.recruiter_response_rate >= 0.5:
        pos += 0.15
    elif s.recruiter_response_rate >= 0.3:
        pos += 0.05
    if s.last_active_date:
        days = _days_ago(s.last_active_date, date.today())
        if days is not None and days <= 30:
            pos += 0.15
        elif days is not None and days <= 90:
            pos += 0.08
    if s.profile_completeness_score >= 80:
        pos += 0.05
    if s.offer_acceptance_rate >= 0.5:
        pos += 0.10
    return min(1.0, pos)


def _expert_share(c: Candidate) -> float:
    """Share of skills at 'expert' level. In [0, 1]."""
    if not c.skills:
        return 0.0
    return sum(1 for sk in c.skills if sk.proficiency == "expert") / len(c.skills)


def _advanced_or_expert_share(c: Candidate) -> float:
    """Share of skills at advanced+expert level."""
    if not c.skills:
        return 0.0
    return sum(1 for sk in c.skills if sk.proficiency in ("advanced", "expert")) / len(c.skills)


def _skill_endorsement_mean(c: Candidate) -> float:
    """Mean endorsement count per skill. In [0, +∞)."""
    if not c.skills:
        return 0.0
    return float(sum(sk.endorsements for sk in c.skills) / len(c.skills))


def _skill_endorsement_max(c: Candidate) -> float:
    """Max endorsement count on any single skill."""
    if not c.skills:
        return 0.0
    return float(max(sk.endorsements for sk in c.skills))


def _skill_count_log(c: Candidate) -> float:
    """log(1 + n_skills)."""
    return safe_log1p(len(c.skills))


def _has_product_company_recent(c: Candidate) -> int:
    """1 if current or most-recent role is at a product company."""
    if not c.career_history:
        return 0
    return int((c.career_history[0].industry or "").lower() in (
        "ai/ml", "saas", "fintech", "ecommerce", "edtech", "adtech", "gaming",
    ))


def _title_yoe_in_band(c: Candidate) -> int:
    """1 if title bucket matches YOE band (no title/YOE mismatch)."""
    yoe = float(c.profile.years_of_experience or 0.0)
    actual = _SENIORITY_BUCKET_INDEX.get(
        title_seniority_bucket(c.profile.current_title or ""), 2
    )
    expected = _expected_seniority_for_yoe(yoe)
    return int(abs(expected - actual) <= 1)


def _seniority_distance_from_ideal(c: Candidate) -> float:
    """Absolute distance from the JD's 5-9 ideal band, in years."""
    yoe = float(c.profile.years_of_experience or 0.0)
    if 5 <= yoe <= 9:
        return 0.0
    if yoe < 5:
        return 5.0 - yoe
    return yoe - 9.0


def _ai_skill_count_log(c: Candidate) -> float:
    """log(1 + count of AI-related skills on profile)."""
    ai_skills = {
        "pytorch", "tensorflow", "transformers", "scikit-learn", "sklearn",
        "numpy", "pandas", "jax", "keras", "huggingface",
    }
    n = sum(1 for sk in c.skills if (sk.name or "").lower() in ai_skills)
    return safe_log1p(n)


def _career_jd_sim_proxy(c: Candidate) -> float:
    """Cheap proxy for career_JD semantic similarity without an embedder.

    Counts the number of unique JD-named keywords found in the career text,
    normalised by the number of JD-named skills. In [0, 1].
    """
    blob = (build_career_text(c) or "").lower()
    hits = sum(1 for s in JD_NAMED_SKILLS if s in blob)
    return min(1.0, hits / max(1, len(JD_NAMED_SKILLS) // 3))




def build_features(c: Candidate, today: date | None = None) -> dict[str, Any]:
    """Build the per-candidate feature dict.

    Every key here is also declared in `configs/ranking.yaml::features` so the
    LTR trainer and the ranker can agree on the schema.
    """
    if today is None:
        today = date.today()

    p = c.profile
    s = c.redrob_signals
    career = build_career_text(c) or ""
    skills_text = build_skills_text(c) or ""

    yoe = float(p.years_of_experience or 0.0)
    career_months = _career_total_months(c)
    career_years = career_months / 12.0 if career_months else 0.0

    current_role = _current_role(c)
    current_title = current_role.title if current_role else p.current_title
    seniority = title_seniority_bucket(current_title)

    ai_hits_career = _ai_keyword_hits(career)
    ai_hits_skills = _ai_keyword_hits(skills_text)

    consulting_share = _consulting_career_share(c)
    product_count = _product_company_count(c)

    features: dict[str, Any] = {
        # ------------------------------------------------------------------
        # Profile
        # ------------------------------------------------------------------
        "candidate_id": c.candidate_id,
        "yoe_reported": yoe,
        "yoe_career_sum": career_years,
        "yoe_diff": abs(yoe - career_years),
        "n_career_roles": len(c.career_history),
        "n_current_roles": _is_current_count(c),
        "avg_tenure_months": _avg_tenure_months(c),
        "n_skills": len(c.skills),
        "n_projects": len(c.projects),
        "n_certifications": len(c.certifications),
        "n_education": len(c.education),
        "education_tier_1": int(any((e.tier or "").lower() == "tier_1" for e in c.education)),
        "education_tier_2": int(any((e.tier or "").lower() == "tier_2" for e in c.education)),
        "education_prestige": _school_prestige_score(c),

        # ------------------------------------------------------------------
        # WS-10: career-JD semantic similarity (BGE cosine).
        # Filled in at build time by `career_jd_sim.attach_similarity_column_inplace`
        # AFTER `build_features` returns. Default 0.0 here so the column is
        # always present in the feature frame.
        # ------------------------------------------------------------------
        "career_jd_semantic_sim": 0.0,

        # ------------------------------------------------------------------
        # Raw signal passthroughs (needed by vectorized behavioral scoring)
        # ------------------------------------------------------------------
        "last_active_date": c.redrob_signals.last_active_date,
        "open_to_work_raw": int(c.redrob_signals.open_to_work_flag),
        "preferred_work_mode_raw": c.redrob_signals.preferred_work_mode,
        "current_industry_raw": c.profile.current_industry,
        "current_title_raw": c.profile.current_title,

        # ------------------------------------------------------------------
        # Seniority
        # ------------------------------------------------------------------
        "seniority_bucket": seniority,
        "seniority_at_least_senior": int(seniority_at_least(seniority, "senior")),
        "seniority_at_least_staff": int(seniority_at_least(seniority, "staff")),
        "yoe_in_5_9_band": int(5 <= yoe <= 9),

        # ------------------------------------------------------------------
        # AI/ML evidence (career-first, skills-second)
        # ------------------------------------------------------------------
        "ai_keyword_hits_career": ai_hits_career,
        "ai_keyword_hits_skills": ai_hits_skills,
        "ai_career_share": min(1.0, ai_hits_career / 8.0),
        "ai_skill_count": sum(
            1
            for sk in c.skills
            if any(k in (sk.name or "").lower() for k in ("machine learning", "ml", "nlp", "llm", "embedding", "deep learning", "neural", "pytorch", "tensorflow", "transformer", "retrieval", "ranker", "xgboost", "lightgbm", "rag", "lora", "fine-tun"))
        ),
        "n_ai_skill_advanced": sum(
            1
            for sk in c.skills
            if sk.proficiency in ("advanced", "expert")
            and any(k in (sk.name or "").lower() for k in ("machine learning", "ml", "nlp", "llm", "embedding", "deep learning", "neural", "pytorch", "tensorflow", "transformer", "retrieval", "ranker", "xgboost", "lightgbm", "rag", "lora", "fine-tun"))
        ),
        "has_retrieval_ranking_evidence": int(any(
            k in career.lower() for k in ("ranking", "retrieval", "search", "recommend", "rerank", "vector search", "embedding")
        )),
        "has_llm_finetune_evidence": int(any(
            k in career.lower() for k in ("lora", "qlora", "peft", "fine-tun", "rlhf", "llm", "language model", "rag")
        )),
        "has_shipped_to_users": int(any(
            k in career.lower() for k in ("shipped", "launched", "production", "deployed", "users")
        )),
        "has_distributed_systems_evidence": int(any(
            k in career.lower() for k in ("distributed", "spark", "kafka", "ray", "kubernetes", "microservice", "grpc", "scalab")
        )),
        "has_open_source_evidence": int(_has_opensource_github(c) or any(
            k in career.lower() for k in ("open source", "open-source", "github.com", "arxiv", "paper", "publication")
        )),

        # ------------------------------------------------------------------
        # New features (WS-5)
        # ------------------------------------------------------------------
        "title_yoe_consistency": _title_yoe_consistency(c, yoe),
        "title_yoe_inconsistent": _title_yoe_inconsistent(c, yoe),
        "career_progression": _career_progression(c),
        "n_distinct_industries": _n_distinct_industries(c),
        "n_named_jd_skills": _has_named_jd_skill_count(c, career, skills_text),
        "n_named_jd_skills_continuous": _n_named_jd_skills_continuous(c, career, skills_text),
        "distributed_systems_count": _distributed_systems_count(c, career),
        "open_source_count": _open_source_count(c, career),
        "consulting_to_product_transition": _consulting_to_product_transition(c),
        "endorsement_entropy": _endorsement_entropy(c),

        # ------------------------------------------------------------------
        # Company / industry profile
        # ------------------------------------------------------------------
        "consulting_share": consulting_share,
        "consulting_only": int(consulting_share >= 0.99 and len(c.career_history) >= 1),
        "product_company_count": product_count,
        "current_industry_normalized": normalize_industry(p.current_industry),
        "current_industry_ai_ml": int(normalize_industry(p.current_industry) == "ai_ml"),
        "current_company_is_consulting": int(is_consulting_company(p.current_company)),

        # ------------------------------------------------------------------
        # Honeypot / consistency
        # ------------------------------------------------------------------
        "skill_expert_zero_months": _skills_with_zero_months_expert(c),
        "skill_zero_months_total": _skills_with_zero_months_total(c),
        "all_skills_zero_endorsements": int(_all_skills_zero_endorsements(c)),
        "perfect_skill_list_with_non_tech_title": int(_perfect_skill_match_with_non_tech_title(c)),
        "expert_skill_count": sum(1 for sk in c.skills if sk.proficiency == "expert"),
        "advanced_skill_count": sum(1 for sk in c.skills if sk.proficiency == "advanced"),
        "yoe_vs_career_sum_anomaly": int(_career_total_months(c) > 0 and abs(yoe - career_years) > 2.0),

        # ------------------------------------------------------------------
        # JD-specific positives
        # ------------------------------------------------------------------
        "has_ai_career_evidence": int(ai_hits_career >= 3),
        "has_shipped_ranking_search_recsys": int(any(
            k in career.lower() for k in ("ranking", "search", "recommend", "retrieval", "embed")
        )),
        "location_is_noida_or_pune": int(is_preferred_location(p.location)),
        "location_tier1_india": int(is_tier1_india(p.country, p.location)),
        "location_is_india": int(is_india(p.country)),
        "willing_to_relocate": int(s.willing_to_relocate),
        "preferred_work_mode_onsite_or_hybrid": int(s.preferred_work_mode in ("onsite", "hybrid")),
        "notice_period_score": _notice_period_score(s.notice_period_days),
        "notice_period_days": s.notice_period_days,

        # ------------------------------------------------------------------
        # JD-specific negatives
        # ------------------------------------------------------------------
        "is_cv_robotics_only": int(_cv_robotics_only(c)),
        "is_langchain_recent_only": int(_langchain_recent_only(c)),
        "is_closed_source_only": int(_closed_source_only(c)),
        "is_title_chaser": int(_title_chaser(c)),
        "is_consulting_chain": int(consulting_share >= 0.7 and product_count == 0),

        # ------------------------------------------------------------------
        # Behavioral availability
        # ------------------------------------------------------------------
        "open_to_work": int(s.open_to_work_flag),
        "recency_score": _recency_score(s.last_active_date, today),
        "recruiter_response_rate": s.recruiter_response_rate,
        "interview_completion_rate": s.interview_completion_rate,
        "offer_acceptance_rate": s.offer_acceptance_rate if s.offer_acceptance_rate >= 0 else 0.0,
        "has_offer_history": int(s.offer_acceptance_rate >= 0),
        "profile_completeness": s.profile_completeness_score,
        "verified_email": int(s.verified_email),
        "verified_phone": int(s.verified_phone),
        "linkedin_connected": int(s.linkedin_connected),
        "profile_views_30d": s.profile_views_received_30d,
        "search_appearance_30d": s.search_appearance_30d,
        "saved_by_recruiters_30d": s.saved_by_recruiters_30d,
        "github_activity_score": max(0.0, s.github_activity_score),

        # ------------------------------------------------------------------
        # Feature zoo v2 (Agent 5): career-shape + JD-literal + behavioral.
        # ------------------------------------------------------------------
        # Career shape
        "n_ai_roles": _n_ai_roles(c),
        "n_senior_roles": _n_senior_roles(c),
        "n_product_roles": _n_product_roles(c),
        "n_india_roles": _n_india_roles(c),
        "avg_role_duration_months": _avg_role_duration_months(c),
        "max_role_duration_months": _max_role_duration_months(c),
        "tenure_variance": _tenure_variance(c),
        "career_progression_slope": _career_progression_slope(c),
        "current_role_tenure_months": _current_role_tenure_months(c),
        "n_career_gaps": _n_career_gaps(c),
        # JD-literal match
        "jd_skill_match_count": _jd_skill_match_count(c, career, skills_text),
        "jd_skill_match_expert_count": _jd_skill_match_expert_count(c),
        "jd_keyword_count_career": _jd_keyword_count_career(career),
        "jd_keyword_count_summary": _jd_keyword_count_summary(c),
        "title_jd_match": _title_jd_match(c),
        "seniority_jd_match": _seniority_jd_match(c),
        "location_jd_match": _location_jd_match(c),
        "industry_jd_match": _industry_jd_match(c),
        "has_product_company_recent": _has_product_company_recent(c),
        "career_jd_sim_proxy": _career_jd_sim_proxy(c),
        # Behavioral composites
        "log_profile_views_30d": _log_profile_views_30d(c),
        "log_search_appearance_30d": _log_search_appearance_30d(c),
        "log_saved_by_recruiters_30d": _log_saved_by_recruiters_30d(c),
        "log_connection_count": _log_connection_count(c),
        "log_endorsements_received": _log_endorsements_received(c),
        "engagement_intensity": _engagement_intensity(c),
        "behavioral_risk_score": _behavioral_risk_score(c),
        "availability_composite": _availability_composite(c),
        # Backwards-compat columns the existing ranker reads directly
        # from the feature store (avoids re-loading candidates at rank
        # time for the ensemble block).
        "behavioral_availability": _behavioral_availability(c),
        "behavioral_positive": _behavioral_positive(c),
        "behavioral_negative": _behavioral_negative(c),
        "behavioral_honeypot": _behavioral_honeypot(c),
        # Skill mix
        "expert_share": _expert_share(c),
        "advanced_or_expert_share": _advanced_or_expert_share(c),
        "skill_endorsement_mean": _skill_endorsement_mean(c),
        "skill_endorsement_max": _skill_endorsement_max(c),
        "skill_count_log": _skill_count_log(c),
        "ai_skill_count_log": _ai_skill_count_log(c),
        # Title / seniority alignment
        "title_yoe_in_band": _title_yoe_in_band(c),
        "seniority_distance_from_ideal": _seniority_distance_from_ideal(c),
    }

    return features


def features_to_dataframe(rows: list[dict[str, Any]]):
    """Materialize a list of feature dicts into a pandas DataFrame."""
    import pandas as pd

    df = pd.DataFrame(rows)
    if "seniority_bucket" in df.columns:
        df["seniority_bucket"] = df["seniority_bucket"].astype("category")
    if "current_industry_normalized" in df.columns:
        df["current_industry_normalized"] = df["current_industry_normalized"].astype("category")
    return df


def feature_columns() -> list[str]:
    """All numeric feature columns the LTR trainer sees. Excludes IDs and labels."""
    return [
        "yoe_reported", "yoe_career_sum", "yoe_diff",
        "n_career_roles", "n_current_roles", "avg_tenure_months",
        "n_skills", "n_projects", "n_certifications", "n_education",
        "education_tier_1", "education_tier_2", "education_prestige",
        "seniority_at_least_senior", "seniority_at_least_staff", "yoe_in_5_9_band",
        "ai_keyword_hits_career", "ai_keyword_hits_skills", "ai_career_share",
        "ai_skill_count", "n_ai_skill_advanced",
        "has_retrieval_ranking_evidence", "has_llm_finetune_evidence",
        "has_shipped_to_users", "has_distributed_systems_evidence", "has_open_source_evidence",
        # WS-5 new features
        "title_yoe_consistency", "title_yoe_inconsistent", "career_progression", "n_distinct_industries",
        "n_named_jd_skills", "n_named_jd_skills_continuous", "distributed_systems_count", "open_source_count",
        "consulting_to_product_transition", "endorsement_entropy",
        # WS-10: BGE cosine similarity between candidate's deep_profile and the JD.
        # Computed once at build time; re-loaded with the feature store.
        "career_jd_semantic_sim",
        "consulting_share", "consulting_only", "product_company_count",
        "current_industry_ai_ml", "current_company_is_consulting",
        "skill_expert_zero_months", "skill_zero_months_total", "all_skills_zero_endorsements",
        "perfect_skill_list_with_non_tech_title", "expert_skill_count", "advanced_skill_count",
        "yoe_vs_career_sum_anomaly",
        "has_ai_career_evidence", "has_shipped_ranking_search_recsys",
        "location_is_noida_or_pune", "location_tier1_india", "location_is_india",
        "willing_to_relocate", "preferred_work_mode_onsite_or_hybrid",
        "notice_period_score", "notice_period_days",
        "is_cv_robotics_only", "is_langchain_recent_only", "is_closed_source_only",
        "is_title_chaser", "is_consulting_chain",
        "open_to_work", "recency_score",
        "recruiter_response_rate", "interview_completion_rate", "offer_acceptance_rate",
        "has_offer_history", "profile_completeness",
        "verified_email", "verified_phone", "linkedin_connected",
        "profile_views_30d", "search_appearance_30d", "saved_by_recruiters_30d",
        "github_activity_score",

        # ------------------------------------------------------------------
        # Feature zoo v2 (Agent 5): career-shape, JD-literal, behavioral.
        # ------------------------------------------------------------------
        # Career shape (10 features)
        "n_ai_roles", "n_senior_roles", "n_product_roles", "n_india_roles",
        "avg_role_duration_months", "max_role_duration_months",
        "tenure_variance", "career_progression_slope",
        "current_role_tenure_months", "n_career_gaps",
        # JD-literal (10 features)
        "jd_skill_match_count", "jd_skill_match_expert_count",
        "jd_keyword_count_career", "jd_keyword_count_summary",
        "title_jd_match", "seniority_jd_match", "location_jd_match",
        "industry_jd_match", "has_product_company_recent",
        "career_jd_sim_proxy",
        # Behavioral (8 features)
        "log_profile_views_30d", "log_search_appearance_30d",
        "log_saved_by_recruiters_30d", "log_connection_count",
        "log_endorsements_received", "engagement_intensity",
        "behavioral_risk_score", "availability_composite",
        # Skill mix (5 features)
        "expert_share", "advanced_or_expert_share",
        "skill_endorsement_mean", "skill_endorsement_max", "skill_count_log",
        "ai_skill_count_log",
        # Title / seniority alignment (2 features)
        "title_yoe_in_band", "seniority_distance_from_ideal",
        # Backwards-compat columns the existing ranker reads from the
        # feature store at rank time (avoids re-streaming candidates).
        "behavioral_availability", "behavioral_positive",
        "behavioral_negative", "behavioral_honeypot",
    ]


def categorical_columns() -> list[str]:
    return ["seniority_bucket", "current_industry_normalized"]


def safe_log1p(x: float) -> float:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return 0.0
    return math.log1p(max(0.0, x))


def evidence_snippet(c: Candidate, min_words: int = 12, max_words: int = 18) -> str:
    """Return a 12-18 word verbatim snippet from the candidate's current role.

    The snippet is what the ranker splices into the template reasoner to
    satisfy the "Specific facts" check in `submission_spec.md:77-79`. If no
    suitable snippet exists, returns an empty string and the reasoner
    falls back to a generic statement.
    """
    role = _current_role(c)
    desc = (role.description or "").strip() if role else ""
    if not desc:
        return ""
    words = desc.split()
    if len(words) <= max_words:
        return " ".join(words)
    # Try to keep a sentence boundary near the middle.
    target = min_words + (max_words - min_words) // 2
    snippet = " ".join(words[:max_words])
    # Re-trim to last full stop if possible.
    if "." in snippet:
        cut = snippet.rfind(".")
        if cut >= min_words:
            snippet = snippet[: cut + 1]
    return snippet.strip()


def pick_named_jd_skill(c: Candidate) -> str:
    """Return the first JD-named skill (from `JD_NAMED_SKILLS`) that actually
    appears in the candidate's career or skills text. Used by the template
    reasoner to satisfy the "JD connection" check. Returns "" if none.
    """
    from src.preprocessing.deep_profile import build_career_text, build_skills_text

    blob = ((build_career_text(c) or "") + " " + (build_skills_text(c) or "")).lower()
    for needle in JD_NAMED_SKILLS:
        if needle in blob:
            return needle
    return ""
