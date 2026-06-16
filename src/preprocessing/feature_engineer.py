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
        "education_tier_1", "education_tier_2",
        "seniority_at_least_senior", "seniority_at_least_staff", "yoe_in_5_9_band",
        "ai_keyword_hits_career", "ai_keyword_hits_skills", "ai_career_share",
        "ai_skill_count", "n_ai_skill_advanced",
        "has_retrieval_ranking_evidence", "has_llm_finetune_evidence",
        "has_shipped_to_users", "has_distributed_systems_evidence", "has_open_source_evidence",
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
    ]


def categorical_columns() -> list[str]:
    return ["seniority_bucket", "current_industry_normalized"]


def safe_log1p(x: float) -> float:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return 0.0
    return math.log1p(max(0.0, x))
