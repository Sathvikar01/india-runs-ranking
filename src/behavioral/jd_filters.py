"""JD-specific negative and positive filters.

The JD is unusually opinionated about what it doesn't want. We encode each
named "do not want" as a hard flag and each "what we want" as a positive
boost.
"""

from __future__ import annotations

from __future__ import annotations

import functools

from src.api.schemas import Candidate
from src.preprocessing.deep_profile import build_career_text
from src.preprocessing.feature_engineer import _avg_tenure_months
from src.preprocessing.normalize import is_consulting_company


@functools.lru_cache(maxsize=1)
def _load_weights() -> dict[str, float]:
    import yaml

    with open("configs/behavior.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["behavioral"]["jd_filters"]
    return {item["rule"]: float(item.get("weight", 0.0)) for item in cfg.get("reject_if", [])}


def is_only_consulting_companies(c: Candidate) -> bool:
    if not c.career_history:
        return False
    return all(is_consulting_company(r.company) for r in c.career_history)


def is_cv_robotics_speech(c: Candidate) -> bool:
    career = (build_career_text(c) or "").lower()
    cv = any(k in career for k in ("computer vision", "image classification", "object detection", "segmentation", "robotics", "opencv", "ros", "autonomous"))
    speech = "speech recognition" in career or "asr" in career or "tts" in career
    nlp = any(k in career for k in ("nlp", "natural language", "language model", "llm", "rag", "retrieval", "search", "ranker", "rank"))
    return (cv or speech) and not nlp


def is_title_chaser(c: Candidate) -> bool:
    return _avg_tenure_months(c) < 18 and len(c.career_history) >= 3


def is_closed_source_only(c: Candidate) -> bool:
    s = c.redrob_signals
    if s.github_activity_score > 0:
        return False
    career = (build_career_text(c) or "").lower()
    return not any(
        k in career
        for k in ("open source", "open-source", "github.com", "arxiv", "publication", "paper", "released")
    )


def is_langchain_recent_only(c: Candidate) -> bool:
    career = (build_career_text(c) or "").lower()
    has_langchain = "langchain" in career
    has_deep = any(k in career for k in ("pytorch", "tensorflow", "transformer", "embedding", "retrieval", "ranker", "xgboost", "lightgbm"))
    return has_langchain and not has_deep


def has_nlp_ir_in_career(c: Candidate) -> bool:
    career = (build_career_text(c) or "").lower()
    return any(
        k in career
        for k in ("nlp", "natural language", "language model", "llm", "rag", "retrieval", "search", "ranker", "rank", "embedding")
    )


def yoe_out_of_band(c: Candidate) -> bool:
    yoe = float(c.profile.years_of_experience or 0.0)
    return yoe < 3 or yoe > 15


NEGATIVE_RULES = {
    "only_consulting_companies": is_only_consulting_companies,
    "only_cv_robotics_speech": is_cv_robotics_speech,
    "title_chaser_avg_tenure_lt_18mo": is_title_chaser,
    "closed_source_only": is_closed_source_only,
    "langchain_recent_only": is_langchain_recent_only,
    "no_nlp_ir_in_career": lambda c: not has_nlp_ir_in_career(c),
    "yoe_lt_3_or_gt_15": yoe_out_of_band,
}


def negative_flags(c: Candidate) -> dict[str, bool]:
    return {name: fn(c) for name, fn in NEGATIVE_RULES.items()}


def negative_penalty(c: Candidate) -> float:
    """Return a 0-1 score. Higher = more penalty."""
    flags = negative_flags(c)
    weights = _load_weights()
    score = sum(weights.get(name, 0.0) for name, hit in flags.items() if hit)
    return float(max(0.0, min(1.0, score)))


def _load_weights() -> dict[str, float]:
    import yaml

    with open("configs/behavior.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["behavioral"]["jd_filters"]
    return {item["rule"]: float(item.get("weight", 0.0)) for item in cfg.get("reject_if", [])}


# ---------------------------------------------------------------------------
# Positive boosters
# ---------------------------------------------------------------------------


def has_ai_career_evidence(c: Candidate) -> bool:
    career = (build_career_text(c) or "").lower()
    ai_keys = (
        "machine learning", "deep learning", "neural network", "nlp",
        "natural language", "language model", "llm", "llama", "gpt",
        "embedding", "embeddings", "vector search", "vector database",
        "retrieval", "ranking", "ranker", "search engine", "rerank",
        "recommendation", "recommender", "rag", "fine-tun", "lora", "qlora",
        "peft", "rlhf", "transformer", "pytorch", "tensorflow",
    )
    return sum(1 for k in ai_keys if k in career) >= 3


def shipped_ranking_or_search_at_scale(c: Candidate) -> bool:
    career = (build_career_text(c) or "").lower()
    has_ranking = any(k in career for k in ("ranking", "ranker", "search", "recommend", "retrieval", "rerank", "vector search", "embedding"))
    has_scale = any(k in career for k in ("million", "scale", "production", "users", "shipped", "launched", "deployed", "k queries", "k requests"))
    return has_ranking and has_scale


def tier_1_or_2_education(c: Candidate) -> bool:
    return any((e.tier or "").lower() in ("tier_1", "tier_2") for e in c.education)


def location_noida_pune_or_relocate(c: Candidate) -> bool:
    p = c.profile
    if "noida" in (p.location or "").lower() or "pune" in (p.location or "").lower():
        return True
    return c.redrob_signals.willing_to_relocate


def github_open_source_evidence(c: Candidate) -> bool:
    s = c.redrob_signals
    if s.github_activity_score >= 50 and s.github_activity_score > 0:
        return True
    career = (build_career_text(c) or "").lower()
    return any(k in career for k in ("open source", "open-source", "github.com", "arxiv", "paper", "publication"))


def hybrid_or_onsite_preferred(c: Candidate) -> bool:
    return c.redrob_signals.preferred_work_mode in ("onsite", "hybrid")


def notice_period_lt_30d(c: Candidate) -> bool:
    return 0 <= c.redrob_signals.notice_period_days <= 30


def profile_active_30d(c: Candidate, today=None) -> bool:
    from datetime import date, datetime

    if today is None:
        today = date.today()
    try:
        d = datetime.strptime(c.redrob_signals.last_active_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return False
    return (today - d).days <= 30


POSITIVE_RULES = {
    "has_ai_career_evidence": has_ai_career_evidence,
    "shipped_ranking_or_search_at_scale": shipped_ranking_or_search_at_scale,
    "tier_1_or_2_education": tier_1_or_2_education,
    "location_noida_pune_or_relocate": location_noida_pune_or_relocate,
    "github_open_source_evidence": github_open_source_evidence,
    "hybrid_or_onsite_preferred": hybrid_or_onsite_preferred,
    "notice_period_lt_30d": notice_period_lt_30d,
    "profile_active_30d": profile_active_30d,
}


def positive_flags(c: Candidate) -> dict[str, bool]:
    return {name: fn(c) for name, fn in POSITIVE_RULES.items()}


def positive_boost(c: Candidate) -> float:
    flags = positive_flags(c)
    weights = _load_positive_weights()
    score = sum(weights.get(name, 0.0) for name, hit in flags.items() if hit)
    return float(max(0.0, min(1.0, score)))


def _load_positive_weights() -> dict[str, float]:
    import yaml

    with open("configs/behavior.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["behavioral"].get("preference", {})
    return {item["rule"]: float(item.get("weight", 0.0)) for item in cfg.get("prefer_if", [])}


# Make the function itself cached too
_load_positive_weights = functools.lru_cache(maxsize=1)(_load_positive_weights)


# ---------------------------------------------------------------------------
# Vectorized batch scoring
# ---------------------------------------------------------------------------


def positive_flags_df(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Vectorized positive rules. The DataFrame is the feature_store.

    The rules are the same as the per-candidate POSITIVE_RULES dict, but
    expressed as DataFrame operations. Where the rule needs the full career
    text (e.g. has_ai_career_evidence), we fall back to the career-derived
    features already in the store.
    """
    return {
        "has_ai_career_evidence": (df["ai_keyword_hits_career"] >= 3).astype(float),
        "shipped_ranking_or_search_at_scale": (
            (df["has_retrieval_ranking_evidence"] == 1) & (df["has_shipped_to_users"] == 1)
        ).astype(float),
        "tier_1_or_2_education": ((df["education_tier_1"] + df["education_tier_2"]) > 0).astype(float),
        "location_noida_pune_or_relocate": (
            (df["location_is_noida_or_pune"] == 1)
            | ((df["location_is_india"] == 1) & (df["willing_to_relocate"] == 1))
        ).astype(float),
        "github_open_source_evidence": (df["github_activity_score"] >= 50).astype(float),
        "hybrid_or_onsite_preferred": df["preferred_work_mode_onsite_or_hybrid"].astype(float),
        "notice_period_lt_30d": (df["notice_period_days"] <= 30).astype(float),
        "profile_active_30d": (df["recency_score"] >= 0.9).astype(float),
    }


def positive_boost_df(df: pd.DataFrame) -> pd.Series:
    flags = positive_flags_df(df)
    weights = _load_positive_weights()
    return sum(flags[k] * float(weights.get(k, 0.0)) for k in flags).clip(0, 1)


def negative_flags_df(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "only_consulting_companies": df["consulting_only"].astype(float),
        "only_cv_robotics_speech": df["is_cv_robotics_only"].astype(float),
        "title_chaser_avg_tenure_lt_18mo": (df["is_title_chaser"]).astype(float),
        "closed_source_only": df["is_closed_source_only"].astype(float),
        "langchain_recent_only": df["is_langchain_recent_only"].astype(float),
        "no_nlp_ir_in_career": (df["ai_keyword_hits_career"] == 0).astype(float),
        "yoe_lt_3_or_gt_15": ((df["yoe_reported"] < 3) | (df["yoe_reported"] > 15)).astype(float),
    }


def negative_penalty_df(df: pd.DataFrame) -> pd.Series:
    flags = negative_flags_df(df)
    weights = _load_weights()
    return sum(flags[k] * float(weights.get(k, 0.0)) for k in flags).clip(0, 1)


# ---------------------------------------------------------------------------
# Positive boosters
# ---------------------------------------------------------------------------
