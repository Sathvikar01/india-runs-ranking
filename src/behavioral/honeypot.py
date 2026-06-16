"""Honeypot and trap detection.

Combines several rule-based heuristics into a single 0-1 risk score, where 1.0
means "this is almost certainly a honeypot / keyword stuffer and should not be
in the top 100".

The rules are calibrated on the JD's explicit warnings:
  * "8 years of experience at a company founded 3 years ago"
  * "expert proficiency in 10 skills with 0 years used"
  * Marketing Manager with all the AI keywords but no career evidence
"""

from __future__ import annotations

import functools
from datetime import date

import pandas as pd
import yaml

from src.api.schemas import Candidate
from src.preprocessing.deep_profile import build_career_text
from src.preprocessing.feature_engineer import _career_total_months
from src.preprocessing.normalize import (
    is_consulting_company,
    normalize_industry,
    normalize_skill,
    title_seniority_bucket,
)


@functools.lru_cache(maxsize=1)
def _load_weights() -> dict:
    with open("configs/behavior.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["behavioral"]["honeypot"]["rule_weights"]


@functools.lru_cache(maxsize=1)
def _load_threshold() -> float:
    with open("configs/behavior.yaml", "r", encoding="utf-8") as f:
        return float(yaml.safe_load(f)["behavioral"]["honeypot"]["risk_threshold"])


def honeypot_signals(c: Candidate) -> dict[str, float]:
    """Return the individual sub-scores. Each is 0-1."""
    career = (build_career_text(c) or "").lower()
    skills_text = " ".join((s.name or "") for s in c.skills).lower()
    current_title = (c.profile.current_title or "").lower()

    # 1. "expert" with zero months of usage.
    n_expert = sum(1 for s in c.skills if s.proficiency == "expert")
    n_expert_zero = sum(1 for s in c.skills if s.proficiency == "expert" and s.duration_months == 0)
    skill_proficiency_vs_duration = (n_expert_zero / n_expert) if n_expert > 0 else 0.0

    # 2. YOE vs career-history sum: large positive gap is suspicious.
    yoe = float(c.profile.years_of_experience or 0.0)
    career_years = _career_total_months(c) / 12.0
    if yoe > 0 and career_years > 0:
        yoe_diff = abs(yoe - career_years)
        yoe_vs_career_sum = min(1.0, yoe_diff / max(yoe, career_years, 1.0))
    else:
        yoe_vs_career_sum = 0.0

    # 3. Perfect skill list with non-technical current_title.
    jd_core = {
        "machine learning", "nlp", "llm", "embedding", "vector search",
        "retrieval augmented generation", "pytorch", "tensorflow", "transformers",
        "ranker", "lora", "qlora", "peft", "xgboost", "lightgbm",
    }
    skill_names = {normalize_skill(s.name) for s in c.skills}
    overlap = skill_names & jd_core
    is_tech_title = any(
        t in current_title for t in (
            "engineer", "scientist", "developer", "analyst", "architect",
            "researcher", "ml", "ai", "data",
        )
    )
    perfect_skill_list_with_non_tech_title = (
        len(overlap) >= 6 and not is_tech_title and n_expert >= 4
    )

    # 4. Multiple is_current positions.
    multiple_current_positions = sum(1 for r in c.career_history if r.is_current) > 1

    # 5. Expert in too many skills.
    expert_in_too_many_skills = 1.0 if n_expert >= 8 else (n_expert / 8.0 if n_expert > 0 else 0.0)

    # 6. All skills with zero endorsements.
    all_skills_zero_endorsements = (
        bool(c.skills) and all(s.endorsements == 0 for s in c.skills) and n_expert >= 4
    )

    # 7. High skill count but no career evidence of any of those skills.
    if c.skills:
        overlap_with_career = sum(1 for s in c.skills if (s.name or "").lower() in career)
        high_skill_count_no_career_evidence = max(
            0.0, (len(c.skills) - overlap_with_career) / max(1, len(c.skills))
        )
    else:
        high_skill_count_no_career_evidence = 0.0

    # Plus two more derived signals that are *not* honeypot per se but typical
    # of suspicious profiles.
    career_with_no_ai_keywords = ("ranker" not in career and "search" not in career
                                  and "retrieval" not in career and "recommend" not in career
                                  and "embedding" not in career and "nlp" not in career
                                  and "llm" not in career and "language model" not in career)
    if perfect_skill_list_with_non_tech_title and career_with_no_ai_keywords:
        high_skill_count_no_career_evidence = max(high_skill_count_no_career_evidence, 0.9)

    return {
        "skill_proficiency_vs_duration": float(skill_proficiency_vs_duration),
        "yoe_vs_career_sum": float(yoe_vs_career_sum),
        "perfect_skill_list_with_non_tech_title": float(perfect_skill_list_with_non_tech_title),
        "multiple_current_positions": float(multiple_current_positions),
        "expert_in_too_many_skills": float(expert_in_too_many_skills),
        "all_skills_zero_endorsements": float(all_skills_zero_endorsements),
        "high_skill_count_no_career_evidence": float(high_skill_count_no_career_evidence),
    }


def honeypot_risk(c: Candidate) -> float:
    """Weighted sum of honeypot sub-signals. 0-1."""
    weights = _load_weights()
    sub = honeypot_signals(c)
    score = sum(weights.get(k, 0.0) * v for k, v in sub.items())
    return float(max(0.0, min(1.0, score)))


def is_honeypot(c: Candidate) -> bool:
    return honeypot_risk(c) >= _load_threshold()


# ---------------------------------------------------------------------------
# Vectorized batch scoring
# ---------------------------------------------------------------------------


def honeypot_signals_df(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized honeypot sub-signals over a feature DataFrame.

    Required columns: yoe_reported, yoe_career_sum, current_title, expert_skill_count,
    skill_expert_zero_months, n_skills, all_skills_zero_endorsements, n_career_roles.
    """
    # The features DataFrame doesn't carry the skill list or the career text,
    # so we can only compute a subset of the signals here. The full per-candidate
    # `honeypot_signals` is used in the ranker for the top shortlist; this batch
    # version is a fast pre-filter for the 100k pool.
    yoe = df["yoe_reported"].astype(float)
    career_years = df["yoe_career_sum"].astype(float)
    yoe_diff = (yoe - career_years).abs()
    yoe_vs_career = (yoe_diff / yoe.clip(lower=1.0)).clip(0, 1)

    expert = df["expert_skill_count"].astype(float)
    n_skills = df["n_skills"].astype(float)
    expert_too_many = (expert / 8.0).clip(0, 1)

    all_zero_endorse = df["all_skills_zero_endorsements"].astype(float)
    skill_expert_zero_months = df["skill_expert_zero_months"].astype(float)
    skill_prof_vs_dur = (skill_expert_zero_months / expert.clip(lower=1.0)).clip(0, 1)

    return pd.DataFrame({
        "skill_proficiency_vs_duration": skill_prof_vs_dur,
        "yoe_vs_career_sum": yoe_vs_career,
        "perfect_skill_list_with_non_tech_title": df["perfect_skill_list_with_non_tech_title"].astype(float),
        "multiple_current_positions": (df["n_current_roles"] > 1).astype(float),
        "expert_in_too_many_skills": expert_too_many,
        "all_skills_zero_endorsements": all_zero_endorse,
        "high_skill_count_no_career_evidence": 0.0,  # can't compute without career text
    })


def honeypot_risk_df(df: pd.DataFrame) -> pd.Series:
    weights = _load_weights()
    sub = honeypot_signals_df(df)
    score = sum(sub[k] * float(weights.get(k, 0.0)) for k in sub.columns)
    return score.clip(0, 1)
