"""Build the `deep_profile` text per candidate.

The JD is explicit: "the right answer is not find candidates whose skills
section contains the most AI keywords." So we build a per-candidate text that
weights **career evidence** much more than the skill list. This is what both the
BM25 index and the dense embedding index operate on.
"""

from __future__ import annotations

from src.api.schemas import Candidate
from src.preprocessing.normalize import clean_text, join_pieces, normalize_skill


def build_deep_profile(c: Candidate) -> str:
    """Concatenate the candidate's career evidence into a single text.

    The order is deliberate:
        1. headline + summary (cheap signal)
        2. career roles in reverse chronological order, with title, company, and
           the role description (the strongest signal)
        3. project names + descriptions
        4. certifications
        5. skills, canonicalized and capped at 30, with proficiency as a prefix
    """
    parts: list[str] = []

    p = c.profile
    parts.append(join_pieces(p.headline, "|", p.summary))

    for role in c.career_history:
        head = f"{role.title} at {role.company}"
        if role.industry:
            head += f" [{role.industry}]"
        if role.duration_months:
            head += f" ({role.duration_months} months)"
        parts.append(join_pieces(head, role.description))

    for proj in c.projects:
        parts.append(join_pieces(f"Project: {proj.name}", proj.description))

    for cert in c.certifications:
        parts.append(join_pieces(f"Certification: {cert.name}", cert.issuer))

    # Skills: prefix proficiency to let the embedding model use it, but cap to
    # 30 entries so the dense retrieval is not dominated by the skills list.
    skills_sorted = sorted(
        c.skills,
        key=lambda s: (
            {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}.get(s.proficiency, 0),
            s.duration_months,
        ),
        reverse=True,
    )
    skills_text = " ".join(
        f"{s.proficiency}:{normalize_skill(s.name)}({s.duration_months}mo)"
        for s in skills_sorted[:30]
    )
    parts.append(f"Skills: {skills_text}")

    # Behavioral highlights as short appendices. Useful for "is this candidate
    # active" retrieval signals.
    s = c.redrob_signals
    parts.append(
        "Behavioral: "
        f"open_to_work={s.open_to_work_flag} "
        f"recruiter_response_rate={s.recruiter_response_rate:.2f} "
        f"notice_period_days={s.notice_period_days} "
        f"preferred_work_mode={s.preferred_work_mode} "
        f"willing_to_relocate={s.willing_to_relocate} "
        f"github_activity_score={s.github_activity_score} "
        f"last_active={s.last_active_date} "
        f"interview_completion_rate={s.interview_completion_rate:.2f}"
    )

    return clean_text(" \n ".join(p for p in parts if p))


def build_skills_text(c: Candidate) -> str:
    """The traditional "skills list" path. Kept as a separate retrieval corpus
    so we can ablate "skills-only" vs "deep profile" in evaluation.
    """
    skills = sorted(
        c.skills,
        key=lambda s: (
            {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}.get(s.proficiency, 0),
            s.duration_months,
        ),
        reverse=True,
    )
    return " ".join(
        f"{normalize_skill(s.name)}:{s.proficiency}" for s in skills
    )


def build_career_text(c: Candidate) -> str:
    """Career-history-only text. The strongest single signal for the JD."""
    parts: list[str] = []
    for role in c.career_history:
        parts.append(join_pieces(f"{role.title} at {role.company}", role.description))
    return clean_text(" \n ".join(parts))


def build_signals_text(c: Candidate) -> str:
    """The 23 behavioral signals, rendered as text for completeness."""
    s = c.redrob_signals
    return (
        f"profile_completeness={s.profile_completeness_score} "
        f"open_to_work={s.open_to_work_flag} "
        f"recruiter_response_rate={s.recruiter_response_rate} "
        f"avg_response_time_hours={s.avg_response_time_hours} "
        f"notice_period_days={s.notice_period_days} "
        f"preferred_work_mode={s.preferred_work_mode} "
        f"willing_to_relocate={s.willing_to_relocate} "
        f"github_activity_score={s.github_activity_score} "
        f"interview_completion_rate={s.interview_completion_rate} "
        f"offer_acceptance_rate={s.offer_acceptance_rate} "
        f"verified_email={s.verified_email} "
        f"verified_phone={s.verified_phone} "
        f"linkedin_connected={s.linkedin_connected}"
    )
