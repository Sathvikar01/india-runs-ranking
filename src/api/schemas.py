"""Pydantic schemas for the candidate pool, JD, and ranking outputs.

These are the typed contracts every other module talks through. Keeping them
here, instead of scattered through the codebase, is what makes the data flow
auditable.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Profile(BaseModel):
    """A candidate's static profile block."""

    model_config = ConfigDict(extra="allow")

    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float = Field(ge=0, le=50)
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str


class CareerRole(BaseModel):
    """One role in the candidate's career history."""

    model_config = ConfigDict(extra="allow")

    company: str
    title: str
    start_date: str
    end_date: str | None = None
    duration_months: int = Field(ge=0)
    is_current: bool = False
    industry: str = ""
    company_size: str = ""
    description: str = ""


class Education(BaseModel):
    """One degree in the candidate's education block."""

    model_config = ConfigDict(extra="allow")

    institution: str
    degree: str = ""
    field_of_study: str = ""
    start_year: int | None = None
    end_year: int | None = None
    grade: str | None = None
    tier: str | None = None


class Skill(BaseModel):
    """One skill entry."""

    model_config = ConfigDict(extra="allow")

    name: str
    proficiency: str  # beginner | intermediate | advanced | expert
    endorsements: int = 0
    duration_months: int = 0


class Certification(BaseModel):
    """A single certification entry."""

    model_config = ConfigDict(extra="allow")

    name: str
    issuer: str = ""
    date: str | None = None
    url: str | None = None


class Project(BaseModel):
    """A project entry."""

    model_config = ConfigDict(extra="allow")

    name: str
    description: str = ""
    technologies: list[str] = Field(default_factory=list)
    url: str | None = None


class Language(BaseModel):
    """A language entry."""

    model_config = ConfigDict(extra="allow")

    name: str = ""
    language: str = ""
    proficiency: str = ""

    @model_validator(mode="after")
    def _coerce_name(self) -> "Language":
        if not self.name and self.language:
            self.name = self.language
        return self


class RedrobSignals(BaseModel):
    """The 23 behavioral signals attached to every candidate."""

    model_config = ConfigDict(extra="allow")

    profile_completeness_score: float
    signup_date: str
    last_active_date: str
    open_to_work_flag: bool
    profile_views_received_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: float
    avg_response_time_hours: float
    skill_assessment_scores: dict[str, float] = Field(default_factory=dict)
    connection_count: int
    endorsements_received: int
    notice_period_days: int
    expected_salary_range_inr_lpa: dict[str, float] = Field(default_factory=dict)
    preferred_work_mode: str
    willing_to_relocate: bool
    github_activity_score: float
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    offer_acceptance_rate: float
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool

    @field_validator("recruiter_response_rate", "interview_completion_rate")
    @classmethod
    def _clip_unit(cls, v: float) -> float:
        if v < -1.0 or v > 1.0:
            return max(-1.0, min(1.0, v))
        return v


class Candidate(BaseModel):
    """A complete candidate record as loaded from candidates.jsonl."""

    model_config = ConfigDict(extra="allow")

    candidate_id: str
    profile: Profile
    career_history: list[CareerRole] = Field(min_length=1, max_length=10)
    education: list[Education] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    languages: list[Language] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    redrob_signals: RedrobSignals


class JobDescription(BaseModel):
    """The job description we're ranking against."""

    model_config = ConfigDict(extra="allow")

    raw_text: str
    title: str = ""
    company: str = ""
    location: str = ""

    def query_text(self) -> str:
        """Build a retrieval query string from the JD."""
        return self.raw_text.strip()


class RankedCandidate(BaseModel):
    """One row of the final submission CSV."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    rank: int = Field(ge=1, le=100)
    score: float
    reasoning: str

    @field_validator("score")
    @classmethod
    def _score_in_range(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError(f"score {v} out of [0, 1]")
        return float(v)


class RankingResult(BaseModel):
    """The full output of the ranker for a single JD run."""

    model_config = ConfigDict(extra="forbid")

    rows: list[RankedCandidate]
    meta: dict[str, Any] = Field(default_factory=dict)
