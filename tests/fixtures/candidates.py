"""A small hand-built candidate used in unit tests. Designed to exercise
multiple paths: deep AI career, honeypot shape, consulting chain, etc.
"""

from __future__ import annotations

from src.api.schemas import (
    Candidate,
    CareerRole,
    Certification,
    Education,
    Language,
    Profile,
    Project,
    RedrobSignals,
    Skill,
)


def make_ai_candidate() -> Candidate:
    return Candidate(
        candidate_id="CAND_0000001",
        profile=Profile(
            anonymized_name="AI Candidate",
            headline="Senior ML Engineer | RAG, Search, Ranking",
            summary="Senior ML engineer with 7 years of applied AI work at product companies.",
            location="Pune, Maharashtra",
            country="India",
            years_of_experience=7.0,
            current_title="Senior ML Engineer",
            current_company="Acme AI",
            current_company_size="201-500",
            current_industry="AI/ML",
        ),
        career_history=[
            CareerRole(
                company="Acme AI",
                title="Senior ML Engineer",
                start_date="2022-01-01",
                end_date=None,
                duration_months=42,
                is_current=True,
                industry="AI/ML",
                company_size="201-500",
                description="Built and shipped a hybrid retrieval + cross-encoder ranking system serving 5M queries/day.",
            ),
            CareerRole(
                company="Beta Co",
                title="ML Engineer",
                start_date="2018-06-01",
                end_date="2021-12-31",
                duration_months=42,
                is_current=False,
                industry="SaaS",
                company_size="51-200",
                description="Implemented RAG with Pinecone, fine-tuned Llama with LoRA.",
            ),
        ],
        education=[
            Education(institution="IIT Bombay", degree="B.Tech", field_of_study="CS", start_year=2013, end_year=2017, tier="tier_1"),
        ],
        skills=[
            Skill(name="PyTorch", proficiency="advanced", endorsements=20, duration_months=72),
            Skill(name="Transformers", proficiency="advanced", endorsements=18, duration_months=60),
            Skill(name="Pinecone", proficiency="intermediate", endorsements=8, duration_months=24),
            Skill(name="LoRA", proficiency="intermediate", endorsements=5, duration_months=18),
            Skill(name="Python", proficiency="advanced", endorsements=40, duration_months=120),
            Skill(name="Elasticsearch", proficiency="intermediate", endorsements=4, duration_months=36),
        ],
        certifications=[
            Certification(name="Deep Learning Specialization", issuer="Coursera"),
        ],
        languages=[Language(name="English", proficiency="native")],
        projects=[
            Project(name="Hybrid RAG", description="Vector + BM25 with cross-encoder rerank for product search."),
        ],
        redrob_signals=RedrobSignals(
            profile_completeness_score=92.0,
            signup_date="2024-08-12",
            last_active_date="2026-06-12",
            open_to_work_flag=True,
            profile_views_received_30d=58,
            applications_submitted_30d=1,
            recruiter_response_rate=0.85,
            avg_response_time_hours=4.0,
            connection_count=400,
            endorsements_received=80,
            notice_period_days=30,
            expected_salary_range_inr_lpa={"min": 55.0, "max": 75.0},
            preferred_work_mode="hybrid",
            willing_to_relocate=True,
            github_activity_score=78.0,
            search_appearance_30d=200,
            saved_by_recruiters_30d=12,
            interview_completion_rate=0.9,
            offer_acceptance_rate=0.7,
            verified_email=True,
            verified_phone=True,
            linkedin_connected=True,
        ),
    )


def make_honeypot_candidate() -> Candidate:
    """8 years of experience + expert in 10 AI skills with 0 months each + non-tech title."""
    skills = [
        Skill(name=k, proficiency="expert", endorsements=0, duration_months=0)
        for k in (
            "PyTorch", "TensorFlow", "Transformers", "HuggingFace", "LangChain",
            "Pinecone", "LoRA", "QLoRA", "XGBoost", "RAG",
        )
    ]
    return Candidate(
        candidate_id="CAND_9999999",
        profile=Profile(
            anonymized_name="Honeypot",
            headline="Marketing Manager | AI Enthusiast",
            summary="Marketing manager with all the AI keywords.",
            location="Mumbai",
            country="India",
            years_of_experience=8.0,
            current_title="Marketing Manager",
            current_company="Acme Marketing",
            current_company_size="51-200",
            current_industry="Marketing",
        ),
        career_history=[
            CareerRole(
                company="Acme Marketing",
                title="Marketing Manager",
                start_date="2022-01-01",
                end_date=None,
                duration_months=42,
                is_current=True,
                industry="Marketing",
                company_size="51-200",
                description="Ran email campaigns and SEO.",
            ),
        ],
        education=[],
        skills=skills,
        certifications=[],
        languages=[],
        projects=[],
        redrob_signals=RedrobSignals(
            profile_completeness_score=99.0,
            signup_date="2025-12-01",
            last_active_date="2026-06-15",
            open_to_work_flag=True,
            profile_views_received_30d=2,
            applications_submitted_30d=80,
            recruiter_response_rate=0.05,
            avg_response_time_hours=300.0,
            connection_count=10,
            endorsements_received=0,
            notice_period_days=120,
            expected_salary_range_inr_lpa={"min": 20.0, "max": 25.0},
            preferred_work_mode="remote",
            willing_to_relocate=False,
            github_activity_score=0.0,
            search_appearance_30d=0,
            saved_by_recruiters_30d=0,
            interview_completion_rate=0.1,
            offer_acceptance_rate=-1.0,
            verified_email=True,
            verified_phone=False,
            linkedin_connected=False,
        ),
    )


def make_consulting_chain_candidate() -> Candidate:
    return Candidate(
        candidate_id="CAND_0000002",
        profile=Profile(
            anonymized_name="TCS Dev",
            headline="Senior Developer @ TCS",
            summary="Senior developer at TCS.",
            location="Bangalore",
            country="India",
            years_of_experience=8.0,
            current_title="Senior Developer",
            current_company="Tata Consultancy Services",
            current_company_size="10001+",
            current_industry="IT Services",
        ),
        career_history=[
            CareerRole(
                company="Tata Consultancy Services",
                title="Senior Developer",
                start_date="2020-01-01",
                end_date=None,
                duration_months=72,
                is_current=True,
                industry="IT Services",
                company_size="10001+",
                description="Worked on internal Java apps.",
            ),
            CareerRole(
                company="Infosys",
                title="Developer",
                start_date="2017-01-01",
                end_date="2019-12-31",
                duration_months=36,
                is_current=False,
                industry="IT Services",
                company_size="10001+",
                description="Worked on Java apps.",
            ),
        ],
        education=[Education(institution="Local College", degree="B.Tech", field_of_study="CS", tier="tier_3")],
        skills=[
            Skill(name="Java", proficiency="advanced", endorsements=10, duration_months=72),
            Skill(name="SQL", proficiency="advanced", endorsements=8, duration_months=72),
        ],
        certifications=[],
        languages=[],
        projects=[],
        redrob_signals=RedrobSignals(
            profile_completeness_score=70.0,
            signup_date="2024-01-01",
            last_active_date="2026-06-10",
            open_to_work_flag=False,
            profile_views_received_30d=10,
            applications_submitted_30d=2,
            recruiter_response_rate=0.4,
            avg_response_time_hours=12.0,
            connection_count=80,
            endorsements_received=18,
            notice_period_days=60,
            expected_salary_range_inr_lpa={"min": 15.0, "max": 22.0},
            preferred_work_mode="onsite",
            willing_to_relocate=False,
            github_activity_score=5.0,
            search_appearance_30d=30,
            saved_by_recruiters_30d=0,
            interview_completion_rate=0.7,
            offer_acceptance_rate=0.6,
            verified_email=True,
            verified_phone=True,
            linkedin_connected=False,
        ),
    )
