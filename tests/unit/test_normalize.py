"""Pure-function unit tests for the normalize module."""

from __future__ import annotations

from src.preprocessing.normalize import (
    has_any,
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


def test_title_seniority_bucket_staff():
    assert title_seniority_bucket("Staff Software Engineer") == "staff"
    assert title_seniority_bucket("Principal Engineer") == "staff"


def test_title_seniority_bucket_senior():
    assert title_seniority_bucket("Senior ML Engineer") == "senior"
    assert title_seniority_bucket("Sr. Data Scientist") == "senior"


def test_title_seniority_bucket_junior():
    assert title_seniority_bucket("Junior Developer") == "junior"
    assert title_seniority_bucket("Associate Analyst") == "junior"


def test_title_seniority_bucket_manager():
    assert title_seniority_bucket("Engineering Manager") == "manager"
    assert title_seniority_bucket("Head of Data") == "manager"


def test_title_seniority_bucket_unknown():
    assert title_seniority_bucket("Artist") == "unknown"
    assert title_seniority_bucket("") == "unknown"


def test_seniority_at_least():
    assert seniority_at_least("staff", "senior")
    assert not seniority_at_least("mid", "senior")


def test_normalize_skill_synonyms():
    assert normalize_skill("Machine Learning") == "machine learning"
    assert normalize_skill("ML") == "machine learning"
    assert normalize_skill("LLMs") == "llm"
    assert normalize_skill("NLP") == "nlp"
    assert normalize_skill("Pinecone") == "vector database"


def test_normalize_skill_keeps_unknown():
    assert normalize_skill("Totally Novel Thing") == "totally novel thing"


def test_normalize_industry():
    assert normalize_industry("IT Services") == "it_services"
    assert normalize_industry("Software") == "saas"
    assert normalize_industry("AI/ML") == "ai_ml"
    assert normalize_industry("FinTech") == "fintech"


def test_is_consulting_company():
    assert is_consulting_company("Tata Consultancy Services")
    assert is_consulting_company("Infosys Limited")
    assert not is_consulting_company("Acme AI")


def test_is_preferred_location():
    assert is_preferred_location("Pune, Maharashtra")
    assert is_preferred_location("Noida, Uttar Pradesh")
    assert not is_preferred_location("Mumbai, Maharashtra")


def test_is_tier1_india():
    assert is_tier1_india("India", "Bangalore, Karnataka")
    assert not is_tier1_india("USA", "Bangalore, Karnataka")
    assert not is_tier1_india("India", "Indore, Madhya Pradesh")


def test_is_india():
    assert is_india("India")
    assert is_india("INDIA")
    assert not is_india("USA")


def test_location_tokens():
    assert location_tokens("Pune, Maharashtra") == ["pune", "maharashtra"]
    assert location_tokens("Bengaluru / Karnataka") == ["bengaluru", "karnataka"]


def test_has_any():
    assert has_any("I built a search engine with Elasticsearch", ("search", "ranker"))
    assert not has_any("nothing here", ("search", "ranker"))
