"""Text normalization, cleaning, and skill/role taxonomy.

Pure functions; no IO. Anything that touches a file lives elsewhere.
"""

from __future__ import annotations

import re
import string
from collections.abc import Iterable

# Title seniority buckets used throughout the pipeline.
# Order matters: we check the *more specific* modifiers first (staff, manager,
# senior) so a "Senior Software Engineer" doesn't get caught by the
# "engineer" pattern in the mid bucket.
SENIORITY_TITLE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("intern", re.compile(r"\b(intern|internship|trainee|apprentice)\b", re.IGNORECASE)),
    ("junior", re.compile(r"\b(junior|associate|entry[-\s]?level|graduate|fresher)\b", re.IGNORECASE)),
    ("manager", re.compile(r"\b(manager|head of|director|vp|vice president|chief)\b", re.IGNORECASE)),
    ("staff", re.compile(r"\b(distinguished|fellow|principal|staff)\b", re.IGNORECASE)),
    ("senior", re.compile(r"\b(senior|sr\.?|lead|architect)\b", re.IGNORECASE)),
    ("mid", re.compile(r"\b(sde|software engineer|ml engineer|mle\b|data scientist|analyst|developer|engineer)\b", re.IGNORECASE)),
]


def title_seniority_bucket(title: str) -> str:
    """Map a free-text title to one of intern/junior/mid/senior/staff/manager."""
    if not title:
        return "unknown"
    for bucket, pat in SENIORITY_TITLE_PATTERNS:
        if pat.search(title):
            return bucket
    return "unknown"


# Seniority ordering for "is at least X" queries.
SENIORITY_ORDER = ["intern", "junior", "mid", "senior", "staff", "manager"]


def seniority_at_least(bucket: str, floor: str) -> bool:
    if bucket not in SENIORITY_ORDER or floor not in SENIORITY_ORDER:
        return False
    return SENIORITY_ORDER.index(bucket) >= SENIORITY_ORDER.index(floor)


# Skill canonicalization: collapse known synonyms and skill-list boilerplate.
SKILL_SYNONYMS: dict[str, str] = {
    "ms excel": "excel",
    "ms office": "office",
    "google workspace": "google workspace",
    "machine learning": "machine learning",
    "ml": "machine learning",
    "ai": "ai",
    "artificial intelligence": "ai",
    "llm": "llm",
    "large language models": "llm",
    "llms": "llm",
    "natural language processing": "nlp",
    "nlp": "nlp",
    "information retrieval": "information retrieval",
    "ir": "information retrieval",
    "rag": "retrieval augmented generation",
    "retrieval augmented generation": "retrieval augmented generation",
    "vector search": "vector search",
    "vector database": "vector database",
    "vector store": "vector database",
    "pinecone": "vector database",
    "weaviate": "vector database",
    "milvus": "vector database",
    "faiss": "vector search",
    "elasticsearch": "search engine",
    "opensearch": "search engine",
    "solr": "search engine",
    "ranker": "learning to rank",
    "learning to rank": "learning to rank",
    "ltr": "learning to rank",
    "lambdarank": "learning to rank",
    "xgboost": "gradient boosting",
    "lightgbm": "gradient boosting",
    "catboost": "gradient boosting",
    "pytorch": "pytorch",
    "tensorflow": "tensorflow",
    "huggingface": "huggingface",
    "transformers": "transformers",
    "sentence-transformers": "sentence transformers",
    "lora": "lora",
    "qlora": "qlora",
    "peft": "peft",
    "rlhf": "rlhf",
    "prompt engineering": "prompt engineering",
    "prompt": "prompt engineering",
    "langchain": "langchain",
    "llamaindex": "llamaindex",
    "openai": "openai api",
    "gpt": "openai api",
    "chatgpt": "openai api",
    "claude": "anthropic api",
    "gemini": "google ai api",
    "aws": "aws",
    "azure": "azure",
    "gcp": "gcp",
    "spark": "spark",
    "hadoop": "hadoop",
    "kafka": "kafka",
    "airflow": "airflow",
    "dbt": "dbt",
    "snowflake": "snowflake",
    "databricks": "databricks",
    "python": "python",
    "java": "java",
    "scala": "scala",
    "golang": "go",
    "rust": "rust",
    "c++": "c++",
    "javascript": "javascript",
    "typescript": "typescript",
    "node.js": "nodejs",
    "react": "react",
    "vue": "vue",
    "next.js": "nextjs",
    "fastapi": "fastapi",
    "flask": "flask",
    "django": "django",
    "docker": "docker",
    "kubernetes": "kubernetes",
    "terraform": "terraform",
    "mlflow": "mlflow",
    "wandb": "wandb",
}


def normalize_skill(name: str) -> str:
    """Lower-case + collapse synonyms. Strips punctuation at the edges."""
    if not name:
        return ""
    n = name.strip().lower()
    n = re.sub(rf"[{re.escape(string.punctuation)}]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return SKILL_SYNONYMS.get(n, n)


def normalize_industry(industry: str) -> str:
    """Map free-text industry to a small canonical set."""
    if not industry:
        return "unknown"
    s = industry.strip().lower()
    if any(k in s for k in ("consult", "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini")):
        return "consulting"
    if "it service" in s or "it services" in s:
        return "it_services"
    if "ai" in s or "ml" in s or "artificial" in s:
        return "ai_ml"
    if "fintech" in s or "finance" in s or "banking" in s:
        return "fintech"
    if "ecommerce" in s or "e-commerce" in s or "marketplace" in s:
        return "ecommerce"
    if "saas" in s or "software" in s:
        return "saas"
    if "manufactur" in s:
        return "manufacturing"
    if "edtech" in s or "education" in s:
        return "edtech"
    if "adtech" in s or "advertis" in s:
        return "adtech"
    if "gaming" in s or "game" in s:
        return "gaming"
    if "transport" in s or "logistic" in s:
        return "transportation"
    if "food" in s or "restaurant" in s:
        return "food"
    if "health" in s or "pharma" in s or "medic" in s:
        return "healthcare"
    if "telecom" in s or "communication" in s:
        return "telecom"
    if "conglomerate" in s:
        return "conglomerate"
    if "paper" in s:
        return "manufacturing"
    return s or "unknown"


# City / location helpers
TIER_1_INDIAN_CITIES = {
    "bangalore", "bengaluru", "mumbai", "delhi", "hyderabad", "pune", "chennai",
    "kolkata", "gurgaon", "gurugram", "noida", "ahmedabad",
}
TIER_2_INDIAN_CITIES = {
    "jaipur", "lucknow", "kanpur", "nagpur", "indore", "bhopal", "coimbatore",
    "kochi", "visakhapatnam", "vizag", "thiruvananthapuram", "trivandrum",
    "bhubaneswar", "chandigarh", "surat", "vadodara", "patna",
}
JOB_PREFERRED_CITIES = {"noida", "pune"}


def location_tokens(location: str) -> list[str]:
    if not location:
        return []
    return [t.strip().lower() for t in re.split(r"[,/]", location) if t.strip()]


def is_preferred_location(location: str) -> bool:
    return any(t in JOB_PREFERRED_CITIES for t in location_tokens(location))


def is_tier1_india(country: str, location: str) -> bool:
    if (country or "").strip().lower() != "india":
        return False
    return any(t in TIER_1_INDIAN_CITIES for t in location_tokens(location))


def is_india(country: str) -> bool:
    return (country or "").strip().lower() == "india"


CONSULTING_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "hcl", "tech mahindra",
    "accenture", "cognizant", "capgemini", "ibm global services", "mindtree",
    "ltimindtree", "persistent", "mphasis", "larsen", "l&t infotech",
    "ltts", "genpact", "hexaware", "kpit", "cyient", "zensar", "slk", "dxc",
}


def is_consulting_company(company: str) -> bool:
    if not company:
        return False
    s = company.strip().lower()
    return any(c in s for c in CONSULTING_COMPANIES)


# Pure-text cleaners used in the deep profile builder.
def clean_text(s: str | None) -> str:
    if not s:
        return ""
    s = s.replace("\u2014", " - ").replace("\u2013", " - ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def join_pieces(*pieces: str | None) -> str:
    return " ".join(p for p in pieces if p).strip()


# Keyword sets used to score career evidence. Kept narrow on purpose.
AI_KEYWORDS: tuple[str, ...] = (
    "machine learning", "deep learning", "neural network", "transformer",
    "embedding", "embeddings", "vector search", "vector database",
    "retrieval", "ranker", "ranking", "reranker", "rerank", "search engine",
    "recommendation", "recommender", "nlp", "natural language",
    "language model", "llm", "fine-tun", "finetune", "lora", "qlora", "peft",
    "rlhf", "prompt", "rag", "retrieval augmented", "knowledge graph",
    "named entity", "question answer", "qa system", "semantic search",
    "text classification", "text generation", "speech", "computer vision",
    "image classification", "object detection", "segmentation",
    "pytorch", "tensorflow", "huggingface", "sentence-transformer",
    "sentence transformer", "openai", "anthropic", "gemini", "claude",
    "stable diffusion", "diffusion model", "generative ai", "genai",
)

PRODUCT_COMPANY_HINTS: tuple[str, ...] = (
    "product company", "saas", "b2b", "b2c", "consumer", "marketplace",
    "users", "production", "deployed", "shipped", "launched",
)


def has_any(haystack: str, needles: Iterable[str]) -> bool:
    if not haystack:
        return False
    h = haystack.lower()
    return any(n in h for n in needles)
