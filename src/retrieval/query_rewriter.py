"""Per-JD query rewriter (WS-7).

Expands the BM25/dense query with skill synonyms so the dense retriever
matches against semantically similar but lexically different text.

The expansion is *only* applied to the dense query (and to the LLM
portrait-prompt if needed). The BM25 index is *not* re-tokenized at rank
time, so adding noise to the BM25 query can hurt. The dense encoder is
robust to query expansion because it embeds whole sentences.

The synonym map lives in `configs/jd_synonyms.yaml` and is intentionally
small + high-signal. Adding more synonyms is easy, but every synonym is
a potential false positive at scale.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


# Default fallback: minimal map (used if configs/jd_synonyms.yaml is missing).
_DEFAULT_SYNONYMS: dict[str, list[str]] = {
    "embeddings": ["vector search", "semantic search", "sentence transformers", "dense retrieval"],
    "ranking": ["learning to rank", "lambdarank", "ltr", "reranker"],
    "rerank": ["cross-encoder", "monoBERT", "monoT5"],
    "llm": ["large language model", "language model", "gpt", "llama", "mistral", "qwen"],
    "fine-tuning": ["lora", "qlora", "peft", "rlhf", "instruction tuning"],
    "rag": ["retrieval augmented generation", "retrieval-augmented", "grounded generation"],
    "retrieval": ["search", "faiss", "elasticsearch", "vector store", "vector database"],
    "ml platform": ["feature store", "model serving", "model registry", "mlops"],
    "eval harness": ["evaluation framework", "offline eval", "online eval", "ab test"],
    "scoring": ["scoring function", "ranker", "match score", "relevance score"],
}


@lru_cache(maxsize=1)
def _load_synonyms() -> dict[str, list[str]]:
    """Load the synonym map; merge on top of the default fallback."""
    if yaml is None:
        return dict(_DEFAULT_SYNONYMS)
    # Search in (a) CWD/configs, (b) the package-relative configs dir. The
    # latter is important for tests where the CWD is not the project root.
    candidates = [
        Path("configs/jd_synonyms.yaml"),
        Path(__file__).resolve().parents[2] / "configs" / "jd_synonyms.yaml",
    ]
    for cfg_path in candidates:
        if cfg_path.exists():
            try:
                data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                merged: dict[str, list[str]] = dict(_DEFAULT_SYNONYMS)
                for k, v in (data.get("synonyms") or {}).items():
                    merged[str(k).lower()] = [str(x).lower() for x in v]
                return merged
            except Exception:
                return dict(_DEFAULT_SYNONYMS)
    return dict(_DEFAULT_SYNONYMS)


def expand_query(text: str, max_extra_terms: int = 24) -> str:
    """Return the input text plus a suffix of synonym expansions.

    Each expansion is appended as a parenthetical clause at the end of the
    query so the BGE-encode prefix (which prepends "Represent this sentence
    for searching relevant passages: ") still wraps a coherent sentence.
    """
    if not text:
        return text
    syns = _load_synonyms()
    lowered = text.lower()
    found: list[str] = []
    for term, expansions in syns.items():
        if term in lowered:
            for e in expansions:
                if e not in lowered and e not in found:
                    found.append(e)
                    if len(found) >= max_extra_terms:
                        break
        if len(found) >= max_extra_terms:
            break
    if not found:
        return text
    # Append as a comma-list inside a single trailing clause.
    return text.rstrip() + " " + ", ".join(found) + "."


def expansion_terms(text: str, max_extra_terms: int = 24) -> list[str]:
    """Return just the expansion terms (no original text). Useful for tests."""
    if not text:
        return []
    syns = _load_synonyms()
    lowered = text.lower()
    found: list[str] = []
    for term, expansions in syns.items():
        if term in lowered:
            for e in expansions:
                if e not in lowered and e not in found:
                    found.append(e)
                    if len(found) >= max_extra_terms:
                        break
        if len(found) >= max_extra_terms:
            break
    return found


def _token_count(s: str) -> int:
    return len(re.findall(r"\w+", s))
