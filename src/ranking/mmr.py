"""Maximal Marginal Relevance (MMR) re-ranker.

Diversifies the final top-K by penalising candidates that are too similar to
already-selected ones. The similarity is computed from cheap-to-derive
features: current title, current company, current industry, and YOE bucket.
We deliberately avoid using dense embeddings here — the goal is diversity,
not semantic similarity, and we want the diversity check to be interpretable.

The lambda parameter (0..1) trades off relevance vs diversity. With λ=1.0
the result equals the input order. With λ=0.7 (the default), the top-K
becomes more diverse while staying close to the LTR-ranked order.
"""
from __future__ import annotations

import re

# Title-seniority buckets, mapped to integers for cheap distance.
_SENIORITY_BUCKET_INDEX: dict[str, int] = {
    "intern": 0,
    "junior": 1,
    "mid": 2,
    "senior": 3,
    "staff": 4,
    "manager": 5,
    "unknown": 2,
}


def _title_bucket(title: str) -> str:
    """Mini seniority-bucket classifier; mirrors normalize.title_seniority_bucket
    but doesn't import it (keeps this module dependency-free)."""
    if not title:
        return "unknown"
    t = title.lower()
    if re.search(r"\b(intern|internship|trainee|apprentice)\b", t):
        return "intern"
    if re.search(r"\b(junior|associate|entry[-\s]?level|graduate|fresher)\b", t):
        return "junior"
    if re.search(r"\b(manager|head of|director|vp|vice president|chief)\b", t):
        return "manager"
    if re.search(r"\b(distinguished|fellow|principal|staff)\b", t):
        return "staff"
    if re.search(r"\b(senior|sr\.?|lead|architect)\b", t):
        return "senior"
    mid_pat = r"\b(sde|software engineer|ml engineer|mle\b|data scientist|analyst|developer|engineer)\b"
    if re.search(mid_pat, t):
        return "mid"
    return "unknown"


def _yoe_bucket(yoe: float) -> int:
    """3-yr YOE buckets for diversity."""
    if yoe < 3:
        return 0
    if yoe < 6:
        return 1
    if yoe < 9:
        return 2
    if yoe < 12:
        return 3
    return 4


def _normalised_title(title: str) -> str:
    """Lowercased, punctuation-stripped, space-collapsed title used for
    bigram overlap."""
    t = (title or "").lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _title_bigrams(title: str) -> set[tuple[str, str]]:
    toks = _normalised_title(title).split()
    if len(toks) < 2:
        return set()
    return {(toks[i], toks[i + 1]) for i in range(len(toks) - 1)}


def _candidate_features(item: dict) -> dict:
    """Extract the cheap features used for the diversity penalty."""
    title = item.get("current_title") or item.get("current_title_raw") or ""
    company = (item.get("current_company") or "").lower()
    industry = (item.get("current_industry") or item.get("current_industry_raw") or "").lower()
    yoe = float(item.get("yoe_reported", 0.0) or 0.0)
    return {
        "title": _normalised_title(title),
        "title_bucket": _title_bucket(title),
        "title_bigrams": _title_bigrams(title),
        "company": company,
        "industry": industry,
        "yoe_bucket": _yoe_bucket(yoe),
    }


def _similarity(a: dict, b: dict) -> float:
    """[0, 1] similarity score between two candidate feature dicts."""
    s = 0.0
    if a["title_bucket"] and a["title_bucket"] == b["title_bucket"]:
        s += 0.20
    if a["title_bigrams"] and b["title_bigrams"]:
        inter = len(a["title_bigrams"] & b["title_bigrams"])
        union = len(a["title_bigrams"] | b["title_bigrams"])
        if union:
            s += 0.35 * (inter / union)
    if a["company"] and a["company"] == b["company"]:
        s += 0.30
    if a["industry"] and a["industry"] == b["industry"]:
        s += 0.10
    if a["yoe_bucket"] == b["yoe_bucket"]:
        s += 0.05
    return min(1.0, s)


def mmr_rerank(
    candidates: list[dict],
    scores: list[float],
    top_k: int = 100,
    lam: float = 0.7,
) -> list[int]:
    """Return the indices of the top-`top_k` candidates after MMR re-ranking.

    `candidates` is a list of dicts each containing at least
    `current_title` / `current_company` / `yoe_reported`. `scores` is the
    relevance score for each candidate (higher = more relevant).
    """
    n = len(candidates)
    if n == 0:
        return []
    feats = [_candidate_features(c) for c in candidates]
    # Normalize relevance to [0, 1] for the MMR formula.
    if scores:
        lo, hi = min(scores), max(scores)
        rng = hi - lo
        if rng < 1e-9:
            rel = [1.0] * n
        else:
            rel = [(s - lo) / rng for s in scores]
    else:
        rel = [0.0] * n

    selected: list[int] = []
    remaining = set(range(n))
    while len(selected) < min(top_k, n):
        if not selected:
            # Pick the highest-relevance candidate first.
            best = max(remaining, key=lambda i: rel[i])
        else:
            best = -1
            best_score = -1e18
            for i in remaining:
                max_sim = 0.0
                for j in selected:
                    sim = _similarity(feats[i], feats[j])
                    max_sim = max(max_sim, sim)
                mmr_score = lam * rel[i] - (1.0 - lam) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best = i
        selected.append(best)
        remaining.discard(best)
    return selected


def mmr_apply(candidates: list[dict], scores: list[float], top_k: int = 100, lam: float = 0.7) -> list[dict]:
    """Convenience wrapper: return the reranked list of `candidates` dicts."""
    idx = mmr_rerank(candidates, scores, top_k=top_k, lam=lam)
    return [candidates[i] for i in idx]
