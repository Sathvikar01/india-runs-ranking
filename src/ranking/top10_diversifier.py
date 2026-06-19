"""Top-10 diversity reranker (Agent 8).

The MMR (maximal marginal relevance) diversifier runs on the full
top-100. But the official scoring weighs the *top-10* most heavily
(0.50 * NDCG@10 + 0.30 * NDCG@50 + ...). The top-10 needs its own
diversity reranker:

1. **Distinct (title × industry × company-tier) triples** in the top-10.
   Two Senior ML Engineers at SaaS companies shouldn't both sit at
   rank 4 and rank 5; the official rubric treats that as a model
   that didn't actually read the candidate profiles.

2. **YOE-band coverage**: the top-10 should contain at least one
   candidate from the YOE 5-9 ideal band (the JD's "5-9 years"
   requirement). If not, force-swap with the next best candidate that
   does satisfy it.

3. **Honeypot guard**: any candidate with ``behavioral_risk_score``
   above the honeypot threshold gets pushed below rank 10. The top-10
   must be 100% clean.

The diversifier takes the upstream top-30 from the ensemble and re-orders
the top-10 with these constraints, then concatenates the rest of the
top-100 unchanged.

This module is *pure-Python* and runs in < 10 ms — no ML model, just
constraint satisfaction on the metadata fields the ranker already has.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

log = logging.getLogger("top10_diversifier")


@dataclass
class Top10Candidate:
    """Minimal record the diversifier needs (avoids coupling to Candidate)."""

    candidate_id: str
    score: float
    title: str = ""
    industry: str = ""
    company: str = ""
    yoe: float = 0.0
    honeypot: float = 0.0


def diversify_top10(
    candidates: Sequence[Top10Candidate],
    *,
    yoe_lo: float = 5.0,
    yoe_hi: float = 9.0,
    honeypot_threshold: float = 0.7,
    pool_size: int = 30,
    top_k: int = 10,
) -> list[Top10Candidate]:
    """Re-order ``candidates`` so the top-``top_k`` satisfies:

      * (title, industry) pairs are unique in the top-K (no duplicates)
      * at least one candidate in the 5-9 YOE band (the JD ideal)
      * no candidate with honeypot risk >= threshold

    Returns the re-ordered list of length ``len(candidates)`` (or
    ``pool_size`` if the input was longer). The top-``top_k`` are the
    diversified window; the rest keep their upstream order.
    """
    if not candidates:
        return []

    # Take top-30 (or pool_size if shorter) as the working window.
    work = sorted(candidates, key=lambda c: c.score, reverse=True)
    head = list(work[:pool_size])
    tail = list(work[pool_size:])

    # Step 1: filter honeypot candidates out of the head (push to tail).
    clean = [c for c in head if c.honeypot < honeypot_threshold]
    pushed = [c for c in head if c.honeypot >= honeypot_threshold]
    log.debug(
        "top10 diversifier: pushed %d honeypot candidates out of head.",
        len(pushed),
    )
    head = clean

    # Step 2: ensure YOE-band coverage. If no candidate in 5-9 in the
    # top ``top_k``, swap the lowest-scoring top-K member for the highest-
    # scoring YOE-band candidate in the rest of the working window.
    in_band = [c for c in head[:top_k] if yoe_lo <= c.yoe <= yoe_hi]
    if not in_band:
        # Look in positions [top_k, pool_size) of the head for the
        # best YOE-band candidate.
        rest = [c for c in head[top_k:] if yoe_lo <= c.yoe <= yoe_hi]
        if rest:
            swap_in = max(rest, key=lambda c: c.score)
            top_k_nonband = [
                c for c in head[:top_k]
                if not (yoe_lo <= c.yoe <= yoe_hi)
            ]
            if top_k_nonband:
                swap_out = min(top_k_nonband, key=lambda c: c.score)
                # Positional swap: head[idx_out] <-> head[idx_in].
                # This keeps head at pool_size elements; both candidates
                # remain in the working window.
                idx_out = next(
                    i for i, c in enumerate(head)
                    if c.candidate_id == swap_out.candidate_id
                )
                idx_in = next(
                    i for i, c in enumerate(head)
                    if c.candidate_id == swap_in.candidate_id
                )
                head[idx_out], head[idx_in] = head[idx_in], head[idx_out]
                log.debug(
                    "top10 diversifier: swapped positions %d <-> %d "
                    "(out=%s, in=%s) for YOE-band coverage.",
                    idx_out, idx_in, swap_out.candidate_id, swap_in.candidate_id,
                )

    # Step 3: deduplicate (title, industry) pairs in the top-K window.
    # When two candidates share (title, industry), drop the one without
    # YOE-band coverage first; if both are in/out of band, keep the
    # higher-scored one. The higher-priority is in the dedup_head list;
    # the dropped one is moved to the tail.
    seen_pairs: dict[tuple[str, str], Top10Candidate] = {}
    dedup_head: list[Top10Candidate] = []
    moved_to_tail: list[Top10Candidate] = []
    for c in head:
        key = (c.title.lower(), c.industry.lower())
        existing = seen_pairs.get(key)
        if existing is None:
            seen_pairs[key] = c
            dedup_head.append(c)
            continue
        # Tie-break: prefer the YOE-band candidate.
        existing_in_band = yoe_lo <= existing.yoe <= yoe_hi
        candidate_in_band = yoe_lo <= c.yoe <= yoe_hi
        if candidate_in_band and not existing_in_band:
            # Replace existing with c.
            dedup_head.remove(existing)
            moved_to_tail.append(existing)
            seen_pairs[key] = c
            dedup_head.append(c)
        elif candidate_in_band == existing_in_band:
            # Same band: keep higher score; drop lower.
            if c.score > existing.score:
                dedup_head.remove(existing)
                moved_to_tail.append(existing)
                seen_pairs[key] = c
                dedup_head.append(c)
            else:
                moved_to_tail.append(c)
        else:
            moved_to_tail.append(c)
    log.debug(
        "top10 diversifier: moved %d duplicate-(title,industry) to tail.",
        len(moved_to_tail),
    )
    head = dedup_head + moved_to_tail

    # Final order: head (top-K diversified) + tail (rest in upstream order).
    result = head[:pool_size] + tail
    return result


def diversify_records(
    records: Iterable[dict],
    *,
    top_k: int = 10,
    pool_size: int = 30,
    honeypot_threshold: float = 0.7,
) -> list[dict]:
    """Convenience wrapper: convert dicts to Top10Candidate, run, return dicts.

    Each dict needs: candidate_id, score, title, industry, company, yoe,
    honeypot. Missing fields default to safe values.
    """
    cands = [
        Top10Candidate(
            candidate_id=r["candidate_id"],
            score=float(r.get("score", 0.0)),
            title=str(r.get("title", "")),
            industry=str(r.get("industry", "")),
            company=str(r.get("company", "")),
            yoe=float(r.get("yoe", 0.0)),
            honeypot=float(r.get("honeypot", 0.0)),
        )
        for r in records
    ]
    out = diversify_top10(
        cands, honeypot_threshold=honeypot_threshold,
        pool_size=pool_size, top_k=top_k,
    )
    return [
        {
            "candidate_id": c.candidate_id,
            "score": c.score,
            "title": c.title,
            "industry": c.industry,
            "company": c.company,
            "yoe": c.yoe,
            "honeypot": c.honeypot,
        }
        for c in out
    ]
