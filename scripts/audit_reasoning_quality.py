"""Reasoning quality self-audit (WS-13).

Reads a submission CSV and scores each row's `reasoning` field on the
six Stage 4 review criteria from `data/raw/submission_spec.md:75-95`:

  1. Specific facts — does it reference candidate facts (yoe, title, skills, signal values)?
  2. JD connection — does it connect to specific JD requirements?
  3. Honest concerns — does it acknowledge gaps where they exist?
  4. No hallucination — does every claim resolve to the candidate's profile?
  5. Variation — are the 10 sampled reasonings substantively different?
  6. Rank consistency — does the tone match the rank?

Produces `reports/reasoning_quality.md` with per-row scores and an
overall pass/fail.

Usage:
    python scripts/audit_reasoning_quality.py outputs/team_xxx.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ingestion.parse_jsonl import iter_candidates_jsonl

log = logging.getLogger("audit_reasoning")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# Stage 4 forbidden superlatives (per submission_spec.md:43).
_SUPERLATIVES = re.compile(
    r"\b(impressive|stellar|exceptional|outstanding|phenomenal|world[-\s]?class|amazing|elite|top[-\s]?tier|best[-\s]?in[-\s]?class)\b",
    re.IGNORECASE,
)

# Specific-fact signals: at least one of these in the reasoning is expected.
_FACT_SIGNALS = (
    "yrs", "year", "experience", "title", "company", "skill", "pytorch",
    "tensorflow", "llm", "rag", "embedding", "retrieval", "ranking",
    "notice", "response rate", "recency", "location",
)

# Honest-concern signals: a Concern: marker or an honest caveat.
_CONCERN_SIGNALS = (
    "concern", "limited", "notice period", "location", "junior", "consulting",
    "no clear", "older than",
)


def _load_candidates(jsonl_path: str) -> dict[str, dict]:
    """Return {candidate_id: {headline, summary, current_company, current_title,
    skills_names, career_companies, career_titles, career_descriptions, yoe,
    location, signals, industry, all_profile_text}."""
    out: dict[str, dict] = {}
    for c in iter_candidates_jsonl(jsonl_path):
        blob_parts = []
        # Current profile
        if c.profile.current_company:
            blob_parts.append(c.profile.current_company)
        if c.profile.current_title:
            blob_parts.append(c.profile.current_title)
        if c.profile.current_industry:
            blob_parts.append(c.profile.current_industry)
        if c.profile.headline:
            blob_parts.append(c.profile.headline)
        if c.profile.summary:
            blob_parts.append(c.profile.summary)
        if c.profile.location:
            blob_parts.append(c.profile.location)
        # Career history
        for r in c.career_history:
            if r.company:
                blob_parts.append(r.company)
            if r.title:
                blob_parts.append(r.title)
            if r.industry:
                blob_parts.append(r.industry)
            if r.description:
                blob_parts.append(r.description)
        # Skills
        for s in c.skills:
            if s.name:
                blob_parts.append(s.name)
        # Projects and certifications
        for p in c.projects:
            if p.name:
                blob_parts.append(p.name)
            if p.description:
                blob_parts.append(p.description)
        for cert in c.certifications:
            if cert.name:
                blob_parts.append(cert.name)
        blob = " ".join(blob_parts).lower()
        out[c.candidate_id] = {
            "current_company": (c.profile.current_company or "").lower(),
            "current_title": (c.profile.current_title or "").lower(),
            "industry": (c.profile.current_industry or "").lower(),
            "yoe": float(c.profile.years_of_experience or 0.0),
            "location": (c.profile.location or "").lower(),
            "skills_names": {s.name.lower() for s in c.skills if s.name},
            "career_companies": {r.company.lower() for r in c.career_history if r.company},
            "career_titles": {r.title.lower() for r in c.career_history if r.title},
            "career_descriptions": " ".join((r.description or "") for r in c.career_history).lower(),
            "signals": {
                "response_rate": c.redrob_signals.recruiter_response_rate,
                "notice_days": c.redrob_signals.notice_period_days,
                "open_to_work": c.redrob_signals.open_to_work_flag,
                "github": c.redrob_signals.github_activity_score,
            },
            "all_profile_text": blob,
        }
    return out


def _has_specific_facts(reasoning: str) -> bool:
    r = reasoning.lower()
    return any(sig in r for sig in _FACT_SIGNALS)


def _has_jd_connection(reasoning: str) -> bool:
    """At least one of the JD's named must-haves is mentioned."""
    jd_terms = (
        "retrieval", "ranking", "rerank", "embeddings", "vector search", "rag",
        "fine-tun", "lora", "peft", "rlhf", "eval", "pytorch", "transformers",
        "faiss", "elasticsearch", "learning to rank", "lambdarank", "llm",
        "machine learning", "ml platform", "ai/ml",
    )
    r = reasoning.lower()
    return any(term in r for term in jd_terms)


def _has_honest_concerns(reasoning: str) -> bool:
    r = reasoning.lower()
    return any(sig in r for sig in _CONCERN_SIGNALS)


def _no_hallucination(reasoning: str, profile: dict | None) -> list[str]:
    """Return a list of hallucination issues (empty == clean)."""
    if profile is None:
        return ["profile_not_loaded"]
    issues: list[str] = []
    r = reasoning.lower()
    if _SUPERLATIVES.search(reasoning):
        issues.append("superlative")
    # Check for any capitalized bigram that doesn't appear in the profile.
    bigrams = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)\b", reasoning)
    for bg in bigrams:
        words = bg.split()
        if any(w.lower() in profile["all_profile_text"] for w in words):
            continue
        issues.append(f"unknown_employer_or_school:{bg}")
    return issues


def _rank_consistency(rank: int, reasoning: str) -> str:
    """Top-30 should not start with 'Concern:'. Bottom-30 should not be all-positive."""
    r = reasoning.lower()
    if rank <= 30 and r.startswith("concern"):
        return "tone_mismatch_top_concern"
    if rank >= 71 and "concern" not in r and "limited" not in r and "notice" not in r and "older" not in r:
        return "tone_mismatch_bottom_no_concern"
    return "ok"


def _bigrams(text: str) -> set[tuple[str, str]]:
    toks = re.findall(r"\w+", text.lower())
    return {(toks[i], toks[i + 1]) for i in range(len(toks) - 1)}


def _variation_score(reasonings: list[str]) -> dict:
    """Mean pairwise Jaccard over bigrams. Lower = more diverse."""
    bags = [_bigrams(r) for r in reasonings]
    n = len(bags)
    if n < 2:
        return {"mean_jaccard": 0.0, "n_unique": len(set(reasonings))}
    pairs = 0
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = bags[i], bags[j]
            if not a and not b:
                continue
            jacc = len(a & b) / max(1, len(a | b))
            total += jacc
            pairs += 1
    return {
        "mean_jaccard": (total / pairs) if pairs else 0.0,
        "n_unique": len(set(reasonings)),
    }


def audit(csv_path: str, candidates_jsonl: str | None, out_md: str) -> dict:
    """Audit the submission CSV. Returns a summary dict (also written to
    `out_md` as a Markdown report)."""
    rows: list[dict] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    profiles: dict[str, dict] = {}
    if candidates_jsonl:
        profiles = _load_candidates(candidates_jsonl)
        log.info("Loaded %d candidate profiles", len(profiles))

    per_row: list[dict] = []
    pass_counts = Counter()
    fail_counts = Counter()
    for r in rows:
        cid = r["candidate_id"]
        rank = int(r["rank"])
        reasoning = r.get("reasoning", "").strip()
        profile = profiles.get(cid)
        issues: list[str] = []
        # 1. Specific facts
        has_facts = _has_specific_facts(reasoning)
        # 2. JD connection
        has_jd = _has_jd_connection(reasoning)
        # 3. Honest concerns (in the bottom decile, expected)
        has_concerns = _has_honest_concerns(reasoning)
        # 4. No hallucination
        hallu_issues = _no_hallucination(reasoning, profile)
        # 5. Rank consistency
        rank_cons = _rank_consistency(rank, reasoning)
        # length check
        if not (1 <= len(reasoning) <= 350):
            issues.append(f"len_out_of_range:{len(reasoning)}")
        if not has_facts:
            issues.append("no_specific_facts")
        if not has_jd:
            issues.append("no_jd_connection")
        if rank >= 71 and not has_concerns:
            issues.append("bottom_rank_no_concern")
        issues.extend(hallu_issues)
        if rank_cons != "ok":
            issues.append(rank_cons)

        per_row.append({
            "rank": rank,
            "candidate_id": cid,
            "score": r.get("score", ""),
            "n_issues": len(issues),
            "issues": "; ".join(issues) if issues else "ok",
            "has_facts": has_facts,
            "has_jd": has_jd,
            "has_concerns": has_concerns,
            "rank_consistency": rank_cons,
            "reasoning_len": len(reasoning),
        })
        if issues:
            fail_counts[len(issues)] += 1
        else:
            pass_counts["all_pass"] += 1

    # 5. Variation score across the whole CSV
    variation = _variation_score([r.get("reasoning", "") for r in rows])
    # 6. Rank consistency distribution
    rank_cons_counts = Counter(pr["rank_consistency"] for pr in per_row)

    # Length distribution
    lens = [pr["reasoning_len"] for pr in per_row]
    len_dist = {
        "min": min(lens), "max": max(lens), "mean": sum(lens) / len(lens),
        "below_50": sum(1 for n in lens if n < 50),
        "above_320": sum(1 for n in lens if n > 320),
    }

    n = len(per_row)
    n_facts = sum(1 for pr in per_row if pr["has_facts"])
    n_jd = sum(1 for pr in per_row if pr["has_jd"])
    n_concerns = sum(1 for pr in per_row if pr["has_concerns"])
    n_clean = sum(1 for pr in per_row if pr["issues"] == "ok")
    n_hallu = sum(1 for pr in per_row if "hallu" in pr["issues"] or "unknown_employer" in pr["issues"] or "superlative" in pr["issues"])

    summary = {
        "n_rows": n,
        "n_clean": n_clean,
        "n_with_facts": n_facts,
        "n_with_jd_connection": n_jd,
        "n_with_honest_concerns": n_concerns,
        "n_hallucination_issues": n_hallu,
        "n_unique_reasonings": variation["n_unique"],
        "mean_pairwise_jaccard": round(variation["mean_jaccard"], 4),
        "length_distribution": len_dist,
        "rank_consistency": dict(rank_cons_counts),
    }

    # Write Markdown report
    out_lines: list[str] = []
    out_lines.append("# Reasoning Quality Audit\n")
    out_lines.append(f"_Source: `{csv_path}`_\n")
    out_lines.append("## Summary\n")
    out_lines.append(f"- Rows: **{n}**")
    out_lines.append(f"- Clean rows (no issues): **{n_clean}** ({100.0 * n_clean / n:.1f} %)")
    out_lines.append(f"- Rows with specific facts: **{n_facts}** ({100.0 * n_facts / n:.1f} %)")
    out_lines.append(f"- Rows with JD connection: **{n_jd}** ({100.0 * n_jd / n:.1f} %)")
    out_lines.append(f"- Rows with honest concerns: **{n_concerns}** ({100.0 * n_concerns / n:.1f} %)")
    out_lines.append(f"- Rows with hallucination issues: **{n_hallu}** ({100.0 * n_hallu / n:.1f} %)")
    out_lines.append(f"- Unique reasonings: **{variation['n_unique']}** / {n}")
    out_lines.append(f"- Mean pairwise bigram Jaccard: **{variation['mean_jaccard']:.4f}** (lower = more diverse)")
    out_lines.append(f"- Length: min {len_dist['min']}, max {len_dist['max']}, mean {len_dist['mean']:.1f}")
    out_lines.append(
        f"- Length violations: {len_dist['below_50']} below 50 chars, {len_dist['above_320']} above 320"
    )
    out_lines.append("\n### Stage 4 checks (per `submission_spec.md:75-95`)\n")
    out_lines.append("| Check | Verdict |")
    out_lines.append("|---|---|")
    out_lines.append(
        f"| Specific facts (yoe/title/skills/signal values) | "
        f"{'PASS' if n_facts >= 0.95 * n else 'WARN'} (n={n_facts}/{n}) |"
    )
    out_lines.append(
        f"| JD connection (retrieval/ranking/LLM/etc.) | "
        f"{'PASS' if n_jd >= 0.90 * n else 'WARN'} (n={n_jd}/{n}) |"
    )
    out_lines.append(
        f"| Honest concerns where expected | "
        f"{'PASS' if n_concerns >= 0.20 * n else 'WARN'} (n={n_concerns}/{n}) |"
    )
    out_lines.append(
        f"| No hallucination | "
        f"{'PASS' if n_hallu == 0 else 'WARN'} (n={n_hallu}/{n}) |"
    )
    out_lines.append(
        f"| Variation (no all-identical reasonings) | "
        f"{'PASS' if variation['n_unique'] >= 0.9 * n else 'WARN'} "
        f"(n={variation['n_unique']}/{n}) |"
    )
    out_lines.append(
        f"| Rank consistency (top positive, bottom has concerns) | "
        f"{'PASS' if rank_cons_counts.get('ok', 0) >= 0.95 * n else 'WARN'} |"
    )
    out_lines.append("\n## Per-row issues\n")
    out_lines.append("| Rank | candidate_id | issues | reasoning_len |")
    out_lines.append("|---:|---|---:|---:|")
    for pr in per_row:
        if pr["issues"] != "ok":
            out_lines.append(
                f"| {pr['rank']} | `{pr['candidate_id']}` | {pr['issues']} | {pr['reasoning_len']} |"
            )
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    log.info("Audit written to %s", out_md)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a submission CSV for reasoning quality.")
    parser.add_argument("csv", help="Path to the submission CSV.")
    parser.add_argument(
        "--candidates",
        default=None,
        help="Path to candidates.jsonl (needed for hallucination check).",
    )
    parser.add_argument(
        "--out",
        default="reports/reasoning_quality.md",
        help="Output Markdown report path.",
    )
    parser.add_argument("--json", action="store_true", help="Also print JSON summary to stdout.")
    args = parser.parse_args()
    summary = audit(args.csv, args.candidates, args.out)
    if args.json:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
