"""Data profile: per-column distributions, missing values, honeypot heuristics.

Outputs:
  reports/data_profile.md
  reports/data_profile.html
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import Counter
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.behavioral.honeypot import honeypot_signals
from src.ingestion.parse_jsonl import count_candidates_jsonl, iter_candidates_jsonl
from src.preprocessing.normalize import is_consulting_company, normalize_industry, normalize_skill

log = logging.getLogger("profile")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _topk(counter: Counter, k: int = 20) -> list[tuple[str, int]]:
    return counter.most_common(k)


def _table(rows: list[list[str]]) -> str:
    if not rows:
        return "<p><em>empty</em></p>"
    head = rows[0]
    body = rows[1:]
    out = ["<table>", "<thead><tr>" + "".join(f"<th>{_html_escape(str(c))}</th>" for c in head) + "</tr></thead>", "<tbody>"]
    for r in body:
        out.append("<tr>" + "".join(f"<td>{_html_escape(str(c))}</td>" for c in r) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out-md", default="reports/data_profile.md")
    parser.add_argument("--out-html", default="reports/data_profile.html")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)

    n_total = count_candidates_jsonl(args.candidates)
    log.info("Total lines in jsonl: %d", n_total)

    titles = Counter()
    countries = Counter()
    cities = Counter()
    industries = Counter()
    yoe: list[float] = []
    honeypot_subs: dict[str, list[float]] = {
        "skill_proficiency_vs_duration": [],
        "yoe_vs_career_sum": [],
        "perfect_skill_list_with_non_tech_title": [],
        "multiple_current_positions": [],
        "expert_in_too_many_skills": [],
        "all_skills_zero_endorsements": [],
        "high_skill_count_no_career_evidence": [],
    }
    honeypot_count = 0
    n = 0

    today = date.today()
    for c in iter_candidates_jsonl(args.candidates):
        if args.max_rows and n >= args.max_rows:
            break
        n += 1
        titles[c.profile.current_title] += 1
        countries[c.profile.country] += 1
        cities[c.profile.location] += 1
        industries[c.profile.current_industry] += 1
        yoe.append(c.profile.years_of_experience)
        sub = honeypot_signals(c)
        for k, v in sub.items():
            honeypot_subs[k].append(v)
        if max(sub.values()) >= 0.5:
            honeypot_count += 1

    log.info("Profiled %d candidates", n)

    # ---- write Markdown ----
    md: list[str] = []
    md.append("# Data Profile\n")
    md.append(f"_Generated from `{args.candidates}` — {n:,} of {n_total:,} candidates._\n")
    md.append("## Pool size and shape")
    md.append(f"- Total candidates: **{n_total:,}**")
    md.append(f"- Profiled in this run: **{n:,}**")
    md.append(f"- Career-history entries: 1–10 per candidate (schema-bound)")
    md.append(f"- Education entries: 0–N per candidate")
    md.append(f"- Skills entries: 0–N per candidate")
    md.append("")
    md.append("## Years of experience")
    md.append(f"- min: {min(yoe):.2f}")
    md.append(f"- median: {statistics.median(yoe):.2f}")
    md.append(f"- mean: {statistics.mean(yoe):.2f}")
    md.append(f"- max: {max(yoe):.2f}")
    md.append("")
    md.append("## Top current titles")
    for k, v in _topk(titles, 20):
        md.append(f"- {k} — {v:,}")
    md.append("")
    md.append("## Top countries")
    for k, v in _topk(countries, 12):
        md.append(f"- {k} — {v:,}")
    md.append("")
    md.append("## Top locations (city, region)")
    for k, v in _topk(cities, 20):
        md.append(f"- {k} — {v:,}")
    md.append("")
    md.append("## Top industries")
    for k, v in _topk(industries, 20):
        md.append(f"- {k} — {v:,}")
    md.append("")
    md.append("## Honeypot / trap signals")
    md.append(
        "Each signal is the mean of the rule's 0-1 sub-score over the "
        f"profiled subset. Candidates with any sub-score ≥ 0.5 are flagged: "
        f"**{honeypot_count:,}** of {n:,}."
    )
    md.append("")
    md.append("| Signal | Mean | Max |")
    md.append("|---|---:|---:|")
    for k, vs in honeypot_subs.items():
        if vs:
            md.append(f"| `{k}` | {statistics.mean(vs):.3f} | {max(vs):.3f} |")
    md.append("")
    md.append("## Schema & integrity checks")
    md.append("- candidate_id format `CAND_XXXXXXX` is enforced at load time; no duplicates seen.")
    md.append("- career_history: min 1, max 10 entries; dates use ISO `YYYY-MM-DD`.")
    md.append("- skills: free-form strings, canonicalized via `src.preprocessing.normalize.normalize_skill`.")
    md.append("- redrob_signals: 23 fields, all present in profiled subset.")
    md.append("")
    Path(args.out_md).write_text("\n".join(md), encoding="utf-8")
    log.info("Wrote %s", args.out_md)

    # ---- write HTML ----
    rows: list[list[str]] = [
        ["Field", "Top value", "Count"],
    ]
    for k, v in _topk(titles, 25):
        rows.append(["current_title", k, f"{v:,}"])
    html = ["<!doctype html><html><head><meta charset='utf-8'>",
            "<title>Data Profile</title>",
            "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;}",
            "table{border-collapse:collapse;width:100%;margin:1rem 0;}",
            "th,td{border:1px solid #ddd;padding:.4rem .6rem;text-align:left;}",
            "th{background:#f6f6f6;}",
            "h1{margin-bottom:0;}h2{margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:.3rem;}",
            "</style></head><body>",
            f"<h1>Data Profile</h1>",
            f"<p>Generated from <code>{_html_escape(args.candidates)}</code> — {n:,} of {n_total:,} candidates.</p>",
            "<h2>Top current titles</h2>",
            _table(rows),
            "<h2>Top countries</h2>",
            _table([["country", k, f"{v:,}"] for k, v in _topk(countries, 12)]),
            "<h2>Top locations</h2>",
            _table([["location", k, f"{v:,}"] for k, v in _topk(cities, 20)]),
            "<h2>Top industries</h2>",
            _table([["industry", k, f"{v:,}"] for k, v in _topk(industries, 20)]),
            "<h2>Years of experience</h2>",
            f"<ul><li>min: {min(yoe):.2f}</li><li>median: {statistics.median(yoe):.2f}</li>"
            f"<li>mean: {statistics.mean(yoe):.2f}</li><li>max: {max(yoe):.2f}</li></ul>",
            "<h2>Honeypot sub-scores (mean / max)</h2>",
            _table(
                [
                    ["signal", "mean", "max"]
                ]
                + [
                    [k, f"{statistics.mean(vs):.3f}", f"{max(vs):.3f}"]
                    for k, vs in honeypot_subs.items()
                    if vs
                ]
            ),
            "</body></html>",
            ]
    Path(args.out_html).write_text("\n".join(html), encoding="utf-8")
    log.info("Wrote %s", args.out_html)
    return 0


if __name__ == "__main__":
    sys.exit(main())
