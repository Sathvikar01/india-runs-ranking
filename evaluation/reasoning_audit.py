"""Reasoning quality scoring (WS-13 audit, wrapped).

Maps the per-row audit results from `scripts/audit_reasoning_quality.py`
into a single 0..1 `reasoning_score` used in the composite.

Each of the 6 Stage 4 checks contributes equally to the reasoning_score
(unless explicitly weighted below). A check passes if >= 90 % of rows
satisfy it; the sub-score scales linearly between 0 % and 90 %.
"""
from __future__ import annotations

import contextlib
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("eval.reasoning_audit")

# Map of check name → how to extract a 0-1 pass-rate from the audit dict.
CHECK_EXTRACTORS: dict[str, callable] = {
    "specific_facts": lambda a: a.get("n_with_facts", 0) / max(1, a.get("n_rows", 1)),
    "jd_connection": lambda a: a.get("n_with_jd_connection", 0) / max(1, a.get("n_rows", 1)),
    "honest_concerns": lambda a: a.get("n_with_honest_concerns", 0) / max(1, a.get("n_rows", 1)),
    "no_hallucination": lambda a: 1.0 - a.get("n_hallucination_issues", 0) / max(1, a.get("n_rows", 1)),
    "variation": lambda a: a.get("n_unique_reasonings", 0) / max(1, a.get("n_rows", 1)),
    "rank_consistency": lambda a: a.get("rank_consistency", {}).get("ok", 0) / max(1, a.get("n_rows", 1)),
}

# Equal weight for each check.
CHECK_WEIGHTS: dict[str, float] = {k: 1.0 / len(CHECK_EXTRACTORS) for k in CHECK_EXTRACTORS}


def _load_audit_json(report_md_path: str | Path) -> dict | None:
    """Find the latest EVAL.json or read a separate audit JSON if present."""
    # The audit script writes a markdown report; we re-run it to get the
    # JSON summary, then return it.

    # The audit script writes the report next to EVAL.json; the run_evaluation
    # script passes the CSV path explicitly. We require the caller to pass
    # both `csv_path` and `candidates_path` via _run_audit_for_eval below.
    return None


def run_audit_for_eval(
    csv_path: str | Path,
    candidates_jsonl: str | Path,
    out_md: str | Path,
) -> dict:
    """Run the audit and return the JSON summary."""
    import subprocess

    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "audit_reasoning_quality.py"),
        str(csv_path),
        "--candidates",
        str(candidates_jsonl),
        "--out",
        str(out_md),
        "--json",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        log.warning("audit failed: %s\nstderr: %s", r.stdout, r.stderr)
    # The audit script prints the summary to stdout (with --json).
    # Parse the last JSON object in stdout.
    summary: dict = {}
    text = r.stdout.strip()
    if text.startswith("{"):
        try:
            summary = json.loads(text)
        except json.JSONDecodeError:
            # Maybe mixed with log lines; try to find the JSON.
            i = text.rfind("{")
            j = text.rfind("}")
            if i >= 0 and j > i:
                with contextlib.suppress(json.JSONDecodeError):
                    summary = json.loads(text[i : j + 1])
    return summary


def reasoning_score(audit_summary: dict) -> dict:
    """Compute the per-check pass-rate and the overall reasoning_score.

    Returns
    -------
    dict
        {
            "reasoning_score_0_1": float,
            "per_check_pass_rate": {check: float},
            "weights": {check: float},
        }
    """
    if not audit_summary:
        return {"reasoning_score_0_1": 0.0, "per_check_pass_rate": {}}
    per_check: dict[str, float] = {}
    weighted_sum = 0.0
    weight_total = 0.0
    for check, extractor in CHECK_EXTRACTORS.items():
        v = float(extractor(audit_summary))
        per_check[check] = round(v, 4)
        w = CHECK_WEIGHTS[check]
        weighted_sum += w * v
        weight_total += w
    score_0_1 = (weighted_sum / weight_total) if weight_total else 0.0
    return {
        "reasoning_score_0_1": round(score_0_1, 4),
        "per_check_pass_rate": per_check,
        "weights": dict(CHECK_WEIGHTS),
    }
