"""Ranking quality metrics (NDCG, MAP, P@k, MRR) against two ground truths.

WS-4 introduced an *independent* eval rubric, deliberately different
from the proxy ground truth, to break circular-eval. This module
scores the ranker's output against BOTH and reports each separately.

The composite `ranking_score` is the *minimum* of (proxy_score,
eval_rubric_score). Defending against proxy-overfit is the whole
point of the eval-rubric split.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

log = logging.getLogger("eval.ranking_metrics")

# Same weights as the official composite: 0.50 * NDCG@10 + 0.30 * NDCG@50
# + 0.15 * MAP + 0.05 * P@10. (submission_spec.md:116)
COMPOSITE_WEIGHTS: dict[str, float] = {
    "ndcg@10": 0.50,
    "ndcg@50": 0.30,
    "map": 0.15,
    "p@10": 0.05,
}


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _ordered_ids(csv_rows: list[dict]) -> list[str]:
    """Return candidate_ids in the ranker's order (lowest rank first)."""
    return [r["candidate_id"] for r in sorted(csv_rows, key=lambda r: int(r["rank"]))]


def compute_metrics(
    csv_path: str | Path,
    relevance: dict[str, float],
) -> dict:
    """Compute NDCG@K, MAP, P@K, MRR for the order in `csv_path`.

    `relevance` maps candidate_id -> graded relevance (0-4).
    """
    from src.evaluation.ndcg import evaluate_ranking

    rows = _read_csv(Path(csv_path))
    if not rows:
        return {"error": "empty csv"}
    ordered = _ordered_ids(rows)
    metrics = evaluate_ranking(ordered, relevance)
    return metrics


def ranking_score(
    csv_path: str | Path,
    proxy_relevance: dict[str, float],
    eval_relevance: dict[str, float],
    jd_literal_relevance: dict[str, float] | None = None,
) -> dict:
    """Compute ranking metrics against two (or three) ground truths and the
    `ranking_score` used in the composite.

    Returns a dict with per-metric scores for each ground truth, plus
    `ranking_score_0_1` — the *minimum* of all available composites,
    defending against overfit to any single ground truth.

    Agent 9 update: when ``jd_literal_relevance`` is provided, the
    ranking_score is the minimum of all three. The ``jd_literal`` entry
    surfaces the spread so a regression in any one rubric is visible.
    """
    proxy = compute_metrics(csv_path, proxy_relevance)
    eval_ = compute_metrics(csv_path, eval_relevance)
    proxy_composite = sum(COMPOSITE_WEIGHTS[k] * proxy.get(k, 0.0) for k in COMPOSITE_WEIGHTS)
    eval_composite = sum(COMPOSITE_WEIGHTS[k] * eval_.get(k, 0.0) for k in COMPOSITE_WEIGHTS)
    composites = [proxy_composite, eval_composite]
    out = {
        "proxy": proxy,
        "proxy_composite": round(proxy_composite, 4),
        "eval_rubric": eval_,
        "eval_rubric_composite": round(eval_composite, 4),
        "composite_weights": COMPOSITE_WEIGHTS,
    }
    if jd_literal_relevance is not None:
        jd = compute_metrics(csv_path, jd_literal_relevance)
        jd_composite = sum(COMPOSITE_WEIGHTS[k] * jd.get(k, 0.0) for k in COMPOSITE_WEIGHTS)
        composites.append(jd_composite)
        out["jd_literal"] = jd
        out["jd_literal_composite"] = round(jd_composite, 4)
    out["ranking_score_0_1"] = round(min(composites), 4)
    return out


def write_ranking_metrics_csv(
    csv_path: str | Path,
    proxy_relevance: dict[str, float],
    eval_relevance: dict[str, float],
    out_path: str | Path,
    jd_literal_relevance: dict[str, float] | None = None,
) -> None:
    """Write a CSV with one row per ground truth (proxy, eval_rubric,
    optionally jd_literal), with NDCG@K, MAP, P@K, MRR."""
    proxy = compute_metrics(csv_path, proxy_relevance)
    eval_ = compute_metrics(csv_path, eval_relevance)
    rows = [
        {"ground_truth": "proxy", **proxy},
        {"ground_truth": "eval_rubric", **eval_},
    ]
    if jd_literal_relevance is not None:
        jd = compute_metrics(csv_path, jd_literal_relevance)
        rows.append({"ground_truth": "jd_literal", **jd})
    fields = sorted({k for r in rows for k in r})
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    log.info("Wrote ranking metrics to %s", out)


def load_candidates(jsonl_path: str | Path, max_n: int | None = None) -> list:
    """Load candidates from the JSONL pool."""
    from src.ingestion.parse_jsonl import iter_candidates_jsonl

    out = []
    for c in iter_candidates_jsonl(jsonl_path):
        out.append(c)
        if max_n is not None and len(out) >= max_n:
            break
    return out
