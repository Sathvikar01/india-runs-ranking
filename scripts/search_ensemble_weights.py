"""Grid search the ensemble weights (Agent 7).

Loads the dev-split candidates, computes per-candidate features from
``feature_store.parquet``, then evaluates many weight combinations
against both proxy and eval_rubric ground truths. Saves the best
``EnsembleWeights`` to ``artifacts/best_ensemble_weights.json``.

Uses a coordinate-descent search (cheaper than full grid):
  1. Start from the current default weights.
  2. For each weight, sweep a small grid of candidates.
  3. Pick the candidate that maximises min(proxy_score, eval_rubric_score).
  4. Move to the next weight; repeat for 3 rounds.

The score for a given weights instance is:
  ranking_score = mean of NDCG@10, NDCG@50, MAP, P@10 against the dev
                   pool (proxy and eval_rubric separately), then min of
                   the two means.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.behavioral.availability import availability_score
from src.behavioral.honeypot import honeypot_risk
from src.behavioral.jd_filters import negative_penalty, positive_boost
from src.evaluation.ndcg import ndcg_at_k, average_precision
from src.evaluation.proxy_ground_truth import proxy_relevance
from src.evaluation.eval_rubric import eval_relevance
from src.ingestion.parse_jsonl import iter_candidates_jsonl
from src.ranking.ensemble import EnsembleWeights, ensemble_score_v2

log = logging.getLogger("search_ensemble")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _eval_composite(
    candidates: list,
    ltr_scores: np.ndarray,
    ce_scores: np.ndarray,
    weights: EnsembleWeights,
) -> tuple[float, float, float]:
    """Return (proxy_composite, eval_rubric_composite, min)."""
    scored = []
    proxy_truth = {}
    eval_truth = {}
    for i, c in enumerate(candidates):
        s = ensemble_score_v2(
            ltr_score=float(ltr_scores[i]),
            ce_score=float(ce_scores[i]),
            availability=availability_score(c),
            positive=positive_boost(c),
            negative=negative_penalty(c),
            honeypot=honeypot_risk(c),
            weights=weights,
        )
        scored.append(s)
        proxy_truth[c.candidate_id] = proxy_relevance(c)
        eval_truth[c.candidate_id] = eval_relevance(c)
    scored = np.asarray(scored)

    # Build top-100 ordering (with tiebreak by id).
    order = np.argsort(-scored, kind="stable")
    top = order[:100]
    proxy_rels = [proxy_truth[candidates[int(i)].candidate_id] for i in top]
    eval_rels = [eval_truth[candidates[int(i)].candidate_id] for i in top]
    proxy_n10 = ndcg_at_k(proxy_rels, 10)
    proxy_n50 = ndcg_at_k(proxy_rels, 50)
    proxy_map = average_precision(proxy_rels)
    proxy_p10 = sum(1 for r in proxy_rels[:10] if r >= 3.0) / 10
    proxy_comp = 0.50 * proxy_n10 + 0.30 * proxy_n50 + 0.15 * proxy_map + 0.05 * proxy_p10

    eval_n10 = ndcg_at_k(eval_rels, 10)
    eval_n50 = ndcg_at_k(eval_rels, 50)
    eval_map = average_precision(eval_rels)
    eval_p10 = sum(1 for r in eval_rels[:10] if r >= 3.0) / 10
    eval_comp = 0.50 * eval_n10 + 0.30 * eval_n50 + 0.15 * eval_map + 0.05 * eval_p10

    return proxy_comp, eval_comp, min(proxy_comp, eval_comp)


def _main_loop(args) -> int:
    t0 = time.perf_counter()
    log.info("Loading dev split …")
    candidates = list(iter_candidates_jsonl(args.candidates))
    log.info("  %d candidates", len(candidates))

    # Build proxy LTR / CE scores. For the dev search we just use the
    # proxy score as the LTR proxy and uniform random for CE (CE is not
    # re-trained for the search; this is a quick weight sanity check).
    rng = np.random.default_rng(0)
    ltr_scores = np.array([
        # LTR proxy = proxy_relevance() in [0, 4] * 2 to make sigmoid reach 0.9+
        float(proxy_relevance(c)) + rng.normal(0, 0.3)
        for c in candidates
    ], dtype=np.float32)
    ce_scores = rng.normal(0, 1, size=len(candidates)).astype(np.float32)

    log.info("Coordinate-descent search over %d weights × %d candidates each …",
             len(args.grid), len(candidates))

    weights = EnsembleWeights()
    base_proxy, base_eval, base_min = _eval_composite(
        candidates, ltr_scores, ce_scores, weights,
    )
    log.info("baseline: proxy=%.4f eval=%.4f min=%.4f", base_proxy, base_eval, base_min)

    history: list[dict] = [{"round": 0, "min": base_min, "weights": weights.__dict__.copy()}]

    for round_idx in range(args.rounds):
        improved = False
        for name in ("w_ltr", "w_ce", "w_avail", "w_positive",
                     "w_negative", "w_honeypot", "w_catboost",
                     "w_multitask", "w_topk"):
            cur = getattr(weights, name)
            best_v = cur
            best_min = base_min
            for v in args.grid:
                setattr(weights, name, v)
                p, e, m = _eval_composite(
                    candidates, ltr_scores, ce_scores, weights,
                )
                if m > best_min + 1e-5:
                    best_min = m
                    best_v = v
            setattr(weights, name, best_v)
            if best_v != cur:
                improved = True
                p, e, m = _eval_composite(
                    candidates, ltr_scores, ce_scores, weights,
                )
                log.info("  round %d: %s %s -> %s (min %.4f)",
                         round_idx, name, cur, best_v, m)
                base_min = m
        history.append({
            "round": round_idx + 1, "min": base_min,
            "weights": weights.__dict__.copy(),
        })
        if not improved:
            log.info("  round %d: no improvement, stopping early.", round_idx)
            break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(weights.to_json(), encoding="utf-8")
    history_path = out_path.with_name(out_path.stem + "_history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    log.info("Best weights saved to %s", out_path)
    log.info("Final weights: %s", weights.to_json())
    log.info("Wall clock: %.1fs", time.perf_counter() - t0)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", default="data/raw/candidates_5k.jsonl")
    p.add_argument("--out", default="artifacts/best_ensemble_weights.json")
    p.add_argument("--grid", nargs="*", type=float,
                   default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6])
    p.add_argument("--rounds", type=int, default=3)
    args = p.parse_args()
    return _main_loop(args)


if __name__ == "__main__":
    sys.exit(main())
