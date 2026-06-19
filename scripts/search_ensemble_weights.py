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


def _precompute_signals(
    candidates: list,
    ltr_scores: np.ndarray,
    ce_scores: np.ndarray,
) -> dict:
    """Precompute per-candidate signals so the inner loop is just numpy."""
    log.info("Precomputing per-candidate signals for %d candidates …", len(candidates))
    n = len(candidates)
    signals = {
        "ltr_sigmoid": 1.0 / (1.0 + np.exp(-ltr_scores)),
        "ce_sigmoid": 1.0 / (1.0 + np.exp(-ce_scores)),
        "avail": np.zeros(n, dtype=np.float32),
        "positive": np.zeros(n, dtype=np.float32),
        "negative": np.zeros(n, dtype=np.float32),
        "honeypot": np.zeros(n, dtype=np.float32),
        "proxy_truth": np.zeros(n, dtype=np.float32),
        "eval_truth": np.zeros(n, dtype=np.float32),
    }
    for i, c in enumerate(candidates):
        signals["avail"][i] = float(availability_score(c))
        signals["positive"][i] = float(positive_boost(c))
        signals["negative"][i] = float(negative_penalty(c))
        signals["honeypot"][i] = float(honeypot_risk(c))
        signals["proxy_truth"][i] = float(proxy_relevance(c))
        signals["eval_truth"][i] = float(eval_relevance(c))
    return signals


def _eval_composite_fast(signals: dict, weights: EnsembleWeights) -> tuple[float, float, float]:
    """Vectorised scoring: ~100x faster than the per-candidate version."""
    base = (
        weights.w_ltr * signals["ltr_sigmoid"]
        + weights.w_ce * signals["ce_sigmoid"]
        + weights.w_avail * np.clip(signals["avail"], 0, 1)
        + weights.w_positive * np.clip(signals["positive"], 0, 1)
        - weights.w_negative * np.clip(signals["negative"], 0, 1)
        - weights.w_honeypot * np.clip(signals["honeypot"], 0, 1)
    )
    base = np.clip(base, 0, 1)
    order = np.argsort(-base, kind="stable")
    top = order[:100]
    proxy_rels = signals["proxy_truth"][top]
    eval_rels = signals["eval_truth"][top]

    def _comp(rels):
        rels = rels.tolist()
        n10 = ndcg_at_k(rels, 10)
        n50 = ndcg_at_k(rels, 50)
        m = average_precision(rels)
        p10 = sum(1 for r in rels[:10] if r >= 3.0) / 10
        return 0.50 * n10 + 0.30 * n50 + 0.15 * m + 0.05 * p10

    proxy_comp = _comp(proxy_rels)
    eval_comp = _comp(eval_rels)
    return proxy_comp, eval_comp, min(proxy_comp, eval_comp)


def _eval_composite(
    candidates: list,
    ltr_scores: np.ndarray,
    ce_scores: np.ndarray,
    weights: EnsembleWeights,
) -> tuple[float, float, float]:
    """Backwards-compatible per-candidate evaluator (slow path)."""
    signals = _precompute_signals(candidates, ltr_scores, ce_scores)
    return _eval_composite_fast(signals, weights)


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

    signals = _precompute_signals(candidates, ltr_scores, ce_scores)
    log.info("Coordinate-descent search over %d weights × %d candidates each …",
             len(args.grid), len(candidates))

    weights = EnsembleWeights()
    base_proxy, base_eval, base_min = _eval_composite_fast(signals, weights)
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
                _, _, m = _eval_composite_fast(signals, weights)
                if m > best_min + 1e-5:
                    best_min = m
                    best_v = v
            setattr(weights, name, best_v)
            if best_v != cur:
                improved = True
                _, _, m = _eval_composite_fast(signals, weights)
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
