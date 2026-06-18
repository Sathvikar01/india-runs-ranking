"""Ablation runner.

Executes a list of ablation configurations against a development split and
writes a Markdown report. Used by `scripts/run_ablation.py` and CI.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.api.schemas import Candidate
from src.evaluation.ndcg import evaluate_ranking


def _build_relevance(candidates: list[Candidate], use_proxy: bool = False) -> dict[str, float]:
    """Return the eval-relevance dict.

    WS-4: defaults to the *independent* eval rubric. Set `use_proxy=True` to
    fall back to the proxy ground truth (the labels the LTR trainer was
    actually trained on). The two functions are intentionally different
    sub-scoring rules so an LTR model that scores well on `eval_rubric` is
    doing more than memorising the proxy.
    """
    if use_proxy:
        from src.evaluation.proxy_ground_truth import build_proxy_ground_truth

        return build_proxy_ground_truth(candidates)
    from src.evaluation.eval_rubric import build_eval_ground_truth

    return build_eval_ground_truth(candidates)


@dataclass
class AblationResult:
    name: str
    metrics: dict[str, float]
    wall_clock_s: float
    config: dict[str, Any]


def _apply_ranking(
    candidates: list[Candidate],
    relevance: dict[str, float],
    rank_fn: Callable[[list[Candidate]], list[str]],
) -> dict[str, float]:
    t0 = time.perf_counter()
    ordered = rank_fn(candidates)
    metrics = evaluate_ranking(ordered, relevance)
    metrics["wall_clock_s"] = time.perf_counter() - t0
    return metrics


def run_ablations(
    candidates: list[Candidate],
    ablation_fns: dict[str, Callable[[list[Candidate]], list[str]]],
    out_dir: str | Path = "outputs/ablations",
    use_proxy_labels: bool = False,
) -> list[AblationResult]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    relevance = _build_relevance(candidates, use_proxy=use_proxy_labels)
    results: list[AblationResult] = []
    for name, fn in ablation_fns.items():
        metrics = _apply_ranking(candidates, relevance, fn)
        results.append(AblationResult(name=name, metrics=metrics, wall_clock_s=metrics.pop("wall_clock_s"), config={}))
        with (out_dir / f"{name}.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    summary = pd.DataFrame([{"ablation": r.name, **r.metrics} for r in results])
    summary.to_csv(out_dir / "summary.csv", index=False)
    return results


def write_markdown_report(results: list[AblationResult], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = ["| Ablation | NDCG@10 | NDCG@50 | MAP | P@10 | Composite | Wall (s) |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        m = r.metrics
        rows.append(
            f"| {r.name} | {m.get('ndcg@10', 0):.4f} | {m.get('ndcg@50', 0):.4f} | "
            f"{m.get('map', 0):.4f} | {m.get('p@10', 0):.4f} | "
            f"{m.get('composite', 0):.4f} | {r.wall_clock_s:.2f} |"
        )
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
