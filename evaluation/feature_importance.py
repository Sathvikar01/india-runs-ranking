"""Feature importance (SHAP-lite, no extra deps).

Uses LightGBM's built-in feature importance (split count + gain) as a
proxy for SHAP. The full SHAP package is heavy and the LightGBM
internal importance is a good enough signal for an evaluation report.

Also computes: top features per "ranked into the top 10" vs
"ranked below 90" cohort. The difference highlights which features
actually move the needle.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("eval.feature_importance")


def ltr_feature_importance(ltr_model_path: str | Path, top_k: int = 20) -> list[dict]:
    """Return the top-K features by LightGBM split gain.

    Output rows: [{feature, gain, split_count, gain_pct}]
    """
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(ltr_model_path))
    gains = booster.feature_importance(importance_type="gain")
    splits = booster.feature_importance(importance_type="split")
    names = booster.feature_name()
    rows = list(zip(names, gains, splits, strict=False))
    total_gain = sum(g for _, g, _ in rows) or 1.0
    rows.sort(key=lambda r: r[1], reverse=True)
    return [
        {
            "feature": n,
            "gain": float(g),
            "split_count": int(s),
            "gain_pct": round(100.0 * g / total_gain, 2),
        }
        for n, g, s in rows[:top_k]
    ]


def cohort_feature_diff(
    features_df: pd.DataFrame,
    csv_path: str | Path,
    top_n: int = 10,
    bottom_n: int = 10,
) -> list[dict]:
    """Compute the per-feature mean difference between the top-N and
    bottom-N cohorts in the ranker output. Features with the largest
    positive deltas are the ones that drove the ranker to the top.
    """
    import csv

    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["rank"]))
    top_ids = [r["candidate_id"] for r in rows[:top_n]]
    bottom_ids = [r["candidate_id"] for r in rows[-bottom_n:]]
    df = features_df.set_index("candidate_id")
    top = df.loc[top_ids].select_dtypes(include="number")
    bot = df.loc[bottom_ids].select_dtypes(include="number")
    diffs = (top.mean() - bot.mean()).sort_values(ascending=False)
    out = []
    for feat, delta in diffs.items():
        out.append({
            "feature": feat,
            "top_mean": round(float(top[feat].mean()), 4),
            "bottom_mean": round(float(bot[feat].mean()), 4),
            "delta": round(float(delta), 4),
        })
    return out


def write_feature_importance_md(
    importance: list[dict],
    cohort_diff: list[dict],
    out_path: str | Path,
) -> None:
    """Write a Markdown report combining global importance + cohort diff."""
    lines = ["# Feature Importance\n"]
    lines.append("## Top 20 features by LightGBM gain (LTR model)\n")
    lines.append("| rank | feature | gain | split count | gain % |")
    lines.append("|---:|---|---:|---:|---:|")
    for i, r in enumerate(importance, 1):
        lines.append(f"| {i} | `{r['feature']}` | {r['gain']:.1f} | {r['split_count']} | {r['gain_pct']:.1f} |")
    lines.append("\n## Top 10 vs bottom 10 — feature mean difference\n")
    lines.append("Positive delta → feature value is higher in the top-10 than the bottom-10.\n")
    lines.append("| feature | top-10 mean | bottom-10 mean | delta |")
    lines.append("|---|---:|---:|---:|")
    for r in cohort_diff[:20]:
        lines.append(
            f"| `{r['feature']}` | {r['top_mean']} | {r['bottom_mean']} | {r['delta']:+.3f} |"
        )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Wrote feature importance to %s", out_path)
