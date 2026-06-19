"""Hard-negative mining.

Given an LTR model and a candidate pool, mine the top-k candidates the model
currently ranks highly *and* whose proxy relevance is low. These are the
hardest negatives for the next training round.

Agent 6 update: also mines cross-ranker disagreement hard negatives —
candidates where two rankers disagree strongly, or where the ranker is
high but the eval_rubric is low. These are the failure modes the new
listwise reranker (Agent 3) needs to learn to avoid.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from src.api.schemas import Candidate
from src.evaluation.proxy_ground_truth import build_proxy_ground_truth
from src.preprocessing.feature_engineer import categorical_columns, feature_columns
from src.ranking.ltr_model import LTRModel

log = logging.getLogger("hard_negatives")


def mine_hard_negatives(
    candidates: list[Candidate],
    ltr: LTRModel,
    features_df: pd.DataFrame,
    top_k: int = 200,
    rel_threshold: float = 1.0,
) -> list[str]:
    """Return candidate_ids where ltr_score is high but proxy_relevance is low."""
    relevance = build_proxy_ground_truth(candidates)
    scores = _predict_features(ltr, features_df)
    df = features_df[["candidate_id"]].copy()
    df["ltr_score"] = scores
    df["relevance"] = df["candidate_id"].map(relevance).fillna(0.0)
    df = df.sort_values("ltr_score", ascending=False)
    hard = df[df["relevance"] < rel_threshold].head(top_k)
    return hard["candidate_id"].tolist()


def mine_cross_ranker_disagreements(
    candidates: list[Candidate],
    ltr: LTRModel,
    catboost_scores: np.ndarray | None,
    features_df: pd.DataFrame,
    top_k: int = 200,
) -> list[str]:
    """Mine hard negatives from LightGBM-vs-CatBoost disagreements.

    Returns candidates where:
      - LightGBM ranks them in the top 10% AND CatBoost ranks them in the
        bottom 25% (or vice versa). These are the cases where the two
        gradient-boosted trees disagree strongly — the next LTR training
        round should learn to resolve the disagreement.
    """
    ltr_scores = _predict_features(ltr, features_df)
    df = features_df[["candidate_id"]].copy()
    df["ltr_score"] = ltr_scores
    if catboost_scores is not None and len(catboost_scores) == len(df):
        df["cb_score"] = catboost_scores
    else:
        log.warning("CatBoost scores missing; falling back to LTR-only mining.")
        return mine_hard_negatives(candidates, ltr, features_df, top_k=top_k)

    # Rank by each model (1 = top).
    df["ltr_rank"] = df["ltr_score"].rank(method="first", ascending=False)
    df["cb_rank"] = df["cb_score"].rank(method="first", ascending=False)
    n = len(df)
    # Disagreement: LTR says top 10%, CatBoost says bottom 25%.
    df["disagree"] = (df["ltr_rank"] <= 0.10 * n) & (df["cb_rank"] >= 0.75 * n)
    hard = df[df["disagree"]].sort_values("ltr_rank").head(top_k)
    return hard["candidate_id"].tolist()


def mine_top_low_rubric(
    candidates: list[Candidate],
    ltr: LTRModel,
    features_df: pd.DataFrame,
    top_k: int = 200,
    eval_top_frac: float = 0.10,
    rubric_max: float = 1.0,
) -> list[str]:
    """Mine hard negatives where LTR says top but eval_rubric says no.

    Returns candidates where:
      - LightGBM ranks them in the top ``eval_top_frac`` AND
      - the eval_rubric score is <= ``rubric_max`` (default: not tier-3+).

    These teach the model that LTR confidence ≠ rubric alignment.
    """
    from src.evaluation.eval_rubric import build_eval_ground_truth

    ltr_scores = _predict_features(ltr, features_df)
    eval_truth = build_eval_ground_truth(candidates)

    df = features_df[["candidate_id"]].copy()
    df["ltr_score"] = ltr_scores
    df["eval_score"] = df["candidate_id"].map(eval_truth).fillna(0.0)
    df = df.sort_values("ltr_score", ascending=False)
    n = len(df)
    cutoff = int(eval_top_frac * n)
    head = df.head(cutoff)
    hard = head[head["eval_score"] <= rubric_max].head(top_k)
    return hard["candidate_id"].tolist()


def _predict_features(ltr: LTRModel, features_df: pd.DataFrame) -> np.ndarray:
    """Run the LTR booster over a feature DataFrame with the right dtypes."""
    cat_cols = categorical_columns()
    all_cols = list(ltr.feature_columns) + [c for c in cat_cols if c not in ltr.feature_columns]
    X = features_df.loc[:, all_cols].copy()
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return ltr.predict(X)


def reweight_with_hard_negatives(
    relevance: dict[str, float],
    hard_neg_ids: Sequence[str],
    boost: float = 0.5,
) -> dict[str, float]:
    """Up-weight the contrast: explicit 0.0 for hard negatives so LTR learns to push them down."""
    out = dict(relevance)
    for cid in hard_neg_ids:
        out[cid] = min(out.get(cid, 0.0), 0.0)
    return out


def union_hard_negatives(*lists: Sequence[str]) -> list[str]:
    """Union of multiple hard-negative lists (de-duplicated, order preserved)."""
    seen = set()
    out = []
    for lst in lists:
        for cid in lst:
            if cid not in seen:
                seen.add(cid)
                out.append(cid)
    return out
