"""Hard-negative mining.

Given an LTR model and a candidate pool, mine the top-k candidates the model
currently ranks highly *and* whose proxy relevance is low. These are the
hardest negatives for the next training round.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.api.schemas import Candidate
from src.evaluation.proxy_ground_truth import build_proxy_ground_truth
from src.preprocessing.feature_engineer import feature_columns
from src.ranking.ltr_model import LTRModel


def mine_hard_negatives(
    candidates: list[Candidate],
    ltr: LTRModel,
    features_df: pd.DataFrame,
    top_k: int = 200,
    rel_threshold: float = 1.0,
) -> list[str]:
    """Return candidate_ids where ltr_score is high but proxy_relevance is low."""
    from src.preprocessing.feature_engineer import categorical_columns

    relevance = build_proxy_ground_truth(candidates)
    # LTRModel.predict expects both numeric feature columns AND categorical
    # columns in the right dtype (categorical as category dtype).
    all_cols = list(ltr.feature_columns) + [c for c in categorical_columns() if c not in ltr.feature_columns]
    X = features_df.loc[:, all_cols].copy()
    # Cast categorical columns to category dtype (LightGBM expects this).
    for c in categorical_columns():
        if c in X.columns:
            X[c] = X[c].astype("category")
    scores = ltr.predict(X)
    df = features_df[["candidate_id"]].copy()
    df["ltr_score"] = scores
    df["relevance"] = df["candidate_id"].map(relevance).fillna(0.0)
    df = df.sort_values("ltr_score", ascending=False)
    hard = df[df["relevance"] < rel_threshold].head(top_k)
    return hard["candidate_id"].tolist()


def reweight_with_hard_negatives(
    relevance: dict[str, float],
    hard_neg_ids: list[str],
    boost: float = 0.5,
) -> dict[str, float]:
    """Up-weight the contrast: explicit 0.0 for hard negatives so LTR learns to push them down."""
    out = dict(relevance)
    for cid in hard_neg_ids:
        out[cid] = min(out.get(cid, 0.0), 0.0)
    return out
