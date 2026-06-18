"""Precompute per-candidate deep_profile BGE embeddings + per-JD embedding.

WS-10: at build time, encode each candidate's `deep_profile` with the same
BGE model used for retrieval, and encode the JD once. The dot product (BGE
vectors are L2-normalised, so dot = cosine) is added as the
`career_jd_semantic_sim` column to the feature store.

This replaces the bag-of-keywords `ai_keyword_hits_career` heuristic for
"is this candidate actually doing AI?" with a real semantic similarity
between the candidate's career evidence and the JD's requirements.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("career_jd_sim")


def encode_with_bge(
    texts: list[str],
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 64,
    max_seq_length: int = 256,
    cache_dir: str | None = "artifacts/cache/hf",
    device: str = "cpu",
    normalize: bool = True,
) -> np.ndarray:
    """Encode a list of texts with BGE; return float32 [N, D]."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, cache_folder=cache_dir, device=device)
    if "bge" in model_name.lower():
        prefix = "Represent this sentence for searching relevant passages: "
        texts = [prefix + t for t in texts]
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    if not isinstance(vecs, np.ndarray):
        vecs = np.asarray(vecs, dtype=np.float32)
    return vecs.astype(np.float32, copy=False)


def precompute_career_jd_similarity(
    feature_parquet: str,
    jd_text: str,
    deep_profiles: list[tuple[str, str]],
    out_parquet: str | None = None,
) -> dict:
    """Compute per-candidate career-JD cosine similarity and merge into the
    feature store.

    Parameters
    ----------
    feature_parquet : str
        Path to the existing feature store (parquet).
    jd_text : str
        The job description (raw markdown).
    deep_profiles : list[tuple[str, str]]
        (candidate_id, deep_profile_text) pairs, one per candidate.
    out_parquet : str | None
        If set, write the augmented feature store to this path. If None,
        overwrites the input.
    """
    import pandas as pd

    t0 = time.perf_counter()
    log.info("Encoding %d deep profiles + 1 JD with BGE …", len(deep_profiles))
    texts = [t for _, t in deep_profiles]
    cand_vecs = encode_with_bge(texts)
    jd_vec = encode_with_bge([jd_text])[0]
    log.info("Encoding done in %.1fs; shape=%s", time.perf_counter() - t0, cand_vecs.shape)

    # Cosine similarity = dot product since vectors are L2-normalised.
    sims = cand_vecs @ jd_vec

    df = pd.read_parquet(feature_parquet)
    id_to_sim = {cid: float(s) for (cid, _), s in zip(deep_profiles, sims, strict=True)}
    df["career_jd_semantic_sim"] = df["candidate_id"].map(id_to_sim).fillna(0.0)

    out_path = out_parquet or feature_parquet
    df.to_parquet(out_path, index=False)
    log.info(
        "Wrote %d rows to %s with career_jd_semantic_sim (mean=%.3f, min=%.3f, max=%.3f)",
        len(df),
        out_path,
        df["career_jd_semantic_sim"].mean(),
        df["career_jd_semantic_sim"].min(),
        df["career_jd_semantic_sim"].max(),
    )
    return {
        "n_candidates": len(df),
        "mean": float(df["career_jd_semantic_sim"].mean()),
        "min": float(df["career_jd_semantic_sim"].min()),
        "max": float(df["career_jd_semantic_sim"].max()),
    }


def attach_similarity_column_inplace(
    df,  # pandas DataFrame
    jd_text: str,
    deep_profiles: list[tuple[str, str]],
) -> dict:
    """In-place variant for use in the build_artifacts pipeline."""
    t0 = time.perf_counter()
    log.info("Encoding %d deep profiles + 1 JD with BGE …", len(deep_profiles))
    cand_vecs = encode_with_bge([t for _, t in deep_profiles])
    jd_vec = encode_with_bge([jd_text])[0]
    sims = cand_vecs @ jd_vec
    log.info("Encoding done in %.1fs; shape=%s", time.perf_counter() - t0, cand_vecs.shape)
    id_to_sim = {cid: float(s) for (cid, _), s in zip(deep_profiles, sims, strict=True)}
    df["career_jd_semantic_sim"] = df["candidate_id"].map(id_to_sim).fillna(0.0)
    return {
        "mean": float(df["career_jd_semantic_sim"].mean()),
        "min": float(df["career_jd_semantic_sim"].min()),
        "max": float(df["career_jd_semantic_sim"].max()),
    }
