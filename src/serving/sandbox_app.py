"""Streamlit sandbox demo.

Accepts a small candidate sample (≤ 100) and the JD, runs the ranking
pipeline against the prebuilt artifacts, and shows the top-100 with
per-candidate reasoning.

Designed to run on HuggingFace Spaces (free tier), Streamlit Cloud, or
locally.
"""

from __future__ import annotations

import json
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api.schemas import Candidate, JobDescription
from src.behavioral.availability import availability_score
from src.behavioral.honeypot import honeypot_risk
from src.behavioral.jd_filters import negative_penalty, positive_boost
from src.preprocessing.deep_profile import build_deep_profile
from src.preprocessing.feature_engineer import (
    build_features,
    categorical_columns,
    feature_columns,
    features_to_dataframe,
)
from src.ranking.ensemble import ensemble_score, make_monotonic_scores
from src.ranking.ltr_model import LTRModel
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense_index import DenseIndex, encode_queries
from src.retrieval.hybrid_fusion import rrf, union_top_k

st.set_page_config(
    page_title="Redrob Candidate Intelligence — Sandbox",
    page_icon="🎯",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def _load_artifacts(artifacts_dir: str):
    p = Path(artifacts_dir)
    bm25 = BM25Index.load(p / "bm25.pkl")
    dense = DenseIndex.load(p / "faiss.index")
    features = pd.read_parquet(p / "feature_store.parquet")
    ltr = LTRModel.load(p / "ltr.cbm")
    portraits: dict[str, dict] = {}
    portraits_path = p / "portraits.jsonl"
    if portraits_path.exists():
        with portraits_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    portraits[obj["candidate_id"]] = obj
                except (json.JSONDecodeError, KeyError):
                    continue
    return bm25, dense, features, ltr, portraits


def _parse_uploaded_candidates(upload) -> list[Candidate]:
    data = upload.read().decode("utf-8")
    try:
        objs = json.loads(data)
        if isinstance(objs, dict):
            objs = [objs]
        return [Candidate.model_validate(o) for o in objs]
    except json.JSONDecodeError:
        # Try JSONL
        out = []
        for line in data.splitlines():
            line = line.strip()
            if line:
                out.append(Candidate.model_validate(json.loads(line)))
        return out


def _run_ranking(
    jd_text: str,
    candidates: list[Candidate],
    bm25: BM25Index,
    dense: DenseIndex,
    features: pd.DataFrame,
    ltr: LTRModel,
    portraits: dict[str, dict],
    cfg: dict,
) -> pd.DataFrame:
    q_vec = encode_queries(
        [jd_text],
        model_name=cfg["embedding"]["model_name"],
        batch_size=1,
        max_seq_length=cfg["embedding"]["max_seq_length"],
        device=cfg["embedding"]["device"],
        cache_dir=cfg["embedding"]["cache_dir"],
        normalize=cfg["embedding"]["normalize"],
    )

    # Restrict to candidate ids the user uploaded
    wanted_ids = {c.candidate_id for c in candidates}
    # The BM25 index is over the full pool, so query the full text but then intersect.
    bm25_top = bm25.query(jd_text, top_k=2000)
    dense_top = dense.query(q_vec[0], top_k=2000)
    fused = rrf([bm25_top, dense_top], k=cfg["retrieval"]["rrf_k"])
    pool = [c for c in candidates if c.candidate_id in {cid for cid, _ in fused} or True]

    # LTR features for uploaded set; if not in the feature table, compute on the fly.
    feats = features[features["candidate_id"].isin(wanted_ids)].copy()
    if len(feats) < len(candidates):
        # Fall back: build features for the missing ones in memory.
        existing = set(feats["candidate_id"])
        missing = [c for c in candidates if c.candidate_id not in existing]
        if missing:
            extra = features_to_dataframe([build_features(c) for c in missing])
            feats = pd.concat([feats, extra], ignore_index=True)

    feats = feats.set_index("candidate_id")
    feats = feats.loc[[c.candidate_id for c in candidates]].reset_index()
    X = feats[feature_columns() + categorical_columns()].copy()
    ltr_scores = ltr.predict(X)
    feats["ltr_score"] = ltr_scores

    # Use the CE-style rerank for the demo over the uploaded set (full CE on
    # the entire pool is too slow for an interactive demo).
    from src.ranking.cross_encoder import rerank
    ce_top = rerank(
        jd_text,
        [(c.candidate_id, build_deep_profile(c)) for c in candidates],
        model_name=cfg["cross_encoder"]["model_name"],
        top_k=len(candidates),
        batch_size=cfg["cross_encoder"]["batch_size"],
        max_length=cfg["cross_encoder"]["max_length"],
        device=cfg["cross_encoder"]["device"],
    )
    ce_map = {cid: float(s) for cid, s in ce_top}

    cand_by_id = {c.candidate_id: c for c in candidates}
    final_scored: list[tuple[str, float, dict]] = []
    for cid in [c.candidate_id for c in candidates]:
        c = cand_by_id[cid]
        ltr_s = float(feats.set_index("candidate_id").loc[cid, "ltr_score"])
        ce_s = ce_map.get(cid, 0.0)
        avail = availability_score(c)
        pos = positive_boost(c)
        neg = negative_penalty(c)
        hon = honeypot_risk(c)
        score = ensemble_score(ltr_s, ce_s, avail, pos, neg, hon)
        final_scored.append((cid, score, {
            "ltr": ltr_s, "ce": ce_s, "availability": avail,
            "positive": pos, "negative": neg, "honeypot": hon,
        }))

    final_scored.sort(key=lambda x: x[1], reverse=True)
    head = final_scored[:100]
    mono = make_monotonic_scores([s for _, s, _ in head])

    rows = []
    for i, ((cid, _raw, breakdown), ms) in enumerate(zip(head, mono, strict=True), 1):
        c = cand_by_id[cid]
        portrait = portraits.get(cid)
        if portrait and portrait.get("reasoning"):
            reasoning = portrait["reasoning"]
        else:
            reasoning = (
                f"{c.profile.current_title or 'Candidate'} with "
                f"{c.profile.years_of_experience:.1f} yrs; "
                f"response rate {c.redrob_signals.recruiter_response_rate:.2f}."
            )
        rows.append({
            "candidate_id": cid,
            "rank": i,
            "score": round(float(ms), 4),
            "reasoning": reasoning,
            "title": c.profile.current_title,
            "company": c.profile.current_company,
            "yoe": c.profile.years_of_experience,
            "honeypot_risk": round(breakdown["honeypot"], 3),
            "availability": round(breakdown["availability"], 3),
        })
    return pd.DataFrame(rows)


def main() -> None:
    st.title("🎯 Redrob Candidate Intelligence — Sandbox")
    st.caption("Upload a small candidate sample (≤ 100) and the JD text. Get a ranked CSV with per-candidate reasoning.")

    with st.sidebar:
        st.header("Configuration")
        cfg_path = st.text_input("Build config", value="configs/build.yaml")
        artifacts_dir = st.text_input("Artifacts directory", value="artifacts")
        try:
            cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            st.error(f"Config not found: {cfg_path}")
            st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Job description")
        jd_text = st.text_area("Paste JD here", height=200,
                                value=Path("data/raw/job_description.md").read_text(encoding="utf-8") if Path("data/raw/job_description.md").exists() else "")
    with col2:
        st.subheader("2. Candidate sample")
        sample_file = st.file_uploader("Upload candidates (JSON or JSONL, ≤ 100)", type=["json", "jsonl"])
        if sample_file is None and Path("data/samples/sample_50.json").exists():
            st.info("Found data/samples/sample_50.json — using it as default sample.")
            with Path("data/samples/sample_50.json").open("r", encoding="utf-8") as f:
                sample_file_content = f.read()
            sample_file = StringIO(sample_file_content)
            sample_file.name = "sample_50.json"
        if sample_file is not None:
            if hasattr(sample_file, "read"):
                content = sample_file.read()
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
                try:
                    obj = json.loads(content)
                    if isinstance(obj, dict):
                        obj = [obj]
                    candidates = [Candidate.model_validate(o) for o in obj]
                except json.JSONDecodeError:
                    candidates = []
                    for line in content.splitlines():
                        line = line.strip()
                        if line:
                            candidates.append(Candidate.model_validate(json.loads(line)))
            else:
                candidates = []
        else:
            candidates = []

    if st.button("Run ranking", type="primary", disabled=not (jd_text and candidates)):
        with st.spinner("Loading artifacts…"):
            try:
                bm25, dense, features, ltr, portraits = _load_artifacts(artifacts_dir)
            except FileNotFoundError as e:
                st.error(f"Artifact missing: {e}. Run scripts/build_artifacts.py first.")
                st.stop()
        with st.spinner("Running ranking pipeline…"):
            t0 = time.perf_counter()
            df = _run_ranking(jd_text, candidates, bm25, dense, features, ltr, portraits, cfg)
            elapsed = time.perf_counter() - t0

        st.success(f"Ranked {len(candidates)} candidates in {elapsed:.1f}s.")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download ranked CSV",
            df[["candidate_id", "rank", "score", "reasoning"]].to_csv(index=False).encode("utf-8"),
            file_name="sandbox_ranking.csv",
            mime="text/csv",
        )
        with st.expander("Score breakdown"):
            st.dataframe(df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
