"""The ranking script that Stage 3 reproduces.

Hard constraints:
  * ≤ 5 min wall-clock
  * ≤ 16 GB RAM
  * CPU only
  * No network

It loads pre-computed artifacts (built once by `scripts/build_artifacts.py`),
runs the hybrid retrieve → cross-encoder → LTR → ensemble pipeline, applies
the honeypot and JD penalty filters, generates the strict monotonically
non-increasing score list, looks up pre-stored per-candidate reasoning text,
and writes a 100-row CSV that passes the official validator.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from src.api.schemas import Candidate, JobDescription
from src.behavioral.availability import availability_score
from src.behavioral.honeypot import honeypot_risk, is_honeypot
from src.behavioral.jd_filters import negative_penalty, positive_boost
from src.ingestion.parse_jsonl import iter_candidates_jsonl
from src.preprocessing.deep_profile import build_deep_profile
from src.preprocessing.feature_engineer import (
    categorical_columns,
    feature_columns,
    features_to_dataframe,
    build_features,
)
from src.ranking.ensemble import ensemble_score, make_monotonic_scores
from src.ranking.ltr_model import LTRModel
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense_index import DenseIndex, encode_queries
from src.retrieval.hybrid_fusion import rrf, union_top_k

log = logging.getLogger("rank")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _load_jd(path: str | Path) -> str:
    p = Path(path)
    if p.suffix.lower() == ".md":
        return p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".docx":
        from docx import Document  # type: ignore

        doc = Document(str(p))
        return "\n".join(par.text for par in doc.paragraphs)
    return p.read_text(encoding="utf-8")


def _load_portraits(path: str | Path) -> dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, dict] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                out[obj["candidate_id"]] = obj
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def _safe_default_reasoning(c) -> str:
    if isinstance(c, MinimalCandidate):
        row = c._row if hasattr(c, "_row") else {}
        yoe = float(row.get("yoe_reported", 0.0))
        title = row.get("current_title_raw") or "Candidate"
        rr = float(row.get("recruiter_response_rate", 0.0))
        return (
            f"{title} with {yoe:.1f} yrs; response rate {rr:.2f}; "
            f"honeypot_risk {float(row.get('behavioral_honeypot', 0.0)):.2f}."
        )
    title = c.profile.current_title or "Candidate"
    yoe = c.profile.years_of_experience
    n_skills = len(c.skills) if hasattr(c, "skills") and c.skills else 0
    return (
        f"{title} with {yoe:.1f} yrs; {n_skills} skills on profile; "
        f"response rate {c.redrob_signals.recruiter_response_rate:.2f}."
    )


def _format_reasoning(c: Candidate, portrait: dict | None, rank: int) -> str:
    if portrait and portrait.get("reasoning"):
        text = (portrait["reasoning"] or "").strip()
        if text:
            if len(text) > 350:
                text = text[:347] + "..."
            return text
    return _rich_default_reasoning(c, rank)


def _rich_default_reasoning(c, rank: int) -> str:
    """Build a 1-2 sentence recruiter note from feature data when the LLM is unavailable."""
    if isinstance(c, MinimalCandidate):
        row = c._row or {}
    else:
        row = {}

    title = row.get("current_title_raw") or "Candidate"
    yoe = float(row.get("yoe_reported", 0.0))
    has_ai = int(row.get("has_ai_career_evidence", 0) or 0) or int(row.get("ai_keyword_hits_career", 0) or 0) >= 3
    has_rank = int(row.get("has_retrieval_ranking_evidence", 0) or 0) == 1
    has_finetune = int(row.get("has_llm_finetune_evidence", 0) or 0) == 1
    has_shipped = int(row.get("has_shipped_to_users", 0) or 0) == 1
    loc_noida_pune = int(row.get("location_is_noida_or_pune", 0) or 0) == 1
    tier1 = int(row.get("location_tier1_india", 0) or 0) == 1
    relocate = int(row.get("willing_to_relocate", 0) or 0) == 1
    notice = int(row.get("notice_period_days", 60) or 60)
    rr = float(row.get("recruiter_response_rate", 0.0) or 0.0)
    recency = float(row.get("recency_score", 0.0) or 0.0)
    honeypot = float(row.get("behavioral_honeypot", 0.0) or 0.0)

    positives: list[str] = []
    if has_ai:
        positives.append("sustained AI/ML career work in the description")
    if has_rank:
        positives.append("ranking or retrieval-system experience")
    if has_finetune:
        positives.append("LLM fine-tuning exposure (LoRA / PEFT)")
    if has_shipped:
        positives.append("clear production-shipped evidence")
    if loc_noida_pune:
        positives.append("based in Noida or Pune")
    elif tier1 and relocate:
        positives.append("based in a Tier-1 Indian city and willing to relocate")
    if notice <= 30:
        positives.append(f"{notice}-day notice period")

    concerns: list[str] = []
    if not has_ai:
        concerns.append("no clear AI/ML evidence in career history")
    if not loc_noida_pune and not (tier1 and relocate):
        concerns.append("location is not Noida/Pune and no willingness to relocate")
    if notice > 60:
        concerns.append(f"{notice}-day notice period is long")
    if rr < 0.3:
        concerns.append(f"recruiter response rate is low ({rr:.0%})")
    if recency < 0.4:
        concerns.append("last activity is older than 6 months")
    if honeypot > 0.5:
        concerns.append("profile shows honeypot-shaped risk signals")

    if positives:
        lead = positives[0]
        tail_pos = positives[1:2]
    else:
        lead = "career history is a partial match"
        tail_pos = []

    bits = [f"{title} with {yoe:.1f} yrs"]
    if lead:
        bits.append(f"with {lead}")
    for t in tail_pos:
        bits.append(f"and {t}")

    sentence1 = ", ".join(bits) + "."
    if concerns:
        sentence2 = "Concern: " + concerns[0] + "."
    else:
        sentence2 = f"Behavioural: open to work, response rate {rr:.0%}, recency score {recency:.0%}."

    text = f"{sentence1} {sentence2}"
    if len(text) > 350:
        text = text[:347] + "..."
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the candidate ranking pipeline.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--job-description", required=True, help="Path to JD (.md or .docx)")
    parser.add_argument("--artifacts", default="artifacts", help="Artifact directory")
    parser.add_argument("--out", required=True, help="Path to output CSV (e.g. outputs/team_xxx.csv)")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--portraits", default=None, help="Path to portraits.jsonl")
    parser.add_argument("--max-candidates", type=int, default=100_000)
    args = parser.parse_args(argv)

    artifacts = Path(args.artifacts)
    bm25_path = artifacts / "bm25.pkl"
    faiss_path = artifacts / "faiss.index"
    feature_path = artifacts / "feature_store.parquet"
    ltr_path = artifacts / "ltr.cbm"
    portraits_path = Path(args.portraits) if args.portraits else artifacts / "portraits.jsonl"

    if not bm25_path.exists():
        raise FileNotFoundError(f"BM25 index not found: {bm25_path}. Run scripts/build_artifacts.py first.")
    if not feature_path.exists():
        raise FileNotFoundError(f"feature store not found: {feature_path}")
    if not ltr_path.exists():
        raise FileNotFoundError(f"LTR model not found: {ltr_path}")

    t_start = time.perf_counter()

    log.info("Loading job description …")
    jd_text = _load_jd(args.job_description)

    log.info("Loading BM25 index …")
    bm25 = BM25Index.load(bm25_path)
    use_dense = faiss_path.exists()
    if use_dense:
        log.info("Loading faiss dense index …")
        dense = DenseIndex.load(faiss_path)
    else:
        log.info("Dense index not found — using BM25-only retrieval.")
        dense = None
    log.info("Loading feature store …")
    import pandas as pd
    features_df = pd.read_parquet(feature_path)
    log.info("Loading LTR model …")
    ltr = LTRModel.load(ltr_path)
    log.info("Loading portraits …")
    portraits = _load_portraits(portraits_path)

    log.info("Encoding query …")
    cfg = yaml.safe_load(Path("configs/build.yaml").read_text(encoding="utf-8"))
    if use_dense:
        q_vec = encode_queries(
            [jd_text],
            model_name=cfg["embedding"]["model_name"],
            batch_size=1,
            max_seq_length=cfg["embedding"]["max_seq_length"],
            device=cfg["embedding"]["device"],
            cache_dir=cfg["embedding"]["cache_dir"],
            normalize=cfg["embedding"]["normalize"],
        )
    log.info("Hybrid retrieval …")
    bm25_top = bm25.query(jd_text, top_k=cfg["retrieval"]["bm25_top_k"])
    if use_dense:
        dense_top = dense.query(q_vec[0], top_k=cfg["retrieval"]["dense_top_k"])
        fused = rrf([bm25_top, dense_top], k=cfg["retrieval"]["rrf_k"])
        union_top = union_top_k(bm25_top, dense_top, k=cfg["retrieval"]["union_top_k"])
    else:
        fused = [(cid, s) for cid, s in bm25_top]
        union_top = bm25_top
    cand_ids = [cid for cid, _ in union_top[: cfg["cross_encoder"]["top_k_from_retrieval"]]]
    cand_set = set(cand_ids)

    log.info("Loading candidate records for shortlist (%d) …", len(cand_ids))
    shortlist: dict[str, Candidate] = {}
    for c in iter_candidates_jsonl(args.candidates):
        if c.candidate_id in cand_set:
            shortlist[c.candidate_id] = c
        if len(shortlist) == len(cand_set):
            break

    log.info("Cross-encoder rerank …")
    from src.ranking.cross_encoder import rerank
    if use_dense:
        ce_scored = rerank(
            jd_text,
            [(cid, build_deep_profile(shortlist[cid])) for cid in cand_ids],
            model_name=cfg["cross_encoder"]["model_name"],
            top_k=cfg["cross_encoder"]["top_k_output"],
            batch_size=cfg["cross_encoder"]["batch_size"],
            max_length=cfg["cross_encoder"]["max_length"],
            device=cfg["cross_encoder"]["device"],
        )
    else:
        # Without dense, the BM25 top-K is already the shortlist. Skip CE
        # rerank and just use BM25 score as a proxy (still useful as a
        # boosting signal vs the LTR).
        from src.ranking.cross_encoder import rerank
        ce_scored = rerank(
            jd_text,
            [(cid, build_deep_profile(shortlist[cid])) for cid in cand_ids],
            model_name=cfg["cross_encoder"]["model_name"],
            top_k=cfg["cross_encoder"]["top_k_output"],
            batch_size=cfg["cross_encoder"]["batch_size"],
            max_length=cfg["cross_encoder"]["max_length"],
            device=cfg["cross_encoder"]["device"],
        )
    ce_scores = {cid: float(s) for cid, s in ce_scored}
    top_ce_ids = [cid for cid, _ in ce_scored]

    log.info("LTR scoring …")
    feats_for_top = features_df[features_df["candidate_id"].isin(top_ce_ids)].copy()
    # Ensure the order matches top_ce_ids for the LTR score join.
    feats_for_top = feats_for_top.set_index("candidate_id").loc[top_ce_ids].reset_index()
    X = feats_for_top[feature_columns() + categorical_columns()].copy()
    ltr_scores = ltr.predict(X)
    ltr_score_map = {cid: float(s) for cid, s in zip(top_ce_ids, ltr_scores)}

    log.info("Ensemble (vectorized) …")
    X_all = features_df[feature_columns() + categorical_columns()].copy()
    ltr_all = ltr.predict(X_all)
    id_to_ltr_all = dict(zip(features_df["candidate_id"], ltr_all))
    id_to_ce = {cid: float(s) for cid, s in ce_scored}
    id_to_avail = dict(zip(features_df["candidate_id"], features_df["behavioral_availability"]))
    id_to_pos = dict(zip(features_df["candidate_id"], features_df["behavioral_positive"]))
    id_to_neg = dict(zip(features_df["candidate_id"], features_df["behavioral_negative"]))
    id_to_hon = dict(zip(features_df["candidate_id"], features_df["behavioral_honeypot"]))

    final_scored: list[tuple[str, float]] = []
    for cid in features_df["candidate_id"].tolist():
        ltr_s = id_to_ltr_all[cid]
        ce_s = id_to_ce.get(cid, 0.0)
        if cid in id_to_ce:
            ce_s = ce_s + 0.5
        score = ensemble_score(
            ltr_s, ce_s, id_to_avail[cid], id_to_pos[cid], id_to_neg[cid], id_to_hon[cid]
        )
        final_scored.append((cid, float(score)))

    final_scored.sort(key=lambda x: x[1], reverse=True)

    # Apply strict monotonicity to the top-K we will output.
    top_k = min(args.top_k, len(final_scored))
    head = final_scored[:top_k]
    monotone_scores = make_monotonic_scores([s for _, s in head])
    rows = []
    feats_idx = features_df.set_index("candidate_id")
    for i, ((cid, _raw), mono) in enumerate(zip(head, monotone_scores, strict=True), 1):
        portrait = portraits.get(cid)
        feat_row = feats_idx.loc[cid].to_dict() if cid in feats_idx.index else None
        reasoning = _format_reasoning(_candidate_minimal(feat_row), portrait, i)
        rows.append(
            {
                "candidate_id": cid,
                "rank": i,
                "score": round(float(mono), 4),
                "reasoning": reasoning,
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    elapsed = time.perf_counter() - t_start
    log.info("Wrote %d rows to %s in %.1fs", len(rows), out_path, elapsed)
    honeypot_count = sum(1 for r in rows if r.get("score", 0) <= 0.20)
    log.info("Bottom-decile count in output: %d", honeypot_count)
    return 0


def _candidate_minimal(feat_row: dict | None) -> "MinimalCandidate":
    """Wrap a feature row so `_format_reasoning` can pull the headline numbers."""
    return MinimalCandidate(feat_row or {})


class MinimalCandidate:
    def __init__(self, row: dict):
        self._row = row
        self.profile = _MinimalProfile(row)
        self.redrob_signals = _MinimalSignals(row)


class _MinimalProfile:
    def __init__(self, row: dict):
        self.current_title = row.get("current_title_raw", "")
        self.years_of_experience = float(row.get("yoe_reported", 0.0))


class _MinimalSignals:
    def __init__(self, row: dict):
        self.recruiter_response_rate = float(row.get("recruiter_response_rate", 0.0))


def _iter_all_candidates(path: str, ids: list[str]):
    """Yield (candidate_id, Candidate) for every id we know about, in pool order."""
    wanted = set(ids)
    for c in iter_candidates_jsonl(path):
        if c.candidate_id in wanted:
            yield c.candidate_id, c


def _candidate_by_id(path: str, cid: str, all_ids: list[str]) -> Candidate:
    """Re-stream the jsonl to find one record. Slow but only used for the 100 outputs."""
    for c in iter_candidates_jsonl(path):
        if c.candidate_id == cid:
            return c
    raise KeyError(cid)


if __name__ == "__main__":
    sys.exit(main())
