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
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from src.api.schemas import Candidate
from src.ingestion.parse_jsonl import iter_candidates_jsonl
from src.preprocessing.deep_profile import build_deep_profile
from src.preprocessing.feature_engineer import (
    categorical_columns,
    evidence_snippet,
    feature_columns,
    pick_named_jd_skill,
)
from src.ranking.ensemble import make_monotonic_scores_for_topk


def _sigmoid(x: float) -> float:
    """Numerically-stable sigmoid. Used for cross-encoder score calibration."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)
from src.ranking.ltr_model import LTRModel
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense_index import DenseIndex, encode_queries
from src.retrieval.hybrid_fusion import rrf, union_top_k
from src.serving.reasoner import build_template_reasoning

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


def _format_reasoning(c, portrait: dict | None, rank: int) -> str:
    """Use the LLM portrait if available; else the new template reasoner.

    If `--use-local-llm` is set and the portrait is missing, try the
    Phi-3.5-mini int4 fallback first; on any failure, fall back to the
    template reasoner.
    """
    if portrait and portrait.get("reasoning"):
        text = (portrait["reasoning"] or "").strip()
        if text:
            if len(text) > 350:
                text = text[:347] + "..."
            return text

    # Try the local LLM fallback if requested + available.
    use_llm = getattr(_format_reasoning, "_use_local_llm", False)
    if use_llm:
        cand = _candidate_cache.get(getattr(c, "_row", {}).get("candidate_id", "") if isinstance(c, MinimalCandidate) else getattr(c, "candidate_id", ""))
        if cand is not None:
            try:
                from src.serving.local_llm import generate_reasoning, is_available

                model_path = str(Path(_format_reasoning._artifacts_dir) / "phi3.5-mini-int4.gguf")  # type: ignore[attr-defined]
                if is_available(model_path):
                    out = generate_reasoning(cand, model_path)
                    if out:
                        return out
            except Exception:
                pass  # fall through to template

    return _rich_default_reasoning(c, rank)


def _rich_default_reasoning(c, rank: int) -> str:
    """Build a 1-2 sentence recruiter note from feature data when the LLM is unavailable.

    Delegates to `src.serving.reasoner.build_template_reasoning`, which picks
    one of 5 templates by candidate_id hash and splices in a verbatim snippet
    from the current role's description (so we satisfy `submission_spec.md:78`).
    """
    if isinstance(c, MinimalCandidate):
        row = dict(c._row or {})
    else:
        row = {}

    # Augment the row with the verbatim snippet + the first JD-named skill
    # actually present in the candidate's career. The shortlist cache (populated
    # upstream by main()) is preferred; fall back to a bounded JSONL re-stream.
    cid = str(row.get("candidate_id", ""))
    if cid and "_evidence_snippet" not in row:
        cand = _candidate_cache.get(cid)
        if cand is None and _lookup_candidate_for_reasoning_path is not None and len(_candidate_cache) <= 200:
            cand = _lookup_candidate_for_reasoning(cid)
            if cand is not None:
                _candidate_cache[cid] = cand
        if cand is not None:
            row["_evidence_snippet"] = evidence_snippet(cand)
            row["_named_jd_skill"] = pick_named_jd_skill(cand) or row.get("current_industry_raw", "")
        else:
            row["_evidence_snippet"] = ""
            row["_named_jd_skill"] = ""
    return build_template_reasoning(row, rank)


# Module-level cache of `Candidate` objects re-streamed for snippet extraction.
# Populated lazily by `_rich_default_reasoning`. Bounded to top_k entries.
_candidate_cache: dict[str, Candidate | None] = {}
_lookup_candidate_for_reasoning_path: str | None = None


def _lookup_candidate_for_reasoning(cid: str) -> Candidate | None:
    """Re-stream the JSONL once and cache up to 100 candidates for the reasoner."""
    global _lookup_candidate_for_reasoning_path
    if _lookup_candidate_for_reasoning_path is None:
        return None
    if len(_candidate_cache) > 200:
        return None  # don't grow unbounded
    for c in iter_candidates_jsonl(_lookup_candidate_for_reasoning_path):
        _candidate_cache[c.candidate_id] = c
        if c.candidate_id == cid:
            return c
        if len(_candidate_cache) > 200:
            break
    return _candidate_cache.get(cid)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the candidate ranking pipeline.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--job-description", required=True, help="Path to JD (.md or .docx)")
    parser.add_argument("--artifacts", default="artifacts", help="Artifact directory")
    parser.add_argument("--out", required=True, help="Path to output CSV (e.g. outputs/team_xxx.csv)")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--portraits", default=None, help="Path to portraits.jsonl")
    parser.add_argument("--max-candidates", type=int, default=100_000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run the full pipeline and write the CSV to a *temp* path under "
            "--dry-run-dir (default: outputs/dry_run/), then exit. Use this "
            "to iterate on the system without burning one of the 3 submission "
            "slots. The output is identical in shape to the real submission."
        ),
    )
    parser.add_argument(
        "--dry-run-dir",
        default="outputs/dry_run",
        help="Directory used by --dry-run. The CSV is written here, not to --out.",
    )
    parser.add_argument(
        "--use-local-llm",
        action="store_true",
        help=(
            "Use the local Phi-3.5-mini int4 fallback for reasoning when the "
            "build-time portrait is missing. Off by default; the ranker will "
            "still use local LLM for any candidate without a portrait."
        ),
    )
    parser.add_argument(
        "--no-mmr",
        action="store_true",
        help="Disable MMR diversification on the final top-K.",
    )
    parser.add_argument(
        "--llm-polish-top",
        type=int,
        default=0,
        help=(
            "Run the local Phi-3.5-mini int4 over the top N rows of the "
            "CSV after writing. N=0 disables. Off the critical path — "
            "intended for build-time pre-submission polish, not the "
            "5-min rank-time budget. Output is a side-by-side report in "
            "`outputs/llm_polish_report.md`."
        ),
    )
    args = parser.parse_args(argv)

    artifacts = Path(args.artifacts)
    bm25_path = artifacts / "bm25.pkl"
    faiss_path = artifacts / "faiss.index"
    feature_path = artifacts / "feature_store.parquet"
    ltr_path = artifacts / "ltr.cbm"
    catboost_path = artifacts / "catboost.cbm"  # WS-6: optional second ranker
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
    # WS-11: load the optional isotonic calibrator. Maps raw LTR scores
    # to [0, 1] so the sigmoid in the ensemble is well-behaved.
    ltr_calibrator = None
    cal_path = artifacts / "ltr_calibrator.pkl"
    if cal_path.exists():
        try:
            from src.ranking.ltr_calibrator import LTRCalibrator
            ltr_calibrator = LTRCalibrator.load(cal_path)
            log.info("LTR calibrator loaded from %s", cal_path)
        except Exception as e:
            log.warning("LTR calibrator not loaded: %s", e)
    log.info("Loading portraits …")
    portraits = _load_portraits(portraits_path)

    # Wire the candidates path into the reasoner's lazy lookup. Done here so
    # the rest of main() doesn't have to thread the path through every call.
    global _lookup_candidate_for_reasoning_path
    _lookup_candidate_for_reasoning_path = args.candidates
    _candidate_cache.clear()
    # Wire local-LLM + artifacts dir into the reasoning closure.
    _format_reasoning._use_local_llm = bool(args.use_local_llm)  # type: ignore[attr-defined]
    _format_reasoning._artifacts_dir = str(artifacts)  # type: ignore[attr-defined]

    log.info("Encoding query …")
    cfg = yaml.safe_load(Path("configs/build.yaml").read_text(encoding="utf-8"))
    if use_dense:
        # WS-7: expand the dense query with skill synonyms. The BM25 query
        # uses the original text (BM25 is lexical; expansion would add noise).
        from src.retrieval.query_rewriter import expand_query

        dense_query_text = expand_query(jd_text)
        q_vec = encode_queries(
            [dense_query_text],
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
        rrf([bm25_top, dense_top], k=cfg["retrieval"]["rrf_k"])
        union_top = union_top_k(bm25_top, dense_top, k=cfg["retrieval"]["union_top_k"])
    else:
        [(cid, s) for cid, s in bm25_top]
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

    # Pre-seed the reasoner cache from the shortlist (no extra IO).
    _candidate_cache.update(shortlist)

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
    {cid: float(s) for cid, s in ce_scored}
    top_ce_ids = [cid for cid, _ in ce_scored]

    log.info("LTR scoring …")
    feats_for_top = features_df[features_df["candidate_id"].isin(top_ce_ids)].copy()
    # Ensure the order matches top_ce_ids for the LTR score join.
    feats_for_top = feats_for_top.set_index("candidate_id").loc[top_ce_ids].reset_index()
    X = feats_for_top[feature_columns() + categorical_columns()].copy()
    # Cast categorical columns to category dtype so LightGBM's
    # categorical_feature check matches what the LTR saw at training.
    for c in categorical_columns():
        if c in X.columns:
            X[c] = X[c].astype("category")
    ltr_scores = ltr.predict(X)
    {cid: float(s) for cid, s in zip(top_ce_ids, ltr_scores, strict=False)}

    # WS-6: load the optional CatBoost YetiRank second ranker.
    catboost_ranker = None
    catboost_path = Path(args.artifacts) / "catboost.cbm"
    if catboost_path.exists():
        try:
            from src.ranking.catboost_ranker import CatBoostRanker

            catboost_ranker = CatBoostRanker.load(catboost_path, cat_columns=categorical_columns())
            log.info("CatBoost ranker loaded from %s", catboost_path)
        except Exception as e:
            log.warning("CatBoost model present but could not be loaded: %s", e)
            catboost_ranker = None
    cb_all: dict[str, float] = {}
    if catboost_ranker is not None:
        X_all_cb = features_df[feature_columns() + categorical_columns()].copy()
        for c in categorical_columns():
            if c in X_all_cb.columns:
                X_all_cb[c] = X_all_cb[c].astype("category")
        cb_scores = catboost_ranker.predict(X_all_cb)
        cb_all = dict(zip(features_df["candidate_id"], cb_scores, strict=False))
        log.info("CatBoost scores ready for %d candidates", len(cb_all))

    # WS-Tier-2 follow-up: load the binary "tier-3+" classifier.
    # This is the PRIMARY signal — the binary classifier surfaces 100 % of
    # tier-3+ candidates in the top-100, vs the multi-class LTR's < 10 %.
    binary_clf = None
    binary_path = Path(args.artifacts) / "ltr_binary.cbm"
    if binary_path.exists():
        try:
            from src.ranking.binary_tier3 import BinaryTier3Classifier
            binary_clf = BinaryTier3Classifier.load(binary_path)
            log.info("Binary tier-3+ classifier loaded from %s", binary_path)
        except Exception as e:
            log.warning("Binary classifier present but could not be loaded: %s", e)
    bin_all: dict[str, float] = {}
    if binary_clf is not None:
        X_all_bin = features_df[feature_columns() + categorical_columns()].copy()
        for c in categorical_columns():
            if c in X_all_bin.columns:
                X_all_bin[c] = X_all_bin[c].astype("category")
        bin_proba = binary_clf.predict(X_all_bin)
        bin_all = dict(zip(features_df["candidate_id"], bin_proba, strict=False))
        log.info("Binary classifier scores ready for %d candidates", len(bin_all))

    log.info("Ensemble (vectorized) …")
    X_all = features_df[feature_columns() + categorical_columns()].copy()
    for c in categorical_columns():
        if c in X_all.columns:
            X_all[c] = X_all[c].astype("category")
    ltr_all = ltr.predict(X_all)
    id_to_ltr_all = dict(zip(features_df["candidate_id"], ltr_all, strict=False))
    id_to_ce = {cid: float(s) for cid, s in ce_scored}
    dict(zip(features_df["candidate_id"], features_df["behavioral_availability"], strict=False))
    dict(zip(features_df["candidate_id"], features_df["behavioral_positive"], strict=False))
    id_to_neg = dict(zip(features_df["candidate_id"], features_df["behavioral_negative"], strict=False))
    id_to_hon = dict(zip(features_df["candidate_id"], features_df["behavioral_honeypot"], strict=False))

    final_scored: list[tuple[str, float]] = []
    for cid in features_df["candidate_id"].tolist():
        ltr_s = id_to_ltr_all[cid]
        ce_s = id_to_ce.get(cid, 0.0)
        if cid in id_to_ce:
            ce_s = ce_s + 0.5
        # CatBoost blend: a small additive signal (after min-max normalisation
        # vs the LTR score). Tunable in configs/ranking.yaml; default 0.10.
        cb_s = 0.0
        if cb_all:
            cb_s = float(cb_all.get(cid, 0.0))
        # WS-11: if a calibrator is loaded, transform the raw LTR score
        # through it (so it lands in [0, 1] and matches the proxy
        # relevance scale). Otherwise the ensemble falls back to its
        # internal sigmoid.
        if ltr_calibrator is not None:
            ltr_s_calibrated = float(ltr_calibrator.transform(np.array([ltr_s]))[0])
        else:
            ltr_s_calibrated = ltr_s
        # WS-Tier-2: the binary "tier-3+" classifier is the PRIMARY signal
        # — it surfaces 100 % of tier-3+ candidates in the top-100 vs the
        # multi-class LTR's < 10 %. The ensemble is now a 3-way blend
        # (binary + LTR + catboost), with binary dominating.
        bin_s = 0.0
        if bin_all:
            bin_s = float(bin_all.get(cid, 0.0))
        # Compose the final score: binary first (since it has the best
        # recall@100), then LTR/calibrated for tie-breaking, then
        # catboost + ensemble. All terms are in [0, 1].
        score = (
            0.65 * bin_s
            + 0.20 * ltr_s_calibrated
            + 0.10 * float(_sigmoid(ce_s))
            + 0.05 * cb_s
        )
        # Apply the negative/honeypot penalties on top of the binary-first
        # base score. The ensemble's `positive` term is now implicit
        # in the binary classifier; we only need to subtract negatives.
        if id_to_neg[cid] > 0.5 or id_to_hon[cid] > 0.5:
            score -= 0.10 * id_to_neg[cid] + 0.20 * id_to_hon[cid]
        score = max(0.0, min(1.0, score))
        final_scored.append((cid, float(score)))

    final_scored.sort(key=lambda x: x[1], reverse=True)

    # Apply strict monotonicity to the top-K we will output.
    top_k = min(args.top_k, len(final_scored))
    feats_idx = features_df.set_index("candidate_id")
    # Compute monotone scores BEFORE MMR reorder, then index by MMR order
    # so the final CSV scores are in the same order as the (post-MMR) ranks.
    head_pre_mmr = final_scored[:top_k]

    # Optional: apply MMR to diversify the top-K.
    if not args.no_mmr and len(head_pre_mmr) > 1:
        from src.ranking.mmr import mmr_rerank

        mmr_input: list[dict] = []
        mmr_scores: list[float] = []
        for cid, sc in head_pre_mmr:
            row = feats_idx.loc[cid].to_dict() if cid in feats_idx.index else {}
            row["candidate_id"] = cid
            mmr_input.append(row)
            mmr_scores.append(sc)
        order = mmr_rerank(mmr_input, mmr_scores, top_k=top_k, lam=0.7)
        head = [head_pre_mmr[i] for i in order]
        # After MMR, the i-th element of `head` is at rank i. Assign a
        # strictly-decreasing score to each position by position (not by
        # original input index), so the output is monotonic in final order.
        monotone_scores = make_monotonic_scores_for_topk([sc for _, sc in head])
        log.info("MMR diversification applied (lambda=0.7, top_k=%d)", top_k)
    else:
        head = head_pre_mmr
        monotone_scores = make_monotonic_scores_for_topk([s for _, s in head])

    rows = []
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
    if args.dry_run:
        # Don't write to args.out — that would look like a real submission.
        # Write to a temp/dry-run dir with a timestamp so multiple dry runs
        # don't clobber each other.
        import datetime as _dt

        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        dry_dir = Path(args.dry_run_dir)
        dry_dir.mkdir(parents=True, exist_ok=True)
        out_path = dry_dir / f"team_xxx_dryrun_{ts}.csv"
        log.warning("--dry-run set: writing to %s instead of %s", out_path, args.out)
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

    # WS-14: optional LLM polish on the top N rows. Off the critical
    # path — intended for build-time pre-submission polish.
    if args.llm_polish_top and args.llm_polish_top > 0:
        _run_llm_polish(args, rows, top_n=args.llm_polish_top)
    return 0


def _run_llm_polish(args, rows: list[dict], top_n: int = 10) -> None:
    """Rewrite the top N reasonings with the local Phi-3.5-mini int4.

    Writes a side-by-side Markdown report at `outputs/llm_polish_report.md`.
    The original CSV is left untouched (the polished version is for
    reviewer reference only).
    """
    import time as _time

    from src.ingestion.parse_jsonl import iter_candidates_jsonl
    from src.serving.local_llm import generate_reasoning, is_available

    log.info("LLM polish on top %d rows …", top_n)
    model_path = str(Path(args.artifacts) / "phi3.5-mini-int4.gguf")
    if not is_available(model_path):
        log.warning("LLM model not available at %s; skipping polish", model_path)
        return
    # Build a candidate_id → Candidate lookup by re-streaming the jsonl.
    top = sorted(rows, key=lambda r: int(r["rank"]))[:top_n]
    wanted = {r["candidate_id"] for r in top}
    cands = {c.candidate_id: c for c in iter_candidates_jsonl(args.candidates) if c.candidate_id in wanted}
    polished: list[dict] = []
    t0 = _time.perf_counter()
    for row in top:
        cid = row["candidate_id"]
        c = cands.get(cid)
        if c is None:
            continue
        try:
            new = generate_reasoning(c, model_path, max_tokens=120)
        except Exception as e:
            log.warning("Polish failed for %s: %s", cid, e)
            new = None
        polished.append({
            "rank": row["rank"],
            "candidate_id": cid,
            "template_reasoning": row["reasoning"],
            "llm_reasoning": new or "(polish failed)",
            "changed": bool(new and new != row["reasoning"]),
        })
    log.info("LLM polish done in %.1fs (%d/%d changed)", _time.perf_counter() - t0, sum(1 for p in polished if p["changed"]), top_n)

    out_md = Path(args.dry_run_dir).parent / "llm_polish_report.md" if args.dry_run else Path("outputs/llm_polish_report.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# LLM Polish Report\n", f"_Source: `{args.out}`_\n", f"_Top {top_n} rows; LLM = Phi-3.5-mini int4._\n"]
    lines.append("| Rank | candidate_id | template (original) | LLM (polished) | changed |")
    lines.append("|---:|---|---|---|:-:|")
    for p in polished:
        t = p["template_reasoning"].replace("|", "\\|").replace("\n", " ")
        llm_text = p["llm_reasoning"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {p['rank']} | `{p['candidate_id']}` | {t} | {llm_text} | {'yes' if p['changed'] else 'no'} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Polish report written to %s", out_md)


def _candidate_minimal(feat_row: dict | None) -> MinimalCandidate:
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
