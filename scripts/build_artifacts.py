"""Offline build phase: produce all artifacts the ranker consumes.

Steps:
  1. Stream candidates.jsonl, build `deep_profile` text per candidate.
  2. Build BM25 (rank_bm25) over `deep_profile` corpus.
  3. Encode `deep_profile` with BGE-large-en-v1.5 → dense vectors.
  4. Build faiss HNSW index over the dense vectors.
  5. Compute the per-candidate feature table → artifacts/feature_store.parquet.
  6. Train the LTR model on the proxy ground truth → artifacts/ltr.cbm.
  7. (Optional) Generate per-candidate recruiter reasoning via Zenmux MiMo v2.5
     → artifacts/portraits.jsonl.

Run with:
    python scripts/build_artifacts.py \
        --candidates data/raw/candidates.jsonl \
        --job-description data/raw/job_description.md \
        --out artifacts

Wall-clock on a 16 GB CPU laptop: ~ 1.5-2.5 h for the first build (mostly
embedding generation and reasoning generation).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api.schemas import Candidate
from src.behavioral.honeypot import honeypot_risk
from src.evaluation.proxy_ground_truth import proxy_relevance
from src.ingestion.parse_jsonl import count_candidates_jsonl, iter_candidates_jsonl
from src.preprocessing.deep_profile import build_deep_profile
from src.preprocessing.feature_engineer import (
    build_features,
    feature_columns,
    features_to_dataframe,
)
from src.retrieval.bm25 import build_bm25
from src.retrieval.dense_index import build_hnsw, encode_corpus
from src.training.train_ltr import train_ltr

log = logging.getLogger("build")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _save_portraits_incremental(portraits: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for p in portraits:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def _generate_reasoning(
    candidates: list[Candidate],
    jd_text: str,
    cfg: dict,
    out_path: Path,
    resume: bool = True,
) -> int:
    """Generate per-candidate reasoning via Zenmux MiMo v2.5. Network required."""
    from openai import OpenAI

    base_url = cfg.get("base_url", "https://api.zenmux.ai/v1")
    api_key = os.environ.get(cfg.get("api_key_env", "ZENMUX_API_KEY"))
    if not api_key:
        log.warning("No Zenmux API key found in env %s. Skipping reasoning generation.", cfg.get("api_key_env"))
        return 0
    model = cfg.get("model", "xiaomi/mimo-v2-5")
    client = OpenAI(base_url=base_url, api_key=api_key)
    prompts = yaml.safe_load(Path("configs/reasoning_prompts.yaml").read_text(encoding="utf-8"))
    user_template = prompts["user_template"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    already: set[str] = set()
    if resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    already.add(json.loads(line)["candidate_id"])
                except (json.JSONDecodeError, KeyError):
                    continue

    todo = [c for c in candidates if c.candidate_id not in already]
    log.info("Reasoning generation: %d new, %d already done.", len(todo), len(already))
    if not todo:
        return 0

    n_done = 0
    n_failed = 0
    for c in todo:
        prompt_user = user_template.format(
            job_description=jd_text[:6000],
            candidate_id=c.candidate_id,
            current_title=c.profile.current_title or "",
            current_company=c.profile.current_company or "",
            current_industry=c.profile.current_industry or "",
            years_of_experience=c.profile.years_of_experience,
            location=c.profile.location or "",
            headline=c.profile.headline or "",
            summary=c.profile.summary or "",
            n_skills=len(c.skills),
            skills=", ".join(s.name for s in c.skills[:30]),
            career_roles="\n".join(
                f"  - {r.title} at {r.company}: {r.description[:300]}"
                for r in c.career_history[:5]
            ),
            education=", ".join(
                f"{e.degree} {e.field_of_study} @ {e.institution} ({e.tier or 'unknown'})"
                for e in c.education
            ),
            open_to_work=c.redrob_signals.open_to_work_flag,
            recruiter_response_rate=c.redrob_signals.recruiter_response_rate,
            last_active=c.redrob_signals.last_active_date,
            notice_period=c.redrob_signals.notice_period_days,
            willing_to_relocate=c.redrob_signals.willing_to_relocate,
            github_activity=c.redrob_signals.github_activity_score,
            ai_relevance=round(proxy_relevance(c) / 4.0, 3),
            honeypot_risk=round(honeypot_risk(c), 3),
            rank_bucket=("top" if proxy_relevance(c) >= 3 else "mid" if proxy_relevance(c) >= 2 else "low" if proxy_relevance(c) >= 1 else "exclude"),
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompts["system"]},
                    {"role": "user", "content": prompt_user},
                ],
                max_tokens=int(cfg.get("max_tokens", 160)),
                temperature=float(cfg.get("temperature", 0.2)),
                top_p=float(cfg.get("top_p", 0.95)),
                timeout=float(cfg.get("timeout_s", 60)),
            )
            content = resp.choices[0].message.content or ""
            obj: dict
            try:
                obj = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract the first JSON object from the content.
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        obj = json.loads(content[start : end + 1])
                    except json.JSONDecodeError:
                        obj = {"reasoning": content.strip()[:600]}
                else:
                    obj = {"reasoning": content.strip()[:600]}
            rec = {
                "candidate_id": c.candidate_id,
                "reasoning": obj.get("reasoning", "").strip(),
                "top_positive": obj.get("top_positive"),
                "top_concern": obj.get("top_concern"),
            }
            _save_portraits_incremental([rec], out_path)
            n_done += 1
        except Exception as e:
            log.warning("Zenmux call failed for %s: %s", c.candidate_id, e)
            n_failed += 1
            # Write a placeholder so we don't loop on the same candidate.
            _save_portraits_incremental(
                [{"candidate_id": c.candidate_id, "reasoning": "", "top_positive": None, "top_concern": None}],
                out_path,
            )
        if (n_done + n_failed) % 50 == 0:
            log.info("Reasoning: %d done, %d failed.", n_done, n_failed)
    return n_done


def main() -> int:
    parser = argparse.ArgumentParser(description="Build offline artifacts for the ranker.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--job-description", required=True)
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-bm25", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-ltr", action="store_true")
    parser.add_argument("--skip-reasoning", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=0,
                        help="If >0, only process the first N candidates (smoke test).")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(Path("configs/build.yaml").read_text(encoding="utf-8"))
    llm_cfg = yaml.safe_load(Path("configs/llm.yaml").read_text(encoding="utf-8"))["llm"]

    log.info("Counting candidates in %s …", args.candidates)
    total = count_candidates_jsonl(args.candidates)
    log.info("Total candidates: %d", total)

    t_total = time.perf_counter()
    candidates: list[Candidate] = []
    deep_profiles: list[str] = []
    log.info("Streaming candidates …")
    for i, c in enumerate(iter_candidates_jsonl(args.candidates), 1):
        if args.max_candidates and i > args.max_candidates:
            break
        candidates.append(c)
        deep_profiles.append(build_deep_profile(c))
    log.info("Loaded %d candidates in %.1fs", len(candidates), time.perf_counter() - t_total)

    if not args.skip_bm25:
        log.info("Building BM25 index …")
        t0 = time.perf_counter()
        bm25 = build_bm25(
            zip([c.candidate_id for c in candidates], deep_profiles, strict=False),
            k1=cfg["bm25"]["k1"],
            b=cfg["bm25"]["b"],
        )
        bm25.save(out / "bm25.pkl")
        log.info("BM25 done in %.1fs.", time.perf_counter() - t0)

    if not args.skip_embeddings:
        log.info("Encoding %d deep profiles with %s …", len(deep_profiles), cfg["embedding"]["model_name"])
        t0 = time.perf_counter()
        emb_path = out / "embeddings.npz"
        if emb_path.exists():
            log.info("Embeddings already exist at %s, skipping encode.", emb_path)
        else:
            vecs = encode_corpus(
                deep_profiles,
                model_name=cfg["embedding"]["model_name"],
                batch_size=cfg["embedding"]["batch_size"],
                max_seq_length=cfg["embedding"]["max_seq_length"],
                device=cfg["embedding"]["device"],
                cache_dir=cfg["embedding"]["cache_dir"],
                normalize=cfg["embedding"]["normalize"],
                show_progress=True,
            )
            np.savez_compressed(emb_path, vectors=vecs, ids=np.array([c.candidate_id for c in candidates]))
            log.info("Embeddings done in %.1fs (shape=%s).", time.perf_counter() - t0, vecs.shape)

            log.info("Building faiss HNSW index …")
            t0 = time.perf_counter()
            faiss_index = build_hnsw(
                vecs,
                [c.candidate_id for c in candidates],
                M=cfg["retrieval"]["faiss_hnsw_M"],
                ef_construction=cfg["retrieval"]["faiss_ef_construction"],
                ef_search=cfg["retrieval"]["faiss_ef_search"],
            )
            faiss_index.save(out / "faiss.index")
            log.info("faiss done in %.1fs.", time.perf_counter() - t0)

    if not args.skip_features:
        log.info("Building feature table …")
        t0 = time.perf_counter()
        feats = features_to_dataframe([build_features(c) for c in candidates])
        feat_path = out / "feature_store.parquet"

        log.info("Adding behavioral score columns (avail, positive, negative, honeypot) …")
        from src.behavioral.availability import availability_score_df
        from src.behavioral.honeypot import honeypot_risk_df
        from src.behavioral.jd_filters import negative_penalty_df, positive_boost_df

        feats["behavioral_availability"] = availability_score_df(feats)
        feats["behavioral_positive"] = positive_boost_df(feats)
        feats["behavioral_negative"] = negative_penalty_df(feats)
        feats["behavioral_honeypot"] = honeypot_risk_df(feats)
        feats.to_parquet(feat_path, index=False)
        log.info("Features done in %.1fs → %s", time.perf_counter() - t0, feat_path)

    if not args.skip_ltr:
        log.info("Training LTR …")
        t0 = time.perf_counter()
        cv = train_ltr(
            candidates_path=args.candidates,
            feature_parquet=str(out / "feature_store.parquet"),
            out_model=str(out / "ltr.cbm"),
            k_folds=int(cfg.get("k_folds", 5)),
            num_boost_round=int(cfg.get("num_boost_round", 600)),
        )
        log.info("LTR done in %.1fs. summary=%s", time.perf_counter() - t0, json.dumps(cv))

    if not args.skip_reasoning:
        log.info("Generating reasoning portraits via Zenmux MiMo v2.5 …")
        t0 = time.perf_counter()
        jd_text = Path(args.job_description).read_text(encoding="utf-8")
        n = _generate_reasoning(
            candidates,
            jd_text,
            llm_cfg,
            out / "portraits.jsonl",
        )
        log.info("Reasoning generation done: %d in %.1fs", n, time.perf_counter() - t0)

    log.info("All artifacts written under %s. Total wall-clock: %.1fs", out, time.perf_counter() - t_total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
