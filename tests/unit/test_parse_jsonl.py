from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ingestion.parse_jsonl import iter_candidates_jsonl
from src.ingestion.schema_validator import validate_against_schema


DATA_DIR = Path(__file__).resolve().parents[3] / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge"


def test_iter_sample_candidates():
    path = DATA_DIR / "sample_candidates.json"
    cs = list(iter_candidates_jsonl(path))
    assert len(cs) > 0
    assert all(c.candidate_id.startswith("CAND_") for c in cs)


def test_validate_against_schema_passes_sample():
    path = DATA_DIR / "sample_candidates.json"
    schema_path = DATA_DIR / "candidate_schema.json"
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    cs = list(iter_candidates_jsonl(path))
    errors = validate_against_schema(cs, schema)
    assert not any("bad candidate_id" in e for e in errors)
