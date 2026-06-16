"""Streaming JSONL parser for the candidate pool.

Designed to be cheap (no full materialisation unless the caller asks) and
strict (raises on malformed JSON or duplicate IDs).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import IO

from src.api.schemas import Candidate


def iter_candidates_jsonl(path: str | Path) -> Iterator[Candidate]:
    """Yield validated Candidate objects from a JSONL file.

    Also handles a JSON array file (the pretty-printed `sample_candidates.json`
    from the challenge bundle) by reading it whole and yielding each element.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("r", encoding="utf-8") as f:
        head = f.read(1)
    if head == "[":
        return _iter_from_json_array(p)
    return _iter_from_handle(p.open("r", encoding="utf-8"))


def _iter_from_json_array(path: Path) -> Iterator[Candidate]:
    import json

    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        objs = json.load(f)
    if not isinstance(objs, list):
        raise ValueError(f"{path}: expected a JSON array of candidate objects")
    for i, obj in enumerate(objs, 1):
        cand = Candidate.model_validate(obj)
        if cand.candidate_id in seen:
            raise ValueError(f"Duplicate candidate_id at index {i}: {cand.candidate_id}")
        seen.add(cand.candidate_id)
        yield cand


def _iter_from_handle(handle: IO[str]) -> Iterator[Candidate]:
    seen: set[str] = set()
    for line_no, line in enumerate(handle, 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON on line {line_no}: {e}") from e
        cand = Candidate.model_validate(obj)
        if cand.candidate_id in seen:
            raise ValueError(f"Duplicate candidate_id on line {line_no}: {cand.candidate_id}")
        seen.add(cand.candidate_id)
        yield cand


def load_candidates_jsonl(path: str | Path) -> list[Candidate]:
    """Load the full pool as a list. Use sparingly for ≤ ~50 k rows."""
    return list(iter_candidates_jsonl(path))


def count_candidates_jsonl(path: str | Path) -> int:
    """Count records in a JSONL without parsing them."""
    p = Path(path)
    n = 0
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            n += chunk.count(b"\n")
    return n
