"""Schema-level validation for the candidate pool.

This is a defense-in-depth check on top of Pydantic: it verifies that the
declared `candidate_schema.json` from the challenge bundle is satisfied, and
flags candidate_id format violations.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

from jsonschema import Draft7Validator

from src.api.schemas import Candidate

CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")


def load_schema(path: str | Path) -> dict:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_against_schema(candidates: Iterable[Candidate], schema: dict) -> list[str]:
    """Return a list of human-readable validation errors (empty if all good)."""
    validator = Draft7Validator(schema)
    errors: list[str] = []
    for c in candidates:
        if not CANDIDATE_ID_PATTERN.match(c.candidate_id):
            errors.append(f"bad candidate_id format: {c.candidate_id}")
        for err in validator.iter_errors(c.model_dump()):
            errors.append(
                f"{c.candidate_id}: {err.message} at {'/'.join(str(x) for x in err.absolute_path)}"
            )
    return errors
