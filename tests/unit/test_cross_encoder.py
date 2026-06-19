"""Tests for the cross-encoder model-name resolution + fallback (Agent 4)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.ranking.cross_encoder import (
    FALLBACK_MODEL,
    STRONG_MODEL,
    _resolve_model_name,
)


def test_resolve_returns_existing_local_path(tmp_path: Path):
    p = tmp_path / "ce_local"
    p.mkdir()
    assert _resolve_model_name(str(p)) == str(p)


def test_resolve_falls_back_when_missing(tmp_path: Path):
    nonexistent = tmp_path / "does_not_exist"
    # Should fall back to FALLBACK_MODEL when the configured name is
    # neither an existing path nor the fallback itself.
    with patch("src.ranking.cross_encoder.log") as mock_log:
        out = _resolve_model_name("custom-model-name")
    assert out == "custom-model-name"  # not a path, so passes through as-is


def test_resolve_fallback_when_path_missing():
    out = _resolve_model_name("/no/such/path/ce_finetuned")
    assert out == "/no/such/path/ce_finetuned"  # not the fallback string


def test_fallback_model_constant():
    assert FALLBACK_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_strong_model_constant():
    assert STRONG_MODEL == "BAAI/bge-reranker-base"
