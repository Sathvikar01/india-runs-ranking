"""Unit tests for hybrid fusion (RRF / weighted)."""

from __future__ import annotations

from src.retrieval.hybrid_fusion import rrf, union_top_k, weighted


def test_rrf_basic():
    a = [("x", 1.0), ("y", 0.5), ("z", 0.2)]
    b = [("y", 0.9), ("x", 0.4), ("w", 0.1)]
    out = rrf([a, b], k=10)
    # x and y should both be top, x slightly higher because it appeared at rank 0 in both
    assert out[0][0] in ("x", "y")
    assert all(s > 0 for _, s in out)


def test_weighted_basic():
    a = [("x", 1.0), ("y", 0.5)]
    b = [("y", 0.9), ("x", 0.4)]
    out = weighted([a, b], weights=[0.5, 0.5])
    assert out[0][0] in ("x", "y")
    assert all(s >= 0 and s <= 1 for _, s in out)


def test_union_top_k():
    a = [("x", 1.0), ("y", 0.5)]
    b = [("z", 0.7), ("x", 0.2)]
    out = union_top_k(a, b, k=3)
    assert {cid for cid, _ in out} == {"x", "y", "z"}


def test_empty_inputs():
    assert rrf([]) == []
    assert weighted([]) == []
    assert union_top_k() == []
