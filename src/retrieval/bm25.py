"""BM25 sparse retriever over the deep profile corpus.

Uses the `rank_bm25` library. We keep the index in memory (one corpus, ~500
MB on disk for tokenised docs) so ranking stays under the 5-min budget.
"""

from __future__ import annotations

import pickle
import re
from collections.abc import Iterable
from pathlib import Path

from rank_bm25 import BM25Okapi

DEFAULT_TOKEN_PATTERN = re.compile(r"(?u)\b\w[a-zA-Z0-9+#.-]{1,}\b")


def tokenize(text: str, pattern: re.Pattern[str] = DEFAULT_TOKEN_PATTERN) -> list[str]:
    if not text:
        return []
    return pattern.findall(text.lower())


class BM25Index:
    """A serialisable BM25Okapi wrapper."""

    def __init__(self, bm25: BM25Okapi, doc_ids: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.bm25 = bm25
        self.doc_ids = doc_ids
        self.k1 = k1
        self.b = b

    def query(self, text: str, top_k: int = 500) -> list[tuple[str, float]]:
        toks = tokenize(text)
        if not toks:
            return []
        scores = self.bm25.get_scores(toks)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.doc_ids[i], float(s)) for i, s in ranked if s > 0]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(
                {
                    "bm25": self.bm25,
                    "doc_ids": self.doc_ids,
                    "k1": self.k1,
                    "b": self.b,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        return cls(obj["bm25"], obj["doc_ids"], obj["k1"], obj["b"])


def build_bm25(
    docs: Iterable[tuple[str, str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> BM25Index:
    """Build the BM25 index from an iterable of (doc_id, text) tuples."""
    doc_ids: list[str] = []
    tokenized: list[list[str]] = []
    for doc_id, text in docs:
        doc_ids.append(doc_id)
        tokenized.append(tokenize(text))
    bm25 = BM25Okapi(tokenized, k1=k1, b=b)
    return BM25Index(bm25, doc_ids, k1=k1, b=b)
