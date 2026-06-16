"""Dense retriever: encode the deep profile corpus with BGE-large-en-v1.5 and
build a faiss HNSW index.

The model is loaded once and reused for both indexing (build) and querying
(ranking). Quantization to int8 is supported but optional; for 100 k vectors at
1024 d, raw float32 is ~400 MB and fits the 5 GB cap with plenty of headroom.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterable
from pathlib import Path

import numpy as np


class DenseIndex:
    """A wrapper over a faiss index + an aligned id list.

    The wrapper handles persistence, querying, and the (de)serialization of the
    id list alongside the index.
    """

    def __init__(self, index, doc_ids: list[str], dim: int) -> None:
        self.index = index
        self.doc_ids = doc_ids
        self.dim = dim

    def query(self, query_vec: np.ndarray, top_k: int = 500) -> list[tuple[str, float]]:
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype(np.float32)
        scores, idx = self.index.search(query_vec, top_k)
        out: list[tuple[str, float]] = []
        for s, i in zip(scores[0].tolist(), idx[0].tolist()):
            if i < 0 or i >= len(self.doc_ids):
                continue
            out.append((self.doc_ids[i], float(s)))
        return out

    def save(self, path: str | Path) -> None:
        import faiss  # local import: faiss is heavy

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p))
        meta = p.with_suffix(".ids.pkl")
        with meta.open("wb") as f:
            pickle.dump({"doc_ids": self.doc_ids, "dim": self.dim}, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "DenseIndex":
        import faiss

        p = Path(path)
        meta = p.with_suffix(".ids.pkl")
        with meta.open("rb") as f:
            obj = pickle.load(f)
        idx = faiss.read_index(str(p))
        return cls(idx, obj["doc_ids"], obj["dim"])


def build_hnsw(
    embeddings: np.ndarray,
    doc_ids: list[str],
    M: int = 32,
    ef_construction: int = 200,
    ef_search: int = 64,
    use_inner_product: bool = True,
) -> DenseIndex:
    """Build an HNSW index from pre-computed embeddings.

    If `use_inner_product` is True (the default), embeddings should be L2-
    normalised. We use cosine similarity == inner product on the unit sphere.
    """
    import faiss

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    dim = embeddings.shape[1]
    if use_inner_product:
        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    else:
        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_L2)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(embeddings)  # type: ignore[attr-defined]
    return DenseIndex(index, doc_ids, dim)


def normalise_l2(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return x / norms


def encode_corpus(
    texts: Iterable[str],
    model_name: str = "BAAI/bge-large-en-v1.5",
    batch_size: int = 32,
    max_seq_length: int = 512,
    device: str = "cpu",
    cache_dir: str | None = None,
    normalize: bool = True,
    show_progress: bool = False,
) -> np.ndarray:
    """Encode an iterable of texts with a sentence-transformers model.

    Quantization is intentionally not applied here. The build phase may choose
    to do it downstream; for the ranking step we keep the model in float32 to
    stay accurate.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device, cache_folder=cache_dir)
    model.max_seq_length = max_seq_length
    vecs = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    if not isinstance(vecs, np.ndarray):
        vecs = np.asarray(vecs, dtype=np.float32)
    return vecs.astype(np.float32, copy=False)


def encode_queries(
    queries: list[str],
    model_name: str = "BAAI/bge-large-en-v1.5",
    batch_size: int = 32,
    max_seq_length: int = 512,
    device: str = "cpu",
    cache_dir: str | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Encode query texts. Uses the BGE "query:" prefix when relevant."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device, cache_folder=cache_dir)
    model.max_seq_length = max_seq_length
    if "bge" in model_name.lower():
        queries = ["Represent this sentence for searching relevant passages: " + q for q in queries]
    vecs = model.encode(
        queries,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    if not isinstance(vecs, np.ndarray):
        vecs = np.asarray(vecs, dtype=np.float32)
    return vecs.astype(np.float32, copy=False)
