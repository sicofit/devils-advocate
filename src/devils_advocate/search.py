"""Retrieval: BM25 over FTS5, dense vectors, and reciprocal-rank fusion of both."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .client import Embedder, LLMError
from .store import Store, StoredPassage

RRF_K = 60

# Retrieval instructions some embedding models were trained with: (query, document).
PREFIXES = {
    "nomic-embed": ("search_query: ", "search_document: "),
    "qwen3-embedding": (
        "Instruct: Given a legal argument, retrieve passages of Swiss Federal Supreme "
        "Court decisions that address it\nQuery: ",
        "",
    ),
    "e5": ("query: ", "passage: "),
}


def prefixes(model: str) -> tuple[str, str]:
    for key, value in PREFIXES.items():
        if key in model:
            return value
    return "", ""


def vectors_path(data_dir: Path, model: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-." else "_" for c in model)
    return data_dir / f"vectors-{safe}.npy"


class VectorIndex:
    """Unit-normalised passage embeddings, one row per passage id, in id order."""

    def __init__(self, ids: np.ndarray, matrix: np.ndarray, model: str, index: str = "") -> None:
        self.ids = ids
        self.matrix = matrix
        self.model = model
        self.index = index  # build_id of the index the passages were embedded from

    @classmethod
    def load(cls, path: Path) -> VectorIndex:
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        # Stored as float16 to halve the file; searched as float32 because numpy
        # has no fast float16 matrix product.
        matrix = np.load(path).astype(np.float32)
        ids = np.load(path.with_suffix(".ids.npy"))
        return cls(ids, matrix, meta["model"], meta.get("index", ""))

    def matches(self, store: Store) -> bool:
        """Whether these vectors were embedded from the passages now in `store`."""
        return self.index == store.meta().get("build_id") and len(self.ids) == store.count()

    def search_many(
        self, queries: np.ndarray, k: int, exclude: Sequence[np.ndarray] | None = None
    ) -> list[list[int]]:
        """Top-k passage ids for each row of `queries`, skipping each row's excluded ids."""
        scores = queries.astype(np.float32) @ self.matrix.T
        k = min(k, scores.shape[1])
        results: list[list[int]] = []
        for row, row_scores in enumerate(scores):
            if exclude is not None and len(exclude[row]):
                row_scores[np.searchsorted(self.ids, exclude[row])] = -np.inf
            top = np.argpartition(-row_scores, k - 1)[:k]
            top = top[np.argsort(-row_scores[top])]
            results.append([int(self.ids[i]) for i in top if np.isfinite(row_scores[i])])
        return results


def build_vectors(
    store: Store,
    embedder: Embedder,
    model: str,
    path: Path,
    *,
    batch: int = 8,
    attempts: int = 5,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Embed every passage into `path`, resuming from a previous partial run.

    A failed batch is retried `attempts` times before giving up; an interrupted
    run continues where it stopped the next time it is started.
    """
    _, doc_prefix = prefixes(model)
    total = store.count()
    index = store.meta().get("build_id", "")
    partial = path.with_name(path.name + ".part")
    meta_path = path.with_suffix(".json")
    dims = None
    done = 0
    if partial.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        same_run = (meta.get("model"), meta.get("count"), meta.get("index")) == (
            model,
            total,
            index,
        )
        if same_run and meta.get("dims"):
            dims = meta["dims"]
            done = partial.stat().st_size // (2 * dims)
            with partial.open("r+b") as fh:  # drop a half-written row
                fh.truncate(done * 2 * dims)
        else:  # left over from another model or another index: start again
            partial.unlink()

    texts: list[str] = []
    with partial.open("ab") as out:

        def flush() -> None:
            nonlocal dims, done
            rows = np.asarray(_embed_with_retry(embedder, texts, attempts), dtype=np.float32)
            rows /= np.linalg.norm(rows, axis=1, keepdims=True) + 1e-12
            if dims is None:
                dims = rows.shape[1]
                meta_path.write_text(
                    json.dumps({"model": model, "dims": dims, "count": total, "index": index}),
                    encoding="utf-8",
                )
            out.write(rows.astype(np.float16).tobytes())
            out.flush()
            done += len(texts)
            texts.clear()
            if on_progress:
                on_progress(done, total)

        for _, text in store.iter_texts(start=done):
            texts.append(doc_prefix + text)
            if len(texts) == batch:
                flush()
        if texts:
            flush()

    assert dims is not None, "the store has no passages"
    matrix = np.fromfile(partial, dtype=np.float16).reshape(-1, dims)
    ids = np.asarray(store.ids(), dtype=np.int64)
    if len(ids) != len(matrix):
        raise RuntimeError(f"embedded {len(matrix)} passages but the index has {len(ids)}")
    np.save(path, matrix)
    np.save(path.with_suffix(".ids.npy"), ids)
    partial.unlink()
    return path


def _embed_with_retry(embedder: Embedder, texts: list[str], attempts: int) -> list[list[float]]:
    for attempt in range(1, attempts + 1):
        try:
            return embedder.embed(texts)
        except LLMError:
            if attempt == attempts:
                raise
            time.sleep(min(2.0**attempt, 30.0))
    raise AssertionError("unreachable")


@dataclass(slots=True)
class Hit:
    passage: StoredPassage
    score: float
    sources: tuple[str, ...]  # which retrievers found it: "bm25", "dense"


class Retriever:
    def __init__(
        self,
        store: Store,
        embedder: Embedder | None = None,
        vectors: VectorIndex | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.vectors = vectors

    @property
    def has_dense(self) -> bool:
        return self.embedder is not None and self.vectors is not None

    def bm25(self, query: str, k: int, exclude: frozenset[str] = frozenset()) -> list[int]:
        return self.store.bm25(query, k, exclude)

    def dense(self, query: str, k: int, exclude: frozenset[str] = frozenset()) -> list[int]:
        return self.dense_many([query], k, exclude)[0]

    def embed_queries(self, queries: Sequence[str]) -> np.ndarray:
        assert self.embedder is not None and self.vectors is not None
        query_prefix, _ = prefixes(self.vectors.model)
        vectors = np.asarray(
            self.embedder.embed([query_prefix + q for q in queries]), dtype=np.float32
        )
        return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)

    def dense_many(
        self, queries: Sequence[str], k: int, exclude: frozenset[str] = frozenset()
    ) -> list[list[int]]:
        if not self.has_dense or not queries:
            return [[] for _ in queries]
        assert self.vectors is not None
        excluded = np.asarray(self.store.ids_of(exclude), dtype=np.int64)
        return self.vectors.search_many(self.embed_queries(queries), k, [excluded] * len(queries))

    def search(
        self,
        queries: Sequence[str],
        k: int,
        *,
        mode: str = "hybrid",
        exclude: frozenset[str] = frozenset(),
        per_decision: int | None = None,
        depth: int = 50,
    ) -> list[Hit]:
        """Fuse the rankings of every query and retriever with reciprocal-rank fusion."""
        rankings: list[tuple[str, list[int]]] = []
        if mode in ("bm25", "hybrid"):
            rankings += [("bm25", self.bm25(q, depth, exclude)) for q in queries]
        if mode in ("dense", "hybrid") and self.has_dense:
            rankings += [("dense", ids) for ids in self.dense_many(queries, depth, exclude)]

        scores: dict[int, float] = {}
        sources: dict[int, set[str]] = {}
        for name, ids in rankings:
            for rank, pid in enumerate(ids):
                scores[pid] = scores.get(pid, 0.0) + 1.0 / (RRF_K + rank + 1)
                sources.setdefault(pid, set()).add(name)
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)

        hits: list[Hit] = []
        per: dict[str, int] = {}
        for passage in self.store.get(ranked[: k * 4 if per_decision else k]):
            if per_decision is not None:
                if per.get(passage.reference, 0) >= per_decision:
                    continue
                per[passage.reference] = per.get(passage.reference, 0) + 1
            hits.append(Hit(passage, scores[passage.id], tuple(sorted(sources[passage.id]))))
            if len(hits) == k:
                break
        return hits
