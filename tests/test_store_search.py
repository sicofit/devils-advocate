from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from devils_advocate import store
from devils_advocate.client import LLMError, ModelNotFound, ModelUnavailable
from devils_advocate.corpus import Decision
from devils_advocate.search import Retriever, VectorIndex, build_vectors, prefixes, vectors_path

from conftest import passage_id
from fakes import DECISIONS, HashEmbedder


def test_fts_query_keeps_content_words_once() -> None:
    query = store.fts_query("Die fristlose Kündigung der Kündigung nach Art. 337 OR, 12 a")
    assert query == '"fristlose" OR "kündigung" OR "art" OR "337" OR "or" OR "12"'
    assert store.fts_query("und oder le la") == ""


def test_fts_query_spells_like_the_court() -> None:
    # The court writes "ss"; FTS5 does not fold "ß".
    assert store.fts_query("Verstoß gemäß") == '"verstoss" OR "gemäss"'
    # Two-letter statutes and single digits are kept, the English "or" is not.
    assert store.fts_query("art. 8 CC") == '"art" OR "8" OR "cc"'
    assert store.fts_query("notice or warning") == '"notice" OR "warning"'


def test_fts_query_is_capped() -> None:
    words = " ".join(f"wort{i}" for i in range(100))
    assert store.fts_query(words).count(" OR ") == store.MAX_QUERY_TERMS - 1


def test_build_records_counts_and_meta(db: store.Store) -> None:
    meta = db.meta()
    assert meta["decisions"] == "3"
    assert meta["passages"] == str(db.count())
    assert meta["revision"] == "test"
    assert db.decision_count() == 3


def test_build_skips_decisions_without_passages(tmp_path: Path) -> None:
    empty = Decision("BGE 1 I 1", "2001-01-01", "de", "Nichts zu segmentieren.", "")
    decisions, _ = store.build(tmp_path / "x.sqlite", [empty, *DECISIONS], {})
    assert decisions == 3


def test_an_empty_build_keeps_the_old_index(db_path: Path) -> None:
    empty = Decision("BGE 1 I 1", "2001-01-01", "de", "Nichts zu segmentieren.", "")
    with pytest.raises(store.IndexProblem, match="old index is kept"):
        store.build(db_path, [empty], {})
    with store.Store(db_path) as db:
        assert db.decision_count() == 3
        db.check()
    assert not db_path.with_name(db_path.name + ".tmp").exists()


def test_an_index_of_another_version_is_refused(db: store.Store) -> None:
    db.check()
    db.db.execute("DELETE FROM meta WHERE key = 'schema'")
    with pytest.raises(store.IndexProblem, match="Run `devil index` again"):
        db.check()


def test_bm25_finds_the_matching_erwaegung(db: store.Store) -> None:
    ids = db.bm25("besonders schwere Verfehlungen trotz Verwarnung", 3)
    assert ids[0] == passage_id(db, "BGE 150 III 100", "E. 2.2")


def test_bm25_ignores_diacritics(db: store.Store) -> None:
    assert db.bm25("Kundigung", 5) == db.bm25("Kündigung", 5) != []


def test_bm25_exclude_drops_a_whole_decision(db: store.Store) -> None:
    ids = db.bm25("fristlose Kündigung", 10, exclude=frozenset({"BGE 150 III 100"}))
    assert ids
    assert {p.reference for p in db.get(ids)} == {"BGE 149 II 200"}


def test_bm25_without_content_words_returns_nothing(db: store.Store) -> None:
    assert db.bm25("und oder", 5) == []


def test_get_keeps_order_and_skips_unknown_ids(db: store.Store) -> None:
    ids = db.ids()
    got = db.get([ids[2], 99999, ids[0]])
    assert [p.id for p in got] == [ids[2], ids[0]]
    assert got[0].language in {"de", "fr"}


@pytest.mark.parametrize("start", [0, 1, 5, 12])
def test_iter_texts_resumes_at_any_position(db: store.Store, start: int) -> None:
    ids = db.ids()
    assert [pid for pid, _ in db.iter_texts(start=start, batch=2)] == ids[start:]


def test_iter_texts_past_the_end_is_empty(db: store.Store) -> None:
    assert list(db.iter_texts(start=db.count() + 5)) == []


# -- dense and hybrid ------------------------------------------------------------


@pytest.fixture
def vectors(db: store.Store, tmp_path: Path) -> VectorIndex:
    path = build_vectors(db, HashEmbedder(), "hash-test", tmp_path / "v.npy", batch=4)
    return VectorIndex.load(path)


def test_vectors_cover_every_passage(db: store.Store, vectors: VectorIndex) -> None:
    assert vectors.ids.tolist() == db.ids()
    assert vectors.matrix.shape == (db.count(), 64)
    assert np.allclose(np.linalg.norm(vectors.matrix, axis=1), 1.0, atol=1e-2)


def test_build_vectors_resumes_after_a_failure(db: store.Store, tmp_path: Path) -> None:
    class Flaky(HashEmbedder):
        def embed(self, texts: list[str]) -> list[list[float]]:
            if self.calls == 2:
                self.calls += 1
                raise LLMError("endpoint went away")
            return super().embed(texts)

    path = tmp_path / "resumed.npy"
    with pytest.raises(LLMError):
        build_vectors(db, Flaky(), "hash-test", path, batch=2, attempts=1)
    assert path.with_name(path.name + ".part").exists()
    assert not path.exists()

    second = HashEmbedder()
    build_vectors(db, second, "hash-test", path, batch=2)
    fresh = build_vectors(db, HashEmbedder(), "hash-test", tmp_path / "fresh.npy", batch=2)
    assert second.calls < HashEmbedder().calls + db.count()  # did not start over
    assert np.array_equal(np.load(path), np.load(fresh))


def test_vectors_belong_to_the_index_they_were_built_from(db_path: Path, tmp_path: Path) -> None:
    path = tmp_path / "v.npy"
    with store.Store(db_path) as db:
        build_vectors(db, HashEmbedder(), "hash-test", path, batch=4)
        assert VectorIndex.load(path).matches(db)
    store.build(db_path, DECISIONS, {"revision": "test"})  # same passages, new build
    with store.Store(db_path) as db:
        assert not VectorIndex.load(path).matches(db)


@pytest.mark.parametrize(
    "leftover",
    [
        None,  # a .part without its .json
        {"model": "other-model", "dims": 64, "count": 0, "index": ""},
        {"model": "hash-test", "dims": 64, "count": 0, "index": "an-earlier-build"},
    ],
)
def test_build_vectors_ignores_a_part_file_from_another_run(
    db: store.Store, tmp_path: Path, leftover: dict | None
) -> None:
    path = tmp_path / "v.npy"
    path.with_name(path.name + ".part").write_bytes(bytes(2 * 64 * 3))  # three rows of zeros
    if leftover is not None:
        leftover["count"] = db.count()
        path.with_name(path.name + ".part.json").write_text(json.dumps(leftover), "utf-8")
    build_vectors(db, HashEmbedder(), "hash-test", path, batch=4)
    fresh = build_vectors(db, HashEmbedder(), "hash-test", tmp_path / "fresh.npy", batch=4)
    assert np.array_equal(np.load(path), np.load(fresh))
    assert VectorIndex.load(path).matches(db)


def test_an_interrupted_embed_does_not_vouch_for_old_vectors(db_path: Path, tmp_path: Path) -> None:
    path = tmp_path / "v.npy"
    with store.Store(db_path) as db:
        build_vectors(db, HashEmbedder(), "hash-test", path, batch=4)
    store.build(db_path, DECISIONS, {"revision": "test"})  # same passages, new build

    class Interrupted(HashEmbedder):
        def embed(self, texts: list[str]) -> list[list[float]]:
            if self.calls == 1:
                raise KeyboardInterrupt
            return super().embed(texts)

    with store.Store(db_path) as db:
        with pytest.raises(KeyboardInterrupt):
            build_vectors(db, Interrupted(), "hash-test", path, batch=4)
        assert not VectorIndex.load(path).matches(db)
        build_vectors(db, HashEmbedder(), "hash-test", path, batch=4)
        assert VectorIndex.load(path).matches(db)


def test_a_failing_embedding_model_can_fall_back_to_bm25(
    db: store.Store, vectors: VectorIndex
) -> None:
    class Broken(HashEmbedder):
        def embed(self, texts: list[str]) -> list[list[float]]:
            raise LLMError("embedding model not found")

    with pytest.raises(LLMError):
        Retriever(db, Broken(), vectors).search(["Verfehlungen"], 3)
    problems: list[str] = []
    retriever = Retriever(db, Broken(), vectors, on_dense_error=problems.append)
    hits = retriever.search(["Verfehlungen"], 3)
    assert hits and all(hit.sources == ("bm25",) for hit in hits)
    assert problems == ["embedding model not found"] and not retriever.has_dense


def test_a_missing_embedding_model_falls_back_but_a_dead_server_does_not(
    db: store.Store, vectors: VectorIndex
) -> None:
    class Missing(HashEmbedder):
        def embed(self, texts: list[str]) -> list[list[float]]:
            raise ModelNotFound("The server has no model 'e' (HTTP 404)")

    class Down(HashEmbedder):
        def embed(self, texts: list[str]) -> list[list[float]]:
            raise ModelUnavailable("No model server answers")

    problems: list[str] = []
    retriever = Retriever(db, Missing(), vectors, on_dense_error=problems.append)
    assert retriever.search(["Verfehlungen"], 3)
    assert problems and not retriever.has_dense
    with pytest.raises(ModelUnavailable):
        Retriever(db, Down(), vectors, on_dense_error=problems.append).search(["Verfehlungen"], 3)


def test_build_vectors_retries_a_failed_batch(
    db: store.Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("devils_advocate.search.time.sleep", lambda _: None)
    embedder = HashEmbedder(fail_first=2)
    build_vectors(db, embedder, "hash-test", tmp_path / "v.npy", batch=100, attempts=3)
    assert embedder.calls == 3


def test_dense_search_excludes_decisions(db: store.Store, vectors: VectorIndex) -> None:
    retriever = Retriever(db, HashEmbedder(), vectors)
    ids = retriever.dense("fristlose Kündigung Verfehlungen", 5, frozenset({"BGE 150 III 100"}))
    assert ids
    assert "BGE 150 III 100" not in {p.reference for p in db.get(ids)}


def test_hybrid_fuses_both_rankings(db: store.Store, vectors: VectorIndex) -> None:
    retriever = Retriever(db, HashEmbedder(), vectors)
    hits = retriever.search(["besonders schweren Verfehlungen"], 3, mode="hybrid")
    assert hits[0].passage.id == passage_id(db, "BGE 150 III 100", "E. 2.2")
    assert hits[0].sources == ("bm25", "dense")
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_bm25_mode_ignores_vectors(db: store.Store, vectors: VectorIndex) -> None:
    retriever = Retriever(db, HashEmbedder(), vectors)
    hits = retriever.search(["besonders schweren Verfehlungen"], 5, mode="bm25")
    assert all(h.sources == ("bm25",) for h in hits)


def test_search_caps_passages_per_decision(retriever: Retriever) -> None:
    queries = ["fristlose Kündigung Arbeitsverhältnis Verfehlungen", "Kündigung Verwarnung"]
    hits = retriever.search(queries, 5, mode="bm25", per_decision=1)
    references = [h.passage.reference for h in hits]
    assert len(references) == len(set(references))


def test_search_without_vectors_falls_back_to_bm25(retriever: Retriever) -> None:
    assert not retriever.has_dense
    assert retriever.search(["Verfehlungen"], 3, mode="hybrid")
    assert retriever.dense_many(["Verfehlungen"], 3) == [[]]


def test_model_prefixes_and_paths() -> None:
    assert prefixes("nomic-embed-text") == ("search_query: ", "search_document: ")
    assert prefixes("unknown-model") == ("", "")
    assert (
        vectors_path(Path("d"), "qwen3-embedding:0.6b").name == "vectors-qwen3-embedding_0.6b.npy"
    )


def test_page_breaks_survive_the_store(tmp_path: Path) -> None:
    text = (
        "Regeste\nKurz.\nErwägungen\n2.\nDie Frist begann mit der Zustellung\n"
        "BGE 141 III 200 S. 201\nund endete zehn Tage später.\n"
    )
    decision = Decision("BGE 141 III 200", "2015-01-01", "de", text, "")
    foreign = Decision("BGE 141 III 300", "2015-01-01", "de", text, "")
    decisions, _ = store.build(tmp_path / "x.sqlite", [decision, foreign], {})
    assert decisions == 1  # the second one carries the page markers of another decision
    with store.Store(tmp_path / "x.sqlite") as db:
        [_, two] = db.passages_of("BGE 141 III 200")
    assert two.breaks == ((two.text.index("und endete"), 201),)
    assert two.citation_of_span(0, len(two.text)) == "BGE 141 III 200 E. 2 S. 200 f."
    with sqlite3.connect(tmp_path / "x.sqlite") as raw:
        stored = [row[0] for row in raw.execute("SELECT breaks FROM passages ORDER BY id")]
    assert stored == [None, f"[[{two.breaks[0][0]}, 201]]"]  # NULL where the page never changes
