from __future__ import annotations

import json
from pathlib import Path

import pytest

from devils_advocate import bench
from devils_advocate.client import LLMError
from devils_advocate.search import Retriever
from devils_advocate.store import Store

from conftest import passage_id
from fakes import FakeModel


def test_sentences_ignore_abbreviations_and_parentheses() -> None:
    text = (
        "Nach Art. 337 Abs. 1 OR ist dies zulässig (vgl. BGE 1 I 1 E. 2. Dort anders). "
        "Das gilt auch hier. Zweiter Absatz\nDritter Satz"
    )
    assert list(bench.sentences(text)) == [
        "Nach Art. 337 Abs. 1 OR ist dies zulässig (vgl. BGE 1 I 1 E. 2. Dort anders).",
        "Das gilt auch hier.",
        "Zweiter Absatz",
        "Dritter Satz",
    ]


def test_words_ending_like_an_abbreviation_still_end_a_sentence() -> None:
    # "n." and "f." are abbreviations, but "décision." and "Bedarf." end sentences.
    text = "Elle a motivé sa décision. Il suffit. Es besteht kein Bedarf. Vgl. N. 3 f. Ende"
    assert list(bench.sentences(text)) == [
        "Elle a motivé sa décision.",
        "Il suffit.",
        "Es besteht kein Bedarf.",
        "Vgl. N. 3 f. Ende",
    ]


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        (
            "Der Anspruch verjährt (BGE 137 III 303 E. 2.1.1 S. 305 mit Hinweisen).",
            "Der Anspruch verjährt.",
        ),
        (
            "Wie in BGE 142 III 579 E. 4.2 ausgeführt, gilt dies (vgl. ATF 140 II 1 consid. 3).",
            "Wie in ausgeführt, gilt dies.",
        ),
        ("Ohne Zitat bleibt der Satz , wie er ist.", "Ohne Zitat bleibt der Satz, wie er ist."),
    ],
)
def test_strip_citations(sentence: str, expected: str) -> None:
    assert bench.strip_citations(sentence) == expected


def test_citation_pattern_reads_the_pinpoint() -> None:
    found = [
        m.groups()
        for m in bench.CITATION.finditer(
            "BGE 120 II 50 E. 3b/aa; ATF 149 II 200 consid. 4.1 p. 203; BGE 96 I 12 Erw. 2"
        )
    ]
    assert found == [
        ("120", "II", "50", "3b/aa"),
        ("149", "II", "200", "4.1"),
        ("96", "I", "12", "2"),
    ]


def test_pinpoint_parts() -> None:
    assert bench.pinpoint_parts("E. 3b/aa") == ("3", "b", "aa")
    assert bench.pinpoint_parts("E. 2.3.3") == ("2", "3", "3")
    assert bench.pinpoint_parts("E.") == ()


def test_target_passages_inside_and_ancestor(db: Store) -> None:
    three = bench.target_passages(db, "BGE 120 II 50", "E. 3")
    assert len(three) == 4  # 3a, 3b, 3b/aa, 3b/bb
    assert bench.target_passages(db, "BGE 120 II 50", "E. 3b") == tuple(
        passage_id(db, "BGE 120 II 50", p) for p in ("E. 3b", "E. 3b/aa", "E. 3b/bb")
    )
    # A deeper Erwägung than the index holds falls back to its closest ancestor.
    assert bench.target_passages(db, "BGE 150 III 100", "E. 2.1.4") == (
        passage_id(db, "BGE 150 III 100", "E. 2.1"),
    )
    assert bench.target_passages(db, "BGE 150 III 100", "E. 9") == ()


def test_build_queries_from_citations(db: Store) -> None:
    [query] = bench.build_queries(db, n=10)
    assert query.source == "BGE 149 II 200"
    assert query.target == "BGE 150 III 100"
    assert query.pinpoint == "E. 2.1"
    assert query.targets == (passage_id(db, "BGE 150 III 100", "E. 2.1"),)
    assert query.cross_language
    assert "ATF" not in query.text
    assert query.text.startswith("Il suffit qu'elle mentionne")


TRANSLATION = "Der Arbeitgeber kann das Arbeitsverhältnis aus wichtigen Gründen fristlos auflösen."


def test_translation_closes_the_language_gap(db: Store, retriever: Retriever) -> None:
    queries = bench.build_queries(db, n=10)
    translations = {queries[0].id: [TRANSLATION, queries[0].text]}
    report = bench.run(retriever, queries, ["bm25", "bm25+translate"], translations=translations)

    plain = report["modes"]["bm25"]["cross_language"]
    translated = report["modes"]["bm25+translate"]["cross_language"]
    assert plain["decision"]["recall@20"] == 0.0
    assert translated["decision"]["recall@1"] == 1.0
    assert translated["pinpoint"]["recall@1"] == 1.0
    assert report["modes"]["bm25"]["same_language"] == {"decision": {"n": 0}, "pinpoint": {"n": 0}}
    [row] = report["per_query"]
    assert row["ranks"] == {
        "bm25": {"decision": None, "pinpoint": None},
        "bm25+translate": {"decision": 1, "pinpoint": 1},
    }
    assert "bm25+translate" in bench.table(report)


def test_translate_mode_needs_translations(db: Store, retriever: Retriever) -> None:
    with pytest.raises(ValueError, match="needs translations"):
        bench.run(retriever, bench.build_queries(db), ["bm25+translate"])


def test_translations_are_cached(db: Store, tmp_path: Path) -> None:
    queries = bench.build_queries(db)
    cache = tmp_path / "translations.json"
    model = FakeModel(lambda m, name: {"de": TRANSLATION, "fr": m[-1]["content"]})
    first = bench.translate(model, queries, cache, model_name="m1")
    assert first == {queries[0].id: [TRANSLATION, queries[0].text]}
    assert json.loads(cache.read_text(encoding="utf-8")) == {
        "model": "m1",
        "items": {queries[0].text: first[queries[0].id]},
    }

    def unreachable(m: list[dict], name: str) -> dict:
        raise AssertionError("should come from the cache")

    assert bench.translate(FakeModel(unreachable), queries, cache, model_name="m1") == first


def test_translations_by_another_model_are_made_again(db: Store, tmp_path: Path) -> None:
    queries = bench.build_queries(db)
    cache = tmp_path / "translations.json"
    # A cache from an earlier version: no model recorded.
    cache.write_text(json.dumps({queries[0].text: ["alt", "ancien"]}), encoding="utf-8")
    model = FakeModel(lambda m, name: {"de": TRANSLATION, "fr": "fr"})
    assert bench.translate(model, queries, cache, model_name="m2") == {
        queries[0].id: [TRANSLATION, "fr"]
    }
    assert bench.translate(model, queries, cache, model_name="m3") == {
        queries[0].id: [TRANSLATION, "fr"]
    }
    assert model.names() == ["translation", "translation"]


def test_failed_translation_is_empty_and_asked_again(db: Store, tmp_path: Path) -> None:
    def failing(m: list[dict], name: str) -> dict:
        raise LLMError("model down")

    queries = bench.build_queries(db)
    cache = tmp_path / "t.json"
    assert bench.translate(FakeModel(failing), queries, cache) == {queries[0].id: []}
    assert json.loads(cache.read_text(encoding="utf-8"))["items"] == {}  # nothing cached
    model = FakeModel(lambda m, name: {"de": TRANSLATION, "fr": ""})
    # A reply with one language is used, but not cached: the next run asks again.
    assert bench.translate(model, queries, cache) == {queries[0].id: [TRANSLATION]}
    assert json.loads(cache.read_text(encoding="utf-8"))["items"] == {}


def test_level_scores() -> None:
    scores = bench._level([1, 3, None, 12])
    assert scores == {
        "n": 4,
        "recall@1": 0.25,
        "recall@5": 0.5,
        "recall@10": 0.5,
        "recall@20": 0.75,
        "mrr@10": round((1 + 1 / 3) / 4, 3),
    }
    assert bench._level([]) == {"n": 0}


def test_a_translation_is_searched_once(
    db: Store, retriever: Retriever, monkeypatch: pytest.MonkeyPatch
) -> None:
    queries = bench.build_queries(db)
    searched: list[list[str]] = []
    search = retriever.search

    def recording(texts: list[str], *args: object, **kwargs: object) -> list:
        searched.append(list(texts))
        return search(texts, *args, **kwargs)

    monkeypatch.setattr(retriever, "search", recording)
    # The model returned the query itself for its own language, and the same text twice.
    translations = {queries[0].id: [queries[0].text, TRANSLATION, TRANSLATION]}
    bench.run(retriever, queries, ["bm25+translate"], translations=translations)
    assert searched == [[queries[0].text, TRANSLATION]]
