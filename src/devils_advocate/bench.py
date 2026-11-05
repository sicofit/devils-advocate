"""Retrieval benchmark built from the court's own citations.

When the Federal Supreme Court writes "Nach der Rechtsprechung ... (BGE 137 IV 105
E. 2.3.3)", it states a proposition and names the Erwägung that supports it. That
gives a labelled retrieval task without anyone writing labels: the sentence with
its citations removed is the query, the cited decision and Erwägung are the
answer. The citing decision itself is excluded from the results, because it
contains the sentence word for word.

Two levels are scored:

* decision: is the cited BGE among the first k distinct decisions retrieved?
* pinpoint: is a passage from the cited Erwägung among the first k passages?

The task is harder than what the devil does in a conversation (one query
instead of a claim plus German and French reformulations) and noisier (the court
often cites a decision for a side point), so absolute numbers understate the
pipeline. They are for comparing retrievers and catching regressions.

A mode with the suffix "+translate" ("bm25+translate") also searches with a
German and a French translation of each query, written by the chat model, and
fuses the rankings. That isolates what the bilingual queries of the analysis
step contribute.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from . import prompts
from .client import ChatModel, LLMError
from .search import Retriever
from .segment import ends_with_abbreviation
from .store import Store

CITATION = re.compile(
    r"\b(?:BGE|ATF|DTF)\s+(\d{2,3})\s+([IVX]+[a-z]?)\s+(\d{1,4})"
    r"(?:\s+(?:E\.|Erw\.|consid\.|c\.)\s*(\d{1,2}(?:\.\d{1,2})*(?:[a-z]{1,2})?(?:/[a-z]{1,3})?))?"
    r"(?:\s+(?:S\.|p\.|pag\.)\s*\d+(?:\s*(?:f\.|ff\.|s\.|ss\.|seg\.|segg\.))?)?"
)
# A parenthetical that holds at least one BGE citation,
# e.g. "(vgl. BGE 140 III 86 E. 4.1 mit Hinweisen)".
CITING_PARENTHETICAL = re.compile(r"\([^()]*\b(?:BGE|ATF|DTF)\s+\d[^()]*\)")
WORD = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
MIN_QUERY_WORDS = 8
MAX_QUERY_CHARS = 600
KS = (1, 5, 10, 20)
DEPTH = 100
SAVE_EVERY = 25


@dataclass(frozen=True, slots=True)
class Query:
    id: str
    source: str  # the citing decision, excluded from retrieval
    text: str  # the citing sentence without its citations
    target: str  # the cited decision
    pinpoint: str | None  # the cited Erwägung, e.g. "E. 2.3.3"
    targets: tuple[int, ...]  # passage ids of that Erwägung in the index
    cross_language: bool  # citing and cited decision are written in different languages


def sentences(text: str) -> Iterable[str]:
    """Split a passage into sentences.

    Breaks inside parentheses and after abbreviations ("vgl.", "E.") do not count.
    """
    start = depth = 0
    for i, char in enumerate(text):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth = max(0, depth - 1)
        elif depth == 0 and (char == "\n" or (char in ".!?" and _ends_sentence(text, start, i))):
            sentence = text[start : i + 1].strip()
            if sentence:
                yield sentence
            start = i + 1
    rest = text[start:].strip()
    if rest:
        yield rest


def _ends_sentence(text: str, start: int, i: int) -> bool:
    following = text[i + 1 : i + 3]
    if not following.startswith((" ", "\n")) or not following[1:2].isupper():
        return False
    return not ends_with_abbreviation(text[start : i + 1])


def strip_citations(sentence: str) -> str:
    previous = None
    while previous != sentence:
        previous = sentence
        sentence = CITING_PARENTHETICAL.sub("", sentence)
    sentence = CITATION.sub("", sentence)
    return re.sub(r"\s+([,.;:])", r"\1", re.sub(r"\s{2,}", " ", sentence)).strip()


def pinpoint_parts(pinpoint: str) -> tuple[str, ...]:
    """'E. 3b/aa' -> ('3', 'b', 'aa'); 'E. 2.3.3' -> ('2', '3', '3')."""
    body = pinpoint.removeprefix("E.").strip()
    return tuple(re.findall(r"\d+|[a-z]+", body))


def target_passages(store: Store, reference: str, pinpoint: str) -> tuple[int, ...]:
    """Passages that belong to the cited Erwägung, or to its closest indexed ancestor."""
    cited = pinpoint_parts(pinpoint)
    if not cited:
        return ()
    passages = [p for p in store.passages_of(reference) if p.pinpoint.startswith("E.")]
    inside = [p.id for p in passages if pinpoint_parts(p.pinpoint)[: len(cited)] == cited]
    if inside:
        return tuple(inside)
    ancestors = [
        (len(parts), p.id)
        for p in passages
        if (parts := pinpoint_parts(p.pinpoint)) and cited[: len(parts)] == parts
    ]
    if not ancestors:
        return ()
    deepest = max(depth for depth, _ in ancestors)
    return tuple(pid for depth, pid in ancestors if depth == deepest)


def build_queries(store: Store, *, n: int = 1000, seed: int = 7) -> list[Query]:
    """Sample `n` citing sentences, at most one per (citing, cited) decision pair."""
    languages = dict(store.db.execute("SELECT reference, language FROM decisions"))
    candidates: dict[tuple[str, str], list[tuple[str, str | None]]] = {}
    rows = store.db.execute(
        "SELECT reference, text FROM passages WHERE pinpoint != 'Regeste' ORDER BY id"
    )
    for source, text in rows:
        if not CITATION.search(text):
            continue
        for sentence in sentences(text):
            cited = [(f"BGE {m[1]} {m[2]} {m[3]}", m[4]) for m in CITATION.finditer(sentence)]
            if not cited:
                continue
            query = strip_citations(sentence)
            if len(WORD.findall(query)) < MIN_QUERY_WORDS or len(query) > MAX_QUERY_CHARS:
                continue
            for target, pin in cited:
                if target != source and target in languages:
                    candidates.setdefault((source, target), []).append((query, pin))

    rng = random.Random(seed)
    pairs = sorted(candidates)
    rng.shuffle(pairs)
    queries: list[Query] = []
    for source, target in pairs:
        if len(queries) == n:
            break
        options = candidates[(source, target)]
        text, pin = options[rng.randrange(len(options))]
        pinpoint = f"E. {pin}" if pin else None
        queries.append(
            Query(
                id=f"q{len(queries) + 1:04d}",
                source=source,
                text=text,
                target=target,
                pinpoint=pinpoint,
                targets=target_passages(store, target, pinpoint) if pinpoint else (),
                cross_language=languages[source] != languages[target],
            )
        )
    return queries


def translate(
    model: ChatModel,
    queries: Sequence[Query],
    cache: Path,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, list[str]]:
    """German and French versions of every query, cached in `cache` by query text.

    The cache is saved every `SAVE_EVERY` new translations, so an interrupted
    run loses little.
    """
    known: dict[str, list[str]] = (
        json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    )

    def save() -> None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(known, ensure_ascii=False, indent=1), encoding="utf-8")

    new = 0
    try:
        for i, query in enumerate(queries, 1):
            if query.text not in known:
                new += 1
                if new % SAVE_EVERY == 0:
                    save()
                try:
                    reply = model.json(
                        prompts.translate_messages(query.text),
                        prompts.TRANSLATE_SCHEMA,
                        name="translation",
                    )
                except LLMError:
                    reply = None  # not cached: the next run asks again
                if reply is not None:
                    known[query.text] = [
                        str(reply.get(lang, "")).strip() for lang in ("de", "fr") if reply.get(lang)
                    ]
            if on_progress:
                on_progress(i, len(queries))
    finally:
        save()
    return {q.id: known.get(q.text, []) for q in queries}


Ranks = tuple[int | None, int | None, bool]  # decision rank, pinpoint rank, pinpoint known


def run(
    retriever: Retriever,
    queries: Sequence[Query],
    modes: Sequence[str],
    on_progress: Callable[[str, int, int], None] | None = None,
    translations: dict[str, list[str]] | None = None,
) -> dict:
    """Score every mode on every query. Returns a JSON-ready report."""
    report: dict = {"queries": len(queries), "modes": {}, "per_query": []}
    ranks: dict[str, list[Ranks]] = {}
    for mode in modes:
        base, _, extra = mode.partition("+")
        if extra and (extra != "translate" or translations is None):
            raise ValueError(f"mode {mode!r} needs translations")
        started = time.perf_counter()
        mode_ranks: list[Ranks] = []
        for i, query in enumerate(queries, 1):
            texts = [query.text]
            if extra:
                texts += [t for t in translations.get(query.id, []) if t != query.text]
            hits = retriever.search(
                texts, DEPTH, mode=base, exclude=frozenset({query.source}), depth=DEPTH
            )
            mode_ranks.append(_ranks(query, [(h.passage.id, h.passage.reference) for h in hits]))
            if on_progress:
                on_progress(mode, i, len(queries))
        ranks[mode] = mode_ranks
        seconds = time.perf_counter() - started
        same = [r for r, q in zip(mode_ranks, queries, strict=True) if not q.cross_language]
        cross = [r for r, q in zip(mode_ranks, queries, strict=True) if q.cross_language]
        report["modes"][mode] = {
            "all": _scores(mode_ranks),
            "same_language": _scores(same),
            "cross_language": _scores(cross),
            "ms_per_query": round(1000 * seconds / max(1, len(queries)), 1),
        }
    for i, query in enumerate(queries):
        row = {k: v for k, v in asdict(query).items() if k != "targets"}
        row["ranks"] = {
            mode: {"decision": r[i][0], "pinpoint": r[i][1]} for mode, r in ranks.items()
        }
        report["per_query"].append(row)
    return report


def _ranks(query: Query, hits: list[tuple[int, str]]) -> Ranks:
    """1-based rank of the cited decision among distinct decisions, and of the cited Erwägung."""
    decisions = list(dict.fromkeys(reference for _, reference in hits))
    decision_rank = decisions.index(query.target) + 1 if query.target in decisions else None
    wanted = set(query.targets)
    pinpoint_rank = next((i for i, (pid, _) in enumerate(hits, 1) if pid in wanted), None)
    return decision_rank, pinpoint_rank, bool(wanted)


def _scores(ranks: list[Ranks]) -> dict:
    return {
        "decision": _level([d for d, _, _ in ranks]),
        # Only queries whose cited Erwägung could be located in the index.
        "pinpoint": _level([p for _, p, known in ranks if known]),
    }


def _level(values: list[int | None]) -> dict:
    if not values:
        return {"n": 0}
    out: dict = {"n": len(values)}
    for k in KS:
        out[f"recall@{k}"] = round(sum(1 for r in values if r and r <= k) / len(values), 3)
    out["mrr@10"] = round(sum(1 / r for r in values if r and r <= 10) / len(values), 3)
    return out


def table(report: dict) -> str:
    """Plain-text summary for the terminal and the README."""
    lines = [f"{report['queries']} citation queries"]
    header = (
        f"{'mode':<16} {'subset':<15} {'level':<9} {'n':>5} "
        + " ".join(f"{'R@' + str(k):>6}" for k in KS)
        + f" {'MRR@10':>7}"
    )
    lines.append(header)
    for mode, result in report["modes"].items():
        for subset in ("all", "same_language", "cross_language"):
            for level in ("decision", "pinpoint"):
                scores = result[subset][level]
                if not scores.get("n"):
                    continue
                lines.append(
                    f"{mode:<16} {subset:<15} {level:<9} {scores['n']:>5} "
                    + " ".join(f"{scores[f'recall@{k}']:>6.3f}" for k in KS)
                    + f" {scores['mrr@10']:>7.3f}"
                )
        lines.append(f"{mode:<16} {result['ms_per_query']} ms per query")
    return "\n".join(lines)
