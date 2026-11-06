"""SQLite store for decisions and passages, with an FTS5 index for BM25 search."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .corpus import Decision
from .segment import PageBreaks, Passage, names_another_decision, segment

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE decisions (
    reference TEXT PRIMARY KEY,
    date TEXT NOT NULL,
    year INTEGER NOT NULL,
    language TEXT NOT NULL,
    url TEXT NOT NULL
);
CREATE TABLE passages (
    id INTEGER PRIMARY KEY,
    reference TEXT NOT NULL REFERENCES decisions(reference),
    pinpoint TEXT NOT NULL,
    page INTEGER,
    text TEXT NOT NULL,
    breaks TEXT  -- JSON [[offset, page], ...] where a later page begins, or NULL
);
CREATE INDEX passages_reference ON passages(reference);
CREATE VIRTUAL TABLE passages_fts USING fts5(
    text, content='passages', content_rowid='id',
    tokenize="unicode61 remove_diacritics 2"
);
"""

# Function words in the three official languages the court writes in. They carry
# no signal for BM25 and only slow down the OR query.
STOPWORDS = frozenset(
    """
    aber als also am an auch auf aus bei beim bis da dabei damit dann das dass dem den der des
    dessen die dies diese diesem diesen dieser dieses doch dort du durch ein eine einem einen
    einer eines er es etwa für gegen hat hatte hier ich ihr im in ist ja jedoch kann kein keine
    man mehr mit muss nach nicht noch nur ob oder ohne sein seine sich sie sind so soll sowie
    über um und uns unter vom von vor war wenn wer werden wie wir wird wo zu zum zur zwar
    au aux avec ce ces cette dans des du elle en est et il ils la le les leur lui mais même ne
    ni nous on ou par pas plus pour qu que qui sa se ses son sont sur un une été être
    al alla alle che con da dal dei del della delle di e gli ha il le lo nel nella non per si
    su una uno
    """.split()
)
TOKEN = re.compile(r"\w+", re.UNICODE)
MAX_QUERY_TERMS = 40
# Statutes with a two-letter abbreviation. Kept in a query when written in capitals
# ("Art. 97 CO"), so that the English "or" and the "a" of "lit. a" are not.
ACTS = frozenset("OR BV CC CO CP LP IV UV EO".split())
# Raised whenever the tables or the cutting of decisions into passages change, so an
# index built by an earlier version is rebuilt instead of failing on a missing column
# or serving passages cut the old way.
SCHEMA_VERSION = "4"


class IndexProblem(Exception):
    """The index cannot be built or used; the message says what to do."""


@dataclass(frozen=True, slots=True)
class StoredPassage:
    id: int
    reference: str
    pinpoint: str
    page: int | None
    text: str
    language: str
    year: int
    breaks: PageBreaks = ()

    def as_passage(self) -> Passage:
        return Passage(self.reference, self.pinpoint, self.page, self.text, self.breaks)

    @property
    def citation(self) -> str:
        return self.as_passage().citation

    def citation_of_span(self, start: int, end: int) -> str:
        return self.as_passage().citation_of_span(start, end)


def build(
    path: Path,
    decisions: Iterable[Decision],
    meta: dict[str, str],
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[int, int]:
    """Segment `decisions` into a fresh store at `path`. Returns (decisions, passages).

    The index at `path` is replaced only by one that holds at least one decision.
    """
    tmp = path.with_name(path.name + ".tmp")
    store = Store.create(tmp)
    decision_count = passage_count = 0
    try:
        for decision in decisions:
            if names_another_decision(decision.reference, decision.text):
                continue
            passages = segment(decision.reference, decision.text)
            if not passages:
                continue
            passage_count += store.add_decision(
                decision.reference, decision.date, decision.language, decision.url, passages
            )
            decision_count += 1
            if on_progress:
                on_progress(decision_count, passage_count)
        store.finish(
            {
                **meta,
                "decisions": str(decision_count),
                "passages": str(passage_count),
                "schema": SCHEMA_VERSION,
                # Vectors record the build they were embedded from, so stale ones are noticed.
                "build_id": uuid.uuid4().hex,
            }
        )
    finally:
        store.close()
    if not decision_count:
        tmp.unlink(missing_ok=True)
        raise IndexProblem("No decision in the corpus qualifies; the old index is kept.")
    try:
        tmp.replace(path)
    except PermissionError:  # Windows: another devil session has the index open
        tmp.unlink(missing_ok=True)
        raise IndexProblem(
            "The index is open in another devil session. Close it and run `devil index` again."
        ) from None
    return decision_count, passage_count


def _stored(row: sqlite3.Row) -> StoredPassage:
    values = dict(row)
    breaks = values.pop("breaks", None)
    pages = tuple((int(at), int(page)) for at, page in json.loads(breaks)) if breaks else ()
    return StoredPassage(**values, breaks=pages)


def fts_query(text: str) -> str:
    """Turn free text into an FTS5 OR query of its content words.

    Article numbers and two-letter statutes stay in. "ß" becomes "ss", as the
    court writes it.
    """
    terms: list[str] = []
    for written in TOKEN.findall(text):
        token = written.casefold()
        if token in STOPWORDS or token in terms:
            continue
        if len(token) < 3 and not token.isdigit() and written not in ACTS:
            continue
        terms.append(token)
    return " OR ".join(f'"{term}"' for term in terms[:MAX_QUERY_TERMS])


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row

    @classmethod
    def create(cls, path: Path) -> Store:
        path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("", "-wal", "-shm"):
            Path(str(path) + suffix).unlink(missing_ok=True)
        store = cls(path)
        store.db.executescript(SCHEMA)
        return store

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writing -------------------------------------------------------------

    def add_decision(
        self, reference: str, date: str, language: str, url: str, passages: Iterable[Passage]
    ) -> int:
        year = int(date[:4]) if date[:4].isdigit() else 0
        self.db.execute(
            "INSERT OR REPLACE INTO decisions VALUES (?, ?, ?, ?, ?)",
            (reference, date, year, language, url),
        )
        rows = [
            (reference, p.pinpoint, p.page, p.text, json.dumps(p.breaks) if p.breaks else None)
            for p in passages
        ]
        self.db.executemany(
            "INSERT INTO passages (reference, pinpoint, page, text, breaks) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        return len(rows)

    def finish(self, meta: dict[str, str]) -> None:
        """Build the full-text index and record how the store was made."""
        self.db.execute("INSERT INTO passages_fts(passages_fts) VALUES ('rebuild')")
        self.db.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)", meta.items())
        self.db.commit()
        self.db.execute("INSERT INTO passages_fts(passages_fts) VALUES ('optimize')")
        self.db.commit()

    # -- reading -------------------------------------------------------------

    def meta(self) -> dict[str, str]:
        return {row["key"]: row["value"] for row in self.db.execute("SELECT * FROM meta")}

    def check(self) -> None:
        """Raise IndexProblem unless this index was built by this version and is not empty."""
        try:
            meta = self.meta()
        except sqlite3.DatabaseError:
            meta = {}
        if meta.get("schema") != SCHEMA_VERSION:
            raise IndexProblem(
                "The index was built by another version of devils-advocate. "
                "Run `devil index` again."
            )
        if meta.get("passages") in (None, "0"):
            raise IndexProblem("The index is empty. Run `devil index` again.")

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM passages").fetchone()[0]

    def decision_count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]

    def get(self, ids: Iterable[int]) -> list[StoredPassage]:
        ids = list(ids)
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        rows = self.db.execute(
            "SELECT p.id, p.reference, p.pinpoint, p.page, p.text, p.breaks, d.language, d.year "
            f"FROM passages p JOIN decisions d USING (reference) WHERE p.id IN ({marks})",
            ids,
        ).fetchall()
        by_id = {row["id"]: _stored(row) for row in rows}
        return [by_id[i] for i in ids if i in by_id]

    def passages_of(self, reference: str) -> list[StoredPassage]:
        rows = self.db.execute(
            "SELECT p.id FROM passages p WHERE p.reference = ? ORDER BY p.id", (reference,)
        ).fetchall()
        return self.get(row["id"] for row in rows)

    def has_decision(self, reference: str) -> bool:
        row = self.db.execute("SELECT 1 FROM decisions WHERE reference = ?", (reference,))
        return row.fetchone() is not None

    def iter_texts(self, start: int = 0, batch: int = 1000) -> Iterable[tuple[int, str]]:
        """Yield (id, text) in id order from the `start`-th passage on."""
        last = -1
        if start:
            row = self.db.execute(
                "SELECT id FROM passages ORDER BY id LIMIT 1 OFFSET ?", (start - 1,)
            ).fetchone()
            if row is None:
                return
            last = row["id"]
        while True:
            rows = self.db.execute(
                "SELECT id, text FROM passages WHERE id > ? ORDER BY id LIMIT ?", (last, batch)
            ).fetchall()
            if not rows:
                return
            for row in rows:
                yield row["id"], row["text"]
            last = rows[-1]["id"]

    def ids(self) -> list[int]:
        return [row[0] for row in self.db.execute("SELECT id FROM passages ORDER BY id")]

    def ids_of(self, references: Iterable[str]) -> list[int]:
        references = list(references)
        if not references:
            return []
        marks = ",".join("?" * len(references))
        rows = self.db.execute(
            f"SELECT id FROM passages WHERE reference IN ({marks}) ORDER BY id", references
        )
        return [row[0] for row in rows]

    def bm25(self, query: str, k: int, exclude: frozenset[str] = frozenset()) -> list[int]:
        match = fts_query(query)
        if not match:
            return []
        marks = ",".join("?" * len(exclude))
        rows = self.db.execute(
            "SELECT f.rowid FROM passages_fts f JOIN passages p ON p.id = f.rowid "
            f"WHERE passages_fts MATCH ? AND p.reference NOT IN ({marks}) "
            "ORDER BY bm25(passages_fts) LIMIT ?",
            (match, *exclude, k),
        ).fetchall()
        return [row[0] for row in rows]
